#!/usr/bin/env python3
"""
Score hypoglycemia risk from saved models.

Train first: python -m modeling.train
Not for clinical use.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.artifacts import load_xgb_artifact
from modeling.config import ARTIFACTS_DIR, HORIZON_H, HYPO_THRESHOLD
from modeling.features import build_feature_matrices
from modeling.windows import build_window_table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import get_dataset_config, load_dataset_config, sync_audit_dataset_root

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DISCLAIMER = (
    "Not for clinical use. Research and education only."
)

TARGET_HYPO_DEFINITION = (
    f"Any historical CGM reading below {HYPO_THRESHOLD} mg/dL in the "
    f"{HORIZON_H} hours after prediction_time"
)


def _target_hypo_label(value: int) -> str:
    if value == 1:
        return f"yes: hypoglycemia (<{HYPO_THRESHOLD} mg/dL) in next {HORIZON_H}h"
    return f"no: no hypoglycemia (<{HYPO_THRESHOLD} mg/dL) in next {HORIZON_H}h"


def _load_manifest(artifacts_dir: Path) -> dict:
    path = artifacts_dir / "artifact_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m modeling.train"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def score_windows(
    windows: pd.DataFrame,
    *,
    artifacts_dir: Path,
    models: list[str],
) -> pd.DataFrame:
    if windows.empty:
        raise ValueError("No eligible prediction windows for the requested input.")

    dense_df, sparse_df, sequences, _, _ = build_feature_matrices(windows)
    out = windows[
        ["participant_id", "prediction_time", "has_prior_scan", "scan_count_6h"]
    ].copy()
    if "target_hypo_2h" in windows.columns:
        out["target_hypo_2h"] = windows["target_hypo_2h"].values
        out["target_hypo_2h_label"] = [
            _target_hypo_label(int(v)) for v in windows["target_hypo_2h"].values
        ]
        out["target_hypo_2h_definition"] = TARGET_HYPO_DEFINITION

    manifest = _load_manifest(artifacts_dir)
    out["horizon_hours"] = manifest.get("horizon_hours", 2)
    out["disclaimer"] = DISCLAIMER

    if "dense_xgb" in models:
        art = load_xgb_artifact(artifacts_dir / "dense_xgb.joblib")
        prob, alert = art.predict_alert(dense_df)
        out["risk_dense"] = prob
        out["alert_dense"] = alert
        out["threshold_dense"] = art.threshold

    if "sparse_xgb" in models:
        art = load_xgb_artifact(artifacts_dir / "sparse_xgb.joblib")
        prob, alert = art.predict_alert(sparse_df)
        out["risk_sparse"] = prob
        out["alert_sparse"] = alert
        out["threshold_sparse"] = art.threshold

    if "dense_gru" in models:
        gru_path = artifacts_dir / "dense_gru.pt"
        if not gru_path.exists():
            log.warning("dense_gru artifact not found; train without --skip-gru")
        else:
            from modeling.gru_model import load_gru_artifact, predict_gru_proba

            gru = load_gru_artifact(gru_path)
            prob = predict_gru_proba(gru, sequences)
            out["risk_dense_gru"] = prob
            out["alert_dense_gru"] = (prob >= gru.threshold).astype(int)
            out["threshold_dense_gru"] = gru.threshold

    return out


def _display_columns(scores: pd.DataFrame) -> list[str]:
    """Columns for terminal output: ids, label, then model risks and alerts."""
    lead = ["participant_id", "prediction_time"]
    if "target_hypo_2h" in scores.columns:
        lead.extend(["target_hypo_2h", "target_hypo_2h_label"])
    model_cols = [c for c in scores.columns if c.startswith(("risk_", "alert_"))]
    return lead + model_cols


def _log_alert_label_summary(scores: pd.DataFrame) -> None:
    if "target_hypo_2h" not in scores.columns:
        return
    y = scores["target_hypo_2h"].astype(int)
    for col in scores.columns:
        if not col.startswith("alert_"):
            continue
        pred = scores[col].astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        n_alert = int(pred.sum())
        log.info(
            "%s: %d alerts (%d true positive, %d false positive, %d missed hypo)",
            col,
            n_alert,
            tp,
            fp,
            fn,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score hypoglycemia risk from saved Glucose Gap artifacts"
    )
    parser.add_argument(
        "--participant",
        required=True,
        help="Participant ID (folder name under dataset root)",
    )
    parser.add_argument(
        "--at",
        default=None,
        help="Optional single prediction timestamp (ISO format). Default: all eligible windows.",
    )
    parser.add_argument(
        "--models",
        default="dense,sparse",
        help="Comma-separated: dense, sparse, gru (default: dense,sparse)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "modeling_outputs" / "artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for alerts (default: print summary to stdout)",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Path to dataset_config.json (default: dataset_config.hupa.json)",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Override dataset root directory from config",
    )
    args = parser.parse_args()

    load_dataset_config(args.dataset_config)
    if args.dataset_root:
        from dataset import DatasetConfig, set_dataset_config

        cfg = get_dataset_config()
        root = Path(args.dataset_root)
        if not root.is_absolute():
            root = (Path(__file__).resolve().parent.parent / root).resolve()
        set_dataset_config(
            DatasetConfig(
                layout=cfg.layout,
                root=root,
                description=cfg.description,
                hupa_ucm=cfg.hupa_ucm,
                canonical=cfg.canonical,
                cohort=cfg.cohort,
                config_path=cfg.config_path,
            )
        )
        sync_audit_dataset_root(root)

    log.info(
        "Dataset layout=%s root=%s",
        get_dataset_config().layout,
        get_dataset_config().root,
    )

    model_map = {
        "dense": "dense_xgb",
        "sparse": "sparse_xgb",
        "gru": "dense_gru",
    }
    models = []
    for token in args.models.split(","):
        key = token.strip().lower()
        if key not in model_map:
            parser.error(f"Unknown model '{token}'. Use: dense, sparse, gru")
        models.append(model_map[key])

    windows = build_window_table([args.participant])
    if windows.empty:
        log.error("No windows for participant %s. Check dataset path and CGM coverage.", args.participant)
        return 1

    if args.at:
        at = pd.Timestamp(args.at)
        windows = windows[windows["prediction_time"] == at]
        if windows.empty:
            log.error("No eligible window at %s for %s", at, args.participant)
            return 1

    try:
        scores = score_windows(windows, artifacts_dir=args.artifacts_dir, models=models)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.output, index=False)
        log.info("Wrote %d alert rows to %s", len(scores), args.output)
    else:
        if "target_hypo_2h_definition" in scores.columns:
            print(f"Label (target_hypo_2h): {scores['target_hypo_2h_definition'].iloc[0]}\n")
        print(scores[_display_columns(scores)].to_string(index=False))
        print(f"\n{DISCLAIMER}")

    _log_alert_label_summary(scores)

    alerts = 0
    for col in scores.columns:
        if col.startswith("alert_"):
            alerts += int(scores[col].sum())
    log.info(
        "Scored %d windows for %s (%d alert flags across selected models)",
        len(scores),
        args.participant,
        alerts,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
