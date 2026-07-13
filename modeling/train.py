#!/usr/bin/env python3
"""
Modeling pipeline: paired dense vs sparse hypoglycemia prediction.

Dense and sparse models share the same participants, timestamps, labels, and
CV folds. Folds are assigned once; preprocessing and thresholds are fit on
training data only.

Writes CSV/NPZ outputs under modeling_outputs/.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feasibility_audit"))
from data_audit import compute_cohort_counts, inventory_dataset  # noqa: E402
from data_audit import audit_participant_glucose, load_participant_freestyle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import (
    common_cohort_ids,
    discover_participants,
    get_dataset_config,
    load_dataset_config,
    sync_audit_dataset_root,
)
from modeling.baselines import (
    compute_baselines,
    save_comparison_figure,
    sparse_probability_direction_check,
)
from modeling.config import (
    INNER_VAL_FRACTION,
    N_BOOTSTRAP,
    N_FOLDS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from modeling.cv_splits import get_or_create_folds, window_fold_column
from modeling.features import build_feature_matrices
from modeling.metrics import (
    bootstrap_participant_difference,
    compute_metrics,
    tune_threshold,
)
from modeling.windows import build_window_table, participant_summaries

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _get_common_cohort() -> list[str]:
    cfg = get_dataset_config()
    if cfg.layout == "hupa_ucm":
        participants = discover_participants(cfg)
        inventory = inventory_dataset(participants)
        rows = [audit_participant_glucose(pid, load_participant_freestyle(pid)) for pid in participants]
        glucose_summary = pd.DataFrame(rows)
        cohort = compute_cohort_counts(inventory, glucose_summary)
        return cohort["common_cohort_ids"]
    return common_cohort_ids(cfg)


def _new_xgb(scale_pos_weight: float):
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def run_xgb_cv(
    name: str,
    X_df: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Grouped CV with per-fold imputation and validation-only threshold tuning.

    Returns (oof_prob, oof_threshold_per_row, fold_metrics_df). Rows outside the
    active mask are left as NaN. No held-out information touches preprocessing,
    model fitting, or threshold selection.
    """
    active = np.ones(len(y), dtype=bool) if mask is None else mask.astype(bool)
    oof = np.full(len(y), np.nan)
    oof_thr = np.full(len(y), np.nan)
    fold_rows = []

    for fold in range(N_FOLDS):
        tr_idx = np.where((folds != fold) & active)[0]
        va_idx = np.where((folds == fold) & active)[0]
        if len(va_idx) == 0 or len(tr_idx) == 0:
            continue

        # Inner validation split of the training folds for threshold tuning.
        y_tr_full = y[tr_idx]
        stratify = y_tr_full if len(np.unique(y_tr_full)) > 1 else None
        core_idx, inner_idx = train_test_split(
            tr_idx,
            test_size=INNER_VAL_FRACTION,
            random_state=RANDOM_SEED,
            stratify=stratify,
        )

        imp = SimpleImputer(strategy="median").fit(X_df.iloc[core_idx].values)
        X_core = imp.transform(X_df.iloc[core_idx].values)
        X_inner = imp.transform(X_df.iloc[inner_idx].values)

        pos = max(y[core_idx].sum(), 1)
        neg = max(len(core_idx) - y[core_idx].sum(), 1)
        model = _new_xgb(neg / pos)
        model.fit(X_core, y[core_idx])
        inner_prob = model.predict_proba(X_inner)[:, 1]
        thr = tune_threshold(y[inner_idx], inner_prob)

        # Refit on the full training folds (preprocessing re-fit on full train).
        imp_full = SimpleImputer(strategy="median").fit(X_df.iloc[tr_idx].values)
        X_tr = imp_full.transform(X_df.iloc[tr_idx].values)
        X_va = imp_full.transform(X_df.iloc[va_idx].values)
        pos = max(y[tr_idx].sum(), 1)
        neg = max(len(tr_idx) - y[tr_idx].sum(), 1)
        model_full = _new_xgb(neg / pos)
        model_full.fit(X_tr, y[tr_idx])

        oof[va_idx] = model_full.predict_proba(X_va)[:, 1]
        oof_thr[va_idx] = thr

        m = compute_metrics(y[va_idx], oof[va_idx], threshold=thr)
        m["fold"], m["model"] = fold, name
        fold_rows.append(m)

    return oof, oof_thr, pd.DataFrame(fold_rows)


def pooled_metrics(y: np.ndarray, prob: np.ndarray, thr: np.ndarray, mask: np.ndarray | None = None):
    """Pooled OOF metrics: threshold-free AUPRC/AUROC and per-row threshold classification."""
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    idx = np.where(mask)[0] if mask is not None else np.where(~np.isnan(prob))[0]
    y_s, p_s, t_s = y[idx], prob[idx], thr[idx]
    out = compute_metrics(y_s, p_s, threshold=0.5)  # AUPRC/AUROC are threshold-free
    y_pred = (p_s >= t_s).astype(int)

    if len(np.unique(y_s)) >= 2:
        tn, fp, fn, tp = confusion_matrix(y_s, y_pred, labels=[0, 1]).ravel()
        out["recall"] = float(recall_score(y_s, y_pred, zero_division=0))
        out["precision"] = float(precision_score(y_s, y_pred, zero_division=0))
        out["f1"] = float(f1_score(y_s, y_pred, zero_division=0))
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
        out["tp"], out["fp"], out["fn"], out["tn"] = float(tp), float(fp), float(fn), float(tn)
    out["mean_threshold"] = float(np.nanmean(t_s)) if len(t_s) else np.nan
    return out


def save_shap(name: str, X_df: pd.DataFrame, y: np.ndarray, feature_names: list[str], fig_dir: Path):
    import shap
    import xgboost as xgb

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_df.values)
    pos = max(y.sum(), 1)
    neg = max(len(y) - y.sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=neg / pos, random_state=RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    sample = X if len(X) <= 500 else X[:500]
    shap_vals = explainer.shap_values(sample)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / f"shap_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def participant_metrics(meta: pd.DataFrame, y: np.ndarray, prob: np.ndarray, thr: np.ndarray,
                        model_name: str) -> pd.DataFrame:
    rows = []
    for pid, grp in meta.groupby("participant_id"):
        idx = grp.index.values
        valid = idx[~np.isnan(prob[idx])]
        if len(valid) == 0:
            continue
        m = compute_metrics(y[valid], prob[valid], threshold=float(np.nanmean(thr[valid])))
        m["participant_id"], m["model"] = pid, model_name
        rows.append(m)
    return pd.DataFrame(rows)


def main(skip_gru: bool = False, dataset_config: str | None = None, dataset_root: str | None = None) -> None:
    load_dataset_config(dataset_config)
    if dataset_root:
        from dataset import DatasetConfig, set_dataset_config

        cfg = get_dataset_config()
        root = Path(dataset_root)
        if not root.is_absolute():
            root = (PROJECT_ROOT / root).resolve()
        cfg = DatasetConfig(
            layout=cfg.layout,
            root=root,
            description=cfg.description,
            hupa_ucm=cfg.hupa_ucm,
            canonical=cfg.canonical,
            cohort=cfg.cohort,
            config_path=cfg.config_path,
        )
        set_dataset_config(cfg)
        sync_audit_dataset_root(cfg.root)

    cfg = get_dataset_config()
    log.info("Dataset layout=%s root=%s", cfg.layout, cfg.root)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. Master paired window table (single source of truth).
    common_ids = _get_common_cohort()
    log.info("Common cohort: %d participants", len(common_ids))
    windows = build_window_table(common_ids)
    log.info("Windows: %d eligible, %d positive (%.1f%%)", len(windows),
             int(windows["target_hypo_2h"].sum()), 100 * windows["target_hypo_2h"].mean())

    # 5. Event-aware folds assigned ONCE and reused unchanged.
    summary = participant_summaries(windows)
    summary.to_csv(OUTPUT_DIR / "participant_fold_summary.csv", index=False)
    participant_folds = get_or_create_folds(summary, OUTPUT_DIR / "fold_assignments.csv")
    windows["fold_id"] = window_fold_column(windows, participant_folds)
    windows.to_csv(OUTPUT_DIR / "paired_windows.csv", index=False)

    y = windows["target_hypo_2h"].values.astype(int)
    folds = windows["fold_id"].values
    meta = windows[["window_id", "participant_id", "prediction_time", "target_hypo_2h",
                    "fold_id", "has_prior_scan", "scan_count_6h"]].reset_index(drop=True)

    # 2-4. Features and sequences, joined without changing row order.
    log.info("Building features...")
    dense_df, sparse_df, sequences, dense_cols, sparse_cols = build_feature_matrices(windows)
    dense_df.to_csv(OUTPUT_DIR / "dense_features.csv", index=False)
    sparse_df.to_csv(OUTPUT_DIR / "sparse_features.csv", index=False)
    np.savez_compressed(OUTPUT_DIR / "dense_sequences.npz", sequences=sequences,
                        window_id=windows["window_id"].values, y=y, fold_id=folds)

    baseline_rows, baseline_probs = compute_baselines(y, folds, dense_df, sparse_df)
    model_metric_rows: list[dict] = list(baseline_rows)
    fold_metric_frames: list[pd.DataFrame] = []
    oof_store: dict[str, np.ndarray] = dict(baseline_probs)
    thr_store: dict[str, np.ndarray] = {
        name: np.full(len(y), 0.5) for name in baseline_probs
    }

    def _record(name, oof, thr, fold_df, mask=None):
        oof_store[name] = oof
        thr_store[name] = thr
        pooled = pooled_metrics(y, oof, thr, mask=mask)
        pooled["model"] = name
        pooled["n_active"] = float(mask.sum()) if mask is not None else float((~np.isnan(oof)).sum())
        model_metric_rows.append(pooled)
        if len(fold_df):
            fold_df = fold_df.copy()
            fold_df["model"] = name
            fold_metric_frames.append(fold_df)
        return pooled

    # 6-7. Experiment 1: dense vs sparse XGBoost (paired).
    log.info("Experiment 1: dense XGBoost")
    d_oof, d_thr, d_fold = run_xgb_cv("dense_xgb", dense_df, y, folds)
    r_dense = _record("dense_xgb", d_oof, d_thr, d_fold)
    save_shap("dense_xgb", dense_df, y, dense_cols, fig_dir)

    log.info("Experiment 1: sparse XGBoost")
    s_oof, s_thr, s_fold = run_xgb_cv("sparse_xgb", sparse_df, y, folds)
    r_sparse = _record("sparse_xgb", s_oof, s_thr, s_fold)
    direction = sparse_probability_direction_check(y, s_oof)
    log.info(
        "Sparse OOF mean prob: positive=%.3f negative=%.3f (AUROC=%.3f)",
        direction["mean_prob_positive"],
        direction["mean_prob_negative"],
        r_sparse["auroc"],
    )
    if r_sparse["auroc"] < 0.5:
        log.warning(
            "Sparse OOF AUROC is below 0.5. Checked alignment and class encoding; "
            "compare against baseline_latest_scan in model_metrics.csv."
        )
    save_shap("sparse_xgb", sparse_df, y, sparse_cols, fig_dir)

    # 8. Sparse stratified by scan availability (reuses same sparse OOF predictions).
    has_scan = windows["has_prior_scan"].values.astype(bool)
    for label, m in [("sparse_all", np.ones(len(y), bool)),
                     ("sparse_has_scan", has_scan), ("sparse_no_scan", ~has_scan)]:
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            continue
        pooled = pooled_metrics(y, s_oof, s_thr, mask=m)
        pooled["model"] = label
        pooled["n_active"] = float(m.sum())
        model_metric_rows.append(pooled)

    # 8. Sensitivity: exclude the two dominant participants (20-participant cohort).
    sens_mask = ~windows["participant_id"].isin(get_dataset_config().sensitivity_exclude).values
    if sens_mask.sum() > 0:
        log.info("Sensitivity: exclude %s", get_dataset_config().sensitivity_exclude)
        sd_oof, sd_thr, sd_fold = run_xgb_cv("sens_dense_xgb", dense_df, y, folds, mask=sens_mask)
        _record("sens_dense_xgb", sd_oof, sd_thr, sd_fold, mask=sens_mask)
        ss_oof, ss_thr, ss_fold = run_xgb_cv("sens_sparse_xgb", sparse_df, y, folds, mask=sens_mask)
        _record("sens_sparse_xgb", ss_oof, ss_thr, ss_fold, mask=sens_mask)

    # Experiment 2: dense XGBoost vs small GRU (dense only).
    if skip_gru:
        log.info("Experiment 2: GRU skipped (--skip-gru)")
    else:
        try:
            from modeling.gru_model import train_gru_oof
            log.info("Experiment 2: GRU (use --skip-gru to skip)")
            g_oof, _ = train_gru_oof(sequences, y, folds, N_FOLDS)
            g_thr = np.full(len(y), 0.5)
            pooled = pooled_metrics(y, g_oof, g_thr)
            pooled["model"] = "dense_gru"
            pooled["n_active"] = float((~np.isnan(g_oof)).sum())
            model_metric_rows.append(pooled)
            oof_store["dense_gru"] = g_oof
            thr_store["dense_gru"] = g_thr
        except Exception as exc:  # torch optional; ML-vs-DL arm is secondary
            log.warning("GRU skipped (%s)", exc)

    # 10. Persist combined artifacts.
    oof_long = meta.copy()
    for name, arr in oof_store.items():
        oof_long[f"prob_{name}"] = arr
        oof_long[f"thr_{name}"] = thr_store[name]
    oof_long.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    pd.DataFrame(model_metric_rows).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    save_comparison_figure(pd.DataFrame(model_metric_rows), fig_dir / "baseline_comparison.png")
    if fold_metric_frames:
        pd.concat(fold_metric_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "fold_metrics.csv", index=False)

    part = pd.concat([
        participant_metrics(meta, y, d_oof, d_thr, "dense_xgb"),
        participant_metrics(meta, y, s_oof, s_thr, "sparse_xgb"),
    ], ignore_index=True)
    part.to_csv(OUTPUT_DIR / "participant_metrics.csv", index=False)

    # 7. Paired comparison and participant-level bootstrap of the difference.
    d_auprc, s_auprc = r_dense["auprc"], r_sparse["auprc"]
    boot = bootstrap_participant_difference(meta, y, d_oof, s_oof, n_boot=N_BOOTSTRAP,
                                            seed=RANDOM_SEED, metric="auprc")
    comparison = {
        "dense_auprc": d_auprc,
        "sparse_auprc": s_auprc,
        "absolute_difference": d_auprc - s_auprc,
        "relative_loss_pct": 100 * (d_auprc - s_auprc) / d_auprc if d_auprc else np.nan,
        "dense_auroc": r_dense["auroc"],
        "sparse_auroc": r_sparse["auroc"],
        "boot_ci_lower": boot["ci_lower"],
        "boot_ci_upper": boot["ci_upper"],
        "boot_p_sparse_ge_dense": boot["p_b_ge_a"],
    }
    pd.DataFrame([comparison]).to_csv(OUTPUT_DIR / "paired_comparison.csv", index=False)
    pd.DataFrame([boot]).to_csv(OUTPUT_DIR / "bootstrap_paired_difference.csv", index=False)
    (OUTPUT_DIR / "experiment_summary.json").write_text(
        json.dumps({"n_windows": int(len(windows)), "n_positive": int(y.sum()),
                    **{k: (None if v != v else v) for k, v in comparison.items()}}, indent=2),
        encoding="utf-8")

    log.info("Done. dense AUPRC=%.3f sparse AUPRC=%.3f (rel. loss %.1f%%, 95%% CI [%.3f, %.3f])",
             d_auprc, s_auprc, comparison["relative_loss_pct"],
             boot["ci_lower"], boot["ci_upper"])
    log.info("Outputs in %s", OUTPUT_DIR)

    from modeling.report import generate_modeling_report
    from modeling.reproducibility import write_run_manifest
    from modeling.artifacts import save_deployable_models

    log.info("Saving deployable artifacts...")
    saved = save_deployable_models(
        dense_df, sparse_df, sequences, y, dense_cols, sparse_cols, skip_gru=skip_gru
    )
    log.info("Saved deployment models: %s", ", ".join(saved))

    generate_modeling_report(OUTPUT_DIR)
    write_run_manifest(["modeling"])
    log.info("Wrote %s", OUTPUT_DIR / "modeling_results.md")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run hypoglycemia modeling experiments")
    parser.add_argument(
        "--skip-gru",
        action="store_true",
        help="Skip Experiment 2 GRU (faster; primary dense-vs-sparse comparison still runs)",
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
    main(skip_gru=args.skip_gru, dataset_config=args.dataset_config, dataset_root=args.dataset_root)
