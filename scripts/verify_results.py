#!/usr/bin/env python3
"""Compare generated model_metrics.csv against tutorial verification targets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGETS_PATH = PROJECT_ROOT / "tutorial" / "verification_targets.json"
METRICS_PATH = PROJECT_ROOT / "modeling_outputs" / "model_metrics.csv"


def _metric_keys(tolerance: dict) -> set[str]:
    return {key for key in tolerance if key != "note"}


def main() -> int:
    if not METRICS_PATH.exists():
        print(f"Missing {METRICS_PATH}. Run: python -m modeling.train")
        return 1

    targets = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
    tol = targets["tolerance"]
    metric_keys = _metric_keys(tol)
    verified_models = targets.get(
        "verified_models",
        [name for name, spec in targets["models"].items() if "note" not in spec],
    )

    import pandas as pd

    metrics = pd.read_csv(METRICS_PATH)
    failures: list[str] = []

    for model in verified_models:
        expected = targets["models"].get(model)
        if expected is None:
            failures.append(f"model '{model}' listed in verified_models but missing from targets")
            continue

        rows = metrics[metrics["model"] == model]
        if rows.empty:
            failures.append(f"model '{model}' not found in {METRICS_PATH.name}")
            continue

        row = rows.iloc[0]
        for key, exp_val in expected.items():
            if key not in metric_keys:
                continue
            got = row.get(key)
            if pd.isna(got):
                failures.append(f"{model}.{key}: missing")
                continue
            t = tol.get(key, tol.get("auprc", 0.02))
            if abs(float(got) - float(exp_val)) > float(t):
                failures.append(f"{model}.{key}: expected {exp_val}, got {got:.3f} (tol {t})")

    paired_path = PROJECT_ROOT / "modeling_outputs" / "paired_comparison.csv"
    if paired_path.exists():
        paired = pd.read_csv(paired_path).iloc[0]
        for key, exp_val in targets["paired_comparison"].items():
            got = paired.get(key)
            t = 1.0 if key.endswith("_pct") else tol.get("auprc", 0.02)
            if got is None or pd.isna(got):
                failures.append(f"paired_comparison.{key}: missing")
            elif abs(float(got) - float(exp_val)) > float(t):
                failures.append(f"paired_comparison.{key}: expected {exp_val}, got {got:.3f}")

    if failures:
        print("Verification FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("Verification PASSED: primary XGBoost metrics match reference targets within tolerance.")
    print(f"  dense AUPRC:  {targets['models']['dense_xgb']['auprc']}")
    print(f"  sparse AUPRC: {targets['models']['sparse_xgb']['auprc']}")

    gru_ref = targets["models"].get("dense_gru")
    gru_rows = metrics[metrics["model"] == "dense_gru"]
    if gru_ref and not gru_rows.empty:
        got = gru_rows.iloc[0].get("auprc")
        if got is not None and not pd.isna(got):
            print(
                f"  dense GRU AUPRC: {float(got):.3f} "
                f"(informational; reference {gru_ref['auprc']}, not verified)"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
