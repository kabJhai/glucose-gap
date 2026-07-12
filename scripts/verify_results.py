#!/usr/bin/env python3
"""Compare generated model_metrics.csv against tutorial verification targets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGETS_PATH = PROJECT_ROOT / "tutorial" / "verification_targets.json"
METRICS_PATH = PROJECT_ROOT / "modeling_outputs" / "model_metrics.csv"


def main() -> int:
    if not METRICS_PATH.exists():
        print(f"Missing {METRICS_PATH}. Run: python -m modeling.train --skip-gru")
        return 1

    targets = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
    tol = targets["tolerance"]

    import pandas as pd

    metrics = pd.read_csv(METRICS_PATH)
    failures: list[str] = []

    for model, expected in targets["models"].items():
        rows = metrics[metrics["model"] == model]
        if rows.empty:
            failures.append(f"model '{model}' not found in {METRICS_PATH.name}")
            continue
        row = rows.iloc[0]
        for key, exp_val in expected.items():
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

    print("Verification PASSED: model metrics match reference targets within tolerance.")
    print(f"  dense AUPRC:  {targets['models']['dense_xgb']['auprc']}")
    print(f"  sparse AUPRC: {targets['models']['sparse_xgb']['auprc']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
