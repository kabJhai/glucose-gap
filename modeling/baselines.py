"""Naive baselines for hypoglycemia prediction (same windows and CV folds)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from modeling.config import N_FOLDS, RANDOM_SEED
from modeling.metrics import compute_metrics


def _single_feature_logistic_oof(
    values: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    """Grouped OOF probabilities from a one-feature logistic model."""
    oof = np.full(len(y), np.nan)
    x = values.reshape(-1, 1)

    for fold in range(N_FOLDS):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        imp = SimpleImputer(strategy="median").fit(x[tr_idx])
        x_tr = imp.transform(x[tr_idx])
        x_va = imp.transform(x[va_idx])

        if len(np.unique(y[tr_idx])) < 2:
            oof[va_idx] = float(y[tr_idx].mean())
            continue

        model = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        model.fit(x_tr, y[tr_idx])
        oof[va_idx] = model.predict_proba(x_va)[:, 1]

    return oof


def compute_baselines(
    y: np.ndarray,
    folds: np.ndarray,
    dense_df: pd.DataFrame,
    sparse_df: pd.DataFrame,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Return pooled baseline metrics and OOF probability arrays."""
    y = np.asarray(y).astype(int)
    thr = np.full(len(y), 0.5)
    rows: list[dict] = []
    probs: dict[str, np.ndarray] = {}

    prev_prob = np.full(len(y), float(y.mean()))
    probs["baseline_prevalence"] = prev_prob
    prev = compute_metrics(y, prev_prob, threshold=0.5)
    prev["model"] = "baseline_prevalence"
    prev["n_active"] = float(len(y))
    rows.append(prev)

    scan_prob = _single_feature_logistic_oof(
        sparse_df["most_recent_scan"].values.astype(float),
        y,
        folds,
    )
    probs["baseline_latest_scan"] = scan_prob
    scan = compute_metrics(y, scan_prob, threshold=0.5)
    scan["model"] = "baseline_latest_scan"
    scan["n_active"] = float((~np.isnan(scan_prob)).sum())
    rows.append(scan)

    cgm_prob = _single_feature_logistic_oof(
        dense_df["glucose_current"].values.astype(float),
        y,
        folds,
    )
    probs["baseline_latest_cgm"] = cgm_prob
    cgm = compute_metrics(y, cgm_prob, threshold=0.5)
    cgm["model"] = "baseline_latest_cgm"
    cgm["n_active"] = float((~np.isnan(cgm_prob)).sum())
    rows.append(cgm)

    return rows, probs


def sparse_probability_direction_check(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Diagnostic: compare mean predicted risk for positive vs negative windows."""
    valid = ~np.isnan(prob)
    y_s = y[valid]
    p_s = prob[valid]
    return {
        "mean_prob_positive": float(p_s[y_s == 1].mean()) if (y_s == 1).any() else float("nan"),
        "mean_prob_negative": float(p_s[y_s == 0].mean()) if (y_s == 0).any() else float("nan"),
    }


def save_comparison_figure(metrics: pd.DataFrame, out_path) -> None:
    """Grouped bar chart for AUPRC, recall, and F1 across models and baselines."""
    import matplotlib.pyplot as plt

    order = [
        "dense_xgb",
        "sparse_xgb",
        "baseline_latest_cgm",
        "baseline_latest_scan",
        "baseline_prevalence",
    ]
    labels = {
        "dense_xgb": "Dense XGBoost",
        "sparse_xgb": "Sparse XGBoost",
        "baseline_latest_cgm": "Latest CGM",
        "baseline_latest_scan": "Latest scan",
        "baseline_prevalence": "Prevalence",
    }
    metric_cols = ["auprc", "recall", "f1"]
    present = [m for m in order if m in set(metrics["model"])]
    if not present:
        return

    sub = metrics.set_index("model").loc[present, metric_cols]
    x = np.arange(len(metric_cols))
    width = 0.8 / len(present)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model in enumerate(present):
        offset = (i - (len(present) - 1) / 2) * width
        vals = sub.loc[model].values
        ax.bar(x + offset, vals, width=width, label=labels.get(model, model))

    ax.set_xticks(x)
    ax.set_xticklabels(["AUPRC", "Recall", "F1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Dense vs sparse models and naive baselines")
    ax.legend(fontsize=8, loc="upper right")
    prev = sub.loc["baseline_prevalence", "auprc"] if "baseline_prevalence" in present else np.nan
    if prev == prev:
        ax.axhline(prev, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(2.35, prev + 0.02, f"prevalence ≈ {prev:.2f}", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
