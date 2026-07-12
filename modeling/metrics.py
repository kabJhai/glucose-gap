"""Evaluation metrics for hypoglycemia prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out: dict[str, float] = {"n": float(len(y_true)), "n_positive": float(y_true.sum())}
    if len(np.unique(y_true)) < 2:
        out.update(
            {
                "auprc": np.nan,
                "auroc": np.nan,
                "recall": np.nan,
                "precision": np.nan,
                "f1": np.nan,
                "specificity": np.nan,
            }
        )
        return out

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["auprc"] = float(average_precision_score(y_true, y_prob))
    out["auroc"] = float(roc_auc_score(y_true, y_prob))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    out["threshold"] = float(threshold)
    out["tp"], out["fp"], out["fn"], out["tn"] = float(tp), float(fp), float(fn), float(tn)
    return out


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the probability threshold that maximises F1 on the given data.

    Intended to be called on a validation subset of the training folds only, so
    the decision threshold never sees the held-out fold used for reporting.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.unique(np.clip(y_prob, 1e-6, 1 - 1e-6))
    if len(candidates) > 200:
        candidates = np.quantile(candidates, np.linspace(0, 1, 200))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def bootstrap_participant_difference(
    meta: pd.DataFrame,
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    metric: str = "auprc",
) -> dict[str, float]:
    """Bootstrap the paired metric difference (A - B) by resampling participants.

    Because dense and sparse predictions are paired at identical timestamps, the
    resampling unit is the participant, not the individual window.
    """
    y_true = np.asarray(y_true).astype(int)
    df = meta.reset_index(drop=True).copy()
    df["_y"] = y_true
    df["_pa"] = np.asarray(prob_a, dtype=float)
    df["_pb"] = np.asarray(prob_b, dtype=float)
    groups = {pid: g.index.values for pid, g in df.groupby("participant_id")}
    pids = list(groups.keys())
    rng = np.random.default_rng(seed)

    def _metric(y, p):
        if len(np.unique(y)) < 2:
            return np.nan
        return average_precision_score(y, p) if metric == "auprc" else roc_auc_score(y, p)

    obs = _metric(y_true, df["_pa"].values) - _metric(y_true, df["_pb"].values)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        idx = np.concatenate([groups[p] for p in sampled])
        yb = df["_y"].values[idx]
        d = _metric(yb, df["_pa"].values[idx]) - _metric(yb, df["_pb"].values[idx])
        if d == d:
            diffs.append(d)
    diffs = np.array(diffs)
    return {
        "observed_diff": float(obs),
        "ci_lower": float(np.percentile(diffs, 2.5)) if len(diffs) else np.nan,
        "ci_upper": float(np.percentile(diffs, 97.5)) if len(diffs) else np.nan,
        "p_b_ge_a": float((diffs <= 0).mean()) if len(diffs) else np.nan,
        "n_boot": float(len(diffs)),
    }
