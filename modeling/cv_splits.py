"""Grouped participant CV with event-aware, multi-criteria fold assignment.

Folds are assigned ONCE from participant-level summaries and persisted to
``fold_assignments.csv``. Every downstream model (dense XGBoost, sparse XGBoost,
dense GRU, sensitivities) must reuse that saved assignment unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modeling.config import N_FOLDS


def assign_event_aware_folds(
    summary: pd.DataFrame,
    n_folds: int = N_FOLDS,
) -> pd.Series:
    """Greedy multi-criteria balance across folds.

    Participants are placed (most positive windows first) into the fold that
    currently has the smallest combined, normalised load across positive
    windows, total windows and hypoglycemic-episode count. Positive balance is
    weighted most heavily because it drives AUPRC stability; total-window and
    episode balance act as tie-breakers so a single fold does not accumulate a
    disproportionate share of the data.

    Note: when one participant contributes a large fraction of all positive
    windows, grouped CV cannot fully balance positives (that participant cannot
    be split). This is a documented property of the cohort, not a defect.
    """
    s = summary.set_index("participant_id")
    for col in ("positive_windows", "total_windows", "episode_count"):
        if col not in s.columns:
            raise ValueError(f"summary missing required column: {col}")

    tot_pos = max(float(s["positive_windows"].sum()), 1.0)
    tot_win = max(float(s["total_windows"].sum()), 1.0)
    tot_ep = max(float(s["episode_count"].sum()), 1.0)
    w_pos, w_win, w_ep = 0.6, 0.3, 0.1

    order = s.sort_values(
        ["positive_windows", "total_windows"], ascending=False
    ).index.tolist()

    load_pos = [0.0] * n_folds
    load_win = [0.0] * n_folds
    load_ep = [0.0] * n_folds
    assignment: dict[str, int] = {}

    for pid in order:
        p = float(s.loc[pid, "positive_windows"])
        w = float(s.loc[pid, "total_windows"])
        e = float(s.loc[pid, "episode_count"])
        scores = [
            w_pos * (load_pos[f] + p) / tot_pos
            + w_win * (load_win[f] + w) / tot_win
            + w_ep * (load_ep[f] + e) / tot_ep
            for f in range(n_folds)
        ]
        fold = int(np.argmin(scores))
        assignment[pid] = fold
        load_pos[fold] += p
        load_win[fold] += w
        load_ep[fold] += e

    return pd.Series(assignment, name="fold").sort_index()


def get_or_create_folds(
    summary: pd.DataFrame,
    path: Path,
    n_folds: int = N_FOLDS,
) -> pd.Series:
    """Load the locked fold assignment if it exists; otherwise create and save it."""
    if path.exists():
        saved = pd.read_csv(path, index_col=0)["fold"]
        saved.index.name = "participant_id"
        # Only reuse if it covers exactly the current cohort.
        if set(saved.index) == set(summary["participant_id"]):
            return saved.astype(int)
    folds = assign_event_aware_folds(summary, n_folds)
    folds.to_frame("fold").to_csv(path, index_label="participant_id")
    return folds


def window_fold_column(windows: pd.DataFrame, participant_folds: pd.Series) -> np.ndarray:
    return windows["participant_id"].map(participant_folds).astype(int).values
