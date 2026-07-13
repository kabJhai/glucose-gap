"""Build paired prediction windows.

Walks a 30-minute grid per participant. Label = any CGM < 70 mg/dL in the
next 2 hours. Requires <=20% missing slots in input (4 h) and label windows.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feasibility_audit"))
from data_audit import (  # noqa: E402
    CGM_NOMINAL_INTERVAL_MIN,
    detect_episodes,
    split_glucose_streams,
)

from dataset import load_participant_records

from modeling.config import (
    DENSE_HISTORY_H,
    EPISODE_SEPARATION_MIN,
    HORIZON_H,
    HYPO_THRESHOLD,
    MISSINGNESS_THRESHOLD,
    SPARSE_HISTORY_H,
    STRIDE_MIN,
)


def _expected_slots(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=f"{CGM_NOMINAL_INTERVAL_MIN}min", inclusive="left")


def _miss_frac(start: pd.Timestamp, end: pd.Timestamp, obs_set: set) -> float:
    expected = _expected_slots(start, end)
    if len(expected) == 0:
        return 1.0
    present = sum(1 for t in expected if t in obs_set)
    return 1.0 - present / len(expected)


def build_participant_windows(pid: str) -> list[dict]:
    """Eligible windows for one participant at 30-min stride."""
    raw = load_participant_records(pid)
    cgm, scans, _ = split_glucose_streams(raw)
    if cgm.empty:
        return []

    cgm = cgm.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    obs_index = pd.DatetimeIndex(cgm["timestamp"])
    obs_set = set(obs_index)
    glucose = cgm.set_index("timestamp")["glucose"]
    t_min, t_max = obs_index.min(), obs_index.max()

    scan_ts = (
        pd.DatetimeIndex(scans.sort_values("timestamp")["timestamp"])
        if not scans.empty
        else pd.DatetimeIndex([])
    )

    rows: list[dict] = []
    grid = pd.date_range(
        t_min + timedelta(hours=DENSE_HISTORY_H),
        t_max - timedelta(hours=HORIZON_H),
        freq=f"{STRIDE_MIN}min",
    )

    for pred_time in grid:
        input_start = pred_time - timedelta(hours=DENSE_HISTORY_H)
        future_end = pred_time + timedelta(hours=HORIZON_H)
        if _miss_frac(input_start, pred_time, obs_set) > MISSINGNESS_THRESHOLD:
            continue
        if _miss_frac(pred_time, future_end, obs_set) > MISSINGNESS_THRESHOLD:
            continue

        future_slots = _expected_slots(pred_time, future_end)
        positive = bool((glucose.reindex(future_slots) < HYPO_THRESHOLD).any())

        sparse_start = pred_time - timedelta(hours=SPARSE_HISTORY_H)
        if len(scan_ts):
            left = scan_ts.searchsorted(sparse_start, side="left")
            right = scan_ts.searchsorted(pred_time, side="left")
            n_prior_scans = int(right - left)
            scan_age_min = (
                (pred_time - scan_ts[right - 1]).total_seconds() / 60 if right > left else np.nan
            )
        else:
            n_prior_scans = 0
            scan_age_min = np.nan

        rows.append(
            {
                "participant_id": pid,
                "prediction_time": pred_time,
                # Canonical plan names
                "target_hypo_2h": int(positive),
                "has_prior_scan": int(n_prior_scans >= 1),
                "scan_count_6h": n_prior_scans,
                # Backward-compatible aliases retained for existing consumers
                "positive": int(positive),
                "n_prior_scans_6h": n_prior_scans,
                "has_prior_scan_6h": int(n_prior_scans >= 1),
                "most_recent_scan_age_min": scan_age_min,
            }
        )
    return rows


def build_window_table(participant_ids: list[str]) -> pd.DataFrame:
    """Build paired windows for all participants in the given list."""
    all_rows: list[dict] = []
    for pid in participant_ids:
        all_rows.extend(build_participant_windows(pid))
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["participant_id", "prediction_time"]).reset_index(drop=True)
    df["window_id"] = np.arange(len(df))
    return df


def participant_summaries(windows: pd.DataFrame) -> pd.DataFrame:
    """Per-participant summaries used to balance event-aware folds.

    Combines window-level counts (from the eligible window table) with the total
    hypoglycemic episode count in each participant's full CGM stream.
    """
    grp = windows.groupby("participant_id")
    summary = pd.DataFrame(
        {
            "total_windows": grp.size(),
            "positive_windows": grp["target_hypo_2h"].sum(),
        }
    )
    summary["positive_rate"] = summary["positive_windows"] / summary["total_windows"]

    ep_counts = {}
    for pid in summary.index:
        raw = load_participant_records(pid)
        cgm, _, _ = split_glucose_streams(raw)
        if cgm.empty:
            ep_counts[pid] = 0
            continue
        eps = detect_episodes(cgm, HYPO_THRESHOLD, EPISODE_SEPARATION_MIN)
        ep_counts[pid] = int(len(eps))
    summary["episode_count"] = pd.Series(ep_counts)
    return summary.reset_index().rename(columns={"index": "participant_id"})
