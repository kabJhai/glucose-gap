"""Dense CGM and sparse scan feature engineering.

All features are computed strictly from observations *before* the prediction
time, so no value from the 2-hour prediction horizon can leak into any feature
or GRU input.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "feasibility_audit"))
from data_audit import CGM_NOMINAL_INTERVAL_MIN, load_participant_freestyle, split_glucose_streams  # noqa: E402

from modeling.config import CGM_SLOT_MIN, DENSE_HISTORY_H, DENSE_SEQ_LEN, SPARSE_HISTORY_H


def _cgm_series(pid: str) -> pd.Series:
    raw = load_participant_freestyle(pid)
    cgm, _, _ = split_glucose_streams(raw)
    cgm = cgm.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    return cgm.set_index("timestamp")["glucose"]


def _scan_frame(pid: str) -> pd.DataFrame:
    raw = load_participant_freestyle(pid)
    _, scans, _ = split_glucose_streams(raw)
    if scans.empty:
        return pd.DataFrame(columns=["timestamp", "glucose"])
    return scans.sort_values("timestamp")[["timestamp", "glucose"]].drop_duplicates("timestamp")


def dense_cgm_slots(pred_time: pd.Timestamp, glucose: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, mask) for 16 slots in [pred_time-4h, pred_time).

    Missing slots are forward-filled within the input window only. ``mask`` marks
    slots that carried an observation (before forward-fill) as True. Leading gaps
    are back-filled from the first observation so the sequence contains no NaN.
    """
    start = pred_time - timedelta(hours=DENSE_HISTORY_H)
    slots = pd.date_range(start, pred_time, freq=f"{CGM_SLOT_MIN}min", inclusive="left")
    vals = glucose.reindex(slots).astype(float).values
    mask = ~np.isnan(vals)
    if mask.any():
        last = np.nan
        for i in range(len(vals)):
            if mask[i]:
                last = vals[i]
            elif last == last:  # not nan -> forward fill within window
                vals[i] = last
        if np.isnan(vals).any():
            first_valid = vals[mask][0]
            vals = np.where(np.isnan(vals), first_valid, vals)
    return vals, mask


def _value_at_or_before(ref: pd.Series, t: pd.Timestamp) -> float:
    sub = ref[ref.index <= t]
    return float(sub.iloc[-1]) if len(sub) else np.nan


def dense_tabular_features(pred_time: pd.Timestamp, glucose: pd.Series) -> dict[str, float]:
    vals, mask = dense_cgm_slots(pred_time, glucose)
    observed = vals[mask]
    n_slots = len(vals)

    feats: dict[str, float] = {
        "dense_missing_slots": float((~mask).sum()),
        "dense_input_missing_frac": float(1.0 - mask.mean()) if n_slots else 1.0,
        "hour_sin": float(np.sin(2 * np.pi * (pred_time.hour + pred_time.minute / 60) / 24)),
        "hour_cos": float(np.cos(2 * np.pi * (pred_time.hour + pred_time.minute / 60) / 24)),
    }

    empty_keys = [
        "glucose_current", "glucose_mean_4h", "glucose_median_4h", "glucose_min_4h",
        "glucose_max_4h", "glucose_std_4h", "glucose_range_4h", "glucose_slope_4h",
        "glucose_change_15m", "glucose_change_30m", "glucose_change_60m",
        "glucose_change_120m", "prop_below_70", "prop_below_80", "prop_below_90",
        "time_since_last_valid_min",
    ]
    if len(observed) == 0:
        feats.update({k: np.nan for k in empty_keys})
        return feats

    # History strictly before the prediction time (no horizon leakage).
    ref = glucose[glucose.index < pred_time]
    latest_val = float(ref.iloc[-1]) if len(ref) else float(observed[-1])
    latest_time = ref.index[-1] if len(ref) else pred_time

    feats["glucose_current"] = latest_val
    feats["glucose_mean_4h"] = float(observed.mean())
    feats["glucose_median_4h"] = float(np.median(observed))
    feats["glucose_min_4h"] = float(observed.min())
    feats["glucose_max_4h"] = float(observed.max())
    feats["glucose_std_4h"] = float(observed.std(ddof=0)) if len(observed) > 1 else 0.0
    feats["glucose_range_4h"] = float(observed.max() - observed.min())

    x = np.arange(len(observed), dtype=float)
    feats["glucose_slope_4h"] = float(np.polyfit(x, observed, 1)[0]) if len(observed) > 1 else 0.0

    for mins, key in [(15, "glucose_change_15m"), (30, "glucose_change_30m"),
                      (60, "glucose_change_60m"), (120, "glucose_change_120m")]:
        prev = _value_at_or_before(ref, pred_time - timedelta(minutes=mins))
        feats[key] = float(latest_val - prev) if prev == prev else np.nan

    for thr, key in [(70, "prop_below_70"), (80, "prop_below_80"), (90, "prop_below_90")]:
        feats[key] = float((observed < thr).mean())

    feats["time_since_last_valid_min"] = float((pred_time - latest_time).total_seconds() / 60)
    return feats


def sparse_scan_features(pred_time: pd.Timestamp, scans: pd.DataFrame) -> dict[str, float]:
    start = pred_time - timedelta(hours=SPARSE_HISTORY_H)
    window = scans[(scans["timestamp"] >= start) & (scans["timestamp"] < pred_time)]
    n = len(window)
    feats: dict[str, float] = {
        "n_scans_6h": float(n),
        "no_scan": float(n == 0),
        "only_one_scan": float(n == 1),
    }
    empty_keys = [
        "most_recent_scan", "most_recent_scan_age_min", "scan_mean_6h", "scan_min_6h",
        "scan_max_6h", "scan_std_6h", "scan_change_last_two",
        "scan_time_between_last_two_min", "scan_slope_last_two",
    ]
    if n == 0:
        feats.update({k: np.nan for k in empty_keys})
        return feats

    last = window.iloc[-1]
    feats["most_recent_scan"] = float(last["glucose"])
    feats["most_recent_scan_age_min"] = float((pred_time - last["timestamp"]).total_seconds() / 60)
    feats["scan_mean_6h"] = float(window["glucose"].mean())
    feats["scan_min_6h"] = float(window["glucose"].min())
    feats["scan_max_6h"] = float(window["glucose"].max())
    feats["scan_std_6h"] = float(window["glucose"].std(ddof=0)) if n > 1 else 0.0

    if n >= 2:
        prev = window.iloc[-2]
        dg = float(last["glucose"] - prev["glucose"])
        dt_min = float((last["timestamp"] - prev["timestamp"]).total_seconds() / 60)
        feats["scan_change_last_two"] = dg
        feats["scan_time_between_last_two_min"] = dt_min
        feats["scan_slope_last_two"] = dg / dt_min if dt_min else np.nan
    else:
        feats["scan_change_last_two"] = np.nan
        feats["scan_time_between_last_two_min"] = np.nan
        feats["scan_slope_last_two"] = np.nan
    return feats


def build_feature_matrices(
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Join dense/sparse features onto the window table without changing row order.

    Returns dense_df, sparse_df, gru_sequences (n, 16, 2 = [value, mask]),
    dense_cols, sparse_cols. Row i of every output corresponds to row i of
    ``windows``.
    """
    cache_glucose: dict[str, pd.Series] = {}
    cache_scans: dict[str, pd.DataFrame] = {}

    dense_rows, sparse_rows, seq_rows = [], [], []
    for row in windows.itertuples():
        pid = row.participant_id
        t = row.prediction_time
        if pid not in cache_glucose:
            cache_glucose[pid] = _cgm_series(pid)
            cache_scans[pid] = _scan_frame(pid)

        dense_rows.append(dense_tabular_features(t, cache_glucose[pid]))
        sparse_rows.append(sparse_scan_features(t, cache_scans[pid]))
        vals, mask = dense_cgm_slots(t, cache_glucose[pid])
        seq = np.stack([vals, mask.astype(np.float32)], axis=-1)  # (16, 2)
        seq_rows.append(seq)

    dense_df = pd.DataFrame(dense_rows)
    sparse_df = pd.DataFrame(sparse_rows)
    sequences = np.stack(seq_rows, axis=0).astype(np.float32)  # (n, 16, 2)
    assert len(dense_df) == len(sparse_df) == len(sequences) == len(windows)
    return dense_df, sparse_df, sequences, list(dense_df.columns), list(sparse_df.columns)
