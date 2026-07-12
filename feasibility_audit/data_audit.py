#!/usr/bin/env python3
"""
HUPA-UCM Diabetes Dataset — Feasibility Audit
Predicting hypoglycemia from continuous vs intermittent glucose observations.

Does NOT train ML models. Produces inventory CSVs, plots, and feasibility_report.md.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(__file__).resolve().parent.parent / "HUPA-UCM Diabetes Dataset"
OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR / "figures"

CGM_NOMINAL_INTERVAL_MIN = 15
HYPO_THRESHOLD = 70
SEVERE_HYPO_THRESHOLD = 54
EPISODE_SEPARATION_MIN = 30
EPISODE_SEPARATION_SENSITIVITY_MIN = 60
PREDICTION_HORIZONS_H = [2, 4]
INPUT_HISTORY_H = [2, 4, 6]
MISSINGNESS_THRESHOLDS = [0.20, 0.10]
SCAN_MATCH_TOLERANCE_MIN = 10
BLAND_ALTMAN_FREESTYLE_ONLY = ["HUPA0001P", "HUPA0005P", "HUPA0025P"]
HIGH_EPISODE_PARTICIPANTS = ["HUPA0027P", "HUPA0028P"]
MODELING_STRIDES_MIN = [15, 30, 60]
MODELING_STRIDE_RECOMMENDED = 30

# Spanish → English column mapping for FreeStyle Libre exports (all known variants)
FREESTYLE_COLUMN_ALIASES: dict[str, str] = {
    "ID": "record_id",
    "Hora": "timestamp",
    "Sello de tiempo del dispositivo": "timestamp",
    "Tipo de registro": "record_type",
    "Histórico glucosa (mg/dL)": "historical_glucose_mg_dl",
    "Historial de glucosa mg/dL": "historical_glucose_mg_dl",
    "Glucosa leída (mg/dL)": "scan_glucose_mg_dl",
    "Escaneo de glucosa mg/dL": "scan_glucose_mg_dl",
    "Glucosa de la tira (mg/dL)": "strip_glucose_mg_dl",
    "Tira reactiva para glucosa mg/dL": "strip_glucose_mg_dl",
    "Insulina de acción rápida sin valor numérico": "rapid_insulin_flag",
    "Insulina de acción rápida no numérica": "rapid_insulin_flag",
    "Insulina de acción rápida (unidades)": "rapid_insulin_units",
    "Alimentos sin valor numérico": "food_flag",
    "Alimento no numérico": "food_flag",
    "Carbohidratos (raciones)": "carbs_exchanges",
    "Carbohidratos (gramos)": "carbs_grams",
    "Carbohidratos (porciones)": "carbs_portions",
    "Insulina de acción lenta sin valor numérico": "slow_insulin_flag",
    "Insulina de acción larga no numérica": "slow_insulin_flag",
    "Insulina de acción lenta (unidades)": "slow_insulin_units",
    "Insulina de acción larga (unidades)": "slow_insulin_units",
    "Notas": "notes",
    "Cetonas (mmol/L)": "ketones_mmol_l",
    "Cuerpos cetónicos mmol/L": "ketones_mmol_l",
    "Insulina comida (unidades)": "meal_insulin_units",
    "Comida e insulina (unidades)": "meal_insulin_units",
    "Insulina corrección (unidades)": "correction_insulin_units",
    "Insulina de corrección (unidades)": "correction_insulin_units",
    "Insulina cambio usuario (unidades)": "user_change_insulin_units",
    "Insulina del cambio de usuario (unidades)": "user_change_insulin_units",
    "Hora anterior": "previous_timestamp",
    "Hora actualizada": "updated_timestamp",
    "Dispositivo": "device",
    "Número de serial": "serial_number",
}

PREPROCESSED_COLUMN_MAP = {
    "time": "timestamp",
    "glucose": "glucose_mg_dl",
    "calories": "calories",
    "heart_rate": "heart_rate",
    "steps": "steps",
    "basal_rate": "basal_rate",
    "bolus_volume_delivered": "bolus_volume_delivered",
    "carb_input": "carb_input",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def discover_participants() -> list[str]:
    raw = DATASET_ROOT / "Raw_Data"
    return sorted(p.name for p in raw.iterdir() if p.is_dir() and p.name.startswith("HUPA"))


def _normalize_col_name(name: str) -> str:
    return name.strip().lstrip("\ufeff").strip('"')


def parse_freestyle_timestamp(series: pd.Series) -> pd.Series:
    """Parse FreeStyle timestamps across export variants."""
    s = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    formats = [
        "%Y/%m/%d %H:%M",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M",
        "%d/%m/%y %H:%M",
        "%d-%m-%y %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]
    remaining = s.notna()
    for fmt in formats:
        if not remaining.any():
            break
        parsed = pd.to_datetime(s[remaining], format=fmt, errors="coerce")
        out.loc[remaining] = out.loc[remaining].fillna(parsed)
        remaining = out.isna() & s.notna()

    if remaining.any():
        out.loc[remaining] = pd.to_datetime(s[remaining], errors="coerce", dayfirst=True)

    return out


def _read_freestyle_truncated_fields(
    path: Path, skiprows: int, sep: str
) -> pd.DataFrame | None:
    """Parse rows by keeping only the first 19 fields when trailing fields contain separators."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if len(lines) <= skiprows + 1:
        return None
    header = lines[skiprows].strip().split(sep)
    ncols = len(header)
    rows = []
    for line in lines[skiprows + 1 :]:
        parts = line.strip().split(sep)
        if len(parts) < 3:
            continue
        if len(parts) > ncols:
            parts = parts[:ncols]
        elif len(parts) < ncols:
            parts = parts + [""] * (ncols - len(parts))
        rows.append(parts)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=header)


def _detect_freestyle_layout(path: Path) -> tuple[int, str] | None:
    """Return (skiprows, separator) for a FreeStyle export."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [fh.readline() for _ in range(5)]

    if not lines:
        return None

    line0 = lines[0].strip()
    if "está vacío" in line0.lower() or "vacio" in line0.lower():
        return None

    # LibreLink report format
    if "Informe del paciente" in line0 or "informe del paciente" in line0.lower():
        header_line = lines[2] if len(lines) > 2 else ""
        sep = ";" if header_line.count(";") > header_line.count(",") else ","
        return 2, sep

    # Standard export: participant row then header
    header_line = lines[1] if len(lines) > 1 else ""
    if "Hora" in header_line or "Sello de tiempo" in header_line:
        if "\t" in header_line:
            return 1, "\t"
        sep = ";" if header_line.count(";") > header_line.count(",") else ","
        return 1, sep

    return None


def read_freestyle_file(path: Path, participant_id: str) -> pd.DataFrame | None:
    """Read one FreeStyle CSV across semicolon, tab, and LibreLink report layouts."""
    layout = _detect_freestyle_layout(path)
    if layout is None:
        log.warning("Skipped empty/placeholder FreeStyle file %s", path)
        return None

    skiprows, sep = layout
    is_librelink = skiprows == 2
    read_kwargs: dict[str, Any] = {
        "sep": sep,
        "skiprows": skiprows,
        "encoding": "utf-8",
    }

    df = None
    for encoding in ("utf-8", "latin-1"):
        read_kwargs["encoding"] = encoding
        try:
            if is_librelink:
                read_kwargs["engine"] = "python"
                read_kwargs["on_bad_lines"] = "warn"
                df = pd.read_csv(path, **read_kwargs)
            else:
                try:
                    df = pd.read_csv(path, **read_kwargs, on_bad_lines="skip")
                except Exception:
                    read_kwargs["engine"] = "python"
                    read_kwargs["on_bad_lines"] = "warn"
                    df = pd.read_csv(path, **read_kwargs)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            log.warning("Skipped malformed FreeStyle file %s: %s", path, e)
            return None

    if df is None:
        return None

    # Recover rows dropped by extra semicolons in trailing note/date fields
    if not is_librelink and len(df) < 50:
        try:
            recovered = _read_freestyle_truncated_fields(path, skiprows, sep)
            if recovered is not None and len(recovered) > len(df):
                df = recovered
        except Exception:
            pass

    df.columns = [_normalize_col_name(c) for c in df.columns]
    rename_map = {
        raw: std for raw, std in FREESTYLE_COLUMN_ALIASES.items() if raw in df.columns
    }
    df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        log.warning("No timestamp column in %s (columns: %s)", path, list(df.columns))
        return None

    df["timestamp"] = parse_freestyle_timestamp(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return None

    if "record_type" in df.columns:
        df["record_type"] = pd.to_numeric(
            df["record_type"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    for col in ("historical_glucose_mg_dl", "scan_glucose_mg_dl", "strip_glucose_mg_dl"):
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    df["participant_id"] = participant_id
    df["source_file"] = path.name
    return df


def load_participant_freestyle(participant_id: str) -> pd.DataFrame:
    folder = DATASET_ROOT / "Raw_Data" / participant_id / "free_style_sensor"
    if not folder.exists():
        return pd.DataFrame()
    frames = []
    for f in sorted(folder.glob("*.csv")):
        df = read_freestyle_file(f, participant_id)
        if df is not None and len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp")
    # Avoid collapsing rows when LibreLink IDs are corrupted (e.g. 1.61E+19)
    dedup_cols = ["timestamp", "record_type", "historical_glucose_mg_dl", "scan_glucose_mg_dl"]
    dedup_cols = [c for c in dedup_cols if c in out.columns]
    out = out.drop_duplicates(subset=dedup_cols, keep="first")
    return out


def read_preprocessed(participant_id: str) -> pd.DataFrame:
    path = DATASET_ROOT / "Preprocessed" / f"{participant_id}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df = df.rename(columns=PREPROCESSED_COLUMN_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["participant_id"] = participant_id
    return df.sort_values("timestamp")


def split_glucose_streams(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return historical CGM, scan, and strip streams."""
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    rt = df["record_type"]
    cgm = df[rt == 0].copy()
    cgm = cgm[cgm["historical_glucose_mg_dl"].notna()].copy()
    cgm["glucose"] = pd.to_numeric(cgm["historical_glucose_mg_dl"], errors="coerce")
    cgm = cgm.dropna(subset=["timestamp", "glucose"])

    scans = df[rt == 1].copy()
    scans = scans[scans["scan_glucose_mg_dl"].notna()].copy()
    scans["glucose"] = pd.to_numeric(scans["scan_glucose_mg_dl"], errors="coerce")
    scans = scans.dropna(subset=["timestamp", "glucose"])

    strips = df[df["strip_glucose_mg_dl"].notna()].copy()
    if len(strips):
        strips["glucose"] = pd.to_numeric(strips["strip_glucose_mg_dl"], errors="coerce")
        strips = strips.dropna(subset=["timestamp", "glucose"])
    else:
        strips = pd.DataFrame()

    return cgm, scans, strips


def interval_distribution_minutes(timestamps: pd.Series) -> pd.Series:
    ts = timestamps.sort_values().drop_duplicates()
    if len(ts) < 2:
        return pd.Series(dtype=float)
    deltas = ts.diff().dropna().dt.total_seconds() / 60.0
    return deltas


def count_glucose_ranges(glucose: pd.Series) -> dict[str, int]:
    g = glucose.dropna()
    return {
        "below_70": int((g < 70).sum()),
        "below_54": int((g < 54).sum()),
        "range_70_180": int(((g >= 70) & (g <= 180)).sum()),
        "above_180": int((g > 180).sum()),
        "above_250": int((g > 250).sum()),
        "total": int(len(g)),
    }


# ---------------------------------------------------------------------------
# Task 1: Inventory
# ---------------------------------------------------------------------------


def inventory_dataset(participants: list[str]) -> pd.DataFrame:
    rows = []
    schema_issues = []

    for pid in participants:
        row: dict[str, Any] = {"participant_id": pid}
        base = DATASET_ROOT / "Raw_Data" / pid

        # FreeStyle
        fs_dir = base / "free_style_sensor"
        fs_files = sorted(fs_dir.glob("*.csv")) if fs_dir.exists() else []
        row["has_freestyle"] = bool(fs_files)
        row["freestyle_file_count"] = len(fs_files)
        row["freestyle_files"] = ";".join(f.name for f in fs_files)
        row["freestyle_total_bytes"] = sum(f.stat().st_size for f in fs_files) if fs_files else 0

        # Pump
        pump_type = None
        pump_files = []
        for pump in ("medtronic_insulin_pump", "roche_insulin_pump"):
            pdir = base / pump
            if pdir.exists():
                pump_type = pump
                pump_files = sorted(pdir.glob("*.csv"))
                break
        row["has_insulin_pump"] = bool(pump_files)
        row["pump_type"] = pump_type or ""
        row["pump_file_count"] = len(pump_files)
        row["pump_files"] = ";".join(f.name for f in pump_files)

        # Fitbit
        fb_dir = base / "fitbit"
        fb_files = sorted(fb_dir.glob("*.csv")) if fb_dir.exists() else []
        row["has_fitbit"] = bool(fb_files)
        row["fitbit_file_count"] = len(fb_files)

        # Dexcom (only HUPA0027P)
        dex_dir = base / "dexcom"
        dex_files = sorted(dex_dir.glob("*.csv")) if dex_dir.exists() else []
        row["has_dexcom"] = bool(dex_files)
        row["dexcom_file_count"] = len(dex_files)

        # Preprocessed
        pp = DATASET_ROOT / "Preprocessed" / f"{pid}.csv"
        row["has_preprocessed"] = pp.exists()
        row["preprocessed_bytes"] = pp.stat().st_size if pp.exists() else 0

        if fs_files:
            try:
                sample = read_freestyle_file(fs_files[0], pid)
                if sample is not None:
                    row["freestyle_columns"] = "|".join(sample.columns.tolist())
                    row["freestyle_row_count"] = len(sample)
            except Exception as e:
                schema_issues.append(f"{pid}: freestyle read error: {e}")

        if pp.exists():
            try:
                ppdf = read_preprocessed(pid)
                row["preprocessed_columns"] = "|".join(ppdf.columns.tolist())
                row["preprocessed_row_count"] = len(ppdf)
            except Exception as e:
                schema_issues.append(f"{pid}: preprocessed read error: {e}")

        rows.append(row)

    inv = pd.DataFrame(rows)

    # Schema consistency check
    if "freestyle_columns" in inv.columns:
        unique_fs_schemas = inv["freestyle_columns"].dropna().unique()
        inv.attrs["freestyle_schema_consistent"] = len(unique_fs_schemas) <= 1
        inv.attrs["freestyle_schema_variants"] = list(unique_fs_schemas)

    if "preprocessed_columns" in inv.columns:
        unique_pp = inv["preprocessed_columns"].dropna().unique()
        inv.attrs["preprocessed_schema_consistent"] = len(unique_pp) <= 1

    inv.attrs["schema_issues"] = schema_issues
    return inv


# ---------------------------------------------------------------------------
# Task 2: Raw glucose audit
# ---------------------------------------------------------------------------


def audit_participant_glucose(pid: str, raw: pd.DataFrame) -> dict[str, Any]:
    cgm, scans, strips = split_glucose_streams(raw)
    result: dict[str, Any] = {"participant_id": pid}

    if raw.empty:
        return {**result, "error": "no_freestyle_data"}

    result["record_type_counts"] = raw["record_type"].value_counts().to_dict()
    result["unique_record_types"] = sorted(raw["record_type"].dropna().unique().tolist())
    result["n_historical_cgm"] = len(cgm)
    result["n_scans"] = len(scans)
    result["n_strip"] = len(strips)

    for stream_name, stream in [("cgm", cgm), ("scans", scans)]:
        if stream.empty:
            for k in ["date_min", "date_max", "recording_days", "dup_timestamps", "dup_readings"]:
                result[f"{stream_name}_{k}"] = np.nan if "days" not in k else 0
            continue
        result[f"{stream_name}_date_min"] = stream["timestamp"].min()
        result[f"{stream_name}_date_max"] = stream["timestamp"].max()
        result[f"{stream_name}_recording_days"] = (
            stream["timestamp"].max() - stream["timestamp"].min()
        ).days + 1
        result[f"{stream_name}_dup_timestamps"] = int(
            stream["timestamp"].duplicated().sum()
        )
        result[f"{stream_name}_dup_readings"] = int(
            stream.duplicated(subset=["timestamp", "glucose"]).sum()
        )

    if not cgm.empty:
        intervals = interval_distribution_minutes(cgm["timestamp"])
        result["cgm_interval_median_min"] = intervals.median()
        result["cgm_interval_mean_min"] = intervals.mean()
        result["cgm_interval_mode_min"] = (
            intervals.round().mode().iloc[0] if len(intervals) else np.nan
        )
        result["cgm_interval_p25"] = intervals.quantile(0.25)
        result["cgm_interval_p75"] = intervals.quantile(0.75)
        result["cgm_interval_pct_15min"] = (
            (intervals.between(14, 16)).mean() * 100 if len(intervals) else 0
        )
        gr = count_glucose_ranges(cgm["glucose"])
        result.update({f"cgm_{k}": v for k, v in gr.items()})
        result["cgm_glucose_min"] = cgm["glucose"].min()
        result["cgm_glucose_max"] = cgm["glucose"].max()
        result["cgm_invalid_glucose"] = int(
            cgm["glucose"].isna().sum()
            + (cgm["glucose"] < 20).sum()
            + (cgm["glucose"] > 600).sum()
        )

    if not scans.empty:
        scan_intervals = interval_distribution_minutes(scans["timestamp"])
        scans_per_day = scans.groupby(scans["timestamp"].dt.date).size()
        result["scans_per_day_mean"] = scans_per_day.mean()
        result["scans_per_day_median"] = scans_per_day.median()
        result["scans_per_day_min"] = scans_per_day.min()
        result["scans_per_day_max"] = scans_per_day.max()
        result["scan_interval_median_min"] = (
            scan_intervals.median() if len(scan_intervals) else np.nan
        )
        result["scan_longest_gap_hours"] = (
            scan_intervals.max() / 60 if len(scan_intervals) else np.nan
        )
        gr = count_glucose_ranges(scans["glucose"])
        result.update({f"scan_{k}": v for k, v in gr.items()})

    return result


# ---------------------------------------------------------------------------
# Task 3: Hypoglycemia episodes
# ---------------------------------------------------------------------------


def detect_episodes(
    cgm: pd.DataFrame,
    threshold: float,
    separation_min: int,
) -> pd.DataFrame:
    """Detect hypoglycemic episodes from historical CGM sorted by time."""
    if cgm.empty:
        return pd.DataFrame(
            columns=["episode_id", "start", "end", "duration_min", "hour_of_day", "min_glucose"]
        )

    ts = cgm.sort_values("timestamp").reset_index(drop=True)
    times = ts["timestamp"].values
    glucose = ts["glucose"].values
    below = glucose < threshold

    episodes = []
    in_episode = False
    ep_start_idx = None
    ep_min_g = None
    last_above_time = times[0] - np.timedelta64(separation_min + 1, "m")

    for i in range(len(ts)):
        t = times[i]
        g = glucose[i]
        if below[i]:
            if not in_episode:
                gap_min = (t - last_above_time) / np.timedelta64(1, "m")
                if i == 0 or gap_min >= separation_min:
                    in_episode = True
                    ep_start_idx = i
                    ep_min_g = g
            else:
                ep_min_g = min(ep_min_g, g)
        else:
            if in_episode:
                ep_start = times[ep_start_idx]
                ep_end = times[i - 1]
                dur = (ep_end - ep_start) / np.timedelta64(1, "m") + CGM_NOMINAL_INTERVAL_MIN
                episodes.append(
                    {
                        "start": pd.Timestamp(ep_start),
                        "end": pd.Timestamp(ep_end),
                        "duration_min": float(dur),
                        "hour_of_day": pd.Timestamp(ep_start).hour
                        + pd.Timestamp(ep_start).minute / 60,
                        "min_glucose": ep_min_g,
                    }
                )
                in_episode = False
            last_above_time = t

    if in_episode:
        ep_start = times[ep_start_idx]
        ep_end = times[-1]
        dur = (ep_end - ep_start) / np.timedelta64(1, "m") + CGM_NOMINAL_INTERVAL_MIN
        episodes.append(
            {
                "start": pd.Timestamp(ep_start),
                "end": pd.Timestamp(ep_end),
                "duration_min": float(dur),
                "hour_of_day": pd.Timestamp(ep_start).hour + pd.Timestamp(ep_start).minute / 60,
                "min_glucose": ep_min_g,
            }
        )

    if not episodes:
        return pd.DataFrame(
            columns=["episode_id", "start", "end", "duration_min", "hour_of_day", "min_glucose"]
        )

    epdf = pd.DataFrame(episodes)
    epdf["episode_id"] = range(len(epdf))
    return epdf


# ---------------------------------------------------------------------------
# Task 4 & 5: Prediction windows and sparse scan condition
# ---------------------------------------------------------------------------


def build_cgm_index(cgm: pd.DataFrame) -> pd.DataFrame:
    """Regular grid at CGM nominal interval for window evaluation."""
    if cgm.empty:
        return pd.DataFrame()
    cgm = cgm.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    cgm = cgm.set_index("timestamp")
    return cgm


def window_missing_fraction(
    grid_times: pd.DatetimeIndex,
    observed_times: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval_min: int,
) -> float:
    """Fraction of expected CGM slots missing in [start, end)."""
    expected = pd.date_range(start, end, freq=f"{interval_min}min", inclusive="left")
    if len(expected) == 0:
        return 1.0
    observed_set = set(observed_times)
    present = sum(1 for t in expected if t in observed_set)
    return 1.0 - present / len(expected)


def evaluate_prediction_windows(
    pid: str,
    cgm: pd.DataFrame,
    scans: pd.DataFrame,
    episodes_70: pd.DataFrame,
    missingness_threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate horizon × history combinations using vectorized missingness checks."""
    if cgm.empty:
        return []

    cgm = cgm.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    obs_index = pd.DatetimeIndex(cgm["timestamp"])
    glucose = cgm.set_index("timestamp")["glucose"]
    obs_set = set(obs_index)

    scan_index = (
        pd.DatetimeIndex(scans.sort_values("timestamp")["timestamp"])
        if not scans.empty
        else pd.DatetimeIndex([])
    )

    max_horizon = max(PREDICTION_HORIZONS_H)
    t_min, t_max = obs_index.min(), obs_index.max()

    # Precompute episode intervals for mapping
    ep_starts = episodes_70["start"].values if len(episodes_70) else np.array([])
    ep_ends = episodes_70["end"].values if len(episodes_70) else np.array([])
    ep_ids = episodes_70["episode_id"].values if len(episodes_70) else np.array([])

    def expected_slots(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq=f"{CGM_NOMINAL_INTERVAL_MIN}min", inclusive="left")

    def miss_frac(start: pd.Timestamp, end: pd.Timestamp) -> float:
        expected = expected_slots(start, end)
        if len(expected) == 0:
            return 1.0
        present = sum(1 for t in expected if t in obs_set)
        return 1.0 - present / len(expected)

    def episode_for_window(pred_time: pd.Timestamp, future_end: pd.Timestamp) -> float:
        if len(ep_starts) == 0:
            return np.nan
        for es, ee, eid in zip(ep_starts, ep_ends, ep_ids):
            es, ee = pd.Timestamp(es), pd.Timestamp(ee)
            if (es >= pred_time and es < future_end) or (es < pred_time and ee >= pred_time):
                return eid
        return np.nan

    results: list[dict[str, Any]] = []

    for history_h in INPUT_HISTORY_H:
        for horizon_h in PREDICTION_HORIZONS_H:
            candidates = pd.date_range(
                t_min + timedelta(hours=history_h),
                t_max - timedelta(hours=horizon_h),
                freq=f"{CGM_NOMINAL_INTERVAL_MIN}min",
            )
            if len(candidates) == 0:
                continue

            for pred_time in candidates:
                input_start = pred_time - timedelta(hours=history_h)
                input_miss = miss_frac(input_start, pred_time)
                if input_miss > missingness_threshold:
                    continue

                future_end = pred_time + timedelta(hours=horizon_h)
                future_miss = miss_frac(pred_time, future_end)
                if future_miss > missingness_threshold:
                    continue

                future_times = expected_slots(pred_time, future_end)
                positive = bool((glucose.reindex(future_times) < HYPO_THRESHOLD).any())

                if len(scan_index):
                    left = scan_index.searchsorted(input_start, side="left")
                    right = scan_index.searchsorted(pred_time, side="left")
                    n_prior_scans = right - left
                    scan_age_min = (
                        (pred_time - scan_index[right - 1]).total_seconds() / 60
                        if right > left
                        else np.nan
                    )
                else:
                    n_prior_scans = 0
                    scan_age_min = np.nan

                results.append(
                    {
                        "participant_id": pid,
                        "prediction_time": pred_time,
                        "history_hours": history_h,
                        "horizon_hours": horizon_h,
                        "missingness_threshold": missingness_threshold,
                        "positive": positive,
                        "input_missing_frac": input_miss,
                        "future_missing_frac": future_miss,
                        "n_prior_scans": n_prior_scans,
                        "most_recent_scan_age_min": scan_age_min,
                        "episode_id": episode_for_window(pred_time, future_end)
                        if positive
                        else np.nan,
                    }
                )

    return results


# ---------------------------------------------------------------------------
# Task 6: CGM vs scan comparison
# ---------------------------------------------------------------------------


def aggregate_window_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize window-level evaluations by participant × horizon × history × threshold."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    summaries = []
    for (pid, horizon, history, miss), grp in df.groupby(
        ["participant_id", "horizon_hours", "history_hours", "missingness_threshold"]
    ):
        n_tot = len(grp)
        n_pos = int(grp["positive"].sum())
        n_episodes = grp.loc[grp["positive"], "episode_id"].dropna().nunique()
        pos_windows_per_ep = n_pos / max(n_episodes, 1)
        summaries.append(
            {
                "participant_id": pid,
                "horizon_hours": horizon,
                "history_hours": history,
                "missingness_threshold": miss,
                "n_eligible_windows": n_tot,
                "n_positive_windows": n_pos,
                "pct_positive_windows": 100 * n_pos / n_tot if n_tot else 0,
                "class_imbalance_ratio": (n_tot - n_pos) / max(n_pos, 1),
                "n_unique_episodes_in_positives": int(n_episodes),
                "avg_positive_windows_per_episode": pos_windows_per_ep,
                "pct_windows_0_scans": 100 * (grp["n_prior_scans"] == 0).mean(),
                "pct_windows_1_scan": 100 * (grp["n_prior_scans"] == 1).mean(),
                "pct_windows_2_scans": 100 * (grp["n_prior_scans"] == 2).mean(),
                "pct_windows_3plus_scans": 100 * (grp["n_prior_scans"] >= 3).mean(),
                "pct_positive_with_prior_scan": (
                    100 * grp.loc[grp["positive"], "n_prior_scans"].ge(1).mean()
                    if grp["positive"].any()
                    else 0
                ),
                "median_scan_age_min": grp["most_recent_scan_age_min"].median(),
            }
        )
    return summaries


def compare_cgm_scans(cgm: pd.DataFrame, scans: pd.DataFrame, tolerance_min: int = 10) -> dict:
    if cgm.empty or scans.empty:
        return {"n_scans": 0, "n_matched": 0, "pct_matched": 0}

    cgm_ts = cgm.sort_values("timestamp")
    matches = []
    for _, srow in scans.iterrows():
        st = srow["timestamp"]
        sg = srow["glucose"]
        diffs = (cgm_ts["timestamp"] - st).abs()
        idx = diffs.idxmin()
        dt_min = diffs.loc[idx].total_seconds() / 60
        if dt_min <= tolerance_min:
            cg = cgm_ts.loc[idx, "glucose"]
            matches.append({"time_diff_min": dt_min, "scan_g": sg, "cgm_g": cg, "abs_diff": abs(sg - cg)})

    if not matches:
        return {
            "n_scans": len(scans),
            "n_matched": 0,
            "pct_matched": 0,
        }

    mdf = pd.DataFrame(matches)
    corr = mdf["scan_g"].corr(mdf["cgm_g"]) if len(mdf) > 1 else np.nan
    return {
        "n_scans": len(scans),
        "n_matched": len(mdf),
        "pct_matched": 100 * len(mdf) / len(scans),
        "time_diff_median_min": mdf["time_diff_min"].median(),
        "time_diff_mean_min": mdf["time_diff_min"].mean(),
        "abs_diff_mean": mdf["abs_diff"].mean(),
        "abs_diff_median": mdf["abs_diff"].median(),
        "correlation": corr,
        "mean_diff": (mdf["scan_g"] - mdf["cgm_g"]).mean(),
        "pct_exact_match": 100 * (mdf["abs_diff"] == 0).mean(),
        "pct_within_5mg": 100 * (mdf["abs_diff"] <= 5).mean(),
    }


# ---------------------------------------------------------------------------
# Task 7: Raw vs preprocessed audit
# ---------------------------------------------------------------------------


def audit_raw_vs_preprocessed(pid: str, cgm: pd.DataFrame, preprocessed: pd.DataFrame) -> dict:
    result = {"participant_id": pid}
    if cgm.empty or preprocessed.empty:
        result["error"] = "missing_data"
        return result

    result["raw_date_min"] = cgm["timestamp"].min()
    result["raw_date_max"] = cgm["timestamp"].max()
    result["raw_row_count"] = len(cgm)
    result["preprocessed_date_min"] = preprocessed["timestamp"].min()
    result["preprocessed_date_max"] = preprocessed["timestamp"].max()
    result["preprocessed_row_count"] = len(preprocessed)

    # Infer resampling interval
    pp_intervals = interval_distribution_minutes(preprocessed["timestamp"])
    result["preprocessed_interval_median_min"] = pp_intervals.median()
    result["preprocessed_interval_mode_min"] = (
        pp_intervals.round().mode().iloc[0] if len(pp_intervals) else np.nan
    )

    # Match raw CGM to preprocessed timestamps (within 2.5 min)
    raw_set = cgm.set_index("timestamp")["glucose"]
    pp = preprocessed.dropna(subset=["glucose_mg_dl"]).copy()

    direct_matches = 0
    interpolated_flags = []
    for _, row in pp.iterrows():
        t = row["timestamp"]
        pg = row["glucose_mg_dl"]
        # Exact or near match to raw CGM
        near = cgm[(cgm["timestamp"] - t).abs() <= timedelta(minutes=2.5)]
        if len(near):
            rg = near.iloc[(near["timestamp"] - t).abs().argmin()]["glucose"]
            if abs(rg - pg) < 0.01:
                direct_matches += 1
                interpolated_flags.append(False)
            else:
                interpolated_flags.append(True)
        else:
            interpolated_flags.append(True)

    result["pct_direct_raw_match"] = 100 * direct_matches / len(pp) if len(pp) else 0
    result["pct_likely_interpolated"] = 100 * sum(interpolated_flags) / len(pp) if len(pp) else 0

    # Interpolation-created hypoglycemia
    raw_low = set(cgm[cgm["glucose"] < HYPO_THRESHOLD]["timestamp"].dt.floor("5min"))
    pp_low = pp[pp["glucose_mg_dl"] < HYPO_THRESHOLD]
    synthetic_low = 0
    for _, row in pp_low.iterrows():
        t = row["timestamp"]
        near_raw = cgm[(cgm["timestamp"] - t).abs() <= timedelta(minutes=7.5)]
        if near_raw.empty or near_raw["glucose"].min() >= HYPO_THRESHOLD:
            synthetic_low += 1
    result["preprocessed_low_readings"] = len(pp_low)
    result["synthetic_low_from_interpolation"] = synthetic_low

    # Timestamp alignment: check floor/round
    offset_secs = []
    for t in cgm["timestamp"].head(500):
        floored = t.floor("5min")
        offset_secs.append((t - floored).total_seconds())
    result["raw_timestamp_median_offset_from_5min_floor_sec"] = (
        np.median(offset_secs) if offset_secs else np.nan
    )

    return result


# ---------------------------------------------------------------------------
# Task 8: Insulin audit (sample participants with pumps)
# ---------------------------------------------------------------------------


def audit_insulin_pump_sample(pid: str) -> dict[str, Any]:
    base = DATASET_ROOT / "Raw_Data" / pid
    result: dict[str, Any] = {"participant_id": pid}
    preprocessed = read_preprocessed(pid)

    for pump in ("medtronic_insulin_pump", "roche_insulin_pump"):
        pdir = base / pump
        if pdir.exists():
            result["pump_type"] = pump
            files = list(pdir.glob("*.csv"))
            if files:
                result["pump_file"] = files[0].name
                try:
                    raw = pd.read_csv(files[0], sep=";", encoding="utf-8", low_memory=False)
                    result["pump_raw_columns"] = list(raw.columns[:10])
                    result["pump_raw_rows"] = len(raw)
                except Exception:
                    raw = pd.read_csv(files[0], encoding="utf-8", low_memory=False)
                    result["pump_raw_columns"] = list(raw.columns[:10])
                    result["pump_raw_rows"] = len(raw)
            break

    if not preprocessed.empty:
        for col in ("basal_rate", "bolus_volume_delivered", "carb_input"):
            s = preprocessed[col]
            result[f"pp_{col}_zero_pct"] = 100 * (s == 0).mean()
            result[f"pp_{col}_nonzero_count"] = int((s != 0).sum())
            result[f"pp_{col}_max"] = s.max()

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_episode_counts_by_participant(episode_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    data = episode_summary.sort_values("episodes_below_70", ascending=True)
    ax.barh(data["participant_id"], data["episodes_below_70"], color="steelblue")
    ax.set_xlabel("Episodes below 70 mg/dL (30-min separation)")
    ax.set_ylabel("Participant")
    ax.set_title("Hypoglycemic Episodes by Participant")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "episode_counts_by_participant.png", dpi=150)
    plt.close(fig)


def plot_scan_frequency(scan_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    data = scan_summary.sort_values("scans_per_day_median", ascending=True)
    ax.barh(data["participant_id"], data["scans_per_day_median"], color="coral")
    ax.set_xlabel("Median scans per day")
    ax.set_ylabel("Participant")
    ax.set_title("User-Initiated Scan Frequency by Participant")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "scan_frequency_by_participant.png", dpi=150)
    plt.close(fig)


def plot_cgm_scan_bland_altman(match_details: list[dict], pid: str) -> None:
    if not match_details:
        return
    mdf = pd.DataFrame(match_details)
    mean_g = (mdf["scan_g"] + mdf["cgm_g"]) / 2
    diff_g = mdf["scan_g"] - mdf["cgm_g"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_g, diff_g, alpha=0.4, s=10)
    ax.axhline(diff_g.mean(), color="red", linestyle="--", label=f"mean diff={diff_g.mean():.1f}")
    ax.axhline(diff_g.mean() + 1.96 * diff_g.std(), color="gray", linestyle=":")
    ax.axhline(diff_g.mean() - 1.96 * diff_g.std(), color="gray", linestyle=":")
    ax.set_xlabel("Mean glucose (mg/dL)")
    ax.set_ylabel("Scan − CGM (mg/dL)")
    ax.set_title(f"Bland-Altman: Scan vs CGM ({pid})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"bland_altman_{pid}.png", dpi=150)
    plt.close(fig)


def plot_prediction_feasibility(window_df: pd.DataFrame) -> None:
    if window_df.empty:
        return
    subset = window_df[window_df["missingness_threshold"] == 0.20]
    pivot = (
        subset.groupby(["horizon_hours", "history_hours"])
        .agg(n_positive=("n_positive_windows", "sum"), n_total=("n_eligible_windows", "sum"))
        .reset_index()
    )
    pivot["positive_pct"] = 100 * pivot["n_positive"] / pivot["n_total"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pivot))
    ax.bar(x, pivot["positive_pct"], color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"H={r.horizon_hours}h, hist={r.history_hours}h" for r in pivot.itertuples()]
    )
    ax.set_ylabel("% positive windows")
    ax.set_title("Positive Prediction Window Rate (20% missingness rule)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "prediction_window_positive_rates.png", dpi=150)
    plt.close(fig)


def plot_scan_coverage_by_window(window_df: pd.DataFrame) -> None:
    if window_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, hist in zip(axes, INPUT_HISTORY_H):
        h = window_df[window_df["history_hours"] == hist]
        cats = pd.cut(
            h["n_prior_scans"],
            bins=[-1, 0, 1, 2, np.inf],
            labels=["0", "1", "2", "3+"],
        )
        pct = cats.value_counts(normalize=True).reindex(["0", "1", "2", "3+"]) * 100
        ax.bar(pct.index.astype(str), pct.values, color="mediumpurple")
        ax.set_title(f"{hist}h history")
        ax.set_xlabel("Prior scans in window")
        ax.set_ylabel("% of windows")
    fig.suptitle("Scan Coverage in Input Windows (2h horizon, 20% missingness)")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "scan_coverage_by_history.png", dpi=150)
    plt.close(fig)


def compute_cohort_counts(
    inventory: pd.DataFrame, glucose_summary: pd.DataFrame
) -> dict[str, Any]:
    """Derive participant funnel counts directly from inventory and glucose summaries."""
    n_discovered = len(inventory)
    n_parseable = int(inventory["freestyle_columns"].notna().sum())
    has_cgm = glucose_summary["n_historical_cgm"].fillna(0) > 0
    has_scans = glucose_summary["n_scans"].fillna(0) > 0
    n_with_cgm = int(has_cgm.sum())
    dense_cohort = glucose_summary.loc[has_cgm, "participant_id"].tolist()
    sparse_cohort = glucose_summary.loc[has_cgm & has_scans, "participant_id"].tolist()
    excluded = glucose_summary.loc[~has_cgm, "participant_id"].tolist()
    sparse_excluded = glucose_summary.loc[has_cgm & ~has_scans, "participant_id"].tolist()
    return {
        "n_discovered": n_discovered,
        "n_parseable_freestyle": n_parseable,
        "n_with_historical_cgm": n_with_cgm,
        "n_dense_cohort": len(dense_cohort),
        "n_sparse_cohort": len(sparse_cohort),
        "n_common_cohort": len(sparse_cohort),
        "dense_cohort_ids": dense_cohort,
        "sparse_cohort_ids": sparse_cohort,
        "common_cohort_ids": sparse_cohort,
        "excluded_no_cgm": excluded,
        "excluded_no_scans": sparse_excluded,
    }


def compute_cohort_stride_summaries(
    all_cgm: dict[str, pd.DataFrame],
    cohort: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """30-min stride window counts for dense, common (paired), and sensitivity cohorts."""
    sensitivity_ids = [
        p for p in cohort["common_cohort_ids"] if p not in HIGH_EPISODE_PARTICIPANTS
    ]
    out: dict[str, dict[str, Any]] = {}
    for label, ids in [
        ("dense_23", cohort["dense_cohort_ids"]),
        ("common_22", cohort["common_cohort_ids"]),
        ("sensitivity_20", sensitivity_ids),
    ]:
        rows = compute_stride_window_summaries(
            all_cgm, ids, strides_min=[MODELING_STRIDE_RECOMMENDED]
        )
        out[label] = rows[0] if rows else {}
        out[label]["cohort_label"] = label
        out[label]["n_participants"] = len(ids)
    return out


def compute_pump_counts(inventory: pd.DataFrame) -> dict[str, int]:
    """Reconcile insulin pump totals directly from dataset_inventory.csv."""
    return {
        "n_with_pump": int(inventory["has_insulin_pump"].sum()),
        "n_medtronic": int((inventory["pump_type"] == "medtronic_insulin_pump").sum()),
        "n_roche": int((inventory["pump_type"] == "roche_insulin_pump").sum()),
        "n_without_pump": int((~inventory["has_insulin_pump"]).sum()),
        "no_pump_ids": inventory.loc[~inventory["has_insulin_pump"], "participant_id"].tolist(),
    }


def compute_match_rate_summary(cgm_scan_compare: pd.DataFrame) -> dict[str, float]:
    """Unweighted (per-participant) and scan-weighted CGM–scan match rates."""
    valid = cgm_scan_compare[cgm_scan_compare["n_scans"].fillna(0) > 0].copy()
    if valid.empty:
        return {
            "unweighted_mean_pct_matched": 0.0,
            "weighted_pct_matched": 0.0,
            "n_participants_in_match_summary": 0,
        }
    return {
        "unweighted_mean_pct_matched": float(valid["pct_matched"].mean()),
        "weighted_pct_matched": 100.0 * valid["n_matched"].sum() / valid["n_scans"].sum(),
        "n_participants_in_match_summary": len(valid),
    }


def audit_dexcom_freestyle_overlap(pid: str = "HUPA0027P") -> dict[str, Any]:
    """Check whether Dexcom and FreeStyle periods overlap for a participant."""
    result: dict[str, Any] = {"participant_id": pid}
    raw = load_participant_freestyle(pid)
    cgm, _, _ = split_glucose_streams(raw)
    dex_dir = DATASET_ROOT / "Raw_Data" / pid / "dexcom"
    if cgm.empty:
        result["freestyle_cgm_range"] = None
    else:
        result["freestyle_cgm_min"] = cgm["timestamp"].min()
        result["freestyle_cgm_max"] = cgm["timestamp"].max()
        result["freestyle_cgm_readings"] = len(cgm)

    if not dex_dir.exists():
        result["has_dexcom"] = False
        return result

    dex_files = list(dex_dir.glob("*.csv"))
    result["has_dexcom"] = bool(dex_files)
    if dex_files:
        dex = pd.read_csv(dex_files[0], sep=";", encoding="utf-8", low_memory=False)
        ts_col = [c for c in dex.columns if "Marca temporal" in c or "timestamp" in c.lower()]
        if ts_col:
            dex_ts = pd.to_datetime(dex[ts_col[0]], errors="coerce").dropna()
            if len(dex_ts):
                result["dexcom_min"] = dex_ts.min()
                result["dexcom_max"] = dex_ts.max()
                result["dexcom_rows_with_timestamp"] = int(dex_ts.notna().sum())
        result["dexcom_file"] = dex_files[0].name

    if result.get("freestyle_cgm_max") is not None and result.get("dexcom_min") is not None:
        fs_max = pd.Timestamp(result["freestyle_cgm_max"])
        dx_min = pd.Timestamp(result["dexcom_min"])
        result["periods_overlap"] = bool(fs_max >= dx_min)
        result["gap_days_freestyle_end_to_dexcom_start"] = (dx_min - fs_max).days
    else:
        result["periods_overlap"] = False

    result["recommended_sensor"] = "free_style_only"
    return result


def _active_cgm_days(cgm: pd.DataFrame) -> int:
    if cgm.empty:
        return 0
    return int(cgm["timestamp"].dt.date.nunique())


def _count_borderline_oscillation(cgm: pd.DataFrame) -> dict[str, int]:
    if cgm.empty:
        return {"crossings_below_70": 0, "crossings_69_71_band": 0}
    g = cgm.sort_values("timestamp")["glucose"].values
    crossings_70 = int(
        ((g[:-1] >= 70) & (g[1:] < 70)).sum() + ((g[:-1] < 70) & (g[1:] >= 70)).sum()
    )
    in_band = (g >= 69) & (g <= 71)
    return {
        "crossings_below_70": crossings_70,
        "crossings_69_71_band": int((in_band[1:] != in_band[:-1]).sum()),
    }


def _check_repeated_glucose_values(cgm: pd.DataFrame) -> dict[str, float | int]:
    if cgm.empty:
        return {"max_identical_streak": 0, "pct_readings_at_40": 0.0, "pct_readings_at_39": 0.0}
    g = cgm.sort_values("timestamp")["glucose"]
    streak = 1
    max_streak = 1
    prev = g.iloc[0]
    for val in g.iloc[1:]:
        if val == prev:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
        prev = val
    return {
        "max_identical_streak": max_streak,
        "pct_readings_at_40": float((g == 40).mean() * 100),
        "pct_readings_at_39": float((g == 39).mean() * 100),
    }


def deep_participant_audit(pid: str) -> dict[str, Any]:
    """Comprehensive audit for high-episode participants."""
    raw = load_participant_freestyle(pid)
    cgm, scans, _ = split_glucose_streams(raw)
    result: dict[str, Any] = {"participant_id": pid}
    if cgm.empty:
        result["error"] = "no_cgm"
        return result

    cgm = cgm.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    ep30 = detect_episodes(cgm, HYPO_THRESHOLD, EPISODE_SEPARATION_MIN)
    ep60 = detect_episodes(cgm, HYPO_THRESHOLD, EPISODE_SEPARATION_SENSITIVITY_MIN)
    act_days = _active_cgm_days(cgm)
    cal_days = (cgm["timestamp"].max() - cgm["timestamp"].min()).days + 1
    osc = _count_borderline_oscillation(cgm)
    rep = _check_repeated_glucose_values(cgm)

    folder = DATASET_ROOT / "Raw_Data" / pid / "free_style_sensor"
    file_ranges = []
    for f in sorted(folder.glob("*.csv")):
        df = read_freestyle_file(f, pid)
        if df is None:
            continue
        fcgm, _, _ = split_glucose_streams(df)
        file_ranges.append(
            {
                "file": f.name,
                "cgm_min": fcgm["timestamp"].min() if len(fcgm) else None,
                "cgm_max": fcgm["timestamp"].max() if len(fcgm) else None,
                "cgm_rows": len(fcgm),
            }
        )

    max_ep_day = 0
    if len(ep30):
        daily = ep30.assign(day=ep30["start"].dt.date).groupby("day").size()
        max_ep_day = int(daily.max())

    intervals_h = cgm["timestamp"].diff().dt.total_seconds() / 3600
    result.update(
        {
            "cgm_readings": len(cgm),
            "scan_readings": len(scans),
            "calendar_span_days": cal_days,
            "active_cgm_days": act_days,
            "cgm_date_min": cgm["timestamp"].min(),
            "cgm_date_max": cgm["timestamp"].max(),
            "duplicate_timestamps": int(cgm["timestamp"].duplicated().sum()),
            "episodes_30min_sep": len(ep30),
            "episodes_60min_sep": len(ep60),
            "episodes_per_active_day_30min": len(ep30) / max(act_days, 1),
            "median_episode_duration_min": float(ep30["duration_min"].median()) if len(ep30) else 0,
            "max_episodes_per_day": max_ep_day,
            "longest_cgm_gap_hours": float(intervals_h.max()) if len(intervals_h) else 0,
            "n_freestyle_files": len(file_ranges),
            "file_ranges": file_ranges,
            "has_duplicate_export_file": _has_duplicate_freestyle_export(file_ranges),
            **osc,
            **rep,
            "low_readings_below_70": int((cgm["glucose"] < HYPO_THRESHOLD).sum()),
            "pct_time_below_70": float((cgm["glucose"] < HYPO_THRESHOLD).mean() * 100),
        }
    )
    if pid == "HUPA0027P":
        dex = audit_dexcom_freestyle_overlap(pid)
        result["dexcom_overlap"] = dex.get("periods_overlap", False)
        result["dexcom_gap_days"] = dex.get("gap_days_freestyle_end_to_dexcom_start")
    return result


def _has_duplicate_freestyle_export(file_ranges: list[dict]) -> bool:
    """True if two exports share identical CGM row count and date span."""
    seen: set[tuple] = set()
    for fr in file_ranges:
        key = (fr.get("cgm_rows"), fr.get("cgm_min"), fr.get("cgm_max"))
        if key in seen:
            return True
        seen.add(key)
    return False


def episode_concentration_table(episode_summary: pd.DataFrame) -> pd.DataFrame:
    ep = episode_summary.sort_values("episodes_below_70", ascending=False).copy()
    total = ep["episodes_below_70"].sum()
    ep["pct_of_all_episodes"] = 100 * ep["episodes_below_70"] / max(total, 1)
    ep["cumulative_pct"] = ep["pct_of_all_episodes"].cumsum()
    return ep


def compute_stride_window_summaries(
    all_cgm: dict[str, pd.DataFrame],
    participant_ids: list[str],
    strides_min: list[int] | None = None,
    history_h: int = 4,
    horizon_h: int = 2,
    miss_thresh: float = 0.20,
) -> list[dict[str, Any]]:
    """Count eligible/positive windows at each prediction stride."""
    strides_min = strides_min or MODELING_STRIDES_MIN
    summaries = []
    for stride_min in strides_min:
        total_elig = total_pos = 0
        for pid in participant_ids:
            cgm = all_cgm.get(pid, pd.DataFrame())
            if cgm.empty:
                continue
            cgm = cgm.sort_values("timestamp").drop_duplicates("timestamp")
            obs_index = pd.DatetimeIndex(cgm["timestamp"])
            t_min, t_max = obs_index.min(), obs_index.max()
            obs_set = set(obs_index)
            glucose = cgm.set_index("timestamp")["glucose"]
            grid = pd.date_range(
                t_min + timedelta(hours=history_h),
                t_max - timedelta(hours=horizon_h),
                freq=f"{stride_min}min",
            )
            for t in grid:
                input_start = t - timedelta(hours=history_h)
                future_end = t + timedelta(hours=horizon_h)
                hist_slots = pd.date_range(
                    input_start, t, freq=f"{CGM_NOMINAL_INTERVAL_MIN}min", inclusive="left"
                )
                fut_slots = pd.date_range(
                    t, future_end, freq=f"{CGM_NOMINAL_INTERVAL_MIN}min", inclusive="left"
                )
                if not len(hist_slots) or not len(fut_slots):
                    continue
                hist_miss = 1 - sum(1 for s in hist_slots if s in obs_set) / len(hist_slots)
                fut_miss = 1 - sum(1 for s in fut_slots if s in obs_set) / len(fut_slots)
                if hist_miss > miss_thresh or fut_miss > miss_thresh:
                    continue
                total_elig += 1
                if (glucose.reindex(fut_slots) < HYPO_THRESHOLD).any():
                    total_pos += 1
        summaries.append(
            {
                "stride_min": stride_min,
                "history_hours": history_h,
                "horizon_hours": horizon_h,
                "n_eligible_windows": total_elig,
                "n_positive_windows": total_pos,
                "positive_rate": total_pos / max(total_elig, 1),
            }
        )
    return summaries


def bland_altman_summary_and_plot(
    cgm: pd.DataFrame,
    scans: pd.DataFrame,
    pid: str,
    *,
    freestyle_only: bool = False,
) -> dict[str, Any]:
    """Create Bland-Altman plot and return summary statistics."""
    _, details = _get_match_details(cgm, scans, SCAN_MATCH_TOLERANCE_MIN)
    if not details:
        return {"participant_id": pid, "n_matched": 0}

    mdf = pd.DataFrame(details)
    mean_g = (mdf["scan_g"] + mdf["cgm_g"]) / 2
    diff_g = mdf["scan_g"] - mdf["cgm_g"]
    md = float(diff_g.mean())
    sd = float(diff_g.std())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_g, diff_g, alpha=0.35, s=12, edgecolors="none")
    ax.axhline(md, color="red", linestyle="--", linewidth=1.2, label=f"mean diff = {md:.1f} mg/dL")
    ax.axhline(md + 1.96 * sd, color="gray", linestyle=":", label=f"+1.96 SD = {md + 1.96 * sd:.1f}")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle=":", label=f"−1.96 SD = {md - 1.96 * sd:.1f}")
    ax.set_xlabel("Mean glucose (mg/dL)")
    ax.set_ylabel("Scan − CGM (mg/dL)")
    suffix = ", FreeStyle-only" if freestyle_only else ""
    ax.set_title(f"Bland-Altman: Scan vs CGM ({pid}{suffix})")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"bland_altman_{pid}.png", dpi=150)
    plt.close(fig)

    return {
        "participant_id": pid,
        "n_matched": len(mdf),
        "mean_diff": md,
        "sd_diff": sd,
        "loa_lower": float(md - 1.96 * sd),
        "loa_upper": float(md + 1.96 * sd),
        "median_abs_diff": float(mdf["abs_diff"].median()),
        "pct_within_15mg": float((mdf["abs_diff"] <= 15).mean() * 100),
    }


def compute_prediction_window_flow(
    pid: str,
    cgm: pd.DataFrame,
    history_h: int,
    horizon_h: int,
    missingness_threshold: float,
) -> dict[str, int]:
    """
    Count how prediction timestamps are filtered for one participant × configuration.

    Candidate timestamps: 15-min grid from (t_min + history_h) to (t_max - horizon_h).
    Removals: input-window missingness, then future-window missingness (> threshold).
    """
    if cgm.empty:
        return {
            "participant_id": pid,
            "history_hours": history_h,
            "horizon_hours": horizon_h,
            "missingness_threshold": missingness_threshold,
            "n_cgm_readings": 0,
            "n_initial_candidate_timestamps": 0,
            "n_removed_insufficient_history_margin": 0,
            "n_removed_insufficient_future_margin": 0,
            "n_removed_input_missingness": 0,
            "n_removed_future_missingness": 0,
            "n_removed_gaps": 0,
            "n_final_eligible_windows": 0,
        }

    cgm = cgm.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    obs_index = pd.DatetimeIndex(cgm["timestamp"])
    obs_set = set(obs_index)
    t_min, t_max = obs_index.min(), obs_index.max()

    full_grid = pd.date_range(t_min, t_max, freq=f"{CGM_NOMINAL_INTERVAL_MIN}min")
    n_cgm = len(cgm)

    history_margin = int(
        max(0, ((t_min + timedelta(hours=history_h)) - t_min).total_seconds() / 60 / CGM_NOMINAL_INTERVAL_MIN)
    )
    future_margin = int(
        max(0, (t_max - (t_max - timedelta(hours=horizon_h))).total_seconds() / 60 / CGM_NOMINAL_INTERVAL_MIN)
    )
    n_removed_history_margin = history_margin
    n_removed_future_margin = future_margin

    candidates = pd.date_range(
        t_min + timedelta(hours=history_h),
        t_max - timedelta(hours=horizon_h),
        freq=f"{CGM_NOMINAL_INTERVAL_MIN}min",
    )
    n_initial = len(candidates)

    def expected_slots(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        return pd.date_range(start, end, freq=f"{CGM_NOMINAL_INTERVAL_MIN}min", inclusive="left")

    def miss_frac(start: pd.Timestamp, end: pd.Timestamp) -> float:
        expected = expected_slots(start, end)
        if len(expected) == 0:
            return 1.0
        present = sum(1 for t in expected if t in obs_set)
        return 1.0 - present / len(expected)

    n_removed_input = 0
    n_removed_future = 0
    n_final = 0

    for pred_time in candidates:
        input_start = pred_time - timedelta(hours=history_h)
        input_miss = miss_frac(input_start, pred_time)
        if input_miss > missingness_threshold:
            n_removed_input += 1
            continue
        future_end = pred_time + timedelta(hours=horizon_h)
        future_miss = miss_frac(pred_time, future_end)
        if future_miss > missingness_threshold:
            n_removed_future += 1
            continue
        n_final += 1

    return {
        "participant_id": pid,
        "history_hours": history_h,
        "horizon_hours": horizon_h,
        "missingness_threshold": missingness_threshold,
        "n_cgm_readings": n_cgm,
        "n_full_grid_timestamps": len(full_grid),
        "n_initial_candidate_timestamps": n_initial,
        "n_removed_insufficient_history_margin": n_removed_history_margin,
        "n_removed_insufficient_future_margin": n_removed_future_margin,
        "n_removed_input_missingness": n_removed_input,
        "n_removed_future_missingness": n_removed_future,
        "n_removed_gaps": n_removed_input + n_removed_future,
        "n_final_eligible_windows": n_final,
    }


def aggregate_window_flow(flow_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate participant-level flow counts by configuration."""
    df = pd.DataFrame(flow_rows)
    if df.empty:
        return df
    agg = (
        df.groupby(["history_hours", "horizon_hours", "missingness_threshold"])
        .agg(
            n_participants=("participant_id", "nunique"),
            total_cgm_readings=("n_cgm_readings", "sum"),
            initial_candidate_timestamps=("n_initial_candidate_timestamps", "sum"),
            removed_history_margin=("n_removed_insufficient_history_margin", "sum"),
            removed_future_margin=("n_removed_insufficient_future_margin", "sum"),
            removed_input_missingness=("n_removed_input_missingness", "sum"),
            removed_future_missingness=("n_removed_future_missingness", "sum"),
            removed_gaps=("n_removed_gaps", "sum"),
            final_eligible_windows=("n_final_eligible_windows", "sum"),
        )
        .reset_index()
    )
    return agg


def generate_feasibility_report(
    inventory: pd.DataFrame,
    glucose_summary: pd.DataFrame,
    scan_summary: pd.DataFrame,
    episode_summary: pd.DataFrame,
    window_df: pd.DataFrame,
    window_flow_agg: pd.DataFrame,
    raw_vs_pp: pd.DataFrame,
    cgm_scan_compare: pd.DataFrame,
    cohort: dict[str, Any],
    pump: dict[str, Any],
    match_summary: dict[str, float],
    dexcom_audit: dict[str, Any],
    assumptions: list[str],
    bland_altman_rows: list[dict[str, Any]],
    high_audits: list[dict[str, Any]],
    stride_rows: list[dict[str, Any]],
    cohort_stride: dict[str, dict[str, Any]],
) -> str:
    """Build the feasibility report from computed CSV summaries and supplemental audits."""
    total_episodes_70 = int(episode_summary["episodes_below_70"].sum())
    total_episodes_54 = int(episode_summary["episodes_below_54"].sum())
    w20 = window_df[window_df["missingness_threshold"] == 0.20]
    flow20 = window_flow_agg[window_flow_agg["missingness_threshold"] == 0.20]

    cgm_interval_median = glucose_summary["cgm_interval_median_min"].median()
    cgm_pct_15 = glucose_summary["cgm_interval_pct_15min"].median()
    med_scans_day = scan_summary["scans_per_day_median"].median()
    valid_cmp = cgm_scan_compare[cgm_scan_compare["n_scans"].fillna(0) > 0]
    corr_mean = valid_cmp["correlation"].mean()
    abs_diff_med = valid_cmp["abs_diff_median"].median()

    scans_with_data = scan_summary[scan_summary["n_scans"].fillna(0) > 0]
    scan_day_min = scans_with_data["scans_per_day_median"].min() if len(scans_with_data) else 0
    scan_day_max = scans_with_data["scans_per_day_median"].max() if len(scans_with_data) else 0

    concentration = episode_concentration_table(episode_summary)
    top3 = concentration.head(3)
    top2_ep = int(concentration.head(2)["episodes_below_70"].sum())
    top2_pct = 100 * top2_ep / max(total_episodes_70, 1)
    others_ep = int(total_episodes_70 - top2_ep)

    a27 = next((a for a in high_audits if a.get("participant_id") == "HUPA0027P"), {})
    a28 = next((a for a in high_audits if a.get("participant_id") == "HUPA0028P"), {})

    stride_30 = next((s for s in stride_rows if s["stride_min"] == MODELING_STRIDE_RECOMMENDED), {})
    common_30 = cohort_stride.get("common_22", stride_30)
    sens_30 = cohort_stride.get("sensitivity_20", {})
    n_win_common = int(common_30.get("n_eligible_windows", 0))
    n_pos_common = int(common_30.get("n_positive_windows", 0))
    rate_common = 100 * common_30.get("positive_rate", 0)
    n_win_sens = int(sens_30.get("n_eligible_windows", 0))
    n_pos_sens = int(sens_30.get("n_positive_windows", 0))
    rate_sens = 100 * sens_30.get("positive_rate", 0)

    lines = [
        "# HUPA-UCM Diabetes Dataset — Feasibility Audit Report",
        "",
        f"Generated by `data_audit.py`. Dataset root: `{DATASET_ROOT}`",
        "",
        "## Assumptions recorded",
        "",
    ]
    for a in assumptions:
        lines.append(f"- {a}")

    lines += [
        "",
        "## 1. Dataset inventory",
        "",
        "### Participant funnel",
        "",
        "| Stage | Count | Notes |",
        "|-------|------:|-------|",
        f"| Participant folders discovered | {cohort['n_discovered']} | Under `Raw_Data/` |",
        f"| Parseable FreeStyle files | {cohort['n_parseable_freestyle']} | Readable CSV with timestamp column; excludes HUPA0009P placeholder |",
        f"| ≥1 usable historical CGM reading (tipo 0) | {cohort['n_with_historical_cgm']} | Excludes HUPA0009P (empty), HUPA0010P (scan-only export) |",
        f"| **Final dense modeling cohort** | **{cohort['n_dense_cohort']}** | Participants with historical CGM for labels and dense input |",
        f"| **Final sparse modeling cohort** | **{cohort['n_sparse_cohort']}** | Dense cohort plus ≥1 user-initiated scan; excludes HUPA0015P (0 scans) |",
        f"| **Common paired comparison cohort** | **{cohort['n_common_cohort']}** | Participants in both dense and sparse arms; same timestamps and folds for Experiment 1 |",
        "",
        f"- Dense cohort IDs: {', '.join(cohort['dense_cohort_ids'])}",
        f"- Sparse / common cohort IDs: {', '.join(cohort['common_cohort_ids'])}",
        f"- Excluded (no historical CGM): {', '.join(cohort['excluded_no_cgm']) or 'none'}",
        f"- Excluded from sparse arm only (no scans): {', '.join(cohort['excluded_no_scans']) or 'none'}",
        "",
        "### Other data sources (from `dataset_inventory.csv`)",
        "",
        f"- Participants with insulin pump: **{pump['n_with_pump']}**",
        f"- Medtronic pump files: **{pump['n_medtronic']}**",
        f"- Roche pump files: **{pump['n_roche']}**",
        f"- No pump folder: **{pump['n_without_pump']}**",
        f"- Participants with Fitbit: **{int(inventory['has_fitbit'].sum())}**",
        f"- Participants with preprocessed file: **{int(inventory['has_preprocessed'].sum())}**",
        f"- No pump folder IDs: {', '.join(pump['no_pump_ids'])}",
        "",
        "## 2. Raw glucose structure",
        "",
        "- `Tipo de registro = 0`: historical CGM (`Histórico glucosa` / `Historial de glucosa` populated).",
        "- `Tipo de registro = 1`: user-initiated scan (`Glucosa leída` / `Escaneo de glucosa` populated).",
        "- Additional types observed: **4** (insulin events), **5** (carbohydrate events), **6** (sensor/time-change events).",
        f"- Median CGM sampling interval across participants: **{cgm_interval_median:.1f} min**",
        f"- Median fraction of CGM intervals at 14–16 min: **{cgm_pct_15:.1f}%**",
        f"- Total historical CGM readings (dense cohort): **{int(glucose_summary['n_historical_cgm'].fillna(0).sum()):,}**",
        f"- Total scan readings: **{int(glucose_summary['n_scans'].fillna(0).sum()):,}**",
        f"- Strip glucose readings (`Glucosa de la tira`): **{int(glucose_summary['n_strip'].fillna(0).sum()):,}**",
        "",
        "## 3. Historical CGM versus user-initiated scans",
        "",
        f"- Unweighted mean participant match rate (±10 min): **{match_summary['unweighted_mean_pct_matched']:.1f}%** ({match_summary['n_participants_in_match_summary']} participants with scans)",
        f"- Overall weighted match rate (matched scans / all scans): **{match_summary['weighted_pct_matched']:.1f}%**",
        f"- Mean Pearson correlation among matched pairs: **{corr_mean:.3f}**",
        f"- Median absolute glucose difference: **{abs_diff_med:.1f} mg/dL**",
        "- Scan readings are intermittent observations of the same FreeStyle sensor signal, not an independent glucose modality.",
        "",
        "**Conceptual framing:** the comparison is **continuous access to sensor history versus intermittent access when the user actively scans**. "
        "This is analogous to the information limitation of finger-prick monitoring, but not biologically or technologically identical to finger-prick testing.",
        "",
        "### Bland-Altman agreement (FreeStyle-only participants)",
        "",
        "Do **not** use HUPA0027P as the sole Bland-Altman illustration (Dexcom folder present; see §10). "
        "Use ordinary FreeStyle-only participants:",
        "",
        "| Participant | Matched pairs | Mean diff | 95% LoA | Median \\|diff\\| | % within ±15 mg/dL |",
        "|-------------|-------------:|----------:|--------:|----------------:|-------------------:|",
    ]
    for b in bland_altman_rows:
        if b.get("n_matched", 0) == 0:
            continue
        lines.append(
            f"| {b['participant_id']} | {b['n_matched']} | {b['mean_diff']:+.1f} | "
            f"[{b['loa_lower']:.1f}, {b['loa_upper']:.1f}] | {b['median_abs_diff']:.1f} | "
            f"{b['pct_within_15mg']:.1f}% |"
        )
    ba_figs = ", ".join(f"`figures/bland_altman_{b['participant_id']}.png`" for b in bland_altman_rows if b.get("n_matched"))
    lines += [
        "",
        f"Figures: {ba_figs}.",
        "",
        f"**Presentation wording:** compare continuous sensor history with **intermittent user-initiated scans**. "
        f"Do not claim scans reproduce a fixed finger-prick schedule (median scan rate is ~{med_scans_day:.0f}/day; "
        f"range ~{scan_day_min:.0f}–{scan_day_max:.0f}/day among participants with scans).",
        "",
    ]

    zero_ep = int((episode_summary["episodes_below_70"] == 0).sum())
    lines += [
        "## 4. Hypoglycemia episode counts",
        "",
        f"- Total episodes <70 mg/dL (30-min separation): **{total_episodes_70:,}**",
        f"- Total episodes <54 mg/dL: **{total_episodes_54:,}**",
        f"- Participants with 0 episodes: **{zero_ep}**",
        f"- Median episode duration: **{episode_summary['median_duration_min_70'].median():.0f} min**",
        "",
        "### Episode concentration (primary data-quality risk)",
        "",
        f"The headline total **{total_episodes_70:,}** overstates how broadly distributed events are. "
        "Most episodes come from a small number of long-recording participants:",
        "",
        "| Participant | Episodes <70 | % of all episodes | Cumulative % |",
        "|-------------|-------------:|------------------:|-------------:|",
    ]
    for _, row in top3.iterrows():
        lines.append(
            f"| {row['participant_id']} | {int(row['episodes_below_70'])} | "
            f"{row['pct_of_all_episodes']:.1f}% | {row['cumulative_pct']:.1f}% |"
        )
    n_other_part = len(concentration) - 3
    others_count = int(total_episodes_70 - top3["episodes_below_70"].sum())
    lines += [
        f"| All others ({n_other_part} participants) | {others_count} | "
        f"{100 * others_count / max(total_episodes_70, 1):.1f}% | 100% |",
        "",
        f"**HUPA0027P + HUPA0028P = {top2_ep:,} episodes ({top2_pct:.1f}%).** "
        "The key statistic is **episodes per active CGM day**, not total episodes alone:",
        "",
        "| Participant | Active CGM days | Episodes (30-min sep) | Episodes/active-day | Episodes (60-min sep) |",
        "|-------------|----------------:|----------------------:|--------------------:|----------------------:|",
    ]
    for audit in high_audits:
        lines.append(
            f"| {audit['participant_id']} | {audit.get('active_cgm_days', 0)} | "
            f"{audit.get('episodes_30min_sep', 0)} | {audit.get('episodes_per_active_day_30min', 0):.2f} | "
            f"{audit.get('episodes_60min_sep', 0)} |"
        )
    ep60_red_27 = 0
    if a27.get("episodes_30min_sep") and a27.get("episodes_60min_sep"):
        ep60_red_27 = int(100 * (1 - a27["episodes_60min_sep"] / a27["episodes_30min_sep"]))
    lines += [
        "",
        "**Risks from concentration:**",
        "1. **Model domination** — a pooled model may primarily learn glucose patterns of HUPA0027P and HUPA0028P.",
        "2. **Unstable CV folds** — one fold containing a high-episode participant may have many positives; another may have almost none.",
        f"3. **Episode-definition sensitivity** — 69–71 mg/dL oscillation inflates counts (60-min separation reduces HUPA0027P episodes by {ep60_red_27}%).",
        "",
        "Figure: `figures/episode_counts_by_participant.png`. Detail: `high_participant_audit.csv`.",
        "",
    ]

    flow_4h2h = flow20[(flow20["history_hours"] == 4) & (flow20["horizon_hours"] == 2)]
    if len(flow_4h2h):
        frow = flow_4h2h.iloc[0]
        n_init = int(frow["initial_candidate_timestamps"])
        n_in_miss = int(frow["removed_input_missingness"])
        n_fut_miss = int(frow["removed_future_missingness"])
        n_final_flow = int(frow["final_eligible_windows"])
        pct_in_miss = 100 * n_in_miss / max(n_init, 1)
    else:
        n_init = n_in_miss = n_fut_miss = n_final_flow = pct_in_miss = 0

    lines += [
        "## 5. Candidate prediction horizons",
        "",
        "### How candidate prediction timestamps are generated",
        "",
        "1. For each participant with historical CGM, deduplicate tipo-0 timestamps.",
        "2. Build a **15-minute grid** from `(t_min + history)` to `(t_max − horizon)` for each configuration.",
        "3. For each grid time `t`, require:",
        "   - input window `[t − history, t)` has ≤20% missing 15-min CGM slots;",
        "   - label window `[t, t + horizon)` has ≤20% missing 15-min CGM slots.",
        "4. Label is **positive** if any historical CGM value in the label window is <70 mg/dL.",
        "5. Scans for the sparse condition must have `timestamp < t` (strictly before prediction time).",
        "",
        f"**Why only {n_final_flow:,} eligible windows (4h history, 2h horizon) despite "
        f"{int(glucose_summary['n_historical_cgm'].fillna(0).sum()):,} CGM readings:**",
        f"- Across the dense cohort there are **{n_init:,}** initial 15-min grid candidates for this configuration.",
        f"- **{n_in_miss:,}** ({pct_in_miss:.1f}%) are removed because the 4h input window has >20% missing CGM slots (gaps between recording sessions).",
        f"- **{n_fut_miss:,}** more are removed because the 2h future label window has >20% missing slots.",
        "- CGM readings ≠ prediction windows; windows require contiguous coverage in both history and future.",
        "- Long-recording participants (HUPA0026P–0028P) contribute most surviving windows.",
        "",
        "### Window eligibility flow (aggregated across dense cohort, 20% missingness)",
        "",
        "| History | Horizon | CGM readings | Initial grid candidates | Removed: history margin | Removed: future margin | Removed: input missingness | Removed: future missingness | Final eligible |",
        "|--------:|--------:|-------------:|------------------------:|------------------------:|-----------------------:|---------------------------:|----------------------------:|---------------:|",
    ]
    for _, row in flow20.sort_values(["horizon_hours", "history_hours"]).iterrows():
        lines.append(
            f"| {int(row['history_hours'])}h | {int(row['horizon_hours'])}h | "
            f"{int(row['total_cgm_readings']):,} | {int(row['initial_candidate_timestamps']):,} | "
            f"{int(row['removed_history_margin']):,} | {int(row['removed_future_margin']):,} | "
            f"{int(row['removed_input_missingness']):,} | {int(row['removed_future_missingness']):,} | "
            f"{int(row['final_eligible_windows']):,} |"
        )
    lines += ["", "Detailed per-participant flow: `prediction_window_flow.csv`.", ""]

    for miss in MISSINGNESS_THRESHOLDS:
        ws = window_df[window_df["missingness_threshold"] == miss]
        lines.append(f"### Eligible windows summary — missingness threshold {miss*100:.0f}%")
        for horizon in PREDICTION_HORIZONS_H:
            for hist in INPUT_HISTORY_H:
                sub = ws[(ws["horizon_hours"] == horizon) & (ws["history_hours"] == hist)]
                if sub.empty:
                    continue
                n_pos = int(sub["n_positive_windows"].sum())
                n_tot = int(sub["n_eligible_windows"].sum())
                imb = (n_tot - n_pos) / max(n_pos, 1)
                n_part_pos = int((sub["n_positive_windows"] > 0).sum())
                lines.append(
                    f"- Horizon {horizon}h, history {hist}h: **{n_tot:,}** eligible, "
                    f"**{n_pos:,}** positive ({100*n_pos/n_tot:.2f}%), "
                    f"imbalance {imb:.1f}:1, **{n_part_pos}** participants with ≥1 positive"
                )
        lines.append("")

    w27 = w20[
        (w20["participant_id"] == "HUPA0027P")
        & (w20["horizon_hours"] == 2)
        & (w20["history_hours"] == 4)
    ]
    avg_pos_per_ep_27 = (
        float(w27["avg_positive_windows_per_episode"].iloc[0]) if len(w27) else 0
    )
    lines += [
        "### Prediction stride and episode duplication (4h history, 2h horizon, 20% missingness)",
        "",
        "One hypoglycemic episode produces multiple overlapping positive windows at 15-min stride "
        f"(e.g. HUPA0027P averages ~{avg_pos_per_ep_27:.0f} positive windows/episode). "
        f"Use a **{MODELING_STRIDE_RECOMMENDED}-min prediction stride** for modeling to limit duplication "
        "without materially changing class balance:",
        "",
        "| Stride | Eligible windows | Positive windows | Positive rate |",
        "|--------|-----------------:|-----------------:|--------------:|",
    ]
    for s in stride_rows:
        label = f"{s['stride_min']} min"
        if s["stride_min"] == 15:
            label += " (audit default)"
        elif s["stride_min"] == MODELING_STRIDE_RECOMMENDED:
            label = f"**{s['stride_min']} min (recommended)**"
        lines.append(
            f"| {label} | {s['n_eligible_windows']:,} | {s['n_positive_windows']:,} | "
            f"{100 * s['positive_rate']:.1f}% |"
        )
    lines += [
        "",
        "Detail: `prediction_stride_summary.csv`.",
        "",
        "**Class balance:** ~14–15% positive at 2h horizon is workable for XGBoost and a small dense GRU, "
        "provided CV folds contain events. The 2h horizon is preferred over 4h (more meaningful early-warning task, "
        "more eligible windows, less event bundling).",
        "",
        "## 6. Dense-condition feasibility",
        "",
        f"- Feasible for **{cohort['n_dense_cohort']}** participants with historical CGM.",
        "- **Recommended input:** previous **4 h** historical CGM at 15-min resolution (16 observations).",
        "- XGBoost on engineered CGM features: feasible.",
        "- Small GRU on 15-min CGM sequences: feasible where coverage passes missingness rules.",
        "",
    ]

    sub6 = w20[(w20["horizon_hours"] == 2) & (w20["history_hours"] == 6)]
    pct_3plus = (
        (sub6["pct_windows_3plus_scans"] * sub6["n_eligible_windows"]).sum()
        / max(sub6["n_eligible_windows"].sum(), 1)
        if len(sub6) else 0
    )
    pct_no_scan_2h = (
        (
            w20[(w20["horizon_hours"] == 2) & (w20["history_hours"] == 2)]["pct_windows_0_scans"]
            * w20[(w20["horizon_hours"] == 2) & (w20["history_hours"] == 2)]["n_eligible_windows"]
        ).sum()
        / max(
            w20[(w20["horizon_hours"] == 2) & (w20["history_hours"] == 2)][
                "n_eligible_windows"
            ].sum(),
            1,
        )
    )
    pct_no_scan_6h = (
        (sub6["pct_windows_0_scans"] * sub6["n_eligible_windows"]).sum()
        / max(sub6["n_eligible_windows"].sum(), 1)
        if len(sub6) else 0
    )
    lines += [
        "## 7. Sparse-condition feasibility",
        "",
        f"- Median participant scan rate: **{med_scans_day:.1f} scans/day** "
        f"(range ~{scan_day_min:.0f}–{scan_day_max:.0f}/day among participants with scans).",
        "- Sparsity is primarily a **window-level** problem: scans are episodic, so many individual prediction times lack a recent prior scan even when daily scan counts are adequate.",
        "",
    ]
    for hist in INPUT_HISTORY_H:
        sub = w20[(w20["horizon_hours"] == 2) & (w20["history_hours"] == hist)]
        if sub.empty:
            continue
        n_tot = sub["n_eligible_windows"].sum()
        no_scan = (sub["pct_windows_0_scans"] * sub["n_eligible_windows"]).sum() / n_tot
        three_plus = (sub["pct_windows_3plus_scans"] * sub["n_eligible_windows"]).sum() / n_tot
        lines.append(
            f"- 2h horizon, {hist}h history: **{no_scan:.1f}%** of windows have 0 prior scans; "
            f"**{three_plus:.1f}%** have ≥3 prior scans"
        )
    lines += [
        "",
        f"**2h sparse history is not practical** (~{pct_no_scan_2h:.0f}% of windows have no prior scan). "
        "**6h sparse history** is the defensible choice "
        f"(~{pct_no_scan_6h:.0f}% zero-scan windows). "
        "Dense and sparse conditions need not use identical history lengths; they represent different observation regimes.",
        "",
        "### Are there enough user-initiated scan observations to build a meaningful sparse-condition model?",
        "",
        f"**Yes, with window-level caveats.** Participant-level scan adherence is generally sufficient "
        f"(median **{med_scans_day:.1f}**/day), but **{pct_no_scan_2h:.1f}%** of 2h/2h-history windows still "
        f"have zero prior scans. With **6h history**, **{pct_3plus:.1f}%** of windows have ≥3 prior scans.",
        "",
        "**Sparse model (XGBoost only)** — engineered features from scans in the previous 6 h:",
        "- most recent scan value",
        "- age of most recent scan",
        "- number of scans in window",
        "- mean, minimum, maximum scan values",
        "- change between last two scans",
        "- time between last two scans",
        "- indicators for no scan or only one scan",
        "",
        "Exclude HUPA0015P from sparse arm (0 scans). Figure: `figures/scan_coverage_by_history.png`.",
        "",
    ]

    interp_pct_wt = (
        (raw_vs_pp["pct_likely_interpolated"] * raw_vs_pp["preprocessed_row_count"]).sum()
        / max(raw_vs_pp["preprocessed_row_count"].sum(), 1)
    )
    synth_low = int(raw_vs_pp["synthetic_low_from_interpolation"].sum())
    lines += [
        "## 8. Preprocessed-data risks",
        "",
        f"- Audited **all {len(raw_vs_pp)}** dense-cohort participants (full audit, not a spot check).",
        "- Preprocessed glucose on a **5-min** grid (median interval 5 min).",
        f"- Row-weighted fraction not matching raw CGM directly: **{interp_pct_wt:.1f}%**.",
        f"- Synthetic hypoglycemia from interpolation: **{synth_low:,}** preprocessed <70 readings without nearby raw <70.",
        "- **Recommendation:** construct labels from raw historical CGM only.",
        "",
        "## 9. Insulin-data interpretation",
        "",
        f"- Pump data present for **{pump['n_with_pump']}** participants ({pump['n_medtronic']} Medtronic, {pump['n_roche']} Roche).",
        "- Preprocessed `basal_rate`: active rate forward-filled across 5-min bins.",
        "- Preprocessed `bolus_volume_delivered`: bolus in that bin; **zero = no bolus**.",
        "- Preprocessed `carb_input`: **zero = no carb entry**.",
        "- Exclude insulin from v1 without reconstructing pump timelines.",
        "",
        "## 10. Leakage risks",
        "",
        "1. Overlapping 15-min windows — random row splits leak future information.",
        "2. Participant leakage — same participant in train and test.",
        "3. Scan timing — scans at/after prediction time leak outcome.",
        "4. Preprocessed interpolation may cross time boundaries.",
        f"5. Episode duplication — one episode yields many near-duplicate positive windows (mitigate with **{MODELING_STRIDE_RECOMMENDED}-min stride** or sample weighting).",
        "",
        "**Recommended split:** **Grouped 5-fold participant cross-validation** as primary strategy "
        f"(~{max(cohort['n_dense_cohort'] // 5, 4)}–{max(cohort['n_dense_cohort'] // 4, 5)} participants per fold). "
        "**Stratify folds manually by each participant's event count** where possible. "
        "**Leave-one-participant-out (LOPO)** as sensitivity analysis. "
        f"LOPO is not clearly superior given uneven episode concentration (top 3 participants = ~{top3['cumulative_pct'].iloc[-1]:.0f}% of episodes); "
        f"5-fold grouped CV balances bias/variance better for n={cohort['n_dense_cohort']}.",
        "",
        "**Evaluation metrics:** AUPRC (primary), hypoglycemia recall, precision, F1, AUROC, and "
        "**participant-level performance distribution** (do not rely on pooled metrics alone).",
        "",
        "### HUPA0027P and HUPA0028P validation",
        "",
        "Pre-modeling audit confirms episode counts are **real but concentrated**, not concatenation-inflated.",
        "",
        "**HUPA0027P**",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| FreeStyle CGM range | {a27.get('cgm_date_min', 'n/a')} to {a27.get('cgm_date_max', 'n/a')} ({a27.get('cgm_readings', 0):,} readings after dedup) |",
        f"| Dexcom overlap | **{a27.get('dexcom_overlap', False)}** ({a27.get('dexcom_gap_days', 'n/a')}-day gap after FreeStyle ends) |",
        f"| Duplicate timestamps in merged CGM | {a27.get('duplicate_timestamps', 0)} |",
        f"| Episodes/active-day | {a27.get('episodes_per_active_day_30min', 0):.2f} ({a27.get('episodes_30min_sep', 0)} over {a27.get('active_cgm_days', 0)} active days) |",
        f"| Median episode duration | {a27.get('median_episode_duration_min', 0):.0f} min |",
        f"| % readings <70 | {a27.get('pct_time_below_70', 0):.1f}% |",
        f"| 69–71 boundary crossings | {a27.get('crossings_69_71_band', 0):,}; 60-min sep → {a27.get('episodes_60min_sep', 0)} episodes |",
        f"| Sensor compression (exactly 40 mg/dL) | {a27.get('pct_readings_at_40', 0):.2f}% |",
        f"| Max identical glucose streak | {a27.get('max_identical_streak', 0)} |",
    ]
    if a27.get("has_duplicate_export_file"):
        lines.append(
            "| Duplicate export file | Two FreeStyle exports are identical; loader deduplicates — "
            "**episode count not inflated**; remove redundant file before modeling |"
        )
    lines += [
        "",
        "**HUPA0028P**",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| Active CGM days | {a28.get('active_cgm_days', 0)} |",
        f"| Episodes/active-day | {a28.get('episodes_per_active_day_30min', 0):.2f} ({a28.get('episodes_30min_sep', 0)} episodes) |",
        f"| % readings <70 | {a28.get('pct_time_below_70', 0):.1f}% |",
        f"| 60-min separation | {a28.get('episodes_60min_sep', 0)} episodes |",
        f"| Max episodes in one day | {a28.get('max_episodes_per_day', 0)} |",
        "",
        "**Use FreeStyle only** for HUPA0027P. Do not use HUPA0027P Bland-Altman "
        "(`figures/bland_altman_HUPA0027P.png`) as the sole agreement illustration.",
        "",
        "## 11. Locked project specification",
        "",
        "**Title:** *Glucose Gap: What Is Lost Between Glucose Checks?*",
        "",
        "**Subtitle:** *Predicting Hypoglycemia from Continuous and Intermittent Glucose Observations*",
        "",
        "**Primary research question:** How much does near-term hypoglycemia-prediction performance decline when a model receives only intermittent user-initiated scans instead of continuous historical CGM?",
        "",
        "**Tutorial contribution:** How to build a leakage-safe healthcare time-series prediction pipeline and measure the value of continuous versus intermittent observation.",
        "",
        "### Data and labeling",
        "",
        "- HUPA-UCM raw FreeStyle exports",
        "- Record type 0: historical CGM (dense features and outcome labels)",
        "- Record type 1: user-initiated scans (sparse features only)",
        "- No preprocessed glucose for labels or features",
        "- Exclude v1: insulin, carbs, Fitbit, Dexcom, participant ID",
        "",
        "### Input, outcome, and windows",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Dense history | Previous **4 h** CGM at 15-min resolution (16 steps) |",
        "| Sparse history | Previous **6 h** user-initiated scans |",
        "| Prediction horizon | Next **2 h** |",
        f"| Prediction stride | **{MODELING_STRIDE_RECOMMENDED} min** |",
        "| Positive target | Any raw historical CGM <70 mg/dL in label window |",
        "| Missingness rule | ≤20% missing CGM slots in input and label windows |",
        "",
        "### Paired comparison requirement (critical)",
        "",
        f"The dense cohort has **{cohort['n_dense_cohort']}** participants; the sparse cohort has **{cohort['n_sparse_cohort']}**. "
        f"For the **headline dense-versus-sparse comparison**, use only the **{cohort['n_common_cohort']} common participants** "
        f"({', '.join(cohort['excluded_no_scans']) or 'none'} excluded from sparse arm) and construct **dense and sparse features at the exact same prediction timestamps** with **identical grouped CV folds** and **identical labels**. "
        "Otherwise performance differences could reflect different participants, windows, or class distributions rather than monitoring density.",
        "",
        f"**Common-cohort window counts ({MODELING_STRIDE_RECOMMENDED}-min stride, 4h history, 2h horizon, 20% rule):** "
        f"**{n_win_common:,}** eligible windows, **{n_pos_common:,}** positive (~{rate_common:.1f}%). "
        "(HUPA0015P contributes no windows under this configuration, so counts match the full dense cohort.)",
        "",
        "### Experiment 1: Core personal-story comparison (primary result)",
        "",
        "**Question:** How much performance is lost when continuous historical CGM is replaced with intermittent user-initiated scans?",
        "",
        "| Component | Specification |",
        "|-----------|---------------|",
        f"| Cohort | Common **{cohort['n_common_cohort']}** participants |",
        "| Timestamps | Same prediction times for dense and sparse XGBoost |",
        "| Folds | Same grouped 5-fold CV assignment |",
        "| Dense model | XGBoost, 4-hour CGM engineered features |",
        "| Sparse model | XGBoost, 6-hour scan-summary features |",
        "| Target | Any raw CGM <70 mg/dL in next 2 h |",
        f"| Stride | {MODELING_STRIDE_RECOMMENDED} minutes |",
        "| Interpretation | SHAP for both XGBoost models |",
        "",
        "### Experiment 2: ML versus DL comparison (dense only)",
        "",
        "**Question:** Does a sequence model outperform engineered tabular features when dense CGM is available?",
        "",
        "| Component | Specification |",
        "|-----------|---------------|",
        f"| Cohort | Common **{cohort['n_common_cohort']}** participants (primary); optional secondary analysis on all **{cohort['n_dense_cohort']}** for dense GRU |",
        "| Models | Dense XGBoost vs small GRU on same dense windows, folds, and outcome |",
        "| GRU design | **Intentionally small:** 1 GRU layer, 16 time steps, small hidden dim, dropout, class weights, early stopping, fixed seeds; **no architecture search** |",
        "| Expectation | GRU may underperform XGBoost given modest sample size and participant heterogeneity — a valid tutorial result |",
        "",
        "### Validation and metrics",
        "",
        "- Grouped **5-fold CV** by participant with **event-aware fold assignment**",
        "- **Identical folds** across all model conditions within an experiment",
        "- Pooled out-of-fold predictions; fold-level and participant-level metrics",
        "- **Primary metric:** AUPRC",
        "- **Secondary:** hypoglycemia recall, precision, F1, AUROC, specificity, confusion matrix",
        "",
        "### Sensitivity analyses (not separate full experiments)",
        "",
        f"1. **Exclude dominant participants** — repeat Experiment 1 XGBoost comparison after removing HUPA0027P and HUPA0028P "
        f"(**{n_win_sens:,}** windows, **{n_pos_sens:,}** positive, ~{rate_sens:.1f}% in remaining "
        f"**{cohort_stride.get('sensitivity_20', {}).get('n_participants', 20)}** participants). "
        "Report whether the conclusion remains directionally similar; emphasize uncertainty given smaller event count.",
        "",
        "2. **Sparse windows by scan availability** — for sparse XGBoost, report performance on: all eligible windows; windows with ≥1 prior scan in 6 h; windows with no prior scan. "
        "Separates performance loss from missing recent observations vs loss despite having intermittent observations.",
        "",
        "## 12. Open questions",
        "",
        "- Libre `Tipo de registro` semantics (inferred, not documented).",
        "- Roche carb units (grams vs exchanges).",
        "- HUPA0010P: preprocessed glucose exists but raw export has no tipo-0 CGM.",
        "",
        "## 13. Final feasibility verdict",
        "",
        "**Feasibility stage complete. Proceed to modeling.**",
        "",
        "The audit provides a transparent participant funnel, validates raw data structure, quantifies interpolation risk, "
        "explains window attrition, documents scan sparsity, and addresses event concentration.",
        "",
        "Locked design summary:",
        "",
        f"- **Experiment 1 (primary):** paired dense vs sparse XGBoost on **{cohort['n_common_cohort']}** participants, same timestamps and folds",
        "- **Experiment 2:** dense XGBoost vs small GRU on same dense windows",
        f"- **Windows:** {n_win_common:,} eligible / {n_pos_common:,} positive at 30-min stride",
        f"- **Robustness:** exclude HUPA0027P/HUPA0028P ({n_win_sens:,} windows); stratify sparse results by scan availability",
        "- **Primary threat remains participant concentration** — report per-participant and fold-level metrics",
        "",
        "## Corrections after consistency review",
        "",
        "- Separated participant funnel: 25 discovered → 24 parseable → 23 with CGM → 22 sparse/common cohort.",
        f"- Insulin pump counts reconciled from `dataset_inventory.csv`: {pump['n_with_pump']} total "
        f"({pump['n_medtronic']} Medtronic, {pump['n_roche']} Roche, {pump['n_without_pump']} none).",
        f"- Added unweighted ({match_summary['unweighted_mean_pct_matched']:.1f}%) and weighted "
        f"({match_summary['weighted_pct_matched']:.1f}%) scan-match rates.",
        "- Corrected sparse feasibility: window-level sparsity, not participant-subset sparsity.",
        f"- Preprocessed audit expanded to all {len(raw_vs_pp)} dense-cohort participants.",
        "- Added prediction-window flow table explaining eligible-window counts.",
        "- Confirmed HUPA0027P Dexcom does not overlap FreeStyle; FreeStyle only.",
        "- Primary CV changed to grouped 5-fold; LOPO demoted to sensitivity analysis.",
        "",
        "## Updates after plot review and high-participant audit",
        "",
        "- Added episode concentration analysis; key metric is episodes/active-day, not headline totals.",
        "- Validated HUPA0027P/HUPA0028P (duplicate export, oscillation sensitivity, episodes/day).",
        f"- Generated Bland-Altman for FreeStyle-only participants ({', '.join(BLAND_ALTMAN_FREESTYLE_ONLY)}).",
        "- Locked sparse history at **6 h** (2 h too sparse); dense at **4 h**.",
        f"- Added **{MODELING_STRIDE_RECOMMENDED}-min prediction stride** and `prediction_stride_summary.csv`.",
        "- Specified sparse XGBoost feature set, evaluation metrics, and presentation wording.",
        "",
        "## Updates after modeling design lock",
        "",
        f"- Defined **common {cohort['n_common_cohort']}-participant cohort** for paired dense-vs-sparse comparison (same timestamps, labels, folds).",
        "- Split experiments: **Experiment 1** (dense vs sparse XGBoost, primary) and **Experiment 2** (dense XGBoost vs small GRU).",
        "- Locked GRU as intentionally small tutorial model; no architecture search.",
        "- Added sensitivity checks: exclude HUPA0027P/HUPA0028P; sparse performance by scan availability.",
        "- Clarified conceptual framing: continuous vs intermittent **sensor access**, not separate measurement modalities.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    ensure_output_dirs()
    participants = discover_participants()
    log.info("Discovered %d participants", len(participants))

    assumptions = [
        "Tipo de registro 0 = historical CGM, 1 = user-initiated scan (inferred from non-null glucose columns).",
        "CGM nominal interval = 15 minutes for window slot expectations.",
        "Hypoglycemia threshold = 70 mg/dL; severe = 54 mg/dL.",
        "Episode separation = 30 min (primary), 60 min (sensitivity).",
        "Scan-CGM matching tolerance = 10 minutes.",
        "Missingness computed as fraction of expected 15-min CGM slots absent in window.",
        "Prediction times: 15-min grid from (t_min + history) to (t_max − horizon) per configuration (audit); 30-min stride recommended for modeling.",
        "Scans used for sparse condition only if timestamp < prediction_time.",
        "Preprocessed insulin zero = no event in 5-min bin (inferred).",
    ]

    # Load all raw data
    log.info("Task 1: Inventory")
    inventory = inventory_dataset(participants)
    inventory.to_csv(OUTPUT_DIR / "dataset_inventory.csv", index=False)

    log.info("Task 2: Raw glucose audit")
    glucose_rows = []
    all_cgm: dict[str, pd.DataFrame] = {}
    all_scans: dict[str, pd.DataFrame] = {}
    all_raw: dict[str, pd.DataFrame] = {}

    for pid in participants:
        raw = load_participant_freestyle(pid)
        all_raw[pid] = raw
        cgm, scans, _ = split_glucose_streams(raw)
        all_cgm[pid] = cgm
        all_scans[pid] = scans
        glucose_rows.append(audit_participant_glucose(pid, raw))
        log.info("  %s: CGM=%d, scans=%d", pid, len(cgm), len(scans))

    glucose_summary = pd.DataFrame(glucose_rows)
    glucose_summary.to_csv(OUTPUT_DIR / "participant_glucose_summary.csv", index=False)

    scan_cols = [
        "participant_id",
        "n_scans",
        "scans_per_day_mean",
        "scans_per_day_median",
        "scans_per_day_min",
        "scans_per_day_max",
        "scan_interval_median_min",
        "scan_longest_gap_hours",
    ]
    for c in scan_cols:
        if c not in glucose_summary.columns:
            glucose_summary[c] = np.nan
    scan_summary = glucose_summary[scan_cols].copy()
    scan_summary.to_csv(OUTPUT_DIR / "scan_frequency_summary.csv", index=False)

    log.info("Task 3: Hypoglycemia episodes")
    episode_rows = []
    all_episodes: dict[str, pd.DataFrame] = {}

    for pid in participants:
        cgm = all_cgm[pid]
        ep70 = detect_episodes(cgm, HYPO_THRESHOLD, EPISODE_SEPARATION_MIN)
        ep54 = detect_episodes(cgm, SEVERE_HYPO_THRESHOLD, EPISODE_SEPARATION_MIN)
        ep70_sens = detect_episodes(cgm, HYPO_THRESHOLD, EPISODE_SEPARATION_SENSITIVITY_MIN)
        all_episodes[pid] = ep70

        rec_days = glucose_summary.loc[
            glucose_summary["participant_id"] == pid, "cgm_recording_days"
        ]
        rec_days = rec_days.iloc[0] if len(rec_days) and pd.notna(rec_days.iloc[0]) else 1

        episode_rows.append(
            {
                "participant_id": pid,
                "episodes_below_70": len(ep70),
                "episodes_below_54": len(ep54),
                "episodes_below_70_60min_sep": len(ep70_sens),
                "median_duration_min_70": ep70["duration_min"].median() if len(ep70) else 0,
                "episodes_per_day_70": len(ep70) / max(rec_days, 1),
            }
        )

    episode_summary = pd.DataFrame(episode_rows)
    episode_summary.to_csv(OUTPUT_DIR / "hypoglycemia_episode_summary.csv", index=False)
    plot_episode_counts_by_participant(episode_summary)
    plot_scan_frequency(scan_summary)

    log.info("Task 4 & 5: Prediction windows")
    cohort = compute_cohort_counts(inventory, glucose_summary)
    pump = compute_pump_counts(inventory)

    window_agg_rows = []
    flow_rows: list[dict] = []
    all_window_rows_for_plots: list[dict] = []
    dense_ids = cohort["dense_cohort_ids"]

    for pid in participants:
        if pid not in dense_ids:
            continue
        for miss in MISSINGNESS_THRESHOLDS:
            rows = evaluate_prediction_windows(
                pid,
                all_cgm[pid],
                all_scans[pid],
                all_episodes[pid],
                missingness_threshold=miss,
            )
            window_agg_rows.extend(aggregate_window_results(rows))
            for history_h in INPUT_HISTORY_H:
                for horizon_h in PREDICTION_HORIZONS_H:
                    flow_rows.append(
                        compute_prediction_window_flow(
                            pid, all_cgm[pid], history_h, horizon_h, miss
                        )
                    )
            if miss == 0.20:
                all_window_rows_for_plots.extend(rows)
        log.info("  %s: done", pid)

    window_df = pd.DataFrame(window_agg_rows)
    window_df.to_csv(OUTPUT_DIR / "prediction_window_feasibility.csv", index=False)
    flow_df = pd.DataFrame(flow_rows)
    flow_df.to_csv(OUTPUT_DIR / "prediction_window_flow.csv", index=False)
    window_flow_agg = aggregate_window_flow(flow_rows)
    window_flow_agg.to_csv(OUTPUT_DIR / "prediction_window_flow_summary.csv", index=False)

    plot_df = pd.DataFrame(all_window_rows_for_plots)
    plot_prediction_feasibility(window_df)
    plot_scan_coverage_by_window(
        plot_df[plot_df["horizon_hours"] == 2] if not plot_df.empty else plot_df
    )

    log.info("Task 6: CGM vs scan comparison")
    compare_rows = []
    for pid in participants:
        cmp = compare_cgm_scans(all_cgm[pid], all_scans[pid], SCAN_MATCH_TOLERANCE_MIN)
        cmp["participant_id"] = pid
        compare_rows.append(cmp)
    cgm_scan_compare = pd.DataFrame(compare_rows)
    cgm_scan_compare.to_csv(OUTPUT_DIR / "cgm_scan_comparison.csv", index=False)
    match_summary = compute_match_rate_summary(cgm_scan_compare)

    log.info("Task 6b: Bland-Altman (FreeStyle-only participants)")
    bland_altman_rows = []
    for pid in BLAND_ALTMAN_FREESTYLE_ONLY:
        summary = bland_altman_summary_and_plot(
            all_cgm[pid], all_scans[pid], pid, freestyle_only=True
        )
        bland_altman_rows.append(summary)
        log.info("  %s: %d matched pairs", pid, summary.get("n_matched", 0))
    pd.DataFrame(bland_altman_rows).to_csv(
        OUTPUT_DIR / "bland_altman_freestyle_only_summary.csv", index=False
    )

    log.info("Task 7: Raw vs preprocessed (all dense-cohort participants)")
    raw_vs_pp_rows = []
    for pid in dense_ids:
        pp = read_preprocessed(pid)
        raw_vs_pp_rows.append(audit_raw_vs_preprocessed(pid, all_cgm[pid], pp))
        log.info("  audited %s", pid)
    raw_vs_pp = pd.DataFrame(raw_vs_pp_rows)
    raw_vs_pp.to_csv(OUTPUT_DIR / "raw_vs_preprocessed_audit.csv", index=False)

    log.info("Task 8: Dexcom vs FreeStyle check")
    dexcom_audit = audit_dexcom_freestyle_overlap("HUPA0027P")

    log.info("Task 9: Episode concentration and stride analysis")
    high_audits = [deep_participant_audit(pid) for pid in HIGH_EPISODE_PARTICIPANTS]
    high_audit_flat = [{k: v for k, v in a.items() if k != "file_ranges"} for a in high_audits]
    pd.DataFrame(high_audit_flat).to_csv(OUTPUT_DIR / "high_participant_audit.csv", index=False)
    episode_concentration_table(episode_summary).to_csv(
        OUTPUT_DIR / "episode_concentration_by_participant.csv", index=False
    )
    stride_rows = compute_stride_window_summaries(all_cgm, dense_ids)
    cohort_stride = compute_cohort_stride_summaries(all_cgm, cohort)
    stride_export = stride_rows + list(cohort_stride.values())
    pd.DataFrame(stride_export).to_csv(OUTPUT_DIR / "prediction_stride_summary.csv", index=False)

    log.info("Generating report")
    report = generate_feasibility_report(
        inventory,
        glucose_summary,
        scan_summary,
        episode_summary,
        window_df,
        window_flow_agg,
        raw_vs_pp,
        cgm_scan_compare,
        cohort,
        pump,
        match_summary,
        dexcom_audit,
        assumptions,
        bland_altman_rows,
        high_audits,
        stride_rows,
        cohort_stride,
    )
    (OUTPUT_DIR / "feasibility_report.md").write_text(report, encoding="utf-8")

    log.info("Done. Outputs in %s", OUTPUT_DIR)


def _get_match_details(
    cgm: pd.DataFrame, scans: pd.DataFrame, tolerance_min: int = 10
) -> tuple[dict, list[dict]]:
    cmp = compare_cgm_scans(cgm, scans, tolerance_min)
    details = []
    if cgm.empty or scans.empty:
        return cmp, details
    cgm_ts = cgm.sort_values("timestamp")
    for _, srow in scans.iterrows():
        st, sg = srow["timestamp"], srow["glucose"]
        diffs = (cgm_ts["timestamp"] - st).abs()
        idx = diffs.idxmin()
        dt_min = diffs.loc[idx].total_seconds() / 60
        if dt_min <= tolerance_min:
            details.append(
                {
                    "time_diff_min": dt_min,
                    "scan_g": sg,
                    "cgm_g": cgm_ts.loc[idx, "glucose"],
                    "abs_diff": abs(sg - cgm_ts.loc[idx, "glucose"]),
                }
            )
    return cmp, details


if __name__ == "__main__":
    main()
