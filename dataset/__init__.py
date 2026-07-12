"""Dataset configuration and loaders for HUPA-UCM and similar CGM datasets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "dataset_config.hupa.json"

CANONICAL_REQUIRED_COLUMNS = {"timestamp", "record_type"}


@dataclass
class DatasetConfig:
    layout: str
    root: Path
    description: str = ""
    hupa_ucm: dict[str, Any] = field(default_factory=dict)
    canonical: dict[str, Any] = field(default_factory=dict)
    cohort: dict[str, list[str]] = field(default_factory=dict)
    config_path: Path | None = None

    @property
    def exclude_sparse_no_scan(self) -> list[str]:
        return list(self.cohort.get("exclude_sparse_no_scan", []))

    @property
    def sensitivity_exclude(self) -> list[str]:
        return list(self.cohort.get("sensitivity_exclude", []))


_active_config: DatasetConfig | None = None


def _resolve_root(raw: str | Path, base: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def load_dataset_config(path: Path | str | None = None) -> DatasetConfig:
    """Load dataset profile from JSON. Falls back to HUPA default."""
    global _active_config

    env_path = os.environ.get("GLUCOSE_GAP_DATASET_CONFIG")
    config_path = Path(path) if path else Path(env_path) if env_path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {config_path}. "
            "Copy dataset_config.example.json or use dataset_config.hupa.json."
        )

    data = json.loads(config_path.read_text(encoding="utf-8"))
    env_root = os.environ.get("GLUCOSE_GAP_DATASET_ROOT")
    root_raw = env_root or data.get("dataset_root", ".")
    cfg = DatasetConfig(
        layout=str(data.get("layout", "hupa_ucm")),
        root=_resolve_root(root_raw, PROJECT_ROOT),
        description=str(data.get("description", "")),
        hupa_ucm=dict(data.get("hupa_ucm", {})),
        canonical=dict(data.get("canonical", {})),
        cohort=dict(data.get("cohort", {})),
        config_path=config_path,
    )
    _active_config = cfg
    _sync_audit_dataset_root(cfg.root)
    return cfg


def sync_audit_dataset_root(root: Path) -> None:
    _sync_audit_dataset_root(root)


def _sync_audit_dataset_root(root: Path) -> None:
    """Keep feasibility_audit loaders aligned with the active dataset root."""
    import sys

    audit_dir = PROJECT_ROOT / "feasibility_audit"
    if str(audit_dir) not in sys.path:
        sys.path.insert(0, str(audit_dir))
    import data_audit

    data_audit.DATASET_ROOT = root


def get_dataset_config() -> DatasetConfig:
    global _active_config
    if _active_config is None:
        return load_dataset_config()
    return _active_config


def set_dataset_config(cfg: DatasetConfig) -> None:
    global _active_config
    _active_config = cfg


def discover_participants(cfg: DatasetConfig | None = None) -> list[str]:
    cfg = cfg or get_dataset_config()
    if cfg.layout == "hupa_ucm":
        raw = cfg.root / cfg.hupa_ucm.get("raw_data_subdir", "Raw_Data")
        prefix = cfg.hupa_ucm.get("participant_prefix", "")
        ids = [p.name for p in raw.iterdir() if p.is_dir()]
        if prefix:
            ids = [pid for pid in ids if pid.startswith(prefix)]
        return sorted(ids)

    if cfg.layout == "canonical":
        base = cfg.root / cfg.canonical.get("participants_subdir", "participants")
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    raise ValueError(f"Unknown dataset layout: {cfg.layout}")


def _normalize_canonical_frame(df: pd.DataFrame, participant_id: str) -> pd.DataFrame:
    missing = CANONICAL_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Canonical glucose.csv missing columns: {sorted(missing)}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["record_type"] = pd.to_numeric(out["record_type"], errors="coerce")
    out = out.dropna(subset=["timestamp", "record_type"])
    out["record_type"] = out["record_type"].astype(int)

    if "glucose_mg_dl" in out.columns and "historical_glucose_mg_dl" not in out.columns:
        out["historical_glucose_mg_dl"] = np.nan
        out["scan_glucose_mg_dl"] = np.nan
        out.loc[out["record_type"] == 0, "historical_glucose_mg_dl"] = pd.to_numeric(
            out.loc[out["record_type"] == 0, "glucose_mg_dl"], errors="coerce"
        )
        out.loc[out["record_type"] == 1, "scan_glucose_mg_dl"] = pd.to_numeric(
            out.loc[out["record_type"] == 1, "glucose_mg_dl"], errors="coerce"
        )

    for col in ("historical_glucose_mg_dl", "scan_glucose_mg_dl", "strip_glucose_mg_dl"):
        if col not in out.columns:
            out[col] = np.nan
        else:
            out[col] = pd.to_numeric(
                out[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

    out["participant_id"] = participant_id
    out["source_file"] = "glucose.csv"
    return out.sort_values("timestamp")


def load_participant_records(
    participant_id: str,
    cfg: DatasetConfig | None = None,
) -> pd.DataFrame:
    """Load one participant's glucose records in the unified FreeStyle-like schema."""
    cfg = cfg or get_dataset_config()

    if cfg.layout == "hupa_ucm":
        import sys

        audit_dir = PROJECT_ROOT / "feasibility_audit"
        if str(audit_dir) not in sys.path:
            sys.path.insert(0, str(audit_dir))
        from data_audit import load_participant_freestyle

        os.environ.setdefault("GLUCOSE_GAP_DATASET_ROOT", str(cfg.root))
        _sync_audit_dataset_root(cfg.root)
        return load_participant_freestyle(participant_id)

    if cfg.layout == "canonical":
        subdir = cfg.canonical.get("participants_subdir", "participants")
        filename = cfg.canonical.get("glucose_filename", "glucose.csv")
        path = cfg.root / subdir / participant_id / filename
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        return _normalize_canonical_frame(df, participant_id)

    raise ValueError(f"Unknown dataset layout: {cfg.layout}")


def split_glucose_streams(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import sys

    audit_dir = PROJECT_ROOT / "feasibility_audit"
    if str(audit_dir) not in sys.path:
        sys.path.insert(0, str(audit_dir))
    from data_audit import split_glucose_streams as _split

    return _split(df)


def participant_has_scans(participant_id: str, cfg: DatasetConfig | None = None) -> bool:
    df = load_participant_records(participant_id, cfg)
    if df.empty or "record_type" not in df.columns:
        return False
    scans = df[df["record_type"] == 1]
    if "scan_glucose_mg_dl" in scans.columns:
        return scans["scan_glucose_mg_dl"].notna().any()
    return len(scans) > 0


def common_cohort_ids(cfg: DatasetConfig | None = None) -> list[str]:
    """Participants with CGM and at least one user scan (paired modeling cohort)."""
    cfg = cfg or get_dataset_config()
    exclude = set(cfg.exclude_sparse_no_scan)
    ids = []
    for pid in discover_participants(cfg):
        raw = load_participant_records(pid, cfg)
        if raw.empty:
            continue
        rt = raw["record_type"]
        if not (rt == 0).any():
            continue
        if pid in exclude:
            continue
        if not participant_has_scans(pid, cfg):
            continue
        ids.append(pid)
    return sorted(ids)
