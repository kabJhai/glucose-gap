#!/usr/bin/env python3
"""Export HUPA (or loaded) participant records to canonical glucose.csv layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import discover_participants, load_dataset_config, load_participant_records


def export_participant(pid: str, out_dir: Path) -> Path:
    df = load_participant_records(pid)
    if df.empty:
        raise ValueError(f"No records for {pid}")

    cols = ["timestamp", "record_type"]
    if "glucose_mg_dl" not in df.columns:
        df = df.copy()
        df["glucose_mg_dl"] = df["historical_glucose_mg_dl"].where(
            df["record_type"] == 0, df.get("scan_glucose_mg_dl")
        )
    cols.append("glucose_mg_dl")

    target = out_dir / "participants" / pid
    target.mkdir(parents=True, exist_ok=True)
    path = target / "glucose.csv"
    df[cols].to_csv(path, index=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export participants to canonical layout")
    parser.add_argument("--dataset-config", default="dataset_config.hupa.json")
    parser.add_argument("--output", default="data/exported_cohort", help="Canonical dataset root")
    parser.add_argument("--participant", default=None, help="Single participant; default all discovered")
    args = parser.parse_args()

    load_dataset_config(args.dataset_config)
    out_root = Path(args.output)
    if not out_root.is_absolute():
        out_root = (Path(__file__).resolve().parent.parent / out_root).resolve()

    ids = [args.participant] if args.participant else discover_participants()
    if not ids:
        print("No participants to export.")
        return 1

    for pid in ids:
        path = export_participant(pid, out_root)
        print(f"Wrote {path}")

    example_config = {
        "layout": "canonical",
        "dataset_root": str(out_root.relative_to(Path(__file__).resolve().parent.parent))
        if out_root.is_relative_to(Path(__file__).resolve().parent.parent)
        else str(out_root),
        "canonical": {"participants_subdir": "participants", "glucose_filename": "glucose.csv"},
        "cohort": {"exclude_sparse_no_scan": [], "sensitivity_exclude": []},
    }
    print("\nUse with:")
    print(f"  python -m modeling.train --dataset-config dataset_config.example.json --dataset-root {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
