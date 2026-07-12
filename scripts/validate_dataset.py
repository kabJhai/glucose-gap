#!/usr/bin/env python3
"""Validate a dataset layout before training or inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import (
    common_cohort_ids,
    discover_participants,
    load_dataset_config,
    load_participant_records,
    sync_audit_dataset_root,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Glucose Gap dataset layout")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-root", default=None)
    args = parser.parse_args()

    cfg = load_dataset_config(args.dataset_config)
    if args.dataset_root:
        root = Path(args.dataset_root)
        if not root.is_absolute():
            root = (Path(__file__).resolve().parent.parent / root).resolve()
        cfg.root = root
        sync_audit_dataset_root(cfg.root)

    participants = discover_participants(cfg)
    if not participants:
        print(f"FAIL: no participants found under {cfg.root} (layout={cfg.layout})")
        return 1

    ok = 0
    for pid in participants:
        df = load_participant_records(pid, cfg)
        if df.empty:
            print(f"  {pid}: empty")
            continue
        n_cgm = int((df["record_type"] == 0).sum()) if "record_type" in df.columns else 0
        n_scan = int((df["record_type"] == 1).sum()) if "record_type" in df.columns else 0
        print(f"  {pid}: rows={len(df)} cgm={n_cgm} scans={n_scan}")
        ok += 1

    cohort = common_cohort_ids(cfg)
    print(f"\nLayout: {cfg.layout}")
    print(f"Root:   {cfg.root}")
    print(f"Participants found: {len(participants)} (non-empty: {ok})")
    print(f"Paired modeling cohort (CGM + scans): {len(cohort)}")
    if cohort:
        print(f"  {', '.join(cohort[:8])}{'...' if len(cohort) > 8 else ''}")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
