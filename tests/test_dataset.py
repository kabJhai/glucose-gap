"""Canonical dataset loader tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dataset import (
    DatasetConfig,
    common_cohort_ids,
    discover_participants,
    load_participant_records,
    set_dataset_config,
)


class TestCanonicalDataset(unittest.TestCase):
    def test_canonical_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid = "P001"
            base = root / "participants" / pid
            base.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": ["2020-01-01 10:00:00", "2020-01-01 10:15:00", "2020-01-01 10:30:00"],
                    "record_type": [0, 0, 1],
                    "glucose_mg_dl": [120, 115, 110],
                }
            ).to_csv(base / "glucose.csv", index=False)

            cfg = DatasetConfig(
                layout="canonical",
                root=root,
                canonical={"participants_subdir": "participants", "glucose_filename": "glucose.csv"},
            )
            set_dataset_config(cfg)

            self.assertEqual(discover_participants(cfg), ["P001"])
            df = load_participant_records("P001", cfg)
            self.assertEqual(len(df), 3)
            self.assertIn("historical_glucose_mg_dl", df.columns)
            self.assertEqual(common_cohort_ids(cfg), ["P001"])


if __name__ == "__main__":
    unittest.main()
