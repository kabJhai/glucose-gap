"""Leakage and cohort design checks."""

from __future__ import annotations

import unittest

import pandas as pd

from modeling.config import RANDOM_SEED, SENSITIVITY_EXCLUDE
from modeling.cv_splits import window_fold_column


class TestLeakage(unittest.TestCase):
    def test_participant_id_not_in_feature_columns(self):
        dense_cols = [
            "glucose_current",
            "glucose_mean_4h",
            "prop_below_70",
            "hour_sin",
        ]
        sparse_cols = [
            "n_scans_6h",
            "most_recent_scan",
            "no_scan",
        ]
        for col in dense_cols + sparse_cols:
            self.assertNotIn("participant", col.lower())

    def test_windows_map_to_single_fold_per_participant(self):
        windows = pd.DataFrame(
            {
                "participant_id": ["P1", "P1", "P2", "P2"],
                "prediction_time": pd.to_datetime(
                    ["2020-01-01 10:00", "2020-01-01 10:30", "2020-01-01 11:00", "2020-01-01 11:30"]
                ),
            }
        )
        participant_folds = pd.Series({"P1": 0, "P2": 1}, name="fold")
        fold_col = window_fold_column(windows, participant_folds)
        p1_folds = set(fold_col[:2])
        p2_folds = set(fold_col[2:])
        self.assertEqual(len(p1_folds), 1)
        self.assertEqual(len(p2_folds), 1)

    def test_fixed_random_seed(self):
        self.assertEqual(RANDOM_SEED, 42)

    def test_sensitivity_excludes_dominant_participants(self):
        self.assertIn("HUPA0027P", SENSITIVITY_EXCLUDE)
        self.assertIn("HUPA0028P", SENSITIVITY_EXCLUDE)


if __name__ == "__main__":
    unittest.main()
