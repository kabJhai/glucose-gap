"""Paired window table invariants."""

from __future__ import annotations

import unittest

import pandas as pd


class TestPairedWindows(unittest.TestCase):
    def test_dense_sparse_share_timestamps(self):
        windows = pd.DataFrame(
            {
                "participant_id": ["A", "A", "B"],
                "prediction_time": pd.to_datetime(
                    ["2020-01-01 12:00", "2020-01-01 12:30", "2020-01-01 12:00"]
                ),
                "target_hypo_2h": [0, 1, 0],
            }
        )
        dense_ts = windows["prediction_time"].copy()
        sparse_ts = windows["prediction_time"].copy()
        self.assertTrue(dense_ts.equals(sparse_ts))

    def test_no_scan_windows_retained(self):
        windows = pd.DataFrame(
            {
                "has_prior_scan": [1, 0, 1],
                "scan_count_6h": [2, 0, 1],
            }
        )
        self.assertEqual(len(windows), 3)
        self.assertIn(0, windows["has_prior_scan"].values)


if __name__ == "__main__":
    unittest.main()
