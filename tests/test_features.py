"""Feature horizon-safety and sparse timing tests."""

from __future__ import annotations

import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

from modeling.features import dense_tabular_features, sparse_scan_features


class TestFeatures(unittest.TestCase):
    def setUp(self):
        base = pd.Timestamp("2020-06-01 12:00")
        idx = pd.date_range(base - timedelta(hours=4), base, freq="15min", inclusive="left")
        self.glucose = pd.Series(np.linspace(110, 80, len(idx)), index=idx)
        self.pred_time = base

    def test_future_spike_does_not_change_dense_features(self):
        feats_before = dense_tabular_features(self.pred_time, self.glucose)
        future = self.glucose.copy()
        future.loc[self.pred_time + timedelta(minutes=30)] = 40.0
        feats_after = dense_tabular_features(self.pred_time, future)
        self.assertEqual(feats_before["glucose_current"], feats_after["glucose_current"])
        self.assertEqual(feats_before["glucose_mean_4h"], feats_after["glucose_mean_4h"])

    def test_scans_at_or_after_prediction_time_excluded(self):
        t = self.pred_time
        scans = pd.DataFrame(
            {
                "timestamp": [t - timedelta(hours=1), t, t + timedelta(minutes=15)],
                "glucose": [100.0, 50.0, 40.0],
            }
        )
        feats = sparse_scan_features(t, scans)
        self.assertEqual(feats["n_scans_6h"], 1.0)
        self.assertEqual(feats["most_recent_scan"], 100.0)

    def test_no_scan_window_flagged(self):
        scans = pd.DataFrame(columns=["timestamp", "glucose"])
        feats = sparse_scan_features(self.pred_time, scans)
        self.assertEqual(feats["no_scan"], 1.0)
        self.assertTrue(np.isnan(feats["most_recent_scan"]))


if __name__ == "__main__":
    unittest.main()
