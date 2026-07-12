"""Grouped fold assignment tests."""

from __future__ import annotations

import unittest

import pandas as pd

from modeling.cv_splits import assign_event_aware_folds


class TestFolds(unittest.TestCase):
    def test_participants_never_split_across_folds(self):
        summary = pd.DataFrame(
            {
                "participant_id": ["P1", "P2", "P3", "P4", "P5"],
                "positive_windows": [40, 30, 20, 10, 5],
                "total_windows": [200, 180, 150, 120, 80],
                "episode_count": [10, 8, 6, 4, 2],
            }
        )
        folds = assign_event_aware_folds(summary, n_folds=5)
        self.assertEqual(len(folds), 5)
        self.assertEqual(set(folds.index), set(summary["participant_id"]))

    def test_fold_assignment_is_deterministic(self):
        summary = pd.DataFrame(
            {
                "participant_id": ["A", "B", "C"],
                "positive_windows": [50, 20, 5],
                "total_windows": [300, 200, 100],
                "episode_count": [12, 6, 2],
            }
        )
        f1 = assign_event_aware_folds(summary, n_folds=3)
        f2 = assign_event_aware_folds(summary, n_folds=3)
        pd.testing.assert_series_equal(f1, f2)


if __name__ == "__main__":
    unittest.main()
