"""Deployment artifact round-trip tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def _has_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


class TestArtifacts(unittest.TestCase):
    @unittest.skipUnless(_has_xgboost(), "xgboost not installed")
    def test_xgb_artifact_save_load_predict(self):
        from modeling.artifacts import fit_deployable_xgb, load_xgb_artifact, save_xgb_artifact

        X, y = make_classification(
            n_samples=80,
            n_features=6,
            weights=[0.9],
            random_state=42,
        )
        cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        art = fit_deployable_xgb(df, y, model_name="dense_xgb", feature_names=cols)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dense_xgb.joblib"
            save_xgb_artifact(art, path)
            loaded = load_xgb_artifact(path)

        prob, alert = loaded.predict_alert(df)
        self.assertEqual(len(prob), len(y))
        self.assertTrue(np.all((prob >= 0) & (prob <= 1)))
        self.assertEqual(set(np.unique(alert)), {0, 1})


if __name__ == "__main__":
    unittest.main()
