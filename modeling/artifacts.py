"""Save and load deployable model artifacts for inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from modeling.config import (
    ARTIFACTS_DIR,
    GRU_BATCH_SIZE,
    GRU_DROPOUT,
    GRU_EPOCHS,
    GRU_HIDDEN,
    GRU_LR,
    GRU_PATIENCE,
    INNER_VAL_FRACTION,
    RANDOM_SEED,
)
from modeling.metrics import tune_threshold


@dataclass
class XGBArtifact:
    model: Any
    imputer: SimpleImputer
    threshold: float
    feature_names: list[str]
    model_name: str
    positive_class: int = 1
    horizon_hours: int = 2

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in X_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for {self.model_name}: {missing}")
        X = self.imputer.transform(X_df[self.feature_names].values)
        return self.model.predict_proba(X)[:, self.positive_class]

    def predict_alert(self, X_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        prob = self.predict_proba(X_df)
        return prob, (prob >= self.threshold).astype(int)


def _new_xgb(scale_pos_weight: float):
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def fit_deployable_xgb(
    X_df: pd.DataFrame,
    y: np.ndarray,
    *,
    model_name: str,
    feature_names: list[str],
) -> XGBArtifact:
    """Fit on all windows for deployment; tune threshold on a held-out slice."""
    y = np.asarray(y).astype(int)
    idx = np.arange(len(y))
    stratify = y if len(np.unique(y)) > 1 else None
    core_idx, inner_idx = train_test_split(
        idx,
        test_size=INNER_VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X_df[feature_names].values)

    pos = max(int(y[core_idx].sum()), 1)
    neg = max(len(core_idx) - int(y[core_idx].sum()), 1)
    model = _new_xgb(neg / pos)
    model.fit(X[core_idx], y[core_idx])
    inner_prob = model.predict_proba(X[inner_idx])[:, 1]
    threshold = tune_threshold(y[inner_idx], inner_prob)

    pos = max(int(y.sum()), 1)
    neg = max(len(y) - int(y.sum()), 1)
    final_model = _new_xgb(neg / pos)
    final_model.fit(X, y)

    return XGBArtifact(
        model=final_model,
        imputer=imp,
        threshold=float(threshold),
        feature_names=list(feature_names),
        model_name=model_name,
    )


def save_xgb_artifact(artifact: XGBArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_xgb_artifact(path: Path) -> XGBArtifact:
    return joblib.load(path)


def write_artifact_manifest(
    out_dir: Path,
    *,
    models_saved: list[str],
    n_training_windows: int,
    n_positive: int,
    skip_gru: bool,
) -> None:
    manifest = {
        "project": "Glucose Gap",
        "purpose": "Saved alert models (research prototype)",
        "saved_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "horizon_hours": 2,
        "positive_class_label": "hypoglycemia_below_70_mg_dl_in_next_2h",
        "models_saved": models_saved,
        "n_training_windows": n_training_windows,
        "n_positive": n_positive,
        "skip_gru": skip_gru,
        "citation": "https://github.com/kabJhai/glucose-gap",
        "license": "MIT",
        "disclaimer": "Research and education only. Not for clinical decision-making.",
    }
    (out_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def save_deployable_models(
    dense_df: pd.DataFrame,
    sparse_df: pd.DataFrame,
    sequences: np.ndarray,
    y: np.ndarray,
    dense_cols: list[str],
    sparse_cols: list[str],
    *,
    skip_gru: bool = False,
    out_dir: Path = ARTIFACTS_DIR,
) -> list[str]:
    """Train deployment models on the full cohort and persist artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    dense_art = fit_deployable_xgb(
        dense_df, y, model_name="dense_xgb", feature_names=dense_cols
    )
    save_xgb_artifact(dense_art, out_dir / "dense_xgb.joblib")
    saved.append("dense_xgb")

    sparse_art = fit_deployable_xgb(
        sparse_df, y, model_name="sparse_xgb", feature_names=sparse_cols
    )
    save_xgb_artifact(sparse_art, out_dir / "sparse_xgb.joblib")
    saved.append("sparse_xgb")

    if not skip_gru:
        try:
            from modeling.gru_model import fit_deployable_gru, save_gru_artifact

            gru_bundle = fit_deployable_gru(sequences, y)
            save_gru_artifact(gru_bundle, out_dir / "dense_gru.pt")
            saved.append("dense_gru")
        except Exception:
            pass

    write_artifact_manifest(
        out_dir,
        models_saved=saved,
        n_training_windows=int(len(y)),
        n_positive=int(np.asarray(y).sum()),
        skip_gru=skip_gru,
    )
    return saved
