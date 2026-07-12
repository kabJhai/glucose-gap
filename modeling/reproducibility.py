"""Centralized random seeds and run manifest for reproducible experiments."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime, timezone

import numpy as np

from modeling.config import OUTPUT_DIR, RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Fix seeds for NumPy, Python random, and PyTorch (if installed)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def write_run_manifest(steps: list[str] | None = None) -> None:
    """Record environment and commands for reproducibility."""
    try:
        import pandas as pd
        import sklearn
        import xgboost

        versions = {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        }
    except ImportError:
        versions = {}

    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        pass

    manifest = {
        "project": "Glucose Gap: What Is Lost Between Glucose Checks?",
        "subtitle": "Predicting Hypoglycemia from Continuous and Intermittent Glucose Observations",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "random_seed": RANDOM_SEED,
        "steps_completed": steps or ["modeling"],
        "package_versions": versions,
        "commands": {
            "audit": "python feasibility_audit/data_audit.py",
            "modeling": "python -m modeling.train",
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
