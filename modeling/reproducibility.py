"""Centralized random seeds for reproducible tutorial runs."""

from __future__ import annotations

import os
import random

import numpy as np

from modeling.config import RANDOM_SEED


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
