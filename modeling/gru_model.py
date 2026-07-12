"""Small single-layer GRU for dense CGM sequences.

Input sequences have shape (n, 16, 2): channel 0 is the (within-window imputed)
glucose value, channel 1 is the observation mask. The value channel is
standardized using training-fold statistics only; the mask channel is left as-is.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from modeling.config import (
    GRU_BATCH_SIZE,
    GRU_DROPOUT,
    GRU_EPOCHS,
    GRU_HIDDEN,
    GRU_LR,
    GRU_PATIENCE,
    RANDOM_SEED,
)


class SmallGRU(nn.Module):
    def __init__(self, input_size: int = 2, hidden: int = GRU_HIDDEN, dropout: float = GRU_DROPOUT):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last)).squeeze(-1)


def _standardize_value_channel(seqs: np.ndarray, mean: float, std: float) -> np.ndarray:
    out = seqs.copy()
    out[:, :, 0] = (out[:, :, 0] - mean) / (std if std > 1e-8 else 1.0)
    return out


def train_gru_oof(
    sequences: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
) -> tuple[np.ndarray, list[dict]]:
    """Train GRU per fold; return out-of-fold probabilities and per-fold metrics.

    Standardization statistics for the value channel are computed on the training
    folds only, then applied to the held-out fold — no held-out information leaks
    into preprocessing.
    """
    torch.manual_seed(RANDOM_SEED)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_metrics: list[dict] = []

    for fold in range(n_folds):
        tr = folds != fold
        va = folds == fold
        if va.sum() == 0 or tr.sum() == 0:
            continue

        # Training-fold statistics for the value channel only.
        tr_vals = sequences[tr][:, :, 0]
        mean = float(tr_vals.mean())
        std = float(tr_vals.std())
        seq_tr = _standardize_value_channel(sequences[tr], mean, std)
        seq_va = _standardize_value_channel(sequences[va], mean, std)

        X_tr = torch.tensor(seq_tr, dtype=torch.float32)
        y_tr = torch.tensor(y[tr], dtype=torch.float32)
        X_va = torch.tensor(seq_va, dtype=torch.float32)
        y_va_np = y[va].astype(int)

        pos = max(y_tr.sum().item(), 1.0)
        neg = max((len(y_tr) - y_tr.sum()).item(), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

        model = SmallGRU(input_size=sequences.shape[-1])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.Adam(model.parameters(), lr=GRU_LR)
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=GRU_BATCH_SIZE, shuffle=True)

        best_auprc, best_state, patience = -1.0, None, 0
        for _ in range(GRU_EPOCHS):
            model.train()
            for xb, yb in loader:
                optim.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optim.step()

            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(X_va)).numpy()
            auprc = average_precision_score(y_va_np, prob) if len(np.unique(y_va_np)) > 1 else 0.0
            if auprc > best_auprc:
                best_auprc, best_state, patience = auprc, model.state_dict(), 0
            else:
                patience += 1
                if patience >= GRU_PATIENCE:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[va] = torch.sigmoid(model(X_va)).numpy()
        fold_metrics.append({"fold": fold, "val_auprc": best_auprc})

    return oof, fold_metrics
