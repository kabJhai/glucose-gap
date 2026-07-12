"""Small single-layer GRU for dense CGM sequences.

Input sequences have shape (n, 16, 2): channel 0 is the (within-window imputed)
glucose value, channel 1 is the observation mask. The value channel is
standardized using training-fold statistics only; the mask channel is left as-is.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

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

log = logging.getLogger(__name__)


def _training_device() -> torch.device:
    """Prefer Apple Silicon GPU (MPS) on M-series Macs, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    folds only, then applied to the held-out fold. No held-out information leaks
    into preprocessing.
    """
    device = _training_device()
    torch.manual_seed(RANDOM_SEED)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_metrics: list[dict] = []
    n_folds_run = sum(1 for f in range(n_folds) if (folds == f).any() and (folds != f).any())
    log.info(
        "GRU: device=%s, %d folds, up to %d epochs each (patience %d, batch %d)",
        device,
        n_folds_run,
        GRU_EPOCHS,
        GRU_PATIENCE,
        GRU_BATCH_SIZE,
    )

    for fold in range(n_folds):
        tr = folds != fold
        va = folds == fold
        if va.sum() == 0 or tr.sum() == 0:
            continue

        log.info("GRU fold %d/%d: train=%d val=%d", fold + 1, n_folds, int(tr.sum()), int(va.sum()))

        # Training-fold statistics for the value channel only.
        tr_vals = sequences[tr][:, :, 0]
        mean = float(tr_vals.mean())
        std = float(tr_vals.std())
        seq_tr = _standardize_value_channel(sequences[tr], mean, std)
        seq_va = _standardize_value_channel(sequences[va], mean, std)

        X_tr = torch.tensor(seq_tr, dtype=torch.float32, device=device)
        y_tr = torch.tensor(y[tr], dtype=torch.float32, device=device)
        X_va = torch.tensor(seq_va, dtype=torch.float32, device=device)
        y_va_np = y[va].astype(int)

        pos = max(float(y_tr.sum().item()), 1.0)
        neg = max(float(len(y_tr) - y_tr.sum().item()), 1.0)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

        model = SmallGRU(input_size=sequences.shape[-1]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optim = torch.optim.Adam(model.parameters(), lr=GRU_LR)
        loader = DataLoader(
            TensorDataset(X_tr.cpu(), y_tr.cpu()),
            batch_size=GRU_BATCH_SIZE,
            shuffle=True,
        )

        best_auprc, best_state, patience = -1.0, None, 0
        for epoch in range(1, GRU_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optim.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item())
                n_batches += 1

            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(X_va)).detach().cpu().numpy()
            auprc = average_precision_score(y_va_np, prob) if len(np.unique(y_va_np)) > 1 else 0.0
            mean_loss = epoch_loss / max(n_batches, 1)
            if auprc > best_auprc:
                best_auprc, best_state, patience = auprc, model.state_dict(), 0
                improved = True
            else:
                patience += 1
                improved = False

            log.info(
                "GRU fold %d epoch %d/%d: loss=%.4f val_auprc=%.3f%s",
                fold + 1,
                epoch,
                GRU_EPOCHS,
                mean_loss,
                auprc,
                " *best*" if improved else f" (patience {patience}/{GRU_PATIENCE})",
            )
            if patience >= GRU_PATIENCE:
                log.info("GRU fold %d early stop at epoch %d (best val_auprc=%.3f)", fold + 1, epoch, best_auprc)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[va] = torch.sigmoid(model(X_va)).detach().cpu().numpy()
        fold_metrics.append({"fold": fold, "val_auprc": best_auprc})
        log.info("GRU fold %d done (best val_auprc=%.3f)", fold + 1, best_auprc)

    log.info("GRU training complete")
    return oof, fold_metrics


@dataclass
class GRUArtifact:
    state_dict: dict
    value_mean: float
    value_std: float
    threshold: float
    input_size: int = 2
    hidden: int = GRU_HIDDEN
    dropout: float = GRU_DROPOUT
    model_name: str = "dense_gru"
    horizon_hours: int = 2


def fit_deployable_gru(sequences: np.ndarray, y: np.ndarray) -> GRUArtifact:
    """Train a deployment GRU on all windows with a held-out slice for early stopping."""
    from sklearn.model_selection import train_test_split

    device = _training_device()
    torch.manual_seed(RANDOM_SEED)
    y = np.asarray(y).astype(int)
    idx = np.arange(len(y))
    stratify = y if len(np.unique(y)) > 1 else None
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=INNER_VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )

    tr_vals = sequences[tr_idx][:, :, 0]
    mean = float(tr_vals.mean())
    std = float(tr_vals.std())
    seq_tr = _standardize_value_channel(sequences[tr_idx], mean, std)
    seq_va = _standardize_value_channel(sequences[va_idx], mean, std)

    X_tr = torch.tensor(seq_tr, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.float32, device=device)
    X_va = torch.tensor(seq_va, dtype=torch.float32, device=device)
    y_va_np = y[va_idx].astype(int)

    pos = max(float(y_tr.sum().item()), 1.0)
    neg = max(float(len(y_tr) - y_tr.sum().item()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    model = SmallGRU(input_size=sequences.shape[-1]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=GRU_LR)
    loader = DataLoader(
        TensorDataset(X_tr.cpu(), y_tr.cpu()),
        batch_size=GRU_BATCH_SIZE,
        shuffle=True,
    )

    best_auprc, best_state, patience = -1.0, None, 0
    for epoch in range(1, GRU_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(X_va)).detach().cpu().numpy()
        auprc = average_precision_score(y_va_np, prob) if len(np.unique(y_va_np)) > 1 else 0.0
        if auprc > best_auprc:
            best_auprc, best_state, patience = auprc, model.state_dict(), 0
        else:
            patience += 1
        if patience >= GRU_PATIENCE:
            break

    if best_state is None:
        best_state = model.state_dict()

    from modeling.metrics import tune_threshold

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(X_va)).detach().cpu().numpy()
    threshold = tune_threshold(y_va_np, val_prob)

    return GRUArtifact(
        state_dict=best_state,
        value_mean=mean,
        value_std=std,
        threshold=float(threshold),
        input_size=sequences.shape[-1],
    )


def save_gru_artifact(artifact: GRUArtifact, path) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(artifact), path)


def load_gru_artifact(path) -> GRUArtifact:
    from pathlib import Path

    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    return GRUArtifact(**data)


def predict_gru_proba(artifact: GRUArtifact, sequences: np.ndarray) -> np.ndarray:
    device = _training_device()
    seq = _standardize_value_channel(sequences, artifact.value_mean, artifact.value_std)
    X = torch.tensor(seq, dtype=torch.float32, device=device)
    model = SmallGRU(
        input_size=artifact.input_size,
        hidden=artifact.hidden,
        dropout=artifact.dropout,
    ).to(device)
    model.load_state_dict(artifact.state_dict)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).detach().cpu().numpy()
