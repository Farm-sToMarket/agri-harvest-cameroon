"""Time-series models for weather-aware yield prediction.

Two architectures:
- **HybridYieldModel**: LSTM + Tabular (fast, good baseline).
- **TransformerYieldModel**: Transformer Encoder + Tabular (SOTA 2025,
  better on long-range climate dependencies like 15-day droughts).

Both share the same Dataset, collate function, and training loop.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from models.v1.config import TIMESERIES_CONFIG


# ── Helpers ──────────────────────────────────────────────────────────────────


def _masked_mean(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean pooling that ignores padding positions.

    Parameters
    ----------
    tensor : (batch, max_seq_len, hidden)
    lengths : (batch,) — actual lengths per sample

    Returns
    -------
    (batch, hidden) — averaged over real timesteps only.
    """
    device = tensor.device
    max_len = tensor.size(1)
    # (batch, max_len)
    mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    # (batch, max_len, 1) for broadcasting
    mask = mask.unsqueeze(2).float()
    summed = (tensor * mask).sum(dim=1)  # (batch, hidden)
    return summed / lengths.unsqueeze(1).float().clamp(min=1)


# ── Dataset ─────────────────────────────────────────────────────────────────


class CropTimeSeriesDataset(Dataset):
    """PyTorch Dataset for hybrid (static + sequence) inputs.

    Parameters
    ----------
    static_features : np.ndarray of shape (n_samples, n_static)
    sequences : list[np.ndarray], each of shape (timesteps_i, n_weather)
    targets : np.ndarray of shape (n_samples,)
    """

    def __init__(
        self,
        static_features: np.ndarray,
        sequences: list[np.ndarray],
        targets: np.ndarray,
    ):
        self.static = torch.tensor(static_features, dtype=torch.float32)
        self.sequences = [
            torch.tensor(s, dtype=torch.float32) for s in sequences
        ]
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.static[idx], self.sequences[idx], self.targets[idx]


def collate_timeseries(batch):
    """Collate function with dynamic padding for variable-length sequences.

    Returns (statics, seqs_padded, targets, lengths).
    """
    statics, seqs, targets = zip(*batch)
    statics = torch.stack(statics)
    targets = torch.stack(targets)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return statics, seqs_padded, targets, lengths


# ── Model ───────────────────────────────────────────────────────────────────


class HybridYieldModel(nn.Module):
    """LSTM (weather sequences) + Dense (static features) -> yield prediction.

    Parameters
    ----------
    static_dim : number of static input features
    weather_dim : number of weather variables per timestep
    config : optional dict overriding TIMESERIES_CONFIG
    """

    def __init__(
        self,
        static_dim: int,
        weather_dim: int | None = None,
        config: dict | None = None,
    ):
        super().__init__()
        cfg = config or TIMESERIES_CONFIG
        weather_dim = weather_dim or len(cfg["weather_features"])
        hidden = cfg["lstm_hidden"]
        n_layers = cfg["lstm_layers"]

        # Tabular branch
        self.tabular = nn.Sequential(
            nn.Linear(static_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=weather_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.3 if n_layers > 1 else 0.0,
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(128 + hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        static: torch.Tensor,
        sequence: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tab_out = self.tabular(static)

        lstm_out, _ = self.lstm(sequence)
        if lengths is not None:
            lstm_out = _masked_mean(lstm_out, lengths)
        else:
            lstm_out = lstm_out.mean(dim=1)

        combined = torch.cat([tab_out, lstm_out], dim=1)
        return self.head(combined)


# ── Transformer model ───────────────────────────────────────────────────────


def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Generate sinusoidal positional encoding (Vaswani et al., 2017)."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, max_len, d_model)


class TransformerYieldModel(nn.Module):
    """Transformer Encoder (weather sequences) + Dense (static) -> yield.

    Uses sinusoidal positional encoding and pre-norm Transformer layers
    with GELU activation. Better than LSTM on long-range climate patterns.

    Parameters
    ----------
    static_dim : number of static input features
    seq_dim : number of weather variables per timestep
    d_model : Transformer embedding dimension
    nhead : number of attention heads
    num_layers : number of Transformer encoder layers
    dim_feedforward : hidden size of feedforward sublayer
    max_seq_len : maximum sequence length (days) for positional encoding
    """

    def __init__(
        self,
        static_dim: int = 70,
        seq_dim: int | None = None,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        max_seq_len: int = 180,
        config: dict | None = None,
    ):
        super().__init__()
        cfg = config or TIMESERIES_CONFIG
        seq_dim = seq_dim or len(cfg["weather_features"])

        # Tabular branch
        self.tabular = nn.Sequential(
            nn.Linear(static_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Weather -> d_model projection
        self.input_proj = nn.Linear(seq_dim, d_model)

        # Sinusoidal positional encoding (fixed, not learned)
        self.register_buffer(
            "pos_encoding", _sinusoidal_encoding(max_seq_len, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.3,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(128 + d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        static: torch.Tensor,
        sequence: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tab_out = self.tabular(static)

        # Project + add positional encoding
        seq = self.input_proj(sequence)
        seq = seq + self.pos_encoding[:, : seq.size(1), :]

        # Build padding mask if lengths provided
        src_key_padding_mask = None
        if lengths is not None:
            max_len = seq.size(1)
            # True = masked (padding position)
            src_key_padding_mask = (
                torch.arange(max_len, device=seq.device).unsqueeze(0)
                >= lengths.unsqueeze(1)
            )

        # Transformer encoder
        trans_out = self.transformer(seq, src_key_padding_mask=src_key_padding_mask)

        # Pooling
        if lengths is not None:
            seq_out = _masked_mean(trans_out, lengths)
        else:
            seq_out = trans_out.mean(dim=1)

        combined = torch.cat([tab_out, seq_out], dim=1)
        return self.head(combined)


# ── Training loop ───────────────────────────────────────────────────────────


def train_hybrid_model(
    train_dataset: CropTimeSeriesDataset,
    val_dataset: CropTimeSeriesDataset | None = None,
    static_dim: int = 70,
    weather_dim: int = 5,
    config: dict | None = None,
    device: str = "cpu",
) -> tuple[HybridYieldModel, list[float]]:
    """Train the hybrid LSTM+Tabular model.

    Returns (model, list_of_epoch_losses).
    """
    cfg = config or TIMESERIES_CONFIG

    model = HybridYieldModel(static_dim, weather_dim, cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_timeseries,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=(device != "cpu"),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=collate_timeseries,
            num_workers=cfg.get("num_workers", 0),
            pin_memory=(device != "cpu"),
        )

    losses: list[float] = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        for static, seq, target, lengths in train_loader:
            static = static.to(device)
            seq = seq.to(device)
            target = target.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            pred = model(static, seq, lengths).squeeze(-1)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Validation for early stopping
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for static, seq, target, lengths in val_loader:
                    static = static.to(device)
                    seq = seq.to(device)
                    target = target.to(device)
                    lengths = lengths.to(device)
                    pred = model(static, seq, lengths).squeeze(-1)
                    val_loss += criterion(pred, target).item()
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            msg = f"  Epoch {epoch:>3d} — Loss: {avg_loss:.4f}"
            if val_loader is not None:
                msg += f"  Val: {val_loss:.4f}"
            print(msg)

    # Restore best model if validation was used
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, losses
