# -*- coding: utf-8 -*-
"""
Standalone demo for the Conv1d + Transformer regression model.

This file does not modify project config, read Excel files, save checkpoints,
or write result files. It only builds synthetic data in memory and runs a tiny
training loop to verify that the model can be instantiated and called.
"""

import random

import numpy as np
import torch

from loss_metrics import calculate_rmse, calculate_rrmse, get_criterion
from net import EnhancedTimeSeriesModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_synthetic_dataset(
    num_samples: int = 128,
    window_size: int = 6,
    feature_dim: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small in-memory regression dataset with shape (B, T, F)."""
    x = np.random.normal(size=(num_samples, window_size, feature_dim)).astype(np.float32)

    recent_mean = x[:, -3:, :3].mean(axis=(1, 2))
    trend = x[:, -1, 3] - x[:, 0, 3]
    interaction = x[:, :, 4].mean(axis=1) * x[:, -1, 5]
    noise = np.random.normal(scale=0.03, size=num_samples)

    y = 2.0 * recent_mean + 0.8 * trend - 0.5 * interaction + noise
    y = y.astype(np.float32).reshape(-1, 1)

    return torch.from_numpy(x), torch.from_numpy(y)


def run_demo() -> None:
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x, y = build_synthetic_dataset()
    split = int(len(x) * 0.8)
    x_train, x_val = x[:split].to(device), x[split:].to(device)
    y_train, y_val = y[:split].to(device), y[split:].to(device)

    model = EnhancedTimeSeriesModel(input_size=x.shape[2]).to(device)
    criterion = get_criterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Demo device:", device)
    print("Input shape :", tuple(x.shape), "(batch, window, features)")
    print("Target shape:", tuple(y.shape))

    for epoch in range(1, 6):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={loss.item():.6f} | "
            f"val_loss={val_loss.item():.6f}"
        )

    model.eval()
    with torch.no_grad():
        pred = model(x_val).cpu().numpy()

    y_true = y_val.cpu().numpy()
    rmse = calculate_rmse(y_true, pred)
    rrmse = calculate_rrmse(y_true, pred)

    print("\nForward output shape:", tuple(pred.shape))
    print(f"Demo RMSE : {rmse:.6f}")
    print(f"Demo RRMSE: {rrmse:.6f}")
    print("Done. No files were written.")


if __name__ == "__main__":
    run_demo()
