"""Standard, leak-free training loop for RUL regressors.

Enforces fair baselines:
  - Trains exclusively on X_train.
  - Early stops and selects checkpoints using X_val (never peeking at test).
  - Evaluates on X_val, X_val_pseudo, and X_test.
  - Records full training curves and timing.
"""
import copy
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rul.metrics import all_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: str, batch: int = 1024, y_scale: float = 1.0) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i : i + batch], device=device)
        out.append(model(xb).cpu())
    return torch.cat(out).numpy() * y_scale


def fit(
    model: nn.Module,
    data: Dict,
    epochs: int = 60,
    lr: float = 1e-3,
    batch: int = 256,
    patience: int = 10,
    cap: int = 125,
    device: Optional[str] = None,
    verbose: bool = True,
    y_scale: float = 1.0,
    select: str = "all",
) -> Dict:
    """select='all': early-stop on every validation window (legacy); select='tl': on the test-like validation set (protocol v2).
    y_scale: train on y/y_scale (e.g. 125) so outputs live near [0,1]; predict() rescales."""
    device = device or pick_device()
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(data["X_train"]), torch.as_tensor(data["y_train"])),
        batch_size=batch,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    bad_epochs = 0
    t0 = time.time()

    train_rmse_history = []
    val_rmse_history = []
    tl_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb / y_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        train_rmse = (total_loss / len(train_loader.dataset)) ** 0.5 * y_scale
        train_rmse_history.append(train_rmse)

        # Validation evaluation (sliding windows)
        val_pred = predict(model, data["X_val"], device, y_scale=y_scale).clip(0, cap)
        val_rmse = all_metrics(data["y_val"], val_pred)["RMSE"]
        val_rmse_history.append(val_rmse)
        if select == "tl":
            val_rmse = all_metrics(data["y_val_tl"], predict(model, data["X_val_tl"], device, y_scale=y_scale).clip(0, cap))["RMSE"]
            tl_history.append(val_rmse)

        # Checkpoint selection strictly on validation
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if verbose:
            mark = "*" if bad_epochs == 0 else " "
            print(f"epoch {epoch:3d}  train RMSE {train_rmse:6.2f}  val RMSE {val_rmse:6.2f} {mark}")

        if bad_epochs >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    model.load_state_dict(best_state)

    val_pred = predict(model, data["X_val"], device, y_scale=y_scale).clip(0, cap)
    test_pred = predict(model, data["X_test"], device, y_scale=y_scale).clip(0, cap)

    results = {
        "val": all_metrics(data["y_val"], val_pred),
        "test": all_metrics(data["y_test"], test_pred),
        "test_pred": test_pred.tolist(),
        "train_rmse_history": train_rmse_history,
        "val_rmse_history": val_rmse_history,
        "epochs_run": epoch,
        "train_seconds": time.time() - t0,
    }

    if "X_val_tl" in data:
        results["val_tl"] = all_metrics(data["y_val_tl"], predict(model, data["X_val_tl"], device, y_scale=y_scale).clip(0, cap))
        results["val_tl_rmse_history"] = tl_history
    if "X_val_pseudo" in data and "y_val_pseudo" in data:
        val_pseudo_pred = predict(model, data["X_val_pseudo"], device, y_scale=y_scale).clip(0, cap)
        results["val_pseudo"] = all_metrics(data["y_val_pseudo"], val_pseudo_pred)

    return results
