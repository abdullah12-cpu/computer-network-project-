"""Training loop with early stopping on validation Macro-F1."""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch: int = 4096, dev=None) -> np.ndarray:
    dev = dev or next(model.parameters()).device
    model.eval()
    out = [torch.softmax(model(torch.from_numpy(X[i:i + batch]).to(dev)), 1).cpu()
           for i in range(0, len(X), batch)]
    return torch.cat(out).numpy()


def fit(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, loss_fn: nn.Module,
        max_epochs: int = 100, patience: int = 10, batch_size: int = 128,
        lr: float = 1e-3, f1_average: str = "macro", verbose: bool = True) -> dict:
    dev = device()
    model.to(dev)
    loss_fn.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train.astype(np.int64))

    best_f1, best_state, wait, history = -1.0, None, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        running = 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            if len(b) < 2:   # BatchNorm needs more than one sample
                continue
            xb, yb = Xt[b].to(dev), yt[b].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(b)

        val_pred = predict_proba(model, X_val, dev=dev).argmax(1)
        val_f1 = f1_score(y_val, val_pred, average=f1_average, zero_division=0)
        history.append({"epoch": epoch, "loss": running / len(perm), "val_f1": val_f1})
        if verbose:
            print(f"  epoch {epoch:3d}  loss {running / len(perm):.4f}  val_f1 {val_f1:.4f}", flush=True)

        if val_f1 > best_f1:
            best_f1, best_state, wait = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return {"history": history, "best_val_f1": best_f1, "epochs": len(history),
            "train_seconds": time.perf_counter() - t0}


def expand_head(model: nn.Module, n_new: int) -> nn.Module:
    """Add `n_new` output classes (e.g. a newly confirmed attack family), keeping old weights."""
    old = model.head[-1]
    new = nn.Linear(old.in_features, old.out_features + n_new)
    with torch.no_grad():
        new.weight[:old.out_features] = old.weight
        new.bias[:old.out_features] = old.bias
    model.head[-1] = new
    return model


def fine_tune(model: nn.Module, X: np.ndarray, y: np.ndarray, loss_fn: nn.Module,
              epochs: int = 5, lr: float = 1e-4, batch_size: int = 128) -> dict:
    """Retraining loop step: brief fine-tuning on recent analyst-labelled flows,
    mixed by the caller with a replay sample of the original training data."""
    dev = device()
    model.to(dev)
    loss_fn.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(Xt[b].to(dev)), yt[b].to(dev)).backward()
            opt.step()
    model.eval()
    return {"fine_tune_seconds": time.perf_counter() - t0}
