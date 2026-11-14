"""Detection and efficiency metrics."""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize


def detection_metrics(y_true: np.ndarray, proba: np.ndarray, class_names: list[str],
                      normal_index: int) -> dict:
    y_pred = proba.argmax(1)
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    normal = y_true == normal_index
    # FPR: share of normal flows flagged as any attack.
    fpr = float((y_pred[normal] != normal_index).mean()) if normal.any() else float("nan")
    attack = ~normal
    # Detection rate: share of attack flows flagged as any attack (class may be wrong).
    dr = float((y_pred[attack] != normal_index).mean()) if attack.any() else float("nan")

    if len(class_names) == 2:
        pos = 1 - normal_index
        pr_auc = average_precision_score(y_true == pos, proba[:, pos])
    else:
        Y = label_binarize(y_true, classes=labels)
        present = Y.sum(0) > 0
        pr_auc = average_precision_score(Y[:, present], proba[:, present], average="macro")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "fpr": fpr,
        "detection_rate": dr,
        "pr_auc": float(pr_auc),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(class_names)},
        "confusion_matrix": cm.tolist(),
    }


def _time_per_flow(predict, x: np.ndarray, n_flows: int) -> tuple[float, float]:
    for i in range(min(50, n_flows)):        # warm-up
        predict(x[i:i + 1])
    t0 = time.perf_counter()
    for i in range(n_flows):                 # one flow at a time, as at a gateway
        predict(x[i:i + 1])
    per_flow_ms = 1000 * (time.perf_counter() - t0) / n_flows
    t0 = time.perf_counter()
    predict(x)
    return per_flow_ms, n_flows / (time.perf_counter() - t0)


def efficiency(model, n_features: int, n_flows: int = 2000) -> dict:
    """Single-thread CPU latency per flow, throughput and model size.
    Works for PyTorch modules and for objects with predict_proba (sklearn/xgboost)."""
    import pickle
    x = np.random.default_rng(0).random((n_flows, n_features), dtype=np.float32)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if isinstance(model, torch.nn.Module):
            model = model.to("cpu").eval()
            with torch.no_grad():
                per_flow, thr = _time_per_flow(lambda a: model(torch.from_numpy(a)), x, n_flows)
            params = sum(p.numel() for p in model.parameters())
            path = os.path.join(tempfile.gettempdir(), "_xda_size.pt")
            torch.save(model.state_dict(), path)
        else:
            per_flow, thr = _time_per_flow(model.predict_proba, x, min(n_flows, 500))
            params = None
            path = os.path.join(tempfile.gettempdir(), "_xda_size.pkl")
            with open(path, "wb") as fh:   # closed before os.remove (required on Windows)
                pickle.dump(model, fh)
        size_mb = os.path.getsize(path) / 2**20
        os.remove(path)
    finally:
        torch.set_num_threads(threads)
    return {"latency_ms_per_flow": per_flow, "throughput_flows_s": thr, "params": params, "model_mb": size_mb}
