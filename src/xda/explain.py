"""SHAP attributions for the detector and the reference profile used by the monitor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class Explainer:
    """Per-flow attributions for the predicted class.

    method="shap"  uses shap.GradientExplainer (expected gradients) on the
                   PyTorch model; method="deep" uses shap.DeepExplainer (DeepLIFT-SHAP,
                   the base paper's choice); method="gxi" uses gradient x (input - baseline),
                   a cheap fallback with the same output shape.
    """

    def __init__(self, model: nn.Module, background: np.ndarray, method: str = "shap"):
        self.model = model.to("cpu").eval()
        self.method = method
        self.background = torch.from_numpy(background)
        if method == "shap":
            import shap
            self._shap = shap.GradientExplainer(self.model, self.background)
        elif method == "deep":
            import shap
            self._shap = shap.DeepExplainer(self.model, self.background)

    def attributions(self, X: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Returns (attributions (N, F) for each flow's predicted class, predicted class (N,))."""
        x = torch.from_numpy(X)
        with torch.no_grad():
            pred = self.model(x).argmax(1).numpy()
        if self.method in ("shap", "deep"):
            sv = (self._shap.shap_values(x, rseed=seed) if self.method == "shap"
                  else self._shap.shap_values(x, check_additivity=False))
            sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)   # (N, F, C)
            return sv[np.arange(len(X)), :, pred], pred
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        logits.gather(1, torch.from_numpy(pred)[:, None]).sum().backward()
        base = self.background.mean(0, keepdim=True)
        return (x.grad * (x - base)).detach().numpy(), pred


def global_ranking(attr: np.ndarray, feature_names: list[str]) -> list[tuple[str, float]]:
    imp = np.abs(attr).mean(0)
    order = np.argsort(-imp)
    return [(feature_names[i], float(imp[i])) for i in order]


def fidelity(predict, X: np.ndarray, y: np.ndarray, attr: np.ndarray, ks=(1, 3, 5, 10),
             n_random: int = 5, seed: int = 0) -> list[dict]:
    """Macro-F1 drop when the top-k SHAP features are zeroed (as in the base paper),
    compared with zeroing k random features. `predict` maps X -> class indices."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    base = f1_score(y, predict(X), average="macro", zero_division=0)
    order = np.argsort(-np.abs(attr).mean(0))
    rows = []
    for k in ks:
        Xm = X.copy()
        Xm[:, order[:k]] = 0.0
        top = f1_score(y, predict(Xm), average="macro", zero_division=0)
        rand = []
        for _ in range(n_random):
            Xr = X.copy()
            Xr[:, rng.choice(X.shape[1], k, replace=False)] = 0.0
            rand.append(f1_score(y, predict(Xr), average="macro", zero_division=0))
        rows.append({"k": k, "f1_full": base, "f1_top_masked": top, "f1_random_masked": float(np.mean(rand)),
                     "drop_top": base - top, "drop_random": base - float(np.mean(rand))})
    return rows


def stability(attr: np.ndarray, k: int = 10, n_boot: int = 20, seed: int = 0) -> dict:
    """Bootstrap the explained samples; Jaccard overlap of each resample's top-k set
    with the full-sample top-k set."""
    rng = np.random.default_rng(seed)
    full = set(np.argsort(-np.abs(attr).mean(0))[:k].tolist())
    jac = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(attr), len(attr))
        top = set(np.argsort(-np.abs(attr[idx]).mean(0))[:k].tolist())
        jac.append(len(full & top) / len(full | top))
    return {"k": k, "n_boot": n_boot, "jaccard_mean": float(np.mean(jac)), "jaccard_std": float(np.std(jac)),
            "jaccard_min": float(np.min(jac))}


@dataclass
class ReferenceProfile:
    """What 'normal reasoning' looks like on drift-free training data."""
    shap: np.ndarray          # (N_ref, F) attributions
    inputs: np.ndarray        # (N_ref, F) scaled inputs
    pred: np.ndarray          # (N_ref,) predicted classes
    topk: set[int]            # indices of the top-k features by mean |SHAP|
    shap_scale: np.ndarray    # per-feature std of SHAP, for normalised distances

    @classmethod
    def build(cls, explainer: Explainer, X_ref: np.ndarray, k: int = 10) -> "ReferenceProfile":
        attr, pred = explainer.attributions(X_ref)
        topk = set(np.argsort(-np.abs(attr).mean(0))[:k].tolist())
        scale = attr.std(0) + 1e-9
        return cls(attr, X_ref, pred, topk, scale)
