"""Class-balanced focal loss (Lin et al., 2017; Cui et al., 2019)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def class_balanced_weights(counts: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """alpha_c proportional to (1 - beta) / (1 - beta^n_c), normalised to sum to C."""
    counts = np.maximum(np.asarray(counts, dtype=float), 1.0)
    w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    return w * len(counts) / w.sum()


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t), on logits."""

    def __init__(self, alpha: np.ndarray | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - logp.exp()) ** self.gamma) * logp
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()
