"""Detector models (PyTorch ports of the base paper's Keras models).

CNN follows the paper's stated stack:
  Conv1D(64,k3)->BN->ReLU->MaxPool(2)->Conv1D(128,k3)->BN->ReLU->Dropout(.5)
  ->Flatten->Dense(64)->ReLU->Dropout(.5)->output
Models output logits; binary tasks use 2 logits + softmax, which is
equivalent to the paper's single sigmoid unit.
"""
from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int, conv1: int = 64, conv2: int = 128,
                 kernel: int = 3, dense: int = 64, dropout: float = 0.5):
        super().__init__()
        # Valid padding as in the base notebook; same padding only when the input is too
        # short for two valid convolutions (e.g. the k=5 SHAP Top-k model).
        pad = 0 if n_features >= 2 * kernel + 2 else kernel // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, conv1, kernel, padding=pad), nn.BatchNorm1d(conv1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv1, conv2, kernel, padding=pad), nn.BatchNorm1d(conv2), nn.ReLU(),
            nn.Dropout(dropout),
        )
        length = (n_features + 2 * pad - kernel + 1) // 2 + 2 * pad - kernel + 1
        if length < 1:
            raise ValueError(f"too few features for the CNN: {n_features}")
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(conv2 * length, dense), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dense, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (N, F) flow vectors; reshape to a 1-channel sequence of length F.
        return self.head(self.features(x.unsqueeze(1)))


class LSTMNet(nn.Module):
    """LSTM(64)->Dropout->output. The base notebook fed each flow as a
    sequence of length 1 (shape (1, F)); `seq_as_features=True` instead
    treats the F features as F time steps."""

    def __init__(self, n_features: int, n_classes: int, hidden: int = 64, dropout: float = 0.5,
                 seq_as_features: bool = False):
        super().__init__()
        self.seq_as_features = seq_as_features
        self.lstm = nn.LSTM(1 if seq_as_features else n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.unsqueeze(-1) if self.seq_as_features else x.unsqueeze(1)
        _, (h, _) = self.lstm(seq)
        return self.out(self.drop(h[-1]))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
