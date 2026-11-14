"""Preprocessing: cleaning, categorical handling, scaling and splits.

Two protocols are supported:
  * "base"  - replicates the base paper's notebooks (categoricals dropped,
              scaler fitted on ALL data before the split, no deduplication).
  * "xda"   - our protocol (deduplicated, categoricals frequency-encoded,
              scaler fitted on the training part only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from .data import NORMAL, Dataset


def clean(ds: Dataset, dedup: bool = True) -> Dataset:
    X = ds.X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1).to_numpy()
    if dedup:
        keep &= ~pd.concat([X, ds.y_multi.rename("__y")], axis=1).duplicated().to_numpy()
    out = Dataset(ds.name, X, ds.y_bin, ds.y_multi, ds.categorical, ds.time).subset(keep)
    return out


class FrequencyEncoder:
    """Replace each category by its relative frequency in the training data.

    One number per categorical column keeps the input compact for edge use.
    Unseen categories map to 0.
    """

    def fit(self, X: pd.DataFrame, cols: list[str]) -> "FrequencyEncoder":
        self.cols = cols
        self.maps = {c: X[c].value_counts(normalize=True).to_dict() for c in cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cols:
            X[c] = X[c].map(self.maps[c]).fillna(0.0).astype(float)
        return X


@dataclass
class Prepared:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    normal_index: int
    transform: "Pipeline"
    idx_test: np.ndarray      # row positions of the test part in the source Dataset


class Pipeline:
    """Fitted feature pipeline: categorical handling + MinMax scaling."""

    def __init__(self, categorical: list[str], encode_categoricals: bool):
        self.categorical = categorical
        self.encode = encode_categoricals

    def fit(self, X: pd.DataFrame) -> "Pipeline":
        X = self._categoricals(X, fit=True)
        self.feature_names = list(X.columns)
        self.scaler = MinMaxScaler().fit(X.values.astype(float))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._categoricals(X, fit=False)[self.feature_names]
        return self.scaler.transform(X.values.astype(float)).astype(np.float32)

    def _categoricals(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        if not self.categorical:
            return X
        if not self.encode:
            return X.drop(columns=self.categorical)
        if fit:
            self.freq = FrequencyEncoder().fit(X, self.categorical)
        return self.freq.transform(X)


def prepare_static(ds: Dataset, task: str, seed: int, protocol: str = "xda",
                   encode_categoricals: bool | None = None,
                   test_size: float = 0.30, val_size: float = 0.10) -> Prepared:
    """Stratified train/val/test split (Split A)."""
    if protocol not in ("base", "xda"):
        raise ValueError(protocol)
    if encode_categoricals is None:
        encode_categoricals = protocol == "xda"

    y_raw = ds.y_bin.map({0: NORMAL, 1: "Attack"}) if task == "binary" else ds.y_multi
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    class_names = list(le.classes_)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    # The base notebook used 20% of train for validation; ours uses 10%.
    vs = 0.20 if protocol == "base" else val_size
    tr, va = train_test_split(tr, test_size=vs, random_state=seed, stratify=y[tr])

    pipe = Pipeline(ds.categorical, encode_categoricals)
    # Base protocol fits the scaler on all rows before splitting (as in the notebooks).
    pipe.fit(ds.X if protocol == "base" else ds.X.iloc[tr])

    return Prepared(
        pipe.transform(ds.X.iloc[tr]), pipe.transform(ds.X.iloc[va]), pipe.transform(ds.X.iloc[te]),
        y[tr], y[va], y[te], pipe.feature_names, class_names,
        class_names.index(NORMAL), pipe, te,
    )


def leave_family_out(ds: Dataset, family: str, seed: int, test_size: float = 0.30,
                     val_size: float = 0.10):
    """Split B: train only on the remaining classes; the held-out family is
    returned separately so a stream can inject it at a known window.

    Returns (Prepared for known classes, known-class test Dataset,
    Dataset of held-out flows).
    """
    held = (ds.y_multi == family).to_numpy()
    known = ds.subset(~held)
    prep = prepare_static(known, "multiclass", seed, "xda", test_size=test_size, val_size=val_size)
    return prep, known.subset(np.isin(np.arange(len(known.X)), prep.idx_test)), ds.subset(held)
