"""Windowed streams with known drift onsets (Split B, experiments E3/E4).

Drift is injected into RAW feature frames and the fitted Pipeline is applied
afterwards, so scale drift really changes the inputs the model sees.
Every window keeps its labels, which only the label-based baselines, the
online learner and the retraining loop are allowed to use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Dataset

# UNSW byte/packet volume columns; scaled to emulate e.g. new firmware or MTU changes.
UNSW_VOLUME = ["sbytes", "dbytes", "spkts", "dpkts", "sload", "dload", "smean", "dmean"]
WUSTL_VOLUME = ["SrcBytes", "DstBytes", "TotBytes", "SrcPkts", "DstPkts", "TotPkts",
                "SrcLoad", "DstLoad", "Load"]


@dataclass
class Stream:
    windows: list[pd.DataFrame]          # raw feature frames, one per window
    labels: list[np.ndarray]             # y_multi strings per window
    onset: int                           # index of the first drifted window (len(windows) if none)
    kind: str
    expected: str = "none"               # ground-truth diagnosis: "scale", "concept" or "none"
    perturbed: list[str] = field(default_factory=list)   # columns actually changed


def _draw(ds: Dataset, n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """n flows from ds without replacement when possible."""
    idx = rng.choice(len(ds.X), n, replace=len(ds.X) < n)
    return ds.X.iloc[idx].reset_index(drop=True), ds.y_multi.to_numpy()[idx]


def _split(X: pd.DataFrame, y: np.ndarray, window: int):
    n = len(X) // window
    return ([X.iloc[i * window:(i + 1) * window].copy() for i in range(n)],
            [y[i * window:(i + 1) * window] for i in range(n)])


def novel_attack_stream(known: Dataset, novel: Dataset, window: int = 5000, n_pre: int = 10,
                        n_post: int = 10, novel_frac: float = 0.2, seed: int = 0) -> Stream:
    """E3: known traffic, then from `onset` each window mixes in `novel_frac` held-out flows."""
    rng = np.random.default_rng(seed)
    X, y = _draw(known, (n_pre + n_post) * window, rng)
    wins, labs = _split(X, y, window)
    n_nov = int(window * novel_frac)
    for i in range(n_pre, n_pre + n_post):
        Xn, yn = _draw(novel, n_nov, rng)
        Xw = pd.concat([wins[i].iloc[: window - n_nov], Xn], ignore_index=True)
        yw = np.concatenate([labs[i][: window - n_nov], yn])
        order = rng.permutation(window)
        wins[i], labs[i] = Xw.iloc[order].reset_index(drop=True), yw[order]
    return Stream(wins, labs, n_pre, "novel_attack", expected="concept")


def scale_drift_stream(base: Dataset, kind: str, cols: list[str] | None = None, factor: float = 3.0,
                       window: int = 5000, n_pre: int = 10, n_post: int = 10, transition: int = 5,
                       seed: int = 0) -> Stream:
    """E4 feature-behaviour drift, four types (following Agbedanu et al., P4).

    sudden      - all post-onset windows scaled by `factor`
    gradual     - share of scaled flows rises linearly to 100% over `transition` windows
    incremental - scaling factor itself rises linearly from 1 to `factor`
    recurring   - drifted and clean blocks alternate every `transition` windows
    """
    rng = np.random.default_rng(seed)
    X, y = _draw(base, (n_pre + n_post) * window, rng)
    wins, labs = _split(X, y, window)
    cols = [c for c in (cols or UNSW_VOLUME + WUSTL_VOLUME) if c in X.columns]

    for t, i in enumerate(range(n_pre, n_pre + n_post)):
        w = wins[i]
        ramp = min(1.0, (t + 1) / transition)
        if kind == "sudden":
            mask, f = np.ones(len(w), bool), factor
        elif kind == "gradual":
            mask, f = rng.random(len(w)) < ramp, factor
        elif kind == "incremental":
            mask, f = np.ones(len(w), bool), 1.0 + (factor - 1.0) * ramp
        elif kind == "recurring":
            mask, f = np.full(len(w), (t // transition) % 2 == 0), factor
        else:
            raise ValueError(kind)
        w[cols] = w[cols].astype(float)
        w.loc[mask, cols] = w.loc[mask, cols] * f
    return Stream(wins, labs, n_pre, kind, expected="scale", perturbed=cols)


def class_mix_stream(base: Dataset, target: str, share: float = 0.5, window: int = 5000,
                     n_pre: int = 10, n_post: int = 10, seed: int = 0) -> Stream:
    """E4 class-mixture drift: from `onset`, `share` of each window is one known class
    (e.g. a DoS campaign), the rest keeps the original mixture."""
    rng = np.random.default_rng(seed)
    X, y = _draw(base, (n_pre + n_post) * window, rng)
    wins, labs = _split(X, y, window)
    tgt = base.subset(base.y_multi == target)
    n_t = int(window * share)
    for i in range(n_pre, n_pre + n_post):
        Xt, yt = _draw(tgt, n_t, rng)
        order = rng.permutation(window)
        wins[i] = pd.concat([wins[i].iloc[: window - n_t], Xt], ignore_index=True).iloc[order].reset_index(drop=True)
        labs[i] = np.concatenate([labs[i][: window - n_t], yt])[order]
    return Stream(wins, labs, n_pre, f"class_mix_{target}", expected="concept")


def chronological_stream(ds: Dataset, window: int = 5000, start: int = 0) -> Stream:
    """Natural order by flow start time (WUSTL has timestamps; UNSW partitions do not).
    No injected drift, so onset = number of windows."""
    if ds.time is None:
        raise ValueError(f"{ds.name} has no timestamps")
    order = np.argsort(ds.time.to_numpy(), kind="stable")[start:]
    X = ds.X.iloc[order].reset_index(drop=True)
    wins, labs = _split(X, ds.y_multi.to_numpy()[order], window)
    return Stream(wins, labs, len(wins), "chronological")
