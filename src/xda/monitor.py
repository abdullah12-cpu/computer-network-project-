"""Label-free drift monitors and baselines (Module C, Step 6).

ShapDriftMonitor (core contribution) compares each window's SHAP attribution
distribution with the reference profile, overall and per predicted class:
  * KS test per feature (Bonferroni over all tests)  -> number of shifted features
  * normalised Wasserstein distance per feature       -> effect size, top shifted features
  * Jaccard overlap of the top-k SHAP feature set     -> reasoning change
The same KS test on raw inputs drives the diagnosis rule:
  inputs shifted but importance ranking stable -> scale drift (re-normalise)
  importance ranking changed                   -> concept drift / unseen attack (retrain)
Raw SHAP values move whenever inputs move, so the diagnosis compares the
relative importance share and the top-k set rather than raw values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

if TYPE_CHECKING:   # avoid importing torch just for the type hint
    from .explain import ReferenceProfile


def _ks_pvalues(ref: np.ndarray, win: np.ndarray) -> np.ndarray:
    return np.array([ks_2samp(ref[:, j], win[:, j]).pvalue for j in range(ref.shape[1])])


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def _share(attr: np.ndarray) -> np.ndarray:
    imp = np.abs(attr).mean(0)
    return imp / (imp.sum() + 1e-12)


@dataclass
class WindowStats:
    n_shap_shifted: int
    n_input_shifted: int
    wasserstein: np.ndarray       # per-feature normalised distance
    jaccard: float
    share_shift: float            # L1 distance between importance-share vectors


@dataclass
class WindowResult:
    alarm: bool                   # XDA system alarm (SHAP alarm or input alarm), with diagnosis
    shap_alarm: bool              # SHAP signals only (compared against input-KS in the paper)
    input_alarm: bool
    diagnosis: str                # "none" | "scale" | "concept"
    stats: WindowStats | None = None
    shifted_features: list[tuple[str, float]] = field(default_factory=list)
    signals: dict = field(default_factory=dict)   # which signal fired and its margin over threshold


class ShapDriftMonitor:
    """mode="full" uses KS + Wasserstein + Jaccard; mode="ks_only" uses the KS count alone
    (ablation "Monitor: SHAP-KS only")."""

    def __init__(self, ref: ReferenceProfile, feature_names: list[str], alpha: float = 0.01,
                 k: int = 10, per_class: bool = True, min_class_n: int = 30, mode: str = "full"):
        self.ref, self.names, self.alpha, self.k = ref, feature_names, alpha, k
        self.per_class, self.min_class_n, self.mode = per_class, min_class_n, mode
        self.ref_share = _share(ref.shap)
        # Defaults until calibrate() is called.
        self.m, self.m_input, self.w_thr, self.tau, self.share_thr = 3, 3, np.inf, 0.5, np.inf

    # ------------------------------------------------------------------ stats
    def _shifted(self, ref_vals, win_vals, ref_pred, win_pred) -> int:
        pairs = [(ref_vals, win_vals)]
        if self.per_class and win_pred is not None:
            for c in np.unique(ref_pred):
                r, w = ref_vals[ref_pred == c], win_vals[win_pred == c]
                if len(r) >= self.min_class_n and len(w) >= self.min_class_n:
                    pairs.append((r, w))
        d = ref_vals.shape[1]
        shifted = np.zeros(d, bool)
        for r, w in pairs:
            shifted |= _ks_pvalues(r, w) < self.alpha / (d * len(pairs))
        return int(shifted.sum())

    def stats(self, shap_win: np.ndarray, x_win: np.ndarray, pred_win: np.ndarray | None = None) -> WindowStats:
        n_shap = self._shifted(self.ref.shap, shap_win, self.ref.pred, pred_win)
        n_in = self._shifted(self.ref.inputs, x_win, self.ref.pred, pred_win)
        w = np.array([wasserstein_distance(self.ref.shap[:, j], shap_win[:, j]) / self.ref.shap_scale[j]
                      for j in range(shap_win.shape[1])])
        topk = set(np.argsort(-np.abs(shap_win).mean(0))[:self.k].tolist())
        return WindowStats(n_shap, n_in, w, _jaccard(self.ref.topk, topk),
                           float(np.abs(_share(shap_win) - self.ref_share).sum()))

    # ------------------------------------------------------------ calibration
    def calibrate(self, windows: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
                  target_far: float = 0.05) -> None:
        """Set thresholds on drift-free windows so that about `target_far` of them alarm.
        The false-alarm budget is split evenly across the active SHAP signals."""
        S = [self.stats(*w) for w in windows]
        n_sig = 3 if self.mode == "full" else 1
        q = 1.0 - target_far / n_sig
        self.m = int(np.floor(np.quantile([s.n_shap_shifted for s in S], q))) + 1
        self.m_input = int(np.floor(np.quantile([s.n_input_shifted for s in S], 1.0 - target_far))) + 1
        self.w_thr = float(np.quantile([s.wasserstein.max() for s in S], q))
        self.tau = float(np.quantile([s.jaccard for s in S], 1.0 - q))
        self.share_thr = float(np.quantile([s.share_shift for s in S], q))

    # ----------------------------------------------------------------- check
    def check(self, shap_win: np.ndarray, x_win: np.ndarray, pred_win: np.ndarray | None = None,
              n_report: int = 5) -> WindowResult:
        s = self.stats(shap_win, x_win, pred_win)
        ks_alarm = s.n_shap_shifted >= self.m
        if self.mode == "full":
            shap_alarm = ks_alarm or s.wasserstein.max() > self.w_thr or s.jaccard < self.tau
        else:
            shap_alarm = ks_alarm
        input_alarm = s.n_input_shifted >= self.m_input

        reasoning_changed = s.share_shift > self.share_thr or s.jaccard < self.tau
        if not (shap_alarm or input_alarm):
            diagnosis = "none"
        elif reasoning_changed:
            diagnosis = "concept"
        else:
            diagnosis = "scale"
        top = np.argsort(-s.wasserstein)[:n_report]
        signals = {
            "ks": {"fired": bool(ks_alarm), "value": s.n_shap_shifted, "threshold": self.m},
            "wasserstein": {"fired": bool(self.mode == "full" and s.wasserstein.max() > self.w_thr),
                            "value": float(s.wasserstein.max()), "threshold": self.w_thr},
            "jaccard": {"fired": bool(self.mode == "full" and s.jaccard < self.tau),
                        "value": s.jaccard, "threshold": self.tau},
            "input_ks": {"fired": bool(input_alarm), "value": s.n_input_shifted, "threshold": self.m_input},
        }
        return WindowResult(bool(shap_alarm or input_alarm), bool(shap_alarm), bool(input_alarm),
                            diagnosis, s, [(self.names[j], float(s.wasserstein[j])) for j in top], signals)


# ====================================================================== baselines
class InputKSMonitor:
    """Per-feature KS on raw inputs (Obaidat et al. style)."""

    def __init__(self, ref_inputs: np.ndarray, alpha: float = 0.01, m: int = 3):
        self.ref, self.alpha, self.m = ref_inputs, alpha, m

    def _count(self, x):
        return int((_ks_pvalues(self.ref, x) < self.alpha / x.shape[1]).sum())

    def calibrate(self, windows: list[np.ndarray], target_far: float = 0.05) -> None:
        self.m = int(np.floor(np.quantile([self._count(w) for w in windows], 1.0 - target_far))) + 1

    def check(self, x_win: np.ndarray) -> bool:
        return self._count(x_win) >= self.m


class ConfidenceKSMonitor:
    """KS test on the detector's max-probability distribution."""

    def __init__(self, ref_conf: np.ndarray, alpha: float = 0.01):
        self.ref, self.alpha = ref_conf, alpha

    def calibrate(self, windows: list[np.ndarray], target_far: float = 0.05) -> None:
        p = np.array([ks_2samp(self.ref, w).pvalue for w in windows])
        self.alpha = min(self.alpha, float(np.quantile(p, target_far)))

    def check(self, conf_win: np.ndarray) -> bool:
        return bool(ks_2samp(self.ref, conf_win).pvalue < self.alpha)


class KSWINMonitor:
    """KSWIN (Raab et al.) on each input feature, as in Aly & Alotaibi (P1).
    Alarms when any feature's detector fires within the window (alpha Bonferroni-split)."""

    def __init__(self, n_features: int, alpha: float = 0.005, window_size: int = 100,
                 stat_size: int = 30, seed: int = 0):
        from river import drift
        a = alpha / n_features
        self.dets = [drift.KSWIN(alpha=a, window_size=window_size, stat_size=stat_size, seed=seed + j)
                     for j in range(n_features)]

    def warm_up(self, x_ref: np.ndarray) -> None:
        for row in x_ref:
            for det, v in zip(self.dets, row):
                det.update(float(v))

    def check(self, x_win: np.ndarray) -> bool:
        fired = False
        for row in x_win:
            for det, v in zip(self.dets, row):
                det.update(float(v))
                fired |= det.drift_detected
        return fired


class ErrorRateMonitor:
    """Label-based upper bound: ADWIN or DDM on the per-flow binary error stream
    (error = attack/normal decision wrong). Needs ground-truth labels."""

    def __init__(self, kind: str = "adwin"):
        from river import drift
        self.binary = kind == "ddm"     # DDM takes booleans, ADWIN takes numbers
        self.det = drift.binary.DDM() if self.binary else drift.ADWIN()

    def check(self, errors: np.ndarray) -> bool:
        fired = False
        for e in errors:
            self.det.update(bool(e) if self.binary else float(e))
            fired |= self.det.drift_detected
        return fired


# ======================================================================= scoring
def drift_scores(alarms: list[bool], onset: int) -> dict:
    """Detection delay (windows from onset to first alarm), false-alarm rate before onset,
    missed detection when no post-onset window alarms."""
    alarms = np.asarray(alarms, dtype=bool)
    pre, post = alarms[:onset], alarms[onset:]
    first = int(np.argmax(post)) if post.any() else None
    return {"detected": first is not None, "missed": bool(len(post) and not post.any()),
            "delay_windows": first,
            "false_alarm_rate": float(pre.mean()) if len(pre) else float("nan"),
            "post_alarm_rate": float(post.mean()) if len(post) else float("nan")}


def diagnosis_accuracy(results: list[WindowResult], onset: int, expected: str) -> float:
    """Share of alarmed post-onset windows whose diagnosis matches the injected drift type."""
    hits = [r.diagnosis == expected for r in results[onset:] if r.alarm]
    return float(np.mean(hits)) if hits else float("nan")


def localisation(shifted: list[tuple[str, float]], perturbed: list[str]) -> float:
    """Precision@k of the reported top shifted features against the columns actually perturbed."""
    if not perturbed:
        return float("nan")
    k = min(len(shifted), len(perturbed))
    return len({n for n, _ in shifted[:k]} & set(perturbed)) / max(1, k)
