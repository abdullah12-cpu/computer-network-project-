"""Non-neural baselines (Step 6).

All batch detectors expose fit(X, y) and predict_proba(X) -> (N, C).
OnlineKNN is the label-dependent adaptive learner (Agbedanu et al., P4),
evaluated prequentially on streams (predict, then learn the true label).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier


class RFDetector:
    def __init__(self, seed: int = 0, n_estimators: int = 100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBDetector:
    def __init__(self, seed: int = 0, n_classes: int = 2):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
                                   random_state=seed, n_jobs=-1,
                                   objective="binary:logistic" if n_classes == 2 else "multi:softprob")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class HybridRFIF:
    """Obaidat et al. style hybrid: supervised RF plus an Isolation Forest trained on
    benign flows only. A flow the RF calls normal but the IF calls anomalous is
    re-assigned to the RF's most likely attack class."""

    def __init__(self, normal_index: int, seed: int = 0, contamination: float = 0.01):
        self.normal = normal_index
        self.rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
        self.iso = IsolationForest(contamination=contamination, random_state=seed, n_jobs=-1)

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.iso.fit(X[y == self.normal])
        return self

    def predict_proba(self, X):
        p = self.rf.predict_proba(X).copy()
        flagged = (p.argmax(1) == self.normal) & (self.iso.predict(X) == -1)
        if flagged.any() and p.shape[1] > 1:
            p[flagged, self.normal] = 0.0
            p[flagged] /= p[flagged].sum(1, keepdims=True) + 1e-12
        return p


class OnlineKNN:
    """Self-Adjusting-Memory KNN from river when available, otherwise a sliding-window KNN.
    Works on dict inputs; learns string labels so unseen families can be added online."""

    def __init__(self, n_neighbors: int = 5, window: int = 1000):
        from river import neighbors
        if hasattr(neighbors, "SAMKNNClassifier"):
            self.model, self.kind = neighbors.SAMKNNClassifier(n_neighbors=n_neighbors), "SAM-KNN"
        else:   # newer river versions dropped SAM-KNN
            self.model = neighbors.KNNClassifier(n_neighbors=n_neighbors,
                                                 engine=neighbors.LazySearch(window_size=window))
            self.kind = "KNN (sliding window)"

    @staticmethod
    def _row(x):
        return {i: float(v) for i, v in enumerate(x)}

    def learn(self, X, y):
        for x, label in zip(X, y):
            self.model.learn_one(self._row(x), label)

    def predict_then_learn(self, X, y) -> np.ndarray:
        """Prequential step: predict each flow, then learn its true label."""
        preds = []
        for x, label in zip(X, y):
            r = self._row(x)
            preds.append(self.model.predict_one(r))
            self.model.learn_one(r, label)
        return np.array(preds, dtype=object)
