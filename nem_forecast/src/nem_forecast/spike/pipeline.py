from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from nem_forecast.spike.quantile_loss import pinball_loss


class SpikeTwoStagePlaceholder:
    """
    两阶段占位：
    1) 分类：是否尖峰（y > threshold）
    2) 回归：sklearn 梯度提升分位数回归（P10 / P50 / P90）
    """

    def __init__(self, spike_threshold: float, seed: int = 42):
        self.spike_threshold = spike_threshold
        self.seed = seed
        self.clf: GradientBoostingClassifier | None = None
        self.q_models: dict[float, GradientBoostingRegressor] = {}

    def fit(self, X: np.ndarray, y_price: np.ndarray) -> None:
        y = np.asarray(y_price, dtype=float).ravel()
        X = np.asarray(X, dtype=float)
        z = (y > self.spike_threshold).astype(int)
        self.clf = GradientBoostingClassifier(random_state=self.seed)
        self.clf.fit(X, z)

        for q in (0.1, 0.5, 0.9):
            m = GradientBoostingRegressor(loss="quantile", alpha=q, random_state=self.seed)
            m.fit(X, y)
            self.q_models[q] = m

    def predict_spike_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("call fit first")
        return self.clf.predict_proba(X)[:, 1]

    def predict_quantiles(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if not self.q_models:
            raise RuntimeError("call fit first")
        out: dict[str, np.ndarray] = {}
        for q, m in self.q_models.items():
            key = f"P{int(q * 100)}"
            out[key] = m.predict(X)
        return out

    @staticmethod
    def evaluate_pinball(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        return pinball_loss(y_true, y_pred, quantile)
