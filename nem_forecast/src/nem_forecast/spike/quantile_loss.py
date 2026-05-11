from __future__ import annotations

import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """分位数（pinball）损失，quantile in (0,1)。"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    e = y_true - y_pred
    return float(np.mean(np.maximum(quantile * e, (quantile - 1) * e)))
