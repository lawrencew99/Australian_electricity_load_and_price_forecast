from __future__ import annotations

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from nem_forecast.config import SarimaConfig


class SarimaBaseline:
    """Univariate SARIMAX 占位：在训练序列末尾 forecast `steps` 步。"""

    def __init__(self, sarima_cfg: SarimaConfig | None = None):
        self._cfg = sarima_cfg or SarimaConfig()
        self._fitted = None

    def fit_forecast(self, y_train: np.ndarray, steps: int) -> np.ndarray:
        y = np.asarray(y_train, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < max(50, steps * 2):
            raise ValueError("SARIMA: insufficient finite training points")

        order = self._cfg.order
        seasonal_order = self._cfg.seasonal_order
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        self._fitted = model.fit(disp=False)
        fc = self._fitted.forecast(steps=steps)
        return np.asarray(fc, dtype=float)
