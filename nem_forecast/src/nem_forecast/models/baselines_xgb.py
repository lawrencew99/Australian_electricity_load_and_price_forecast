from __future__ import annotations

import numpy as np
import xgboost as xgb

from nem_forecast.config import XGBoostConfig


class XGBBaseline:
    def __init__(self, xgb_cfg: XGBoostConfig | None = None):
        self._cfg = xgb_cfg or XGBoostConfig()
        self._model: xgb.XGBRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = xgb.XGBRegressor(
            n_estimators=self._cfg.n_estimators,
            max_depth=self._cfg.max_depth,
            learning_rate=self._cfg.learning_rate,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call fit before predict")
        return self._model.predict(X)
