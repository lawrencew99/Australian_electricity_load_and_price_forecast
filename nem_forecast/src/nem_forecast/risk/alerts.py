from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AlertConfig:
    spike_prob_high: float = 0.35
    p90_margin_ratio: float = 1.05  # 预测 P90 相对阈值的缓冲


def interval_to_alerts(
    spike_prob: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    spike_threshold: float,
    cfg: AlertConfig | None = None,
) -> pd.DataFrame:
    """
    将分位数区间与尖峰概率转为简易预警与敞口建议占位。
    """
    cfg = cfg or AlertConfig()
    spike_prob = np.asarray(spike_prob, dtype=float).ravel()
    p10 = np.asarray(p10, dtype=float).ravel()
    p50 = np.asarray(p50, dtype=float).ravel()
    p90 = np.asarray(p90, dtype=float).ravel()

    alert_level = np.where(spike_prob >= cfg.spike_prob_high, "high", "normal")
    alert_level = np.where(
        (spike_prob < cfg.spike_prob_high) & (p90 >= spike_threshold * cfg.p90_margin_ratio),
        "elevated",
        alert_level,
    )

    exposure_hint = np.where(
        alert_level == "high",
        "reduce_exposure",
        np.where(alert_level == "elevated", "tighten_risk_limits", "maintain"),
    )

    return pd.DataFrame(
        {
            "spike_prob": spike_prob,
            "P10": p10,
            "P50": p50,
            "P90": p90,
            "alert_level": alert_level,
            "exposure_hint": exposure_hint,
        }
    )
