from __future__ import annotations

import pandas as pd

from nem_forecast.data.schemas import DEMAND_COL, RRP_COL, TIMESTAMP_COL


def forward_backward_fill(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """缺失插补：前向填充后向后填充（按时间排序后的序列）。"""
    out = df.copy()
    use = cols or [c for c in out.columns if c != TIMESTAMP_COL and pd.api.types.is_numeric_dtype(out[c])]
    for c in use:
        if c in out.columns:
            out[c] = out[c].ffill().bfill()
    return out


def clip_spikes(
    df: pd.DataFrame,
    col: str,
    upper_cap: float,
    *,
    suffix: str = "_clipped",
) -> pd.DataFrame:
    """
    尖峰裁剪：将超过 upper_cap 的值截断至 upper_cap（可用于训练稳定性）。
    原始列保留，裁剪结果写入 col + suffix（默认 rrp_clipped）。
    """
    out = df.copy()
    clipped_col = f"{col}{suffix}"
    out[clipped_col] = out[col].clip(upper=upper_cap)
    return out


def rolling_standardize(
    df: pd.DataFrame,
    cols: list[str],
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    滚动 z-score：(x - rolling_mean) / rolling_std。
    输出列名为 {col}_rz。
    """
    out = df.copy()
    mp = min_periods or max(window // 4, 2)
    for c in cols:
        if c not in out.columns:
            continue
        roll = out[c].rolling(window=window, min_periods=mp)
        mu = roll.mean()
        sigma = roll.std().replace(0, pd.NA)
        out[f"{c}_rz"] = (out[c] - mu) / sigma
    return out


def pipeline_preprocess(
    df: pd.DataFrame,
    spike_threshold: float,
    rrp_clip_cap: float | None = None,
    rolling_window: int = 288,
) -> pd.DataFrame:
    """
    组合：插补 -> RRP 可选截断列 -> 对 rrp（及裁剪版）与 demand 滚动标准化占位。
    """
    out = forward_backward_fill(df)
    cap = rrp_clip_cap if rrp_clip_cap is not None else spike_threshold
    out = clip_spikes(out, RRP_COL, upper_cap=cap)
    rrp_use = f"{RRP_COL}_clipped" if f"{RRP_COL}_clipped" in out.columns else RRP_COL
    cols_std = [c for c in [rrp_use, DEMAND_COL] if c in out.columns]
    out = rolling_standardize(out, cols_std, window=rolling_window)
    return out
