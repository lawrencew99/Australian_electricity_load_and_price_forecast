from __future__ import annotations

import pandas as pd

from nem_forecast.data.schemas import REGION_COL, TIMESTAMP_COL, WEATHER_PREFIX


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    日历特征：小时、星期、月份、是否周末。依赖 TIMESTAMP_COL。
    """
    out = df.copy()
    ts = pd.to_datetime(out[TIMESTAMP_COL], utc=True)
    local = ts.dt.tz_convert("Australia/Sydney") if ts.dt.tz is not None else ts
    out["hour"] = local.dt.hour.astype(float)
    out["dow"] = local.dt.dayofweek.astype(float)
    out["month"] = local.dt.month.astype(float)
    out["is_weekend"] = (local.dt.dayofweek >= 5).astype(float)
    # 节假日占位：全 0；可替换为 holidays 库或自定义表 merge
    out["is_holiday"] = 0.0
    return out


def merge_weather_placeholder(
    df: pd.DataFrame,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    气象对齐：合并键 (timestamp, region)。若 weather 为 None，生成占位列 wx_temp / wx_wspd 全 NaN。
    """
    out = df.copy()
    if weather is None:
        out[f"{WEATHER_PREFIX}temp"] = pd.NA
        out[f"{WEATHER_PREFIX}wspd"] = pd.NA
        return out

    w = weather.copy()
    w[TIMESTAMP_COL] = pd.to_datetime(w[TIMESTAMP_COL], utc=True)
    key = [TIMESTAMP_COL, REGION_COL]
    return out.merge(w, on=key, how="left", suffixes=("", "_wx"))
