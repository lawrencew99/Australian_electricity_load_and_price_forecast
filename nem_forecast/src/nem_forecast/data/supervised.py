from __future__ import annotations

import numpy as np
import pandas as pd

def build_supervised_for_horizon(
    df: pd.DataFrame,
    target_col: str,
    n_lags: int,
    horizon_steps: int,
    extra_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造监督学习表：每行特征为 target 滞后 n_lags 及可选 extra 在索引 i 处的取值，
    标签为 target[i + horizon_steps]。
    返回 (feature_df, X_numeric, y, meta_row_index)，meta_row_index 为原始 df 行位置 i。
    """
    extra_feature_cols = extra_feature_cols or []
    df = df.reset_index(drop=True)
    y_series = df[target_col].astype(float)
    rows: list[dict[str, float]] = []
    y_list: list[float] = []
    meta_rows: list[int] = []

    for i in range(n_lags, len(df) - horizon_steps):
        window = y_series.iloc[i - n_lags : i].to_numpy()
        if np.any(~np.isfinite(window)):
            continue
        row = {f"lag_{j + 1}": float(window[-(j + 1)]) for j in range(n_lags)}
        for c in extra_feature_cols:
            if c in df.columns:
                row[c] = float(df.iloc[i][c]) if pd.notna(df.iloc[i][c]) else np.nan
        yt = y_series.iloc[i + horizon_steps]
        if not np.isfinite(yt):
            continue
        rows.append(row)
        y_list.append(float(yt))
        meta_rows.append(i)

    if not rows:
        empty = pd.DataFrame()
        return empty, np.array([]), np.array([]), np.array([], dtype=int)

    X_df = pd.DataFrame(rows)
    y_arr = np.array(y_list, dtype=float)
    meta_arr = np.array(meta_rows, dtype=int)
    mask = np.isfinite(X_df.to_numpy(dtype=float)).all(axis=1)
    X_df = X_df.loc[mask].reset_index(drop=True)
    y_arr = y_arr[mask]
    meta_arr = meta_arr[mask]
    X_numeric = X_df.to_numpy(dtype=float)
    return X_df, X_numeric, y_arr, meta_arr


def time_indices_for_supervised(df: pd.DataFrame, n_lags: int, horizon_steps: int) -> np.ndarray:
    """与 build_supervised 对齐的最后一个索引 i（用于对齐滚动 CV 长度）。"""
    idxs = []
    for i in range(n_lags, len(df) - horizon_steps):
        idxs.append(i)
    return np.array(idxs, dtype=int)
