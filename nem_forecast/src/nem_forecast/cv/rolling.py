from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from nem_forecast.config import Config


def iter_roll_split_indices(supervised_len: int, cfg: Config) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    对监督样本索引 [0, supervised_len) 生成滚动训练/验证索引。
    验证段起点相对训练末端留出 gap_steps（以样本行计，假定一行对应一个 5min tick）。
    """
    tm = cfg.rolling_cv.train_min_periods
    gap = cfg.rolling_cv.gap_steps
    K = cfg.rolling_cv.n_splits

    if supervised_len <= tm + gap + K + 1:
        raise ValueError(
            f"supervised_len={supervised_len} too small for train_min={tm}, gap={gap}, n_splits={K}"
        )

    usable = supervised_len - tm - gap
    step = max(usable // (K + 1), 1)

    for k in range(K):
        train_end = tm + k * step
        val_start = train_end + gap
        val_end = min(val_start + step, supervised_len)
        if val_start >= val_end:
            continue
        train_idx = np.arange(0, train_end, dtype=int)
        val_idx = np.arange(val_start, val_end, dtype=int)
        yield train_idx, val_idx


def train_val_masks_from_absolute_indices(
    supervised_positions: np.ndarray,
    train_idx_relative: np.ndarray,
    val_idx_relative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """将相对监督索引映射到 supervised_positions 上的布尔掩码。"""
    train_set = set(train_idx_relative.tolist())
    val_set = set(val_idx_relative.tolist())
    train_mask = np.array([i in train_set for i in range(len(supervised_positions))])
    val_mask = np.array([i in val_set for i in range(len(supervised_positions))])
    return train_mask, val_mask
