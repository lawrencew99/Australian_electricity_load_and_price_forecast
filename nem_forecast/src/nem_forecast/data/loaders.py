from __future__ import annotations

from pathlib import Path

import pandas as pd

from nem_forecast.config import Config
from nem_forecast.data.schemas import DEMAND_COL, REGION_COL, REQUIRED_COLUMNS, RRP_COL, TIMESTAMP_COL


def _default_column_mapping() -> dict[str, str]:
    """若 CSV 已使用规范列名，可直接 identity；否则在此处扩展别名。"""
    return {
        TIMESTAMP_COL: TIMESTAMP_COL,
        REGION_COL: REGION_COL,
        RRP_COL: RRP_COL,
        DEMAND_COL: DEMAND_COL,
    }


def discover_region_csv(cfg: Config, region: str) -> Path | None:
    raw = cfg.resolve_path(cfg.paths.raw_dir)
    pattern = cfg.csv_glob.format(region=region)
    matches = sorted(raw.glob(pattern))
    if not matches:
        alt = list(raw.glob(f"*{region}*.csv"))
        matches = sorted(alt)
    return matches[0] if matches else None


def load_region_csv(
    cfg: Config,
    region: str,
    column_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    读取单个区域 CSV。column_mapping: 原始列名 -> 规范列名（schemas）。
    """
    path = discover_region_csv(cfg, region)
    if path is None:
        raise FileNotFoundError(
            f"No CSV found for region {region} under {cfg.resolve_path(cfg.paths.raw_dir)} "
            f"(pattern: {cfg.csv_glob.format(region=region)})"
        )

    df = pd.read_csv(path)
    cmap = column_mapping or _default_column_mapping()
    rename = {src: dst for src, dst in cmap.items() if src in df.columns}
    df = df.rename(columns=rename)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns after rename: {missing}")

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True, errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COL])
    df[REGION_COL] = df[REGION_COL].astype(str)
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return df


def load_all_regions(cfg: Config, column_mapping: dict[str, str] | None = None) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for region in cfg.regions:
        parts.append(load_region_csv(cfg, region, column_mapping=column_mapping))
    return pd.concat(parts, ignore_index=True)


def raw_dir_has_any_csv(cfg: Config) -> bool:
    raw = cfg.resolve_path(cfg.paths.raw_dir)
    return raw.is_dir() and any(raw.glob("*.csv"))
