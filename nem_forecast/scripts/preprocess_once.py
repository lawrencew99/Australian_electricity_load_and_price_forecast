from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nem_forecast.config import load_config
from nem_forecast.data.features import add_calendar_features, merge_weather_placeholder
from nem_forecast.data.loaders import discover_region_csv, load_region_csv, raw_dir_has_any_csv
from nem_forecast.data.preprocess import pipeline_preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw CSV -> processed table (single concat file).")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置路径（相对工程根或绝对路径）")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = load_config(cfg_path)

    raw_dir = cfg.resolve_path(cfg.paths.raw_dir)
    if not raw_dir_has_any_csv(cfg):
        print(f"ERROR: 在 {raw_dir} 下未发现任何 .csv。")
        print("请将按区域的 CSV 放入该目录（见 configs/default.yaml 中的 csv_glob 约定）。")
        raise SystemExit(1)

    parts = []
    for region in cfg.regions:
        if discover_region_csv(cfg, region) is None:
            print(f"Skip region {region}: no CSV matched.")
            continue
        parts.append(load_region_csv(cfg, region))
    if not parts:
        print("ERROR: no region CSV could be loaded.")
        raise SystemExit(1)
    df = pd.concat(parts, ignore_index=True)
    df = pipeline_preprocess(df, cfg.spike_threshold_aud)
    df = add_calendar_features(df)
    df = merge_weather_placeholder(df)

    out_dir = cfg.resolve_path(cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "nem_all_regions_processed.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
