from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from nem_forecast.config import load_config
from nem_forecast.cv.rolling import iter_roll_split_indices
from nem_forecast.data.features import add_calendar_features, merge_weather_placeholder
from nem_forecast.data.loaders import discover_region_csv, load_region_csv, raw_dir_has_any_csv
from nem_forecast.data.preprocess import pipeline_preprocess
from nem_forecast.data.schemas import DEMAND_COL, RRP_COL
from nem_forecast.data.supervised import build_supervised_for_horizon
from nem_forecast.metrics import mape, rmse

N_LAGS = 48


def _resolve_target_col(df: pd.DataFrame, target: str) -> str:
    if target == "demand":
        return DEMAND_COL
    clipped = f"{RRP_COL}_clipped"
    return clipped if clipped in df.columns else RRP_COL


def _prepare_region_frame(cfg, region: str) -> pd.DataFrame:
    df = load_region_csv(cfg, region)
    df = pipeline_preprocess(df, cfg.spike_threshold_aud)
    df = add_calendar_features(df)
    df = merge_weather_placeholder(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="滚动 CV 评估（占位管线）")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["xgb", "sarima", "lstm", "tft"], required=True)
    parser.add_argument("--target", choices=["rrp", "demand"], default="rrp")
    parser.add_argument("--horizon", choices=["24h", "48h"], default="24h")
    parser.add_argument("--region", default=None, help="单个区域；默认对 regions 逐个运行")
    parser.add_argument("--out", default=None, help="可选：汇总 CSV 路径")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = load_config(cfg_path)

    if not raw_dir_has_any_csv(cfg):
        print(f"ERROR: 在 {cfg.resolve_path(cfg.paths.raw_dir)} 下未发现 CSV。")
        print(f"示例文件名：{cfg.csv_glob.format(region=cfg.regions[0])}")
        raise SystemExit(1)

    horizon_steps = int(cfg.horizons[args.horizon])
    regions = [args.region] if args.region else list(cfg.regions)

    rows_out: list[dict[str, object]] = []

    for region in regions:
        if discover_region_csv(cfg, region) is None:
            print(f"Skip region {region}: no CSV matched.")
            continue
        df = _prepare_region_frame(cfg, region)
        target_col = _resolve_target_col(df, args.target)
        extras = [c for c in ["hour", "dow", "month", "is_weekend"] if c in df.columns]
        _, X_num, y, meta = build_supervised_for_horizon(
            df, target_col, N_LAGS, horizon_steps, extra_feature_cols=extras
        )
        if len(y) == 0:
            print(f"Region {region}: no supervised rows; skip.")
            continue

        supervised_len = len(y)
        try:
            splits = list(iter_roll_split_indices(supervised_len, cfg))
        except ValueError as e:
            print(f"Region {region}: rolling CV failed: {e}")
            continue

        fold_id = 0
        for train_idx, val_idx in splits:
            fold_id += 1
            n_val_points = int(len(val_idx))
            if args.model == "sarima":
                y_raw = df[target_col].astype(float).to_numpy()
                train_max_i = int(meta[train_idx].max())
                train_series = y_raw[: train_max_i + 1]
                train_series = train_series[np.isfinite(train_series)]
                try:
                    from nem_forecast.models.baselines_sarima import SarimaBaseline

                    sar = SarimaBaseline(cfg.sarima)
                    fc = sar.fit_forecast(train_series, horizon_steps)
                except Exception as exc:
                    print(f"Region {region} fold {fold_id} SARIMA fit failed: {exc}")
                    continue
                start = train_max_i + 1
                end = start + horizon_steps
                if end > len(y_raw):
                    print(f"Region {region} fold {fold_id}: insufficient tail for SARIMA actuals")
                    continue
                actual = y_raw[start:end]
                pred = fc[: len(actual)]
                r = rmse(actual, pred)
                m = mape(actual, pred)
                n_val_points = int(len(actual))
            else:
                X_tr, X_va = X_num[train_idx], X_num[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]
                n_extra = max(X_tr.shape[1] - N_LAGS, 0)
                if args.model == "xgb":
                    from nem_forecast.models.baselines_xgb import XGBBaseline

                    mdl = XGBBaseline(cfg.xgboost)
                    mdl.fit(X_tr, y_tr)
                    pred = mdl.predict(X_va)
                elif args.model == "lstm":
                    from nem_forecast.models.lstm_pt import LSTMRegressor, numpy_to_lstm_sequences

                    X_seq_tr = numpy_to_lstm_sequences(X_tr, N_LAGS, n_extra)
                    X_seq_va = numpy_to_lstm_sequences(X_va, N_LAGS, n_extra)
                    in_dim = X_seq_tr.shape[-1]
                    mdl = LSTMRegressor(
                        in_dim,
                        hidden_size=cfg.torch.hidden_size,
                        num_layers=cfg.torch.num_layers,
                        dropout=cfg.torch.dropout,
                        lr=cfg.torch.learning_rate,
                        max_epochs=cfg.torch.max_epochs,
                        batch_size=min(cfg.torch.batch_size, len(X_seq_tr)),
                        seed=cfg.random_seed,
                    )
                    mdl.fit(X_seq_tr, y_tr)
                    pred = mdl.predict(X_seq_va)
                elif args.model == "tft":
                    from nem_forecast.models.lstm_pt import numpy_to_lstm_sequences
                    from nem_forecast.models.tft_pt import TFTRegressorSklearnLike

                    X_seq_tr = numpy_to_lstm_sequences(X_tr, N_LAGS, n_extra)
                    X_seq_va = numpy_to_lstm_sequences(X_va, N_LAGS, n_extra)
                    in_dim = X_seq_tr.shape[-1]
                    mdl = TFTRegressorSklearnLike(
                        in_dim,
                        hidden_size=cfg.torch.hidden_size,
                        horizon=1,
                        lr=cfg.torch.learning_rate,
                        max_epochs=cfg.torch.max_epochs,
                        batch_size=min(cfg.torch.batch_size, len(X_seq_tr)),
                        seed=cfg.random_seed,
                    )
                    mdl.fit(X_seq_tr, y_tr)
                    pred = mdl.predict(X_seq_va)
                else:
                    raise RuntimeError("unknown model")
                r = rmse(y_va, pred)
                m = mape(y_va, pred)

            rows_out.append(
                {
                    "region": region,
                    "fold": fold_id,
                    "model": args.model,
                    "target": args.target,
                    "horizon": args.horizon,
                    "rmse": r,
                    "mape": m,
                    "n_train": int(len(train_idx)),
                    "n_val": n_val_points,
                }
            )
            print(
                f"{region} fold {fold_id} | RMSE={r:.4g} MAPE={m:.4g} | "
                f"train={len(train_idx)} val={n_val_points}"
            )

    if not rows_out:
        print("No metrics computed (check data length vs rolling_cv settings).")
        raise SystemExit(1)

    summary = pd.DataFrame(rows_out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out, index=False)
        print(f"Wrote summary -> {args.out}")


if __name__ == "__main__":
    main()
