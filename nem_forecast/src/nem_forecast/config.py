from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RollingCVConfig:
    n_splits: int = 3
    train_min_periods: int = 5000
    gap_steps: int = 288


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


@dataclass
class TorchConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    max_epochs: int = 5
    batch_size: int = 64


@dataclass
class SarimaConfig:
    order: tuple[int, int, int] = (1, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 288)


@dataclass
class XGBoostConfig:
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1


@dataclass
class RiskConfig:
    alert_upper_quantile: float = 0.9
    exposure_scale_hint: str = "reduce_when_spike_prob_high"


@dataclass
class Config:
    regions: list[str] = field(default_factory=lambda: ["NSW", "VIC", "QLD", "SA"])
    freq: str = "5min"
    freq_minutes: int = 5
    horizons: dict[str, int] = field(default_factory=lambda: {"24h": 288, "48h": 576})
    spike_threshold_aud: float = 300.0
    rolling_cv: RollingCVConfig = field(default_factory=RollingCVConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    csv_glob: str = "nem_{region}_5min.csv"
    random_seed: int = 42
    torch: TorchConfig = field(default_factory=TorchConfig)
    sarima: SarimaConfig = field(default_factory=SarimaConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    _project_root: Path | None = None

    def resolve_path(self, p: str | Path) -> Path:
        root = self._project_root or Path.cwd()
        path = Path(p)
        return path if path.is_absolute() else (root / path).resolve()


def _tuple_from_list(x: Any, length: int | None = None) -> tuple[int, ...]:
    if isinstance(x, (list, tuple)):
        t = tuple(int(v) for v in x)
        if length is not None and len(t) != length:
            raise ValueError(f"expected length {length}, got {len(t)}")
        return t
    raise TypeError(f"expected list/tuple, got {type(x)}")


def load_config(path: str | Path, project_root: str | Path | None = None) -> Config:
    path = Path(path)
    root = Path(project_root).resolve() if project_root else path.parent.parent.resolve()

    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    rc = raw.get("rolling_cv", {}) or {}
    pc = raw.get("paths", {}) or {}
    tc = raw.get("torch", {}) or {}
    sc = raw.get("sarima", {}) or {}
    xc = raw.get("xgboost", {}) or {}
    risk_c = raw.get("risk", {}) or {}

    cfg = Config(
        regions=list(raw["regions"]),
        freq=str(raw.get("freq", "5min")),
        freq_minutes=int(raw.get("freq_minutes", 5)),
        horizons={str(k): int(v) for k, v in (raw.get("horizons") or {}).items()},
        spike_threshold_aud=float(raw.get("spike_threshold_aud", 300)),
        rolling_cv=RollingCVConfig(
            n_splits=int(rc.get("n_splits", 3)),
            train_min_periods=int(rc.get("train_min_periods", 5000)),
            gap_steps=int(rc.get("gap_steps", 288)),
        ),
        paths=PathsConfig(
            raw_dir=str(pc.get("raw_dir", "data/raw")),
            processed_dir=str(pc.get("processed_dir", "data/processed")),
        ),
        csv_glob=str(raw.get("csv_glob", "nem_{region}_5min.csv")),
        random_seed=int(raw.get("random_seed", 42)),
        torch=TorchConfig(
            hidden_size=int(tc.get("hidden_size", 64)),
            num_layers=int(tc.get("num_layers", 2)),
            dropout=float(tc.get("dropout", 0.1)),
            learning_rate=float(tc.get("learning_rate", 1e-3)),
            max_epochs=int(tc.get("max_epochs", 5)),
            batch_size=int(tc.get("batch_size", 64)),
        ),
        sarima=SarimaConfig(
            order=_tuple_from_list(sc.get("order", [1, 0, 1]), 3),  # type: ignore[arg-type]
            seasonal_order=_tuple_from_list(sc.get("seasonal_order", [1, 0, 1, 288]), 4),  # type: ignore[arg-type]
        ),
        xgboost=XGBoostConfig(
            n_estimators=int(xc.get("n_estimators", 100)),
            max_depth=int(xc.get("max_depth", 6)),
            learning_rate=float(xc.get("learning_rate", 0.1)),
        ),
        risk=RiskConfig(
            alert_upper_quantile=float(risk_c.get("alert_upper_quantile", 0.9)),
            exposure_scale_hint=str(risk_c.get("exposure_scale_hint", "reduce_when_spike_prob_high")),
        ),
    )
    cfg._project_root = root
    return cfg
