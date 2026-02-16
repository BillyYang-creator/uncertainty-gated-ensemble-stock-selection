from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig
from .feature_io import load_factor_list, load_feature_frame
from .model_service import ModelBundle
from .scoring import build_score_frame, select_top_n


@dataclass(frozen=True)
class InspectionResult:
    factor_count: int
    prediction_shape: tuple[int, int]
    prediction_min: float
    prediction_max: float
    prediction_mean: float


def inspect_models(root: str | Path) -> InspectionResult:
    cfg = AppConfig.from_root(root)
    factors = load_factor_list(cfg.factors_csv)
    models = ModelBundle.from_paths(cfg.model_reg, cfg.model_cls, cfg.model_dir)

    x = pd.DataFrame(np.random.randn(12, len(factors)), columns=factors)
    preds = models.predict_all(x)
    return InspectionResult(
        factor_count=len(factors),
        prediction_shape=preds.shape,
        prediction_min=float(preds.min()),
        prediction_max=float(preds.max()),
        prediction_mean=float(preds.mean()),
    )


def rank_stocks(
    root: str | Path,
    features_path: str | Path,
    top_n: int = 10,
) -> pd.DataFrame:
    cfg = AppConfig.from_root(root)
    factors = load_factor_list(cfg.factors_csv)
    models = ModelBundle.from_paths(cfg.model_reg, cfg.model_cls, cfg.model_dir)

    features = load_feature_frame(features_path, factors)
    preds = models.predict_all(features[factors])
    score_df = build_score_frame(features, preds)
    return select_top_n(score_df, n=top_n)
