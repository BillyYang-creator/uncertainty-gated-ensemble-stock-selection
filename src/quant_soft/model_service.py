import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ModelBundle:
    def __init__(self, model_reg: Any, model_cls: Any, model_dir: Any) -> None:
        self.model_reg = model_reg
        self.model_cls = model_cls
        self.model_dir = model_dir

    @staticmethod
    def _load(path: str | Path) -> Any:
        with open(path, "rb") as f:
            try:
                return pickle.load(f)
            except ModuleNotFoundError as e:
                if "lightgbm" in str(e):
                    raise RuntimeError(
                        "lightgbm is required to load model pickle files. "
                        "Install dependencies: pip install -r requirements.txt"
                    ) from e
                raise

    @classmethod
    def from_paths(
        cls,
        model_reg_path: str | Path,
        model_cls_path: str | Path,
        model_dir_path: str | Path,
    ) -> "ModelBundle":
        return cls(
            model_reg=cls._load(model_reg_path),
            model_cls=cls._load(model_cls_path),
            model_dir=cls._load(model_dir_path),
        )

    @staticmethod
    def _infer(model: Any, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            y = model.predict_proba(x)
            y = y[:, 1] if np.ndim(y) > 1 else y
        else:
            y = model.predict(x)
        return np.asarray(y, dtype=float).reshape(-1)

    def predict_all(self, x: pd.DataFrame) -> np.ndarray:
        p_reg = self._infer(self.model_reg, x)
        p_cls = self._infer(self.model_cls, x)
        p_dir = self._infer(self.model_dir, x)
        return np.column_stack([p_reg, p_cls, p_dir])
