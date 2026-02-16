from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    root: Path
    model_reg: Path
    model_cls: Path
    model_dir: Path
    factors_csv: Path

    @staticmethod
    def from_root(root: str | Path) -> "AppConfig":
        base = Path(root).resolve()
        return AppConfig(
            root=base,
            model_reg=base / "model_reg_final.pkl",
            model_cls=base / "model_cls_final.pkl",
            model_dir=base / "model_dir_final.pkl",
            factors_csv=base / "selected_factors.csv",
        )
