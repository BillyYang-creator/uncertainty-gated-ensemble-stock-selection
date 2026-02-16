from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_soft.pipeline import inspect_models


def main() -> int:
    result = inspect_models(ROOT)
    print(f"factor_count={result.factor_count}")
    print(f"prediction_shape={result.prediction_shape}")
    print(f"prediction_min={result.prediction_min:.6f}")
    print(f"prediction_max={result.prediction_max:.6f}")
    print(f"prediction_mean={result.prediction_mean:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
