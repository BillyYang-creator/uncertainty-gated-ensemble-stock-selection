from pathlib import Path

import pandas as pd


def load_factor_list(path: str | Path) -> list[str]:
    df = pd.read_csv(path)
    if "factor" not in df.columns:
        raise ValueError("selected_factors.csv must contain a `factor` column.")
    factors = [str(x).strip() for x in df["factor"].tolist() if str(x).strip()]
    if not factors:
        raise ValueError("No factors found in selected_factors.csv.")
    return factors


def load_feature_frame(path: str | Path, factors: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "code" not in df.columns:
        raise ValueError("Feature file must contain `code` column.")

    missing = [f for f in factors if f not in df.columns]
    if missing:
        first = ", ".join(missing[:10])
        raise ValueError(f"Missing factor columns: {first}")

    out = df[["code", *factors]].copy()
    out = out.dropna(subset=factors)
    return out.set_index("code")
