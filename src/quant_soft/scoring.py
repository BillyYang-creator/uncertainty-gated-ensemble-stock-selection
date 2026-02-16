import numpy as np
import pandas as pd


def cross_section_rank_scale(series: pd.Series) -> pd.Series:
    # Rank-scale to [0, 1] to align outputs with different value ranges.
    ranked = series.rank(method="average", pct=True)
    return ranked.astype(float)


def build_score_frame(features: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    if preds.shape[1] != 3:
        raise ValueError("preds must have shape (n, 3).")

    df = pd.DataFrame(index=features.index)
    df["pred_reg"] = preds[:, 0]
    df["pred_cls"] = preds[:, 1]
    df["pred_dir"] = preds[:, 2]

    df["reg_scaled"] = cross_section_rank_scale(df["pred_reg"])
    df["cls_scaled"] = cross_section_rank_scale(df["pred_cls"])
    df["dir_scaled"] = cross_section_rank_scale(df["pred_dir"])

    scaled_cols = ["reg_scaled", "cls_scaled", "dir_scaled"]
    df["ai_score"] = df[scaled_cols].mean(axis=1)
    df["consistency"] = df[scaled_cols].var(axis=1, ddof=0)
    return df.sort_values(["ai_score", "consistency"], ascending=[False, True])


def select_top_n(score_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    n = max(1, int(n))
    return score_df.head(n).copy()
