import argparse
from pathlib import Path

from .pipeline import inspect_models, rank_stocks


def cmd_inspect(args: argparse.Namespace) -> int:
    result = inspect_models(args.root)
    print(f"factor_count={result.factor_count}")
    print(f"prediction_shape={result.prediction_shape}")
    print(f"prediction_min={result.prediction_min:.6f}")
    print(f"prediction_max={result.prediction_max:.6f}")
    print(f"prediction_mean={result.prediction_mean:.6f}")
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    top_df = rank_stocks(args.root, args.features, top_n=args.top_n)

    out = Path(args.out) if args.out else Path(args.root) / "rank_result.csv"
    top_df.to_csv(out, encoding="utf-8-sig")
    print(f"saved={out}")
    print(top_df.head(args.preview).to_string())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-soft",
        description="AI quant stock scoring app for standalone and JoinQuant integration.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect-models", help="Inspect model loading and output shape.")
    p_inspect.add_argument("--root", default=".", help="Project root path.")
    p_inspect.set_defaults(func=cmd_inspect)

    p_rank = sub.add_parser("rank", help="Score and rank stocks using local feature CSV.")
    p_rank.add_argument("--root", default=".", help="Project root path.")
    p_rank.add_argument("--features", required=True, help="Input CSV with `code` and factor columns.")
    p_rank.add_argument("--top-n", type=int, default=10, help="Number of selected stocks.")
    p_rank.add_argument("--out", default="", help="Output CSV path.")
    p_rank.add_argument("--preview", type=int, default=10, help="Preview row count.")
    p_rank.set_defaults(func=cmd_rank)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))
