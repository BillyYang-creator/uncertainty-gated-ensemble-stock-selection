"""
JoinQuant dynamic-threshold strategy template.

Upload files:
  - model_reg_final.pkl
  - model_cls_final.pkl
  - model_dir_final.pkl
  - selected_factors.csv
"""
from jqdata import *  # noqa: F401,F403
from jqfactor import *  # noqa: F401,F403
import numpy as np
import pandas as pd
import pickle
from six import BytesIO


def model_inference(model, data):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(data)
        return p[:, 1] if p.ndim > 1 else p
    return model.predict(data)


def rank_scale(series):
    return series.rank(method="average", pct=True)


def initialize(context):
    set_benchmark("000985.XSHG")
    set_option("use_real_price", True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0))
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.001,
            open_commission=0.0003,
            close_commission=0.0003,
            close_today_commission=0,
            min_commission=5,
        ),
        type="stock",
    )
    log.set_level("order", "error")

    g.stock_num = 10
    g.hold_list = []
    g.yesterday_HL_list = []
    g.avg_ai_score = 0.0
    g.avg_consistency = 0.0
    g.factor_alignment = 0.5
    g.effective_threshold = 0.005

    g.consistency_history = []
    g.portfolio_value_history = []
    g.daily_returns = []
    g.lookback = 20
    g.base_threshold = 0.005
    g.max_history = 60
    g.factor_batch_size = 20

    try:
        g.model_reg = pickle.loads(read_file("model_reg_final.pkl"))
        g.model_cls = pickle.loads(read_file("model_cls_final.pkl"))
        g.model_dir = pickle.loads(read_file("model_dir_final.pkl"))
        g.factor_list = list(
            pd.read_csv(BytesIO(read_file("selected_factors.csv")))["factor"]
        )
    except Exception as e:
        log.error("Initialization error: {}".format(e))
        g.factor_list = []

    run_daily(prepare_stock_list, "9:05")
    run_daily(update_daily_context, "9:31")
    run_monthly(rebalance, 1, "9:35")
    run_daily(check_limit_up, "14:00")
    run_daily(record_metrics, "14:55")


def prepare_stock_list(context):
    g.hold_list = [p.security for p in context.portfolio.positions.values()]
    if not g.hold_list:
        g.yesterday_HL_list = []
        return

    df = get_price(
        g.hold_list,
        end_date=context.previous_date,
        frequency="daily",
        fields=["close", "high_limit"],
        count=1,
        panel=False,
        fill_paused=False,
    )
    g.yesterday_HL_list = df[df["close"] == df["high_limit"]]["code"].tolist()


def filter_universe(context, stock_list):
    curr = get_current_data()
    return [
        s
        for s in stock_list
        if not (
            s.startswith(("3", "68", "4", "8"))
            or curr[s].paused
            or curr[s].is_st
            or "ST" in curr[s].name
            or "*" in curr[s].name
            or curr[s].day_open == curr[s].high_limit
            or curr[s].day_open == curr[s].low_limit
        )
    ]


def get_feature_frame(context):
    if not g.factor_list:
        return pd.DataFrame()

    universe = get_index_stocks("000985.XSHG", context.current_dt)
    universe = filter_universe(context, universe)
    if not universe:
        return pd.DataFrame()

    factor_series = {}
    for i in range(0, len(g.factor_list), g.factor_batch_size):
        batch = g.factor_list[i : i + g.factor_batch_size]
        fd = get_factor_values(universe, batch, end_date=context.previous_date, count=1)
        for f in batch:
            factor_series[f] = fd[f].iloc[0, :]

    df = pd.DataFrame({f: factor_series[f] for f in g.factor_list})
    return df.dropna()


def get_combined_predictions(context):
    df = get_feature_frame(context)
    if df.empty:
        return None, None

    preds = np.column_stack(
        [
            model_inference(g.model_reg, df),
            model_inference(g.model_cls, df),
            model_inference(g.model_dir, df),
        ]
    )
    return df, preds


def compute_factor_alignment(df, preds):
    if df is None or preds is None or df.empty:
        return 0.5

    x = df.copy()
    x["_p_reg"] = preds[:, 0]
    x["_p_cls"] = preds[:, 1]
    x["_p_dir"] = preds[:, 2]

    sens_reg = x[g.factor_list].corrwith(x["_p_reg"])
    sens_cls = x[g.factor_list].corrwith(x["_p_cls"])
    sens_dir = x[g.factor_list].corrwith(x["_p_dir"])

    raw = (sens_reg.corr(sens_cls) + sens_reg.corr(sens_dir) + sens_cls.corr(sens_dir)) / 3.0
    if pd.isna(raw):
        return 0.5
    return (float(np.clip(raw, -1, 1)) + 1.0) / 2.0


def get_dynamic_threshold():
    base = g.base_threshold
    if len(g.daily_returns) < 5:
        return base

    recent_ret = np.mean(g.daily_returns[-g.lookback :])
    if recent_ret < -0.01:
        base *= 1.25
    elif recent_ret > 0.02:
        base *= 0.85

    if len(g.consistency_history) >= 10:
        recent_cons = g.consistency_history[-g.lookback :]
        p55 = float(np.percentile(recent_cons, 55))
        base = 0.6 * base + 0.4 * p55
    return base


def update_daily_context(context):
    v = context.portfolio.portfolio_value
    g.portfolio_value_history.append(v)

    if len(g.portfolio_value_history) >= 2:
        prev_v = g.portfolio_value_history[-2]
        if prev_v > 0:
            g.daily_returns.append((v - prev_v) / prev_v)

    if len(g.portfolio_value_history) > g.max_history:
        g.portfolio_value_history.pop(0)
    if len(g.daily_returns) > g.max_history:
        g.daily_returns.pop(0)

    df, preds = get_combined_predictions(context)
    if preds is None:
        return

    g.avg_ai_score = float(preds.mean(axis=1).mean())
    g.avg_consistency = float(preds.var(axis=1, ddof=0).mean())
    g.consistency_history.append(g.avg_consistency)

    g.factor_alignment = compute_factor_alignment(df, preds)

    th = get_dynamic_threshold() * (1.0 - 0.25 * g.factor_alignment)
    g.effective_threshold = max(0.002, min(0.015, th))

    if len(g.consistency_history) > g.max_history:
        g.consistency_history.pop(0)


def filter_paused_stock(lst):
    c = get_current_data()
    return [s for s in lst if not c[s].paused]


def filter_limitup_stock(context, lst):
    c = get_current_data()
    return [
        s
        for s in lst
        if s in context.portfolio.positions
        or history(1, "1m", "close", [s])[s][-1] < c[s].high_limit
    ]


def filter_limitdown_stock(context, lst):
    c = get_current_data()
    return [
        s
        for s in lst
        if s in context.portfolio.positions
        or history(1, "1m", "close", [s])[s][-1] > c[s].low_limit
    ]


def get_target_list(context):
    df, preds = get_combined_predictions(context)
    if preds is None:
        return []

    score = pd.DataFrame(index=df.index)
    score["p_reg"] = preds[:, 0]
    score["p_cls"] = preds[:, 1]
    score["p_dir"] = preds[:, 2]
    score["ai_score"] = (
        rank_scale(score["p_reg"]) + rank_scale(score["p_cls"]) + rank_scale(score["p_dir"])
    ) / 3.0
    score["consistency"] = preds.var(axis=1, ddof=0)

    g.factor_alignment = compute_factor_alignment(df, preds)
    dynamic_th = get_dynamic_threshold()
    g.effective_threshold = max(
        0.002, min(0.015, dynamic_th * (1.0 - 0.25 * g.factor_alignment))
    )

    if score["consistency"].mean() > g.effective_threshold:
        n_cand = max(1, int(0.1 * len(score)))
        n_pick = max(1, int(0.2 * n_cand))
        selected = (
            score.sort_values("consistency")
            .head(n_cand)
            .sort_values("ai_score", ascending=False)
            .head(n_pick)
        )
    else:
        n_pick = max(g.stock_num, int(0.2 * len(score)))
        selected = score.sort_values("ai_score", ascending=False).head(n_pick)

    lst = selected.index.tolist()
    lst = filter_paused_stock(lst)
    lst = filter_limitup_stock(context, lst)
    lst = filter_limitdown_stock(context, lst)
    return lst[: g.stock_num]


def rebalance(context):
    target_list = get_target_list(context)
    if not target_list:
        return

    for s in g.hold_list:
        if (s not in target_list) and (s not in g.yesterday_HL_list):
            order_target(s, 0)

    position_count = len(context.portfolio.positions)
    if len(target_list) > position_count:
        buy_num = min(len(target_list), g.stock_num) - position_count
        if buy_num <= 0:
            return

        cash_each = context.portfolio.cash / buy_num
        for s in target_list:
            if s in context.portfolio.positions:
                continue
            order_target_value(s, cash_each)


def check_limit_up(context):
    for s in g.yesterday_HL_list:
        if s not in context.portfolio.positions:
            continue
        curr = get_price(
            s,
            end_date=context.current_dt,
            frequency="1m",
            fields=["close", "high_limit"],
            count=1,
        )
        if (not curr.empty) and (curr.iloc[0, 0] < curr.iloc[0, 1]):
            order_target(s, 0)


def record_metrics(context):
    record(Consistency=g.avg_consistency)
    record(Threshold=g.effective_threshold)
    record(Alignment=g.factor_alignment)
    record(AIScore=g.avg_ai_score)