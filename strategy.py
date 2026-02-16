
"""
ARC 策略增强版：动态阈值 + 特征交叉验证
核心逻辑：
1. 动态阈值：基于收益反馈与一致性分位数的自适应信心门槛
2. 特征校验：多模型因子敏感度相关性验证（Factor Alignment）
"""
from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import pickle
from six import BytesIO


def model_inference(model, data):
    """统一推理接口，支持 Booster / sklearn"""
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(data)
        return p[:, 1] if p.ndim > 1 else p
    return model.predict(data)


def initialize(context):
    set_benchmark('000985.XSHG')
    set_option('use_real_price', True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0))
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003,
                             close_today_commission=0, min_commission=5), type='stock')
    log.set_level('order', 'error')

    # 策略参数
    g.stock_num = 10
    g.hold_list = []
    g.yesterday_HL_list = []
    g.avg_ai_score = 0
    g.avg_consistency = 0
    g.effective_threshold = 0.005
    g.factor_alignment = 0.5

    # 统计序列
    g.consistency_history = []      
    g.portfolio_value_history = []  
    g.daily_returns = []            
    g.lookback = 20                 
    g.base_threshold = 0.005        
    g.max_history = 60              

    try:
        g.model_reg = pickle.loads(read_file('model_reg_final.pkl'))
        g.model_cls = pickle.loads(read_file('model_cls_final.pkl'))
        g.model_dir = pickle.loads(read_file('model_dir_final.pkl'))
        g.factor_list = list(pd.read_csv(BytesIO(read_file('selected_factors.csv')))['factor'])
    except Exception as e:
        log.error(f"Initialization Error: {e}")

    # 任务调度
    run_daily(prepare_stock_list, '9:05')
    run_daily(update_daily_context, '9:06')   
    run_monthly(weekly_adjustment, 1, '9:30')
    run_daily(record_portfolio_value_consistency, '9:31')
    run_daily(check_limit_up, '14:00')


def prepare_stock_list(context):
    g.hold_list = [p.security for p in context.portfolio.positions.values()]
    if g.hold_list:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily', fields=['close', 'high_limit'],
                       count=1, panel=False, fill_paused=False)
        g.yesterday_HL_list = df[df['close'] == df['high_limit']]['code'].tolist()
    else:
        g.yesterday_HL_list = []


def get_combined_predictions(context):
    yesterday = context.previous_date
    initial_list = get_index_stocks('000985.XSHG', context.current_dt)
    initial_list = filter_all_stock2(context, initial_list)

    f1, f2 = g.factor_list[:30], g.factor_list[30:70]
    fd1 = get_factor_values(initial_list, f1, end_date=yesterday, count=1)
    fd2 = get_factor_values(initial_list, f2, end_date=yesterday, count=1)

    # 保持因子对齐
    df_f1 = pd.DataFrame({f: fd1[f].iloc[0, :] for f in f1})
    df_f2 = pd.DataFrame({f: fd2[f].iloc[0, :] for f in f2})
    df_combined = pd.concat([df_f1, df_f2], axis=1).dropna()

    if df_combined.empty:
        return None, None

    preds = np.column_stack([
        model_inference(g.model_reg, df_combined),
        model_inference(g.model_cls, df_combined),
        model_inference(g.model_dir, df_combined)
    ])

    return df_combined, preds


def compute_factor_alignment(df_combined, preds):
    if df_combined.empty or preds is None or len(g.factor_list) < 2:
        return 0.5

    df = df_combined.copy()
    df['_p_reg'], df['_p_cls'], df['_p_dir'] = preds[:, 0], preds[:, 1], preds[:, 2]

    sens_reg = df[g.factor_list].corrwith(df['_p_reg'])
    sens_cls = df[g.factor_list].corrwith(df['_p_cls'])
    sens_dir = df[g.factor_list].corrwith(df['_p_dir'])

    raw = (sens_reg.corr(sens_cls) + sens_reg.corr(sens_dir) + sens_cls.corr(sens_dir)) / 3.0
    return (float(np.clip(raw, -1, 1)) + 1) / 2.0


def get_dynamic_threshold(context):
    base = g.base_threshold
    if len(g.daily_returns) < 5:
        return base

    recent_ret = np.mean(g.daily_returns[-g.lookback:])
    if recent_ret < -0.01:
        base *= 1.25
    elif recent_ret > 0.02:
        base *= 0.85

    if len(g.consistency_history) >= 10:
        recent_cons = g.consistency_history[-g.lookback:]
        p55 = float(np.percentile(recent_cons, 55))
        base = 0.6 * base + 0.4 * p55

    return base


def update_daily_context(context):
    v = context.portfolio.portfolio_value
    g.portfolio_value_history.append(v)
    if len(g.portfolio_value_history) >= 2:
        last_v = g.portfolio_value_history[-2]
        if last_v > 0:
            g.daily_returns.append((v - last_v) / last_v)
    
    # 窗口截断
    if len(g.portfolio_value_history) > g.max_history: g.portfolio_value_history.pop(0)
    if len(g.daily_returns) > g.max_history: g.daily_returns.pop(0)

    df_combined, preds = get_combined_predictions(context)
    if preds is not None:
        g.avg_consistency = float(preds.var(axis=1).mean())
        g.avg_ai_score = float(preds.mean(axis=1).mean())
        g.consistency_history.append(g.avg_consistency)
        g.factor_alignment = compute_factor_alignment(df_combined, preds)
        
        # 阈值计算逻辑
        th = get_dynamic_threshold(context) * (1 - 0.25 * g.factor_alignment)
        g.effective_threshold = max(0.002, min(0.015, th))
        
    if len(g.consistency_history) > g.max_history: g.consistency_history.pop(0)


def get_stock_list(context):
    df_combined, preds = get_combined_predictions(context)
    if df_combined is None or preds is None:
        return []

    df_combined['AI_score'] = preds.mean(axis=1)
    df_combined['consistency'] = preds.var(axis=1, ddof=0)
    
    g.factor_alignment = compute_factor_alignment(df_combined, preds)
    dynamic_th = get_dynamic_threshold(context)
    g.effective_threshold = max(0.002, min(0.015, dynamic_th * (1 - 0.25 * g.factor_alignment)))

    if df_combined['consistency'].mean() > g.effective_threshold:
        df_candidate = df_combined.sort_values('consistency').head(int(0.1 * len(df_combined)))
        df_selected = df_candidate.sort_values('AI_score', ascending=False).head(int(0.2 * len(df_candidate)))
    else:
        n_sel = max(int(0.2 * len(df_combined)), g.stock_num)
        df_selected = df_combined.sort_values('AI_score', ascending=False).head(n_sel)

    lst = df_selected.index.tolist()
    lst = filter_paused_stock(lst)
    lst = filter_limitup_stock(context, lst)
    lst = filter_limitdown_stock(context, lst)
    return lst[:g.stock_num]


def weekly_adjustment(context):
    target_list = get_stock_list(context)
    for s in g.hold_list:
        if (s not in target_list) and (s not in g.yesterday_HL_list):
            order_target(s, 0)

    position_count = len(context.portfolio.positions)
    if len(target_list) > position_count:
        buy_num = min(len(target_list), g.stock_num) - position_count
        if buy_num > 0:
            value = context.portfolio.cash / buy_num
            for s in target_list:
                if s not in context.portfolio.positions:
                    order_target_value(s, value)


def check_limit_up(context):
    for s in g.yesterday_HL_list:
        if s in context.portfolio.positions:
            curr = get_price(s, end_date=context.current_dt, frequency='1m', fields=['close', 'high_limit'], count=1)
            if not curr.empty and curr.iloc[0, 0] < curr.iloc[0, 1]:
                order_target(s, 0)


def filter_all_stock2(context, stock_list):
    curr = get_current_data()
    return [s for s in stock_list if not (
        s.startswith(('3', '68', '4', '8')) or curr[s].paused or curr[s].is_st or
        'ST' in curr[s].name or '*' in curr[s].name or
        curr[s].day_open == curr[s].high_limit or curr[s].day_open == curr[s].low_limit
    )]


def filter_paused_stock(lst):
    c = get_current_data()
    return [s for s in lst if not c[s].paused]


def filter_limitup_stock(context, lst):
    c = get_current_data()
    return [s for s in lst if s in context.portfolio.positions or history(1, '1m', 'close', [s])[s][-1] < c[s].high_limit]


def filter_limitdown_stock(context, lst):
    c = get_current_data()
    return [s for s in lst if s in context.portfolio.positions or history(1, '1m', 'close', [s])[s][-1] > c[s].low_limit]


def record_portfolio_value_consistency(context):
    record(Consistency=g.avg_consistency)
    record(Threshold=g.effective_threshold)
    record(Alignment=g.factor_alignment)