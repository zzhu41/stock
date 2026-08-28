# -*- coding: utf-8 -*-
"""v8 Ablation Study: 逐个关闭组件, 量化每个组件的贡献 (2014起)。"""
import backtest
import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
S = "2014-01-01"
orig_decide = strategy.decide
orig_enter_ok = strategy._enter_ok


def run(name, kw=None, decide=None, enter_ok=None, glob_pool=None):
    strategy.decide = decide or orig_decide
    strategy._enter_ok = enter_ok or orig_enter_ok
    old_glob = strategy.GLOBAL_POOL
    if glob_pool is not None:
        strategy.GLOBAL_POOL = glob_pool
    r = backtest.backtest(histories, calendar, start=S, **(kw or {}))
    strategy.decide = orig_decide
    strategy._enter_ok = orig_enter_ok
    strategy.GLOBAL_POOL = old_glob
    yd = dict(backtest.yearly(r["daily"]))
    calmar = r["ann"] / abs(r["max_dd"]) if r["max_dd"] else 0
    print("%-22s: %+5.1f%% | %6.1f%% | %4.2f | %4.2f | %+.0f%% | %3d || 18:%+.0f 20:%+.0f 22:%+.0f 25:%+.0f 26:%+.0f"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], calmar, (r["nav"] - 1) * 100,
             r["switches"], yd.get("2018", 0) * 100, yd.get("2020", 0) * 100,
             yd.get("2022", 0) * 100, yd.get("2025", 0) * 100, yd.get("2026", 0) * 100), flush=True)
    return r


def decide_bear_open(table, holding, holding_days=99):
    """去牛熊开关: 竞赛池恒为全部风险标的。"""
    info = {c: ind for c, ind in table}
    pool = STOCK_POOL + GLOBAL_POOL + [GOLD]
    best = next((t for t in table if t[0] in pool and strategy._enter_ok(t[1], t[0])), None)
    target, reason = (best[0], "全池") if best else (GOLD if GOLD in info else CASH, "兜底")
    if best is None and GOLD not in info:
        target = CASH
    if holding and holding != target and holding in info:
        h = info[holding]
        ew = strategy._exit_hit(h)
        if ew:
            return target, ew
        c_mom = info[target]["mom20"] if target in info else 0.0
        if c_mom - h["mom20"] < strategy.BUFFER:
            return holding, "缓冲"
    return target, reason


def decide_no_gold_spare(table, holding, holding_days=99):
    """去牛市黄金备胎: 牛市竞赛池无人达标直接走永不空仓。"""
    info = {c: ind for c, ind in table}
    bull = strategy._is_bull(info)
    comp_pool = STOCK_POOL + GLOBAL_POOL if bull else GLOBAL_POOL + [GOLD]
    best = next((t for t in table if t[0] in comp_pool and strategy._enter_ok(t[1], t[0])), None)
    if best:
        target, reason = best[0], "竞赛池第一"
    else:
        cand = next((x for x in table if x[0] in GLOBAL_POOL + [GOLD]), None)
        target, reason = (cand[0], "避险最强") if cand else (CASH, "空仓")
    if holding and holding != target and holding in info:
        h = info[holding]
        if not bull and holding in STOCK_POOL:
            return target, "熊市离场"
        ew = strategy._exit_hit(h)
        if ew:
            return target, ew
        c_mom = info[target]["mom20"] if target in info else 0.0
        if c_mom - h["mom20"] < strategy.BUFFER:
            return holding, "缓冲"
    return target, reason


def decide_no_gold_bear(table, holding, holding_days=99):
    """去黄金熊市参赛: 熊市竞赛池仅跨境。"""
    info = {c: ind for c, ind in table}
    bull = strategy._is_bull(info)
    comp_pool = STOCK_POOL + GLOBAL_POOL if bull else list(GLOBAL_POOL)
    best = next((t for t in table if t[0] in comp_pool and strategy._enter_ok(t[1], t[0])), None)
    if best:
        target, reason = best[0], "竞赛池第一"
    elif bull and GOLD in info and info[GOLD]["mom20"] > 0:
        target, reason = GOLD, "黄金备胎"
    else:
        cand = next((x for x in table if x[0] in GLOBAL_POOL + [GOLD]), None)
        target, reason = (cand[0], "避险最强") if cand else (CASH, "空仓")
    if holding and holding != target and holding in info:
        h = info[holding]
        if not bull and holding in STOCK_POOL:
            return target, "熊市离场"
        ew = strategy._exit_hit(h)
        if ew:
            return target, ew
        c_mom = info[target]["mom20"] if target in info else 0.0
        if c_mom - h["mom20"] < strategy.BUFFER:
            return holding, "缓冲"
    return target, reason


def enter_ok_no_absmom(ind, code=None):
    """去绝对动量门: MOM20<=0 也可进场(恒持最强)。"""
    if ind["mom20"] > strategy.OVERHEAT and ind["mom5"] <= 0:
        return False
    return True


print("组件消融 (年化 | 回撤 | 夏普 | 卡玛 | 终值 | 换手 || 关键年)")
run("基线 v8 (全组件)     ")
run("去 WLS 排名(回v7)     ", {"score_wls": False})
run("去牛熊开关           ", decide=decide_bear_open)
run("去滞回缓冲(BUF=0)    ", {"buffer": 0.0})
run("去 panic 急跌离场     ", {"panic_drop": 0.0})
run("去过热快离场         ", {"overheat": 9.9})
run("去永不空仓           ", {"never_empty": False})
run("去牛市黄金备胎       ", decide=decide_no_gold_spare)
run("去跨境池(纯A股+黄金) ", glob_pool=[])
run("去黄金熊市参赛       ", decide=decide_no_gold_bear)
run("去绝对动量门(恒持最强)", enter_ok=enter_ok_no_absmom)
run("v5 形态(全关)        ", {"score_wls": False, "panic_drop": 0.0, "overheat": 9.9, "never_empty": False})
print("DONE")
