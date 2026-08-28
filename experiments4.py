# -*- coding: utf-8 -*-
"""v7 激进变体(不扩池): 杠杆 / 纯MOM排名 / 永不空仓 / 熊市放开A股 / 组合。基线=v6。"""
import backtest
import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
orig_decide = strategy.decide


def show(name, r):
    yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in backtest.yearly(r["daily"]))
    print("%s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%% || %s"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100, yrs))


def decide_never_empty(table, holding, holding_days=99):
    """永不空仓: 竞赛池无人达标时无条件持黄金(黄金也不在表则货币)。"""
    t, reason = orig_decide(table, holding, holding_days)
    if t == CASH:
        info = {c: ind for c, ind in table}
        if GOLD in info:
            return GOLD, reason + " (永不空仓->黄金)"
    return t, reason


def decide_bear_open(table, holding, holding_days=99):
    """熊市放开A股池: 竞赛池恒为全部风险标的, 无制度性离场。"""
    info = {c: ind for c, ind in table}
    pool = STOCK_POOL + GLOBAL_POOL + [GOLD]
    best = next((t for t in table if t[0] in pool and strategy._enter_ok(t[1], t[0])), None)
    if best:
        target, reason = best[0], "全池动量第一"
    else:
        target, reason = CASH, "无人达标"
    if holding and holding != target and holding in info:
        h = info[holding]
        exit_why = strategy._exit_hit(h)
        if exit_why:
            return target, exit_why
        c_mom = info[target]["mom20"] if target in info else 0.0
        if c_mom - h["mom20"] < strategy.BUFFER:
            return holding, "滞回缓冲"
    return target, reason


S = "2014-01-01"
print("== v7 激进变体 (%s起, 不扩池) ==" % S[:4])
show("基线 v6(满仓)      ", backtest.backtest(histories, calendar, start=S))

print("\n-- A. 杠杆(含6%%融资成本, 空仓期不上杠杆) --")
for lv in (1.2, 1.3, 1.5):
    show("杠杆 %.1fx          " % lv, backtest.backtest(histories, calendar, start=S, leverage=lv))

print("\n-- B. 纯MOM排名(不除波动率) --")
show("纯MOM排名         ", backtest.backtest(histories, calendar, start=S, score_plain_mom=True))

print("\n-- C. 永不空仓 --")
strategy.decide = decide_never_empty
show("永不空仓          ", backtest.backtest(histories, calendar, start=S))
strategy.decide = orig_decide

print("\n-- D. 熊市放开A股池(关开关) --")
strategy.decide = decide_bear_open
show("熊市放开A股       ", backtest.backtest(histories, calendar, start=S))
strategy.decide = orig_decide

print("\n-- E. 组合 --")
show("纯MOM+杠杆1.2x    ", backtest.backtest(histories, calendar, start=S, score_plain_mom=True, leverage=1.2))
show("杠杆1.2x+永不空仓 ", end=" ")
strategy.decide = decide_never_empty
show2 = backtest.backtest(histories, calendar, start=S, leverage=1.2)
strategy.decide = orig_decide
yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in backtest.yearly(show2["daily"]))
print("年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%% || %s"
      % (show2["ann"] * 100, show2["max_dd"] * 100, show2["sharpe"], (show2["nav"] - 1) * 100, yrs))
print("DONE")
