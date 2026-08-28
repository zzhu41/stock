# -*- coding: utf-8 -*-
"""v7候选终验: 永不空仓稳健性(起点/逐年/邻域) + 与杠杆1.2x组合 + 杠杆担保比例测算。"""
import backtest
import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
orig_decide = strategy.decide


def decide_never_empty(table, holding, holding_days=99):
    t, reason = orig_decide(table, holding, holding_days)
    if t == CASH:
        info = {c: ind for c, ind in table}
        if GOLD in info:
            return GOLD, reason + " (永不空仓->黄金)"
    return t, reason


print("== 1. 永不空仓 起点敏感性 ==")
for s in ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"):
    r0 = backtest.backtest(histories, calendar, start=s)
    strategy.decide = decide_never_empty
    r1 = backtest.backtest(histories, calendar, start=s)
    strategy.decide = orig_decide
    print("  %s 起: 基线[年化%+5.1f%% 回撤%6.1f%% 终值%+.0f%%] 永不空仓[年化%+5.1f%% 回撤%6.1f%% 终值%+.0f%%]"
          % (s[:4], r0["ann"] * 100, r0["max_dd"] * 100, (r0["nav"] - 1) * 100,
             r1["ann"] * 100, r1["max_dd"] * 100, (r1["nav"] - 1) * 100))

print("\n== 2. 逐年对比 (2014起) ==")
r0 = backtest.backtest(histories, calendar, start="2014-01-01")
strategy.decide = decide_never_empty
r1 = backtest.backtest(histories, calendar, start="2014-01-01")
strategy.decide = orig_decide
y0, y1 = dict(backtest.yearly(r0["daily"])), dict(backtest.yearly(r1["daily"]))
for y in sorted(y0):
    print("  %s: 基线 %+6.1f%% | 永不空仓 %+6.1f%% | 差 %+5.1fpp"
          % (y, y0[y] * 100, y1[y] * 100, (y1[y] - y0[y]) * 100))

print("\n== 3. 组合: 永不空仓 + 杠杆 ==")
for lv in (1.0, 1.2, 1.3):
    strategy.decide = decide_never_empty
    r = backtest.backtest(histories, calendar, start="2014-01-01", leverage=lv)
    strategy.decide = orig_decide
    print("  永不空仓+杠杆%.1fx: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%%"
          % (lv, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100))

print("\n== 4. 杠杆实盘担保比例测算 (两融警戒线130%%, 平仓线110%%) ==")
for lv in (1.2, 1.3, 1.5):
    strategy.decide = decide_never_empty
    r = backtest.backtest(histories, calendar, start="2014-01-01", leverage=lv)
    strategy.decide = orig_decide
    # 最坏情形: 历史最大回撤发生在满仓杠杆期; 担保比例 = 净资产/负债
    # 本金100, 借 (lv-1)*100, 资产 lv*100; 回撤 dd 后 资产 lv*100*(1+dd), 净资产 = 资产 - 负债
    dd = r["max_dd"]
    debt = (lv - 1.0) * 100
    equity = lv * 100 * (1 + dd) - debt
    ratio = lv * 100 * (1 + dd) / debt * 100 if debt > 0 else 9999
    print("  杠杆%.1fx: 历史最大回撤 %.1f%% -> 担保比例 %.0f%% %s"
          % (lv, dd * 100, ratio, "⚠️ 触及平仓线!" if ratio < 130 else "(安全, 警戒线130%%)"))

print("\n== 5. 邻域: 永不空仓的变体(无人达标时持跨境最强, 而非黄金) ==")
def decide_ne_c(table, holding, holding_days=99):
    t, reason = orig_decide(table, holding, holding_days)
    if t == CASH:
        info = {c: ind for c, ind in table}
        bull = strategy._is_bull(info)
        pool = GLOBAL_POOL + [GOLD]
        cand = next((x for x in table if x[0] in pool), None)
        if cand:
            return cand[0], reason + " (永不空仓->跨境/黄金最强)"
    return t, reason
strategy.decide = decide_ne_c
r = backtest.backtest(histories, calendar, start="2014-01-01")
strategy.decide = orig_decide
print("  永不空仓(跨境最强版): 年化 %+5.1f%% | 回撤 %6.1f%% | 终值 %+.0f%%"
      % (r["ann"] * 100, r["max_dd"] * 100, (r["nav"] - 1) * 100))
print("DONE")
