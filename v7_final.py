# -*- coding: utf-8 -*-
"""v7定型对比: 永不空仓两版本(固定黄金 vs 跨境/黄金最强) 起点敏感性+逐年+换手。"""
import backtest
import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
orig_decide = strategy.decide


def make_ne(mode):
    def decide_ne(table, holding, holding_days=99):
        t, reason = orig_decide(table, holding, holding_days)
        if t == CASH:
            info = {c: ind for c, ind in table}
            if mode == "gold":
                if GOLD in info:
                    return GOLD, reason + " (永不空仓->黄金)"
            else:
                cand = next((x for x in table if x[0] in GLOBAL_POOL + [GOLD]), None)
                if cand:
                    return cand[0], reason + " (永不空仓->避险最强)"
        return t, reason
    return decide_ne


print("== 起点敏感性: 基线 vs 黄金版 vs 避险最强版 ==")
for s in ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"):
    r0 = backtest.backtest(histories, calendar, start=s)
    strategy.decide = make_ne("gold")
    r1 = backtest.backtest(histories, calendar, start=s)
    strategy.decide = make_ne("best")
    r2 = backtest.backtest(histories, calendar, start=s)
    strategy.decide = orig_decide
    print("  %s起: 基线[年化%+5.1f%% 回撤%6.1f%% 终值%+.0f%%] 黄金版[%+5.1f%% %6.1f%% %+.0f%%] 避险最强[%+5.1f%% %6.1f%% %+.0f%%]"
          % (s[:4], r0["ann"] * 100, r0["max_dd"] * 100, (r0["nav"] - 1) * 100,
             r1["ann"] * 100, r1["max_dd"] * 100, (r1["nav"] - 1) * 100,
             r2["ann"] * 100, r2["max_dd"] * 100, (r2["nav"] - 1) * 100))

print("\n== 逐年 (2014起): 黄金版 vs 避险最强版 ==")
strategy.decide = make_ne("gold")
r1 = backtest.backtest(histories, calendar, start="2014-01-01")
strategy.decide = make_ne("best")
r2 = backtest.backtest(histories, calendar, start="2014-01-01")
strategy.decide = orig_decide
y1, y2 = dict(backtest.yearly(r1["daily"])), dict(backtest.yearly(r2["daily"]))
for y in sorted(y1):
    print("  %s: 黄金版 %+6.1f%% | 避险最强 %+6.1f%%" % (y, y1[y] * 100, y2[y] * 100))

print("\n== 换手次数 (2014起) ==")
print("  基线 %d | 黄金版 %d | 避险最强 %d" % (r0["switches"], r1["switches"], r2["switches"]))
print("DONE")
