# -*- coding: utf-8 -*-
"""深跌抄底终验: 参数网格/起点敏感性/逐年/抄底记录审计。"""
import backtest
import strategy
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

print("== 1. 参数网格 (mom5阈值 x 深跌阈值, 2014起) ==")
print("       mom5:  -6%%            -8%%            -10%%")
for bm in (0.12, 0.15, 0.18, 0.20, 0.25, 0.30):
    row = "  深跌%4.0f%%: " % (bm * 100)
    for cm in (-0.06, -0.08, -0.10):
        r = backtest.backtest(histories, calendar, start="2014-01-01",
                              crash_mom5=cm, crash_lock=5, crash_below_ma=bm)
        row += "%+5.1f%%/%+.0f%%/dd%4.1f  " % (r["ann"] * 100, (r["nav"] - 1) * 100, r["max_dd"] * 100)
    print(row, flush=True)

print("\n== 2. 锁仓邻域 (mom5-8%%, 深跌20%%) ==")
for lk in (3, 5, 7, 10):
    r = backtest.backtest(histories, calendar, start="2014-01-01",
                          crash_mom5=-0.08, crash_lock=lk, crash_below_ma=0.20)
    print("  锁%d天: 年化 %+5.1f%% | 回撤 %6.1f%% | 终值 %+.0f%%"
          % (lk, r["ann"] * 100, r["max_dd"] * 100, (r["nav"] - 1) * 100), flush=True)

print("\n== 3. 起点敏感性 (最优组合) ==")
for s in ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"):
    r0 = backtest.backtest(histories, calendar, start=s)
    r1 = backtest.backtest(histories, calendar, start=s, crash_mom5=-0.08, crash_lock=5, crash_below_ma=0.20)
    print("  %s起: 基线[%+5.1f%% %+.0f%% dd%5.1f] 抄底[%+5.1f%% %+.0f%% dd%5.1f] %s"
          % (s[:4], r0["ann"] * 100, (r0["nav"] - 1) * 100, r0["max_dd"] * 100,
             r1["ann"] * 100, (r1["nav"] - 1) * 100, r1["max_dd"] * 100,
             "✓" if r1["nav"] >= r0["nav"] else "✗"), flush=True)

print("\n== 4. 逐年 (2014起) ==")
r0 = backtest.backtest(histories, calendar, start="2014-01-01")
r1 = backtest.backtest(histories, calendar, start="2014-01-01", crash_mom5=-0.08, crash_lock=5, crash_below_ma=0.20)
y0, y1 = dict(backtest.yearly(r0["daily"])), dict(backtest.yearly(r1["daily"]))
for y in sorted(y0):
    d = (y1[y] - y0[y]) * 100
    print("  %s: 基线 %+6.1f%% | 抄底 %+6.1f%% | 差 %+5.1fpp" % (y, y0[y] * 100, y1[y] * 100, d))
print("DONE")
