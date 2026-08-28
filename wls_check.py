# -*- coding: utf-8 -*-
"""WLS 终验: 起点敏感性 / 窗口邻域 / 完整逐年。附成交量变体。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

print("== 1. WLS 起点敏感性 (vs 基线) ==")
for s in ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"):
    r0 = backtest.backtest(histories, calendar, start=s)
    r1 = backtest.backtest(histories, calendar, start=s, score_wls=True)
    print("  %s起: 基线[年化%+5.1f%% 回撤%6.1f%% 终值%+.0f%%] WLS[年化%+5.1f%% 回撤%6.1f%% 终值%+.0f%%] %s"
          % (s[:4], r0["ann"] * 100, r0["max_dd"] * 100, (r0["nav"] - 1) * 100,
             r1["ann"] * 100, r1["max_dd"] * 100, (r1["nav"] - 1) * 100,
             "✓" if r1["nav"] > r0["nav"] else "✗"), flush=True)

print("\n== 2. 完整逐年对比 (2014起) ==")
r0 = backtest.backtest(histories, calendar, start="2014-01-01")
r1 = backtest.backtest(histories, calendar, start="2014-01-01", score_wls=True)
y0, y1 = dict(backtest.yearly(r0["daily"])), dict(backtest.yearly(r1["daily"]))
neg = 0
for y in sorted(y0):
    d = (y1[y] - y0[y]) * 100
    neg += d < -1
    print("  %s: 基线 %+6.1f%% | WLS %+6.1f%% | 差 %+5.1fpp" % (y, y0[y] * 100, y1[y] * 100, d))
print("受损>1pp的年份数: %d" % neg)

print("\n== 3. 成交量变体 (2014起) ==")
for name, kw in (("量比>=1.0进场", {"vol_in_min": 1.0}),
                 ("量比>=1.5进场", {"vol_in_min": 1.5}),
                 ("量比>2.5禁追 ", {"vol_in_max": 2.5}),
                 ("量比加权score ", {"vol_score_mix": True})):
    r = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
    print("  %s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%% | 换手 %d"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100, r["switches"]), flush=True)
print("DONE")
