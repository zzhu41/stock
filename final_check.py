# -*- coding: utf-8 -*-
"""最终验证: panic=4% 起点敏感性 / 回撤区间对比 / 逐年对比。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

print("== 1. 起点敏感性: 基线 vs panic4% ==")
for s in ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"):
    line = "  %s 起: " % s[:4]
    for pd_ in (0.0, 0.04):
        r = backtest.backtest(histories, calendar, start=s, panic_drop=pd_)
        line += "%s[年化%+5.1f%% 回撤%6.1f%% 夏普%4.2f] " % (
            "基线" if pd_ == 0 else "panic", r["ann"] * 100, r["max_dd"] * 100, r["sharpe"])
    print(line)

print("\n== 2. panic4% 的 Top5 回撤区间 ==")
r = backtest.backtest(histories, calendar, start="2018-01-01", panic_drop=0.04)
daily = r["daily"]
peak_nav, peak_date = daily[0][1], daily[0][0]
trough_nav, trough_date = daily[0][1], daily[0][0]
eps, in_dd = [], False
for d, nav, h in daily:
    if nav >= peak_nav:
        if in_dd:
            eps.append((trough_nav / peak_nav - 1, peak_date, trough_date, d))
            in_dd = False
        peak_nav, peak_date, trough_nav, trough_date = nav, d, nav, d
    else:
        in_dd = True
        if nav < trough_nav:
            trough_nav, trough_date = nav, d
if in_dd:
    eps.append((trough_nav / peak_nav - 1, peak_date, trough_date, None))
eps.sort()
for depth, p, t, rec in eps[:5]:
    print("  %6.1f%% | %s -> %s | 修复 %s" % (depth * 100, p, t, rec or "未修复"))

print("\n== 3. 逐年对比 ==")
r0 = backtest.backtest(histories, calendar, start="2018-01-01")
y0 = dict(backtest.yearly(r0["daily"]))
y1 = dict(backtest.yearly(r["daily"]))
for y in sorted(y0):
    print("  %s: 基线 %+6.1f%% | panic4%% %+6.1f%% | 差 %+5.1fpp"
          % (y, y0[y] * 100, y1[y] * 100, (y1[y] - y0[y]) * 100))

print("\n== 4. 交易统计 ==")
print("  基线: 换手 %d 次 | panic4%%: 换手 %d 次" % (r0["switches"], r["switches"]))
