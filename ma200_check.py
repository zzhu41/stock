# -*- coding: utf-8 -*-
"""MA200 候选终验: 邻域细化 / 起点敏感性 / 逐年对比 / 牛熊切换次数。附 PANIC 细化补跑。"""
import backtest
import strategy
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

print("== 1. MA 邻域细化 (2014起) ==")
for ma in (150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250):
    r = backtest.backtest(histories, calendar, start="2014-01-01", ma_bull=ma)
    print("  MA=%-3d: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 总收益 %+.0f%%"
          % (ma, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100))

print("\n== 2. MA200 vs MA250 起点敏感性 ==")
for s in ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"):
    line = "  %s 起: " % s[:4]
    for ma in (250, 200):
        r = backtest.backtest(histories, calendar, start=s, ma_bull=ma)
        line += "MA%d[年化%+5.1f%% 回撤%6.1f%% 夏普%4.2f 终值%+.0f%%] " % (
            ma, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100)
    print(line)

print("\n== 3. 逐年对比 (2014起) ==")
r250 = backtest.backtest(histories, calendar, start="2014-01-01", ma_bull=250)
r200 = backtest.backtest(histories, calendar, start="2014-01-01", ma_bull=200)
y250, y200 = dict(backtest.yearly(r250["daily"])), dict(backtest.yearly(r200["daily"]))
for y in sorted(y250):
    d = (y200[y] - y250[y]) * 100
    print("  %s: MA250 %+6.1f%% | MA200 %+6.1f%% | 差 %+5.1fpp" % (y, y250[y] * 100, y200[y] * 100, d))

print("\n== 4. 牛熊体制切换次数对比 (2014起) ==")
for ma in (250, 200):
    rows = histories["510300"]
    closes_all = [x[2] for x in rows]
    dates_all = [x[0] for x in rows]
    state, switches, n_days = None, 0, {}
    for i in range(ma, len(dates_all)):
        if dates_all[i] < "2014-01-01":
            continue
        above = closes_all[i] > sum(closes_all[i - ma + 1:i + 1]) / ma
        if state is not None and above != state:
            switches += 1
        state = above
        n_days[above] = n_days.get(above, 0) + 1
    print("  MA%d: 牛熊切换 %d 次 | 牛市 %d 天 / 熊市 %d 天"
          % (ma, switches, n_days.get(True, 0), n_days.get(False, 0)))

print("\n== 5. PANIC 平台细化补跑 ==")
for p in (0.035, 0.0375, 0.04, 0.0425, 0.045):
    r = backtest.backtest(histories, calendar, start="2014-01-01", panic_drop=p)
    print("  panic=%.3f: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 总收益 %+.0f%%"
          % (p, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100))
print("DONE")
