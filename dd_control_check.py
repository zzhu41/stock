# -*- coding: utf-8 -*-
"""回撤控制候选终验: B高波禁进场 / E熊市门槛。全周期+起点+邻域+逐年+组合。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

print("== 1. 阈值邻域 (2016起) ==")
for x in (0.32, 0.35, 0.37, 0.40, 0.45, 0.50):
    r = backtest.backtest(histories, calendar, start="2016-01-01", enter_vol_max=x)
    print("  vol>%4.0f%%: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%%"
          % (x * 100, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100), flush=True)

print("\n== 2. 起点敏感性: vol35/vol40/熊门5 (2014/2016/2018/2020/2022/2024) ==")
for s in ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"):
    line = "  %s起: " % s[:4]
    r0 = backtest.backtest(histories, calendar, start=s)
    line += "基线[%+5.1f%% %6.1f%% %+.0f%%] " % (r0["ann"] * 100, r0["max_dd"] * 100, (r0["nav"] - 1) * 100)
    for tag, kw in (("vol35", {"enter_vol_max": 0.35}), ("vol40", {"enter_vol_max": 0.40}),
                    ("熊门5", {"bear_enter_mom": 0.05})):
        r = backtest.backtest(histories, calendar, start=s, **kw)
        line += "%s[%+5.1f%% %6.1f%% %+.0f%%] " % (tag, r["ann"] * 100, r["max_dd"] * 100, (r["nav"] - 1) * 100)
    print(line, flush=True)

print("\n== 3. 完整逐年 (2016起) ==")
r0 = backtest.backtest(histories, calendar, start="2016-01-01")
r35 = backtest.backtest(histories, calendar, start="2016-01-01", enter_vol_max=0.35)
r40 = backtest.backtest(histories, calendar, start="2016-01-01", enter_vol_max=0.40)
rb5 = backtest.backtest(histories, calendar, start="2016-01-01", bear_enter_mom=0.05)
y0 = dict(backtest.yearly(r0["daily"]))
y35 = dict(backtest.yearly(r35["daily"]))
y40 = dict(backtest.yearly(r40["daily"]))
yb5 = dict(backtest.yearly(rb5["daily"]))
for y in sorted(y0):
    print("  %s: 基线 %+6.1f%% | vol35 %+6.1f%% | vol40 %+6.1f%% | 熊门5 %+6.1f%%"
          % (y, y0[y] * 100, y35[y] * 100, y40[y] * 100, yb5[y] * 100))

print("\n== 4. 组合: vol35+熊门5 (2016起) ==")
r = backtest.backtest(histories, calendar, start="2016-01-01", enter_vol_max=0.35, bear_enter_mom=0.05)
print("  组合: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%%"
      % (r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100))
print("\n== 5. 机制检查: vol35禁进场在 2020-03/2026-01 的行为 ==")
import strategy
for code, d0, d1 in (("512400", "2026-01-20", "2026-02-05"), ("513100", "2020-02-20", "2020-03-25")):
    rows = [x for x in histories[code] if x[0] <= d1]
    closes = [x[2] for x in rows]
    dates = [x[0] for x in rows]
    hits = 0
    for i in range(len(dates) - 1, -1, -1):
        if dates[i] < d0:
            break
        ind = strategy.indicators(closes[:i + 1])
        if ind:
            va = ind["vol"] * 244 ** 0.5
            if va > 0.35:
                hits += 1
    print("  %s %s~%s: vol>35%% 的天数 = %d" % (code, d0, d1, hits))
print("DONE")
