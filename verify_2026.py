# -*- coding: utf-8 -*-
"""回测可复核性导出: 2026年全部交易/持仓段/逐日明细CSV/价格抽样/手工验算。"""
import csv
import backtest
from market_data import UNIVERSE, CASH, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
r = backtest.backtest(histories, calendar, start="2014-01-01")  # 全周期路径, 取2026段
daily, trades = r["daily"], r["trades"]
close_of = {c: {row[0]: row[2] for row in rows} for c, rows in histories.items()}

d26 = [(d, nav, h) for d, nav, h in daily if d >= "2026-01-01"]
t26 = [t for t in trades if t[0] >= "2026-01-01"]

print("== 2026 年换仓日志 (共 %d 次) ==" % len(t26))
print("日期       | 操作                          | 成交价  | 当日净值")
for d, frm, to, nav in t26:
    p = close_of[to].get(d, 0.0)
    frm_s = UNIVERSE[frm][0] if frm else "(建仓)"
    print("%s | %s -> %s %s | %7.3f | %.4f" % (d, frm_s, UNIVERSE[to][0], to, p, nav))

# 持仓段: 相邻换仓之间
print("\n== 2026 年持仓段明细 ==")
all_t = list(trades) + [(daily[-1][0], None, None, daily[-1][1])]
for i in range(1, len(all_t)):
    d0, frm0, to0, nav0 = all_t[i - 1]
    d1 = all_t[i][0]
    if d1 < "2026-01-01" or d0 > "2026-08-17":
        continue
    code = to0
    p_in = close_of[code].get(d0, 0.0) if code else 0.0
    p_out = close_of[code].get(d1, 0.0) if code else 0.0
    days_held = len([x for x, _, _ in daily if d0 < x <= d1])
    seg_ret = (p_out / p_in - 1) * 100 if p_in and p_out else 0.0
    print("%s ~ %s | %-7s %-6s | %d天 | %7.3f -> %7.3f | 段收益 %+6.2f%%"
          % (d0, d1, code, UNIVERSE[code][0], days_held, p_in, p_out, seg_ret))

print("\n== 2026 年逐月净值 ==")
mon = {}
for d, nav, h in d26:
    mon[d[:7]] = nav
base = [nav for d, nav, _ in daily if d < "2026-01-01"][-1]
for m in sorted(mon):
    print("  %s: 净值 %.4f | 月收益 %+6.2f%%" % (m, mon[m], (mon[m] / base - 1) * 100))
    base = mon[m]
y26 = d26[-1][1] / [nav for d, nav, _ in daily if d < "2026-01-01"][-1] - 1
print("  2026 累计: %+6.2f%% (全周期净值 %.4f)" % (y26 * 100, daily[-1][1]))

# 逐日明细 CSV
with open("/root/stock/backtest_2026_daily.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["date", "holding", "name", "close", "daily_ret%", "nav"])
    for i, (d, nav, h) in enumerate(d26):
        p = close_of[h].get(d, 0.0)
        prev_nav = d26[i - 1][1] if i else [n for x, n, _ in daily if x < "2026-01-01"][-1]
        w.writerow([d, h, UNIVERSE[h][0], "%.3f" % p, "%+.2f" % ((nav / prev_nav - 1) * 100), "%.4f" % nav])
print("\n逐日明细已存 /root/stock/backtest_2026_daily.csv (%d 行)" % len(d26))

print("\n== 价格抽样 (请对照行情软件, 前复权收盘价) ==")
for c, d in (("512400", "2026-01-29"), ("512400", "2026-01-30"), ("512400", "2026-02-02"),
             ("588080", "2026-07-09"), ("588080", "2026-07-10"),
             ("513100", "2026-06-01"), ("510300", "2026-08-17")):
    print("  %s %s %s: %.3f" % (d, c, UNIVERSE[c][0], close_of[c].get(d, -1)))

print("\n== 手工验算示例: 2026 第一段持仓 ==")
i0 = next(i for i, t in enumerate(trades) if t[0] >= "2026-01-01")
d0, _, c0, nav0 = trades[i0]
d1, _, c1, nav1 = trades[i0 + 1]
p0, p1 = close_of[c0][d0], close_of[c0][d1]
gross = p1 / p0
fee = (1 - 0.0002)
print("  %s 买 %s @%.3f (净值 %.4f) -> %s 卖 @%.3f" % (d0, c0, p0, nav0, d1, p1))
print("  价格涨幅 %.4f%% | 扣双边费后 %.4f%%" % ((gross - 1) * 100, (gross * fee - 1) * 100))
print("  回测净值 %.4f -> %.4f = %+.4f%% (含换入下一只的费用, 尾差属正常)" % (nav0, nav1, (nav1 / nav0 - 1) * 100))
