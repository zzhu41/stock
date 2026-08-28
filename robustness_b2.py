# -*- coding: utf-8 -*-
"""B2(单日急跌离场)稳健性检验: 参数扫描 / 子区间 / 触发日志及触发后走势。"""
import backtest
import strategy
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

print("== 1. panic_drop 参数扫描 (2018起) ==")
for pd_ in (0.0, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08):
    r = backtest.backtest(histories, calendar, start="2018-01-01", panic_drop=pd_)
    print("  panic=%4.1f%%: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 卡玛 %4.2f | 年换手 %4.1f"
          % (pd_ * 100, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], r["calmar"], r["sw_per_year"]))

print("\n== 2. 子区间表现 (基线 vs panic5%) ==")
for s, e in (("2018-01-01", "2021-12-31"), ("2022-01-01", "2026-08-17"),
             ("2018-01-01", "2019-12-31"), ("2020-01-01", "2021-12-31"),
             ("2022-01-01", "2023-12-31"), ("2024-01-01", "2026-08-17")):
    line = "  %s~%s: " % (s[:4], e[:4] if e < "2026" else "now")
    for pd_ in (0.0, 0.05):
        r = backtest.backtest(histories, calendar, start=s, end=e, panic_drop=pd_)
        line += "panic%3.1f%%[年化%+5.1f%% 回撤%6.1f%%] " % (
            pd_ * 100, r["ann"] * 100, r["max_dd"] * 100)
    print(line)

print("\n== 3. panic5% 触发日志: 每次触发后标的5日/10日实际走势 ==")
r = backtest.backtest(histories, calendar, start="2018-01-01", panic_drop=0.05)
daily, trades = r["daily"], r["trades"]
# 重放找出 panic 触发日: 持仓单日 ret1 <= -5% 且次日换仓
idx = {d: i for i, (d, _, _) in enumerate(daily)}
close_of = {c: {row[0]: row[2] for row in rows} for c, rows in histories.items()}
cal_idx = {d: i for i, d in enumerate(calendar)}
n_dodged = n_whipsaw = 0
for i in range(1, len(daily)):
    d, nav, h = daily[i]
    d_prev, _, h_prev = daily[i - 1]
    if h != h_prev and h_prev and h_prev != "511880":
        rows = close_of[h_prev]
        p0, p1 = rows.get(d_prev), rows.get(d)
        if p0 and p1 and p1 / p0 - 1 <= -0.05:
            # 触发后该标的 5/10 日走势
            j = cal_idx.get(d)
            fwd = []
            for k in (5, 10):
                if j and j + k < len(calendar) and calendar[j + k] in rows:
                    fwd.append(rows[calendar[j + k]] / p1 - 1)
                else:
                    fwd.append(None)
            dodged = fwd[0] is not None and fwd[0] < 0
            n_dodged += dodged
            n_whipsaw += not dodged
            print("  %s %s %s 单日 %+.1f%% -> 离场, 后5日 %s 后10日 %s %s"
                  % (d, h_prev, UNIVERSE[h_prev][0], (p1 / p0 - 1) * 100,
                     "%+.1f%%" % (fwd[0] * 100) if fwd[0] is not None else "  n/a",
                     "%+.1f%%" % (fwd[1] * 100) if fwd[1] is not None else "  n/a",
                     "躲开" if dodged else "卖飞"))
print("触发 %d 次: 躲开 %d / 卖飞 %d" % (n_dodged + n_whipsaw, n_dodged, n_whipsaw))
