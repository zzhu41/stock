# -*- coding: utf-8 -*-
"""vol目标仓位细化: 每日调仓 vs 换仓日锁定; 参数细化 + 起点敏感性。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

print("== 1. 每日调仓 vs 换仓日锁定 (2014起) ==")
for vt in (0.15, 0.18, 0.20, 0.22, 0.25):
    line = "  vol目标%2.0f%%: " % (vt * 100)
    for lock in (False, True):
        r = backtest.backtest(histories, calendar, start="2014-01-01",
                              vol_target=vt, vol_target_lock=lock)
        line += "%s[年化%+5.1f%% 回撤%6.1f%% 夏普%4.2f 卡玛%4.2f] " % (
            "每日" if not lock else "锁定", r["ann"] * 100, r["max_dd"] * 100,
            r["sharpe"], r["calmar"])
    print(line)

print("\n== 2. 起点敏感性: vol20%%每日 vs vol20%%锁定 vs 满仓基线 ==")
for s in ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"):
    line = "  %s 起: " % s[:4]
    for kw, tag in (({}, "满仓"), ({"vol_target": 0.20}, "每日20"),
                    ({"vol_target": 0.20, "vol_target_lock": True}, "锁定20")):
        r = backtest.backtest(histories, calendar, start=s, **kw)
        line += "%s[年化%+5.1f%% 回撤%6.1f%% 夏普%4.2f] " % (
            tag, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"])
    print(line)

print("\n== 3. 逐年: 锁定20%% vs 满仓 ==")
r1 = backtest.backtest(histories, calendar, start="2014-01-01",
                       vol_target=0.20, vol_target_lock=True)
r0 = backtest.backtest(histories, calendar, start="2014-01-01")
y0, y1 = dict(backtest.yearly(r0["daily"])), dict(backtest.yearly(r1["daily"]))
for y in sorted(y0):
    print("  %s: 满仓 %+6.1f%% | 锁定20%% %+6.1f%%" % (y, y0[y] * 100, y1[y] * 100))
