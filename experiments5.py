# -*- coding: utf-8 -*-
"""借鉴平台机制实测(本框架+v7基线): 风控离场冷却 exit_cooldown / 绝对止损 abs_stop。
先验证默认参数复现 v7 基线(39.4/-28.1/6411)。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
S = "2014-01-01"


def show(name, **kw):
    r = backtest.backtest(histories, calendar, start=S, **kw)
    yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in backtest.yearly(r["daily"]))
    print("%s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%% | 换手 %d || %s"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100,
             r["switches"], yrs))


print("== 基线复现检查 ==")
show("v7 基线(应39.4/-28.1)")

print("\n== A. 风控离场冷却 ==")
for n in (2, 3, 5, 8):
    show("冷却 %d 天      " % n, exit_cooldown=n)

print("\n== B. 绝对止损 ==")
for x in (0.05, 0.07, 0.09, 0.12):
    show("止损 -%4.1f%%    " % (x * 100), abs_stop=x)

print("\n== C. 组合 ==")
for n in (3, 5):
    for x in (0.07, 0.09):
        show("冷却%d+止损%4.1f%% " % (n, x * 100), exit_cooldown=n, abs_stop=x)
print("DONE")
