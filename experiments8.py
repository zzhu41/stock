# -*- coding: utf-8 -*-
"""第八轮: 止损专项(针对2022)。基线=v7。重点看 2022/2018/2020 逐年。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
S = "2014-01-01"


def show(name, **kw):
    r = backtest.backtest(histories, calendar, start=S, **kw)
    yd = dict(backtest.yearly(r["daily"]))
    key = "22:%+.0f 18:%+.0f 20:%+.0f 25:%+.0f 26:%+.0f" % (
        yd.get("2022", 0) * 100, yd.get("2018", 0) * 100, yd.get("2020", 0) * 100,
        yd.get("2025", 0) * 100, yd.get("2026", 0) * 100)
    print("%s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%% | 换手 %d || %s"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100,
             r["switches"], key), flush=True)


print("== 基线 ==")
show("v7 基线            ")

print("\n== A. 兜底质量门(治2022: 最强避险MOM20<X持币) ==")
for x in (-0.02, -0.03, -0.05, -0.08):
    show("兜底门 MOM>%4.0f%%  " % (x * 100), ne_min_mom=x)

print("\n== B. 熊市关永不空仓(牛市保留) ==")
show("熊市持币/牛市NE   ", ne_bull_only=True)

print("\n== C. 唐奇安低点止损 ==")
for n in (5, 10, 15):
    show("跌破前%2d日最低   " % n, donchian_n=n)

print("\n== D. MOM20阈值离场(提前) ==")
for x in (0.01, 0.02, 0.03):
    show("MOM20<+%.0f%%离场  " % (x * 100), exit_mom_floor=x)

print("\n== E. 动量衰减止损(峰值回落) ==")
for x in (0.08, 0.12, 0.16):
    show("衰减>%4.0f%%离场   " % (x * 100), mom_decay=x)

print("\n== F. 组合抽查 ==")
show("兜底门-3%%+唐奇安10", ne_min_mom=-0.03, donchian_n=10)
show("兜底门-3%%+衰减12%% ", ne_min_mom=-0.03, mom_decay=0.12)
print("DONE")
