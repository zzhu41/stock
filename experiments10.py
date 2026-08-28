# -*- coding: utf-8 -*-
"""第十一轮: 回撤控制普适方法 (2016-01-01起, v8基线=+36.6%/-27.0%/2624%)。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
S = "2016-01-01"


def show(name, **kw):
    r = backtest.backtest(histories, calendar, start=S, **kw)
    yd = dict(backtest.yearly(r["daily"]))
    print("%s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%% | 换手 %d || 20:%+.0f 22:%+.0f 25:%+.0f 26:%+.0f"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100,
             r["switches"], yd.get("2020", 0) * 100, yd.get("2022", 0) * 100,
             yd.get("2025", 0) * 100, yd.get("2026", 0) * 100), flush=True)


print("== 基线 ==")
show("v8 基线             ")

print("\n== A. 信号平滑(score N日均值) ==")
for n in (2, 3, 5):
    show("score %d 日均值     " % n, score_smooth=n)

print("\n== B. 高波禁进场(年化vol>X不开新仓) ==")
for x in (0.30, 0.35, 0.40):
    show("vol>%4.0f%%禁进场   " % (x * 100), enter_vol_max=x)

print("\n== C. 极端高波临时半仓 ==")
for x in (0.30, 0.35, 0.40):
    show("vol>%4.0f%%→半仓    " % (x * 100), high_vol_half=x)

print("\n== D. 组合回撤风控(回撤X%%后仅避险, 修复一半解除) ==")
for x in (0.10, 0.12, 0.15):
    show("回撤>%4.0f%%风控    " % (x * 100), dd_guard=x)

print("\n== E. 熊市进场加门槛 ==")
for x in (0.03, 0.05, 0.08):
    show("熊市MOM20>+%.0f%%    " % (x * 100), bear_enter_mom=x)

print("\n== F. 保守兜底(min vol) ==")
show("兜底=避险最低波动   ", safe_min_vol=True)

print("\n== G. ATR缩放缓冲 ==")
for k in (0.2, 0.3, 0.5):
    show("buf=%.1f×vol×√20    " % k, vol_buffer=k)

print("\n== H. 组合抽查 ==")
show("平滑3+高波半仓35%%   ", score_smooth=3, high_vol_half=0.35)
show("平滑3+回撤风控12%%   ", score_smooth=3, dd_guard=0.12)
show("平滑3+熊市门槛5%%    ", score_smooth=3, bear_enter_mom=0.05)
print("DONE")
