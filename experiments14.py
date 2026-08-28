# -*- coding: utf-8 -*-
"""v10 第二轮矩阵: 牛熊滞后带/体制确认/抄底动态解锁/大涨止盈/WLS自适应窗口/回归线乖离。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]


def show(name, **kw):
    r14 = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
    r16 = backtest.backtest(histories, calendar, start="2016-01-01", **kw)
    y14 = dict(backtest.yearly(r14["daily"]))
    print("%-22s: 14起[%+5.1f%% %6.1f%% %+.0f%%] 16起[%+5.1f%% %6.1f%% %+.0f%%] || 18:%+.0f 20:%+.0f 22:%+.0f 24:%+.0f 26:%+.0f"
          % (name, r14["ann"] * 100, r14["max_dd"] * 100, (r14["nav"] - 1) * 100,
             r16["ann"] * 100, r16["max_dd"] * 100, (r16["nav"] - 1) * 100,
             y14.get("2018", 0) * 100, y14.get("2020", 0) * 100, y14.get("2022", 0) * 100,
             y14.get("2024", 0) * 100, y14.get("2026", 0) * 100), flush=True)


print("== 基线 ==")
show("v9 基线            ")

print("\n== A. 牛熊滞后带 ==")
for b in (0.005, 0.01, 0.02, 0.03):
    show("滞后带 %4.1f%%       " % (b * 100), bull_hyst=b)

print("\n== B. 体制切换确认 ==")
for n in (2, 3):
    show("体制确认 %d 天       " % n, bull_confirm=n)

print("\n== C. 抄底动态解锁(MOM5转正+浮盈3%%) ==")
show("动态解锁            ", crash_dyn_unlock=True)

print("\n== D. 锁仓大涨止盈 ==")
for x in (0.03, 0.04, 0.05):
    show("单日 +%.0f%% 止盈     " % (x * 100), crash_tp=x)

print("\n== E. WLS 波动自适应窗口 ==")
show("自适应窗口(vol>30→30)", wls_adaptive=True)

print("\n== F. 回归线乖离减分 ==")
show("乖离>2σ score减半   ", resid_penalty=True)

print("\n== G. 组合 ==")
show("滞后带1%%+确认2天     ", bull_hyst=0.01, bull_confirm=2)
show("动态解锁+大涨4%%      ", crash_dyn_unlock=True, crash_tp=0.04)
print("DONE")
