# -*- coding: utf-8 -*-
"""v9 候选矩阵 (v8.1基线: 14起+42.2/+8334, 16起+38.2/+2982)。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]


def show(name, **kw):
    r14 = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
    r16 = backtest.backtest(histories, calendar, start="2016-01-01", **kw)
    y14 = dict(backtest.yearly(r14["daily"]))
    print("%-20s: 14起[%+5.1f%% %6.1f%% %+.0f%%] 16起[%+5.1f%% %6.1f%% %+.0f%%] || 22:%+.0f 25:%+.0f 26:%+.0f | 换手%d"
          % (name, r14["ann"] * 100, r14["max_dd"] * 100, (r14["nav"] - 1) * 100,
             r16["ann"] * 100, r16["max_dd"] * 100, (r16["nav"] - 1) * 100,
             y14.get("2022", 0) * 100, y14.get("2025", 0) * 100, y14.get("2026", 0) * 100,
             r14["switches"]), flush=True)


print("== 基线 v8.1 ==")
show("基线")

print("\n== A. vol分母窗口统一25(与WLS一致) ==")
show("vol窗口25", vol_days=25)
show("vol窗口30", vol_days=30)

print("\n== B. R²趋势质量加权 ==")
show("score×(0.5+0.5R²)", score_r2=True)

print("\n== C. WLS双窗口混合 ==")
for w in (0.2, 0.3, 0.5):
    show("WLS25+60 w=%.1f" % w, wls60_mix=w)

print("\n== D. EWM波动率(半衰10日) ==")
show("vol EWM", vol_ewm=True)

print("\n== E. 牛熊双确认(年线+MOM20>0) ==")
show("牛熊双确认", bull_dual=True)

print("\n== F. 组合抽查 ==")
show("R²+vol25", score_r2=True, vol_days=25)
show("R²+WLS60(0.3)", score_r2=True, wls60_mix=0.3)
print("DONE")
