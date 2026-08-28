# -*- coding: utf-8 -*-
"""开放A股+恐慌抄底组合检验 (v8.1基线, 2014/2016双口径)。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]


def show(name, **kw):
    r14 = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
    r16 = backtest.backtest(histories, calendar, start="2016-01-01", **kw)
    y14 = dict(backtest.yearly(r14["daily"]))
    print("%-24s: 14起[%+5.1f%% %6.1f%% %+.0f%%] 16起[%+5.1f%% %6.1f%% %+.0f%%] || 15:%+.0f 18:%+.0f 20:%+.0f 22:%+.0f 24:%+.0f 26:%+.0f"
          % (name, r14["ann"] * 100, r14["max_dd"] * 100, (r14["nav"] - 1) * 100,
             r16["ann"] * 100, r16["max_dd"] * 100, (r16["nav"] - 1) * 100,
             y14.get("2015", 0) * 100, y14.get("2018", 0) * 100, y14.get("2020", 0) * 100,
             y14.get("2022", 0) * 100, y14.get("2024", 0) * 100, y14.get("2026", 0) * 100), flush=True)


print("== 基线与单件 ==")
show("v8.1基线(禁入)          ")
show("开放A股(无抄底)         ", bear_open_stock=True)
show("禁入+抄底8%锁5         ", crash_mom5=-0.08, crash_lock=5)

print("\n== 开放+抄底组合 ==")
show("开放+抄底6%锁5         ", bear_open_stock=True, crash_mom5=-0.06, crash_lock=5)
show("开放+抄底8%锁5         ", bear_open_stock=True, crash_mom5=-0.08, crash_lock=5)
show("开放+抄底10%锁5        ", bear_open_stock=True, crash_mom5=-0.10, crash_lock=5)
show("开放+抄底8%锁7         ", bear_open_stock=True, crash_mom5=-0.08, crash_lock=7)
show("开放+抄底8%锁3         ", bear_open_stock=True, crash_mom5=-0.08, crash_lock=3)
show("开放+抄底8%锁5仅A股    ", bear_open_stock=True, crash_mom5=-0.08, crash_lock=5, crash_stock_only=True)

print("\n== 组合+熊市门槛强化 ==")
show("开放+抄底8%锁5+熊门15% ", bear_open_stock=True, crash_mom5=-0.08, crash_lock=5, bear_enter_mom=0.15)
print("DONE")
