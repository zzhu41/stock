# -*- coding: utf-8 -*-
"""第六轮: 技术指标变体 (v7基线=39.4/-28.1/6411)。先复现基线确认默认行为不变。"""
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
             r["switches"], yrs), flush=True)


print("== 基线复现 ==")
show("v7 基线          ")
print("\n== MACD ==")
show("MACD金叉进场过滤 ", macd_filter=True)
show("MACD死叉离场     ", macd_exit=True)
show("MACD金叉+死叉    ", macd_filter=True, macd_exit=True)
print("\n== KDJ ==")
show("KDJ J>100禁追高  ", kdj_nochase=True)
show("KDJ J<10放宽进场 ", kdj_dip_buy=True)
show("KDJ 双向         ", kdj_nochase=True, kdj_dip_buy=True)
print("\n== RSI ==")
show("RSI>80禁追高     ", rsi_nochase=True)
print("\n== 均线 ==")
show("多头排列进场     ", ma_align=True)
print("\n== 组合抽查 ==")
show("KDJ追高+RSI追高  ", kdj_nochase=True, rsi_nochase=True)
show("MACD金叉+KDJ追高 ", macd_filter=True, kdj_nochase=True)
print("DONE")
