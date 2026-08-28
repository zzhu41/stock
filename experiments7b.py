# -*- coding: utf-8 -*-
"""第七轮补跑: 剩余10个变体(均线族/乖离/布林/CCI/打分混合/组合)。基线39.2/6309。"""
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


print("== D. 均线/乖离/布林/CCI ==")
show("close>MA20进场   ", above_ma20=True)
show("MA20上行进场     ", ma20_rising=True)
show("BIAS20>8%禁追高  ", bias_nochase=0.08)
show("BIAS20>12%禁追高 ", bias_nochase=0.12)
show("%B>1.0禁追高     ", pctb_nochase=1.0)
show("CCI>100禁追高    ", cci_nochase=100.0)
print("\n== E. 打分端混合 ==")
show("score混合MACD柱  ", score_tech_mix=True)
print("\n== F. 离场端组合 ==")
show("MACD死叉+KDJ死叉 ", macd_exit=True, kdj_dead_exit=True)
print("\n== G. 进场端组合 ==")
show("金叉+RSI>50      ", macd_filter=True, rsi_gt50=True)
show("金叉+MA20上      ", macd_filter=True, above_ma20=True)
print("DONE")
