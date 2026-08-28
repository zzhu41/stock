# -*- coding: utf-8 -*-
"""第七轮: 技术指标穷举矩阵 (v7基线=38.9%/-28.1%/6129%, 数据截止2026-08-18)。
覆盖: 单指标多形态x阈值 + 离场端 + 打分端 + 两两组合。"""
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
show("v7 基线             ")

print("\n== A. MACD 形态 ==")
show("MACD金叉进场        ", macd_filter=True)
show("MACD DIF>0进场      ", macd_dif_pos=True)
show("MACD死叉离场        ", macd_exit=True)

print("\n== B. KDJ 形态 ==")
show("KDJ金叉(K>D)进场    ", kdj_golden=True)
show("KDJ死叉离场         ", kdj_dead_exit=True)
show("KDJ J>90禁追高      ", kdj_nochase=True, kdj_nc_thr=90)
show("KDJ J>110禁追高     ", kdj_nochase=True, kdj_nc_thr=110)

print("\n== C. RSI 形态 ==")
show("RSI>50进场          ", rsi_gt50=True)
show("RSI<50离场          ", rsi_lt50_exit=True)
show("RSI>70禁追高        ", rsi_nochase=True, rsi_nc_thr=70)
show("RSI>85禁追高        ", rsi_nochase=True, rsi_nc_thr=85)

print("\n== D. 均线/乖离/布林/CCI ==")
show("close>MA20进场      ", above_ma20=True)
show("MA20上行进场        ", ma20_rising=True)
show("BIAS20>8%禁追高     ", bias_nochase=0.08)
show("BIAS20>12%禁追高    ", bias_nochase=0.12)
show("%B>1.0禁追高        ", pctb_nochase=1.0)
show("CCI>100禁追高       ", cci_nochase=100.0)

print("\n== E. 打分端混合 ==")
show("score混合MACD柱     ", score_tech_mix=True)

print("\n== F. 离场端组合(多重离场线) ==")
show("MACD死叉+KDJ死叉    ", macd_exit=True, kdj_dead_exit=True)

print("\n== G. 进场端组合(过滤叠加) ==")
show("金叉+RSI>50         ", macd_filter=True, rsi_gt50=True)
show("MA20上+RSI>50       ", above_ma20=True, rsi_gt50=True)
show("金叉+MA20上         ", macd_filter=True, above_ma20=True)
show("金叉+BIAS<12        ", macd_filter=True, bias_nochase=0.12)
print("DONE")
