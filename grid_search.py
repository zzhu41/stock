# -*- coding: utf-8 -*-
"""参数网格枚举: MOM_MAIN x BUFFER 二维 / OVERHEAT / MA_BULL / PANIC 细化。基线=v6。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
S = "2014-01-01"


def run(**kw):
    r = backtest.backtest(histories, calendar, start=S, **kw)
    return r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], r["calmar"], (r["nav"] - 1) * 100


print("== A. MOM_MAIN x BUFFER 网格 (行=MOM窗口, 列=缓冲, 值=年化|回撤|夏普) ==")
bufs = (0.01, 0.015, 0.02, 0.025, 0.03)
print("        " + "".join("BUF=%-5.3f          " % b for b in bufs))
for w in (10, 15, 20, 25, 30, 40):
    row = "MOM=%-3d " % w
    for b in bufs:
        ann, dd, sh, ca, nav = run(mom_main=w, buffer=b)
        row += "%+.1f/%+.1f/%.2f  " % (ann, dd, sh)
    print(row)

print("\n== B. OVERHEAT 扫描 (过热阈值) ==")
for oh in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 9.9):
    ann, dd, sh, ca, nav = run(overheat=oh)
    print("  OH=%4.2f: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 卡玛 %4.2f | 总收益 %+.0f%%"
          % (oh, ann, dd, sh, ca, nav))

print("\n== C. MA_BULL 扫描 (牛熊开关均线) ==")
for ma in (120, 150, 200, 250, 300):
    ann, dd, sh, ca, nav = run(ma_bull=ma)
    print("  MA=%-3d: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 卡玛 %4.2f | 总收益 %+.0f%%"
          % (ma, ann, dd, sh, ca, nav))

print("\n== D. PANIC 平台细化 ==")
for pd_ in (0.035, 0.0375, 0.04, 0.0425, 0.045):
    ann, dd, sh, ca, nav = run(panic_drop=pd_)
    print("  panic=%5.3f: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 总收益 %+.0f%%"
          % (pd_, ann, dd, sh, ca, nav))

print("\n== E. 最优组合抽查 (网格里挑出的头部组合) ==")
combos = [
    ("MOM15+BUF1.5%", {"mom_main": 15, "buffer": 0.015}),
    ("MOM15+BUF2%",   {"mom_main": 15, "buffer": 0.02}),
    ("MOM25+BUF2%",   {"mom_main": 25, "buffer": 0.02}),
    ("MOM15+OH35%",   {"mom_main": 15, "overheat": 0.35}),
    ("MOM15+OH45%",   {"mom_main": 15, "overheat": 0.45}),
]
for name, kw in combos:
    ann, dd, sh, ca, nav = run(**kw)
    print("  %-14s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 卡玛 %4.2f | 总收益 %+.0f%%"
          % (name, ann, dd, sh, ca, nav))
print("DONE")
