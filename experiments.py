# -*- coding: utf-8 -*-
"""改进变体矩阵回测：针对回撤三大成因的试验。默认参数必须复现基线。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

variants = [
    ("基线 v5            ", {}),
    ("A 跨境须站年线     ", {"global_ma_filter": True}),
    ("A2 跨境须MOM60>0   ", {"global_mom60_filter": True}),
    ("B1 单日-7%离场     ", {"panic_drop": 0.07}),
    ("B2 单日-5%离场     ", {"panic_drop": 0.05}),
    ("C 过热回看10日     ", {"overheat_lookback": 10}),
    ("C2 回看10+过热35%  ", {"overheat_lookback": 10, "overheat": 0.35}),
    ("A+B1              ", {"global_ma_filter": True, "panic_drop": 0.07}),
    ("A+C               ", {"global_ma_filter": True, "overheat_lookback": 10}),
    ("A+C2              ", {"global_ma_filter": True, "overheat_lookback": 10, "overheat": 0.35}),
    ("A+B1+C            ", {"global_ma_filter": True, "panic_drop": 0.07, "overheat_lookback": 10}),
    ("A+B2+C            ", {"global_ma_filter": True, "panic_drop": 0.05, "overheat_lookback": 10}),
]

print("区间 2018 ~ %s | 变体: 年化 | 回撤 | 夏普 | 卡玛 | 年换手 || 逐年" % calendar[-1])
for name, kw in variants:
    r = backtest.backtest(histories, calendar, start="2018-01-01", **kw)
    yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in backtest.yearly(r["daily"]))
    print("%s: %+5.1f%% | %6.1f%% | %4.2f | %4.2f | %4.1f || %s"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"],
             r["calmar"], r["sw_per_year"], yrs))
