# -*- coding: utf-8 -*-
"""第二轮文献变体矩阵: FIP / MAX / Sortino / vol目标仓位。基线=v6(panic4%)。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

variants = [
    ("基线 v6 (panic4%)      ", {}),
    # Frog-in-the-Pan: 上涨天数占比过滤
    ("FIP pos_frac>=0.50     ", {"pos_frac_min": 0.50}),
    ("FIP pos_frac>=0.55     ", {"pos_frac_min": 0.55}),
    ("FIP pos_frac>=0.60     ", {"pos_frac_min": 0.60}),
    # MAX 彩票效应: 近20日最大单日涨幅上限
    ("MAX cap 6%             ", {"max_cap": 0.06}),
    ("MAX cap 7%             ", {"max_cap": 0.07}),
    ("MAX cap 8%             ", {"max_cap": 0.08}),
    # Sortino 排名
    ("Sortino 下行波动排名   ", {"use_downside_vol": True}),
    # 波动率目标仓位 (Moreira & Muir 2017)
    ("vol目标 年化15%        ", {"vol_target": 0.15}),
    ("vol目标 年化20%        ", {"vol_target": 0.20}),
    ("vol目标 年化25%        ", {"vol_target": 0.25}),
]

print("区间 2014 ~ %s | 变体: 年化 | 回撤 | 夏普 | 卡玛 | 年换手 || 逐年" % calendar[-1])
for name, kw in variants:
    r = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
    yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in backtest.yearly(r["daily"]))
    print("%s: %+5.1f%% | %6.1f%% | %4.2f | %4.2f | %4.1f || %s"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"],
             r["calmar"], r["sw_per_year"], yrs))
