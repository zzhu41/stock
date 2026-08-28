# -*- coding: utf-8 -*-
"""v8方向1: QQQ混合动量 (513100信号序列混入QQQ人民币口径, 收益仍用ETF真实价)。"""
import json
import backtest
import ext_data
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
etf_rows = histories["513100"]

results = []
for w in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0):
    mixed = ext_data.build_mixed_close(etf_rows, w)
    sig = dict(histories)
    sig["513100"] = [(d, mixed[d], mixed[d]) for d, _, _ in etf_rows if d in mixed]
    r = backtest.backtest(histories, calendar, start="2014-01-01", signal_histories=sig)
    yrs = {y: x for y, x in backtest.yearly(r["daily"])}
    results.append({
        "w": w, "ann": round(r["ann"] * 100, 2), "dd": round(r["max_dd"] * 100, 2),
        "sharpe": round(r["sharpe"], 3), "nav_pct": round((r["nav"] - 1) * 100, 1),
        "switches": r["switches"],
        "y18": round(yrs.get("2018", 0) * 100, 1), "y20": round(yrs.get("2020", 0) * 100, 1),
        "y22": round(yrs.get("2022", 0) * 100, 1), "y25": round(yrs.get("2025", 0) * 100, 1),
        "y26": round(yrs.get("2026", 0) * 100, 1),
    })
    print("w=%.1f: 年化 %+.2f%% 回撤 %.2f%% 夏普 %.2f 终值 %+.0f%%" %
          (w, results[-1]["ann"], results[-1]["dd"], results[-1]["sharpe"], results[-1]["nav_pct"]), flush=True)
print("JSON:" + json.dumps(results))
