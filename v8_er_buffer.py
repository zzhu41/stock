# -*- coding: utf-8 -*-
"""v8方向2: ER震荡市自适应缓冲 (持仓ER20<阈值时缓冲x2/x3)。"""
import json
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
results = []
for thr in (0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
    kw = {} if thr == 0 else {"er_buffer_on": True, "kaufman_thr": thr}
    r = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
    yrs = {y: x for y, x in backtest.yearly(r["daily"])}
    results.append({
        "thr": thr, "ann": round(r["ann"] * 100, 2), "dd": round(r["max_dd"] * 100, 2),
        "sharpe": round(r["sharpe"], 3), "nav_pct": round((r["nav"] - 1) * 100, 1),
        "switches": r["switches"],
        "y22": round(yrs.get("2022", 0) * 100, 1), "y26": round(yrs.get("2026", 0) * 100, 1),
    })
    print("ER阈值=%.2f: 年化 %+.2f%% 回撤 %.2f%% 夏普 %.2f 终值 %+.0f%% 换手 %d" %
          (thr, results[-1]["ann"], results[-1]["dd"], results[-1]["sharpe"],
           results[-1]["nav_pct"], r["switches"]), flush=True)
print("JSON:" + json.dumps(results))
