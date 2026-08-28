# -*- coding: utf-8 -*-
"""v8方向3: 同因子防切 (挑战者与持仓近60日相关系数>0.85时缓冲x2)。"""
import json
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
r0 = backtest.backtest(histories, calendar, start="2014-01-01")
r1 = backtest.backtest(histories, calendar, start="2014-01-01", rel_buffer_on=True)
results = []
for tag, r in (("off", r0), ("on", r1)):
    yrs = {y: x for y, x in backtest.yearly(r["daily"])}
    results.append({
        "mode": tag, "ann": round(r["ann"] * 100, 2), "dd": round(r["max_dd"] * 100, 2),
        "sharpe": round(r["sharpe"], 3), "nav_pct": round((r["nav"] - 1) * 100, 1),
        "switches": r["switches"],
        "y22": round(yrs.get("2022", 0) * 100, 1), "y26": round(yrs.get("2026", 0) * 100, 1),
    })
    print("%s: 年化 %+.2f%% 回撤 %.2f%% 夏普 %.2f 终值 %+.0f%% 换手 %d" %
          (tag, results[-1]["ann"], results[-1]["dd"], results[-1]["sharpe"],
           results[-1]["nav_pct"], r["switches"]), flush=True)
print("JSON:" + json.dumps(results))
