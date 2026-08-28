# -*- coding: utf-8 -*-
"""v8方向4: vol缩放panic (阈值=clip(2*vol20,3%,6%) 替代固定4%)。"""
import json
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
results = []
for on in (False, True):
    r = backtest.backtest(histories, calendar, start="2014-01-01", vol_panic_on=on)
    yrs = {y: x for y, x in backtest.yearly(r["daily"])}
    results.append({
        "vol_panic": on, "ann": round(r["ann"] * 100, 2), "dd": round(r["max_dd"] * 100, 2),
        "sharpe": round(r["sharpe"], 3), "nav_pct": round((r["nav"] - 1) * 100, 1),
        "switches": r["switches"],
        "y20": round(yrs.get("2020", 0) * 100, 1), "y25": round(yrs.get("2025", 0) * 100, 1),
        "y26": round(yrs.get("2026", 0) * 100, 1),
    })
    print("vol_panic=%s: 年化 %+.2f%% 回撤 %.2f%% 夏普 %.2f 终值 %+.0f%% 换手 %d" %
          (on, results[-1]["ann"], results[-1]["dd"], results[-1]["sharpe"],
           results[-1]["nav_pct"], r["switches"]), flush=True)
print("JSON:" + json.dumps(results))
