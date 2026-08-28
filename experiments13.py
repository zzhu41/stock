# -*- coding: utf-8 -*-
"""v10 候选矩阵 (v9基线: 14起+49.2/+15367, 16起+42.0/+3997)。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]


def show(name, **kw):
    r14 = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
    r16 = backtest.backtest(histories, calendar, start="2016-01-01", **kw)
    y14 = dict(backtest.yearly(r14["daily"]))
    print("%-24s: 14起[%+5.1f%% %6.1f%% %+.0f%%] 16起[%+5.1f%% %6.1f%% %+.0f%%] || 15:%+.0f 16:%+.0f 22:%+.0f 24:%+.0f 26:%+.0f"
          % (name, r14["ann"] * 100, r14["max_dd"] * 100, (r14["nav"] - 1) * 100,
             r16["ann"] * 100, r16["max_dd"] * 100, (r16["nav"] - 1) * 100,
             y14.get("2015", 0) * 100, y14.get("2016", 0) * 100, y14.get("2022", 0) * 100,
             y14.get("2024", 0) * 100, y14.get("2026", 0) * 100), flush=True)


print("== 基线 ==")
show("v9基线                ")

print("\n== A. 锁仓期止损补丁(抄底成本再跌X%提前砍) ==")
show("锁仓止损-5%           ", crash_stop=0.05)
show("锁仓止损-8%           ", crash_stop=0.08)

print("\n== B. 抄底对象优选 ==")
show("选跌最狠的            ", crash_pick="lowest")

print("\n== C. t统计量排名 ==")
show("score=t值             ", score_tstat=True)

print("\n== D. 抄底仓位 ==")
show("抄底半仓              ", crash_alloc=0.5)
show("抄底3/4仓             ", crash_alloc=0.75)

print("\n== E. 组合 ==")
show("锁仓止损5%%+抄底半仓  ", crash_stop=0.05, crash_alloc=0.5)
print("DONE")
