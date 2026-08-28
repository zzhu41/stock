# -*- coding: utf-8 -*-
"""v9.1 池子组合扫描(带基线保护): LOO + 窄池 + 精选 + 去弱留强。
只临时patch strategy.STOCK_POOL/GLOBAL_POOL, 跑完恢复, 不动默认参数。"""
import backtest
import strategy
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
ORIG_S = list(strategy.STOCK_POOL)
ORIG_G = list(strategy.GLOBAL_POOL)


def show(name, stock, glob):
    strategy.STOCK_POOL, strategy.GLOBAL_POOL = stock, glob
    r14 = backtest.backtest(histories, calendar, start="2014-01-01")
    r16 = backtest.backtest(histories, calendar, start="2016-01-01")
    strategy.STOCK_POOL, strategy.GLOBAL_POOL = ORIG_S, ORIG_G
    y14 = dict(backtest.yearly(r14["daily"]))
    print("%-20s: 14起[%+5.1f%% %6.1f%% %+.0f%%] 16起[%+5.1f%% %6.1f%% %+.0f%%] || 17:%+.0f 19:%+.0f 21:%+.0f 22:%+.0f 25:%+.0f"
          % (name, r14["ann"] * 100, r14["max_dd"] * 100, (r14["nav"] - 1) * 100,
             r16["ann"] * 100, r16["max_dd"] * 100, (r16["nav"] - 1) * 100,
             y14.get("2017", 0) * 100, y14.get("2019", 0) * 100, y14.get("2021", 0) * 100,
             y14.get("2022", 0) * 100, y14.get("2025", 0) * 100), flush=True)


A7 = ["159915", "588080", "510300", "510500", "563300", "512400", "512890"]
G2 = ["513100", "513120"]

print("== 基线 ==")
show("现池(A7+跨2)      ", A7, G2)

print("\n== 1. leave-one-out ==")
for c in A7 + G2:
    show("剔%s" % UNIVERSE[c][0][:5], [x for x in A7 if x != c], [x for x in G2 if x != c])

print("\n== 2. 窄池与精选 ==")
show("平台3只(纳指黄金创业)", ["159915"], ["513100"])
show("窄池+有色(4只)     ", ["159915", "512400"], ["513100"])
show("窄池+有色+创新药(5)", ["159915", "512400"], ["513100", "513120"])
show("进攻全明星6只      ", ["159915", "588080", "512400"], G2)
show("防守精选6只        ", ["510300", "510500", "512890"], G2)
show("去三宽基留4A股     ", ["159915", "588080", "512400", "512890"], G2)
show("仅跨境+黄金(纯防御)", [], G2)
print("DONE")
