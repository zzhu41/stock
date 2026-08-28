# -*- coding: utf-8 -*-
"""v8.1 池子组合扫描: leave-one-out + 结构组合。histories全量(510300牛熊判定不受影响), 只patch竞赛池。"""
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
    print("%-22s: 14起[%+5.1f%% %6.1f%% %+.0f%%] 16起[%+5.1f%% %6.1f%% %+.0f%%] 换手%d"
          % (name, r14["ann"] * 100, r14["max_dd"] * 100, (r14["nav"] - 1) * 100,
             r16["ann"] * 100, r16["max_dd"] * 100, (r16["nav"] - 1) * 100,
             r14["switches"]), flush=True)


A7 = ["159915", "588080", "510300", "510500", "563300", "512400", "512890"]
G2 = ["513100", "513120"]

print("== 基线 ==")
show("现池(A7+跨2)         ", A7, G2)

print("\n== 1. leave-one-out ==")
for c in A7 + G2:
    show("剔%s %s" % (c, UNIVERSE[c][0][:5]), [x for x in A7 if x != c], [x for x in G2 if x != c])

print("\n== 2. 结构组合 ==")
show("高弹性(创业科创有色)", ["159915", "588080", "512400"], G2)
show("小盘进攻(2k有色创业)", ["563300", "512400", "159915"], G2)
show("宽基核心(300/500/创)", ["510300", "510500", "159915"], G2)
show("防御型(300+红利)    ", ["510300", "512890"], G2)
show("行业双雄(有色+红利)", ["512400", "512890"], G2)
show("无跨境纯A股         ", A7, [])
show("跨境仅纳指          ", A7, ["513100"])
show("跨境仅创新药        ", A7, ["513120"])
show("单A股: 仅创业       ", ["159915"], G2)
show("单A股: 仅有色       ", ["512400"], G2)
print("DONE")
