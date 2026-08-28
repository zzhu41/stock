# -*- coding: utf-8 -*-
"""缩池实验: 平台同款/精选小队/leave-one-out/leave-one-in。v7规则不变, 只改池子构成。
牛熊判定用全量histories(510300始终在), 只patch strategy.STOCK_POOL/GLOBAL_POOL。"""
import backtest
import strategy
import market_data
from market_data import UNIVERSE, GOLD, CASH, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
ORIG_STOCK = list(strategy.STOCK_POOL)
ORIG_GLOBAL = list(strategy.GLOBAL_POOL)


def set_pools(stock, glob):
    strategy.STOCK_POOL = stock
    strategy.GLOBAL_POOL = glob


def show(name, stock, glob, start="2014-01-01"):
    set_pools(stock, glob)
    r = backtest.backtest(histories, calendar, start=start)
    set_pools(ORIG_STOCK, ORIG_GLOBAL)
    yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in backtest.yearly(r["daily"]))
    print("%s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%% | 换手 %d || %s"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100,
             r["switches"], yrs), flush=True)


A7 = ["159915", "588080", "510300", "510500", "563300", "512400", "512890"]
G2 = ["513100", "513120"]

print("== 0. 基准: 现池11只 ==")
show("现池(A7+跨2)     ", A7, G2)

print("\n== 1. 精选小队 ==")
show("平台款(创业+纳指) ", ["159915"], ["513100"])
show("平台款+有色      ", ["159915", "512400"], ["513100"])
show("科技小队         ", ["159915", "588080"], ["513100"])
show("宽基核心         ", ["510300", "510500", "159915"], ["513100"])
show("近年强者(后视!)  ", ["512400", "563300", "159915"], ["513100", "513120"])
show("无跨境纯A股      ", A7, [])

print("\n== 2. leave-one-out (逐个剔除, 谁是负贡献) ==")
for c in A7 + G2:
    show("剔 %s%s " % (c, UNIVERSE[c][0][:4]), [x for x in A7 if x != c], [x for x in G2 if x != c])

print("\n== 3. leave-one-in (A股只留一只+跨境2) ==")
for c in A7:
    show("仅 %s%s " % (c, UNIVERSE[c][0][:4]), [c], G2)
print("DONE")
