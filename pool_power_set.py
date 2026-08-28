# -*- coding: utf-8 -*-
"""A股7只幂集穷举 v2 (127组合, 断点续跑): 每组合立即追加写 pool_power_result.jsonl。
中断后重跑自动跳过已完成组合。跨境513100+513120固定, 黄金保留。"""
import itertools
import json
import os
import backtest
import strategy
from market_data import UNIVERSE, fetch_history

OUT = "/root/stock/pool_power_result.jsonl"
histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
ORIG_S = list(strategy.STOCK_POOL)
ORIG_G = list(strategy.GLOBAL_POOL)
A7 = ["159915", "588080", "510300", "510500", "563300", "512400", "512890"]

done = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(tuple(json.loads(line)["combo"]))
print("已完成 %d 组合, 续跑剩余" % len(done))

combos = []
for k in range(1, 8):
    combos.extend(itertools.combinations(A7, k))
todo = [c for c in combos if tuple(sorted(c)) not in done]
print("共 %d 组合, 待跑 %d" % (len(combos), len(todo)))

f = open(OUT, "a", encoding="utf-8")
for i, sub in enumerate(todo):
    strategy.STOCK_POOL = list(sub)
    r = backtest.backtest(histories, calendar, start="2014-01-01")
    strategy.STOCK_POOL = list(ORIG_S)
    rec = {"combo": sorted(sub), "size": len(sub),
           "names": "+".join(UNIVERSE[c][0][:3] for c in sorted(sub)),
           "ann": round(r["ann"] * 100, 2), "dd": round(r["max_dd"] * 100, 2),
           "sharpe": round(r["sharpe"], 3), "nav": round((r["nav"] - 1) * 100, 0)}
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()
    if (i + 1) % 10 == 0:
        print("  进度 %d/%d" % (i + 1, len(todo)), flush=True)
f.close()
strategy.STOCK_POOL, strategy.GLOBAL_POOL = ORIG_S, ORIG_G
print("DONE")
