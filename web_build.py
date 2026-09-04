# -*- coding: utf-8 -*-
"""网页看板数据构建: 对策略版本链(v7/v8/v8.1/v9/v9.1)各跑一次全量回测,
逐日净值/持仓 + 换仓记录 + 基准(沪深300/黄金买入持有) 原子写 signals/web_cache.json。

版本链(与 README 口径一致, 关闭组件即回退旧版):
  v7   = MOM20/VOL20 排名(score_wls=False), 无熊市门槛, 无深跌抄底, 统一2%缓冲
  v8   = v7 + WLS25 斜率排名
  v8.1 = v8 + 熊市进场门槛 MOM20>7%
  v9   = v8.1 + 深跌恐慌抄底
  v9.1 = v9 + 分池缓冲(A股2%/跨境3%/黄金3%)
  注: backtest(pool_buffer={}) 空dict在 decide() 中为 falsy -> 回退统一 BUFFER=0.02;
      crash_mom5=0.0 -> `crash_mom5 < 0` 不成立 -> 抄底关闭。

由 cron(交易日15:25, 信号与数据更新之后)或 web_app.py(启动发现缓存过期/手动刷新)调用。
全量约 3 分钟(单版本约 30s)。用法: python3.8 web_build.py
"""
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)  # 允许从其他 cwd 运行

from market_data import UNIVERSE, fetch_history  # noqa: E402
from backtest import backtest, buy_and_hold  # noqa: E402
from live_utils import atomic_write  # noqa: E402

CACHE = os.path.join(BASE, "signals", "web_cache.json")
LOCK = os.path.join(BASE, "signals", "web_build.lock")
START = "2014-01-01"  # README 回测口径起点

VERSIONS = [
    ("v7",   "v7  MOM20/VOL20 排名", {"score_wls": False, "bear_enter_mom": 0.0,
                                      "crash_mom5": 0.0, "pool_buffer": {}}),
    ("v8",   "v8  WLS25 斜率排名",   {"bear_enter_mom": 0.0, "crash_mom5": 0.0,
                                      "pool_buffer": {}}),
    ("v8.1", "v8.1  +熊市门槛7%",    {"crash_mom5": 0.0, "pool_buffer": {}}),
    ("v9",   "v9  +深跌恐慌抄底",    {"pool_buffer": {}}),
    ("v9.1", "v9.1 +分池缓冲(实盘版)", {}),
]

BENCHMARKS = [("510300", "沪深300ETF 买入持有"), ("518880", "黄金ETF 买入持有")]


def metrics(res):
    return {
        "nav": round(res["nav"], 4),
        "ann": round(res["ann"] * 100, 2),
        "max_dd": round(res["max_dd"] * 100, 2),
        "sharpe": round(res["sharpe"], 2),
        "switches": res["switches"],
    }


def build():
    t0 = time.time()
    codes = list(UNIVERSE)
    histories = {c: fetch_history(c) for c in codes}
    calendar = [r[0] for r in histories["510300"]]
    names = {c: UNIVERSE[c][0] for c in codes}

    versions = {}
    for vid, label, kw in VERSIONS:
        t1 = time.time()
        res = backtest(histories, calendar, start=START, **kw)
        trades = [[d, frm, to, round(nav, 4)] for d, frm, to, nav in res["trades"]]
        if res["daily"]:  # 引擎不计首日建仓为换手, 看板补一条建仓记录
            d0, nav0, h0 = res["daily"][0]
            trades.insert(0, [d0, None, h0, round(nav0, 4)])
        versions[vid] = {
            "label": label,
            "daily": [[d, round(nav, 4), h] for d, nav, h in res["daily"]],
            "trades": trades,
            "crash_buys": [[d, c] for d, c in res["crash_buys"]],
            "metrics": metrics(res),
        }
        print("%-5s 年化 %+5.1f%% 回撤 %6.1f%% 夏普 %4.2f 换手 %3d | %ds"
              % (vid, res["ann"] * 100, res["max_dd"] * 100, res["sharpe"],
                 res["switches"], time.time() - t1), flush=True)

    benchmarks = {}
    for code, label in BENCHMARKS:
        nav, ann, dd = buy_and_hold(histories, calendar, code, start=START)
        close_of = {r[0]: r[2] for r in histories[code]}
        days = [d for d in calendar if START <= d and d in close_of]
        base = close_of[days[0]]
        benchmarks[code] = {
            "label": label,
            "daily": [[d, round(close_of[d] / base, 4)] for d in days],
            "metrics": {"ann": round(ann * 100, 2), "max_dd": round(dd * 100, 2)},
        }

    payload = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start": START,
        "data_range": [calendar[0], calendar[-1]],
        "names": names,
        "versions": versions,
        "benchmarks": benchmarks,
    }
    atomic_write(CACHE, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print("缓存写入 %s (%.1f KB), 总耗时 %ds"
          % (CACHE, os.path.getsize(CACHE) / 1024, time.time() - t0), flush=True)


if __name__ == "__main__":
    import fcntl
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_WRONLY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("已有构建在进行, 退出")
        sys.exit(0)
    try:
        build()
    finally:
        os.close(lock_fd)
