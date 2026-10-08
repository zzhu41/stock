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
import argparse
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


def append_v10(payload):
    """Read frozen research only; preserve all legacy curves and live accounts."""
    from web_v10 import export_versions
    exported = export_versions()
    payload["versions"].update(exported)
    payload["v10_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload.pop("v10_error", None)
    return payload


def build_v10_only():
    """Fast initial rollout: do not refetch/recalculate any old strategy path."""
    with open(CACHE, encoding="utf-8") as stream:
        payload = json.load(stream)
    append_v10(payload)
    atomic_write(CACHE, json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    print("V10-H及同口径v9.2已追加，旧版本曲线保持不变", flush=True)


def _read_closes(code):
    """date -> 原始收盘价, 来自 data/<code>.csv (date,open,close,volume 无表头)。"""
    closes = {}
    with open(os.path.join(BASE, "data", code + ".csv"), encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[0][:1].isdigit():
                closes[parts[0]] = float(parts[2])
    return closes


def graft_v12_forward(payload):
    """冻结研究段之后拼接独立前向账户(signals/shadow_v12_r2.json)净值。

    冻结段本身一字节不动, 仅向网页载荷追加延伸行:
      gap 段(研究末日→账户起算日之间的交易日)按持仓标的原始收盘价比值
      延伸(持有期无换仓费), 账户段以 events 净值(起算日复位 1.0)锚定乘法
      衔接; 基准(买入持有)按同一日历延伸。任一输入缺失保留冻结段原样。
    """
    ver = payload.get("versions", {}).get("v12-r2")
    if not ver or not ver.get("daily"):
        return
    md = ver.setdefault("metadata", {})
    try:
        with open(os.path.join(BASE, "signals", "shadow_v12_r2.json"), encoding="utf-8") as stream:
            state = json.load(stream)
        daily = ver["daily"]
        end_date, end_nav, hold_code = daily[-1][0], float(daily[-1][1]), daily[-1][2]
        by_date = {}  # 每日取最后一条事件, 只要研究末日之后的
        for event in state.get("events") or []:
            day = event.get("date")
            if day and day > end_date and event.get("nav"):
                by_date[day] = event
        if not by_date:
            return
        days = sorted(by_date)
        account_start = days[0]
        anchor = end_nav
        closes = _read_closes(hold_code) if hold_code else {}
        gap_days = []
        if end_date in closes:
            gap_days = [d for d in sorted(closes) if end_date < d < account_start]
            for day in gap_days:
                daily.append([day, anchor * closes[day] / closes[end_date], hold_code])
            if gap_days:
                anchor = daily[-1][1]
        for day in days:  # 账户段锚定衔接; 账户内真实换仓追加到换仓记录
            event = by_date[day]
            grafted = anchor * float(event["nav"])
            daily.append([day, grafted, event.get("to") or hold_code])
            if event.get("from") and event["from"] != event.get("to"):
                ver.setdefault("trades", []).append([day, event["from"], event.get("to"), grafted])
        last_day = days[-1]
        for code, bench in (ver.get("benchmarks") or {}).items():  # 基准同历延伸
            bdaily = bench.get("daily") or []
            b_closes = _read_closes(code)
            if not bdaily or bdaily[-1][0] not in b_closes:
                continue
            b_end_date, b_end_nav = bdaily[-1][0], float(bdaily[-1][1])
            for day in sorted(b_closes):
                if b_end_date < day <= last_day:
                    bdaily.append([day, b_end_nav * b_closes[day] / b_closes[b_end_date]])
        md["warnings"] = [w for w in md.get("warnings", []) if "不含前向账户净值" not in w]
        md["warnings"].append(
            "%s 之后为独立前向账户实时延伸(%s 起算, 14:50 观察价口径), 与冻结研究段口径不同。"
            % (end_date, account_start))
        md["forward"] = {"from": account_start, "through": last_day,
                         "source": "shadow_v12_r2", "gap_days": gap_days}
        md["data_as_of"] = last_day
    except Exception as exc:
        md["forward_error"] = type(exc).__name__
        print("V12前向嫁接失败: %s" % type(exc).__name__, flush=True)


def append_v12(payload):
    """Append the user's frozen R2 choice without altering research selection."""
    from web_v12 import export_versions
    payload["versions"].update(export_versions())
    payload["default_version"] = "v12-r2"
    payload["v12_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload.pop("v12_error", None)
    graft_v12_forward(payload)
    return payload


def build_v12_only():
    with open(CACHE, encoding="utf-8") as stream:
        payload = json.load(stream)
    append_v12(payload)
    atomic_write(CACHE, json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    print("V12-R2冻结曲线已追加；旧曲线与独立前向账户保持不变", flush=True)


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

    # 影子版本曲线(v9.1-0906 / v9.2): v10/lab 钩子引擎, 独立子进程跑(monkey-patch 隔离);
    # 失败仅跳过影子曲线, 主版本链照常
    try:
        import subprocess
        out = subprocess.check_output(
            [sys.executable, os.path.join(BASE, "v10", "web_shadows.py")],
            cwd=BASE, timeout=900)
        for k, v in json.loads(out.decode("utf-8")).items():
            versions[k] = v
            print("%-9s 年化 %+5.1f%% 回撤 %6.1f%% 夏普 %4.2f 换手 %3d (影子)"
                  % (k, v["metrics"]["ann"], v["metrics"]["max_dd"],
                     v["metrics"]["sharpe"], v["metrics"]["switches"]), flush=True)
    except Exception as e:
        print("影子曲线构建失败(跳过, 不影响主版本链): %r" % e, flush=True)

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
    try:
        append_v10(payload)
    except Exception as exc:
        # Keep a previously verified frozen snapshot visibly dated if a rebuild
        # fails; never replace it with an unverified variant or live NAV.
        payload["v10_error"] = type(exc).__name__
        try:
            with open(CACHE, encoding="utf-8") as stream:
                previous = json.load(stream)
            for vid in ("v10-h", "v9.2-tr"):
                if vid in previous.get("versions", {}):
                    payload["versions"][vid] = previous["versions"][vid]
                    warnings = payload["versions"][vid].setdefault("metadata", {}).setdefault("warnings", [])
                    note = "本次研究快照核验未完成，显示上一次已保存的历史数据"
                    if note not in warnings:
                        warnings.append(note)
            payload["v10_updated_at"] = previous.get("v10_updated_at")
        except (OSError, ValueError):
            pass
        print("V10研究曲线更新失败: %s" % type(exc).__name__, flush=True)
    try:
        append_v12(payload)
    except Exception as exc:
        payload["v12_error"] = type(exc).__name__
        try:
            with open(CACHE, encoding="utf-8") as stream:
                previous = json.load(stream)
            if "v12-r2" in previous.get("versions", {}):
                payload["versions"]["v12-r2"] = previous["versions"]["v12-r2"]
                warnings = payload["versions"]["v12-r2"].setdefault("metadata", {}).setdefault("warnings", [])
                note = "本次研究快照核验未完成，显示上一次已保存的历史数据"
                if note not in warnings: warnings.append(note)
                payload["default_version"] = "v12-r2"
            payload["v12_updated_at"] = previous.get("v12_updated_at")
        except (OSError, ValueError):
            pass
        print("V12研究曲线更新失败: %s" % type(exc).__name__, flush=True)
    atomic_write(CACHE, json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    print("缓存写入 %s (%.1f KB), 总耗时 %ds"
          % (CACHE, os.path.getsize(CACHE) / 1024, time.time() - t0), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    fast = parser.add_mutually_exclusive_group()
    fast.add_argument("--v10-only", action="store_true", help="Append frozen V10 curves without rebuilding legacy versions")
    fast.add_argument("--v12-only", action="store_true", help="Append frozen V12-R2 path without rebuilding legacy versions")
    args = parser.parse_args()
    import fcntl
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_WRONLY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("已有构建在进行, 退出")
        sys.exit(0)
    try:
        build_v12_only() if args.v12_only else build_v10_only() if args.v10_only else build()
    finally:
        os.close(lock_fd)
