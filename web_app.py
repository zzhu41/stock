# -*- coding: utf-8 -*-
"""动量轮动策略跟踪看板 (端口 8081, 纯标准库, 无第三方依赖)。

只读 signals/web_cache.json(由 web_build.py 构建, cron 交易日15:25 刷新),
本进程从不 import strategy/backtest —— 回测在子进程跑, 避免 strategy 全局状态污染。

路由:
  GET  /               看板页面 (web/index.html)
  GET  /static/echarts.min.js
  GET  /api/meta       版本列表 / 数据区间 / 构建时间 / 构建中标记
  GET  /api/series?version=v9.1&start=2024-01-01&end=2026-09-04&compare=v7
                       区间净值(原始)+对比序列+区间指标+逐年+换仓记录+当前持仓
  POST /api/refresh    后台子进程重建缓存(幂等, 构建中重复调用直接返回)

运行: python3.8 web_app.py   (或 systemd: stock-web.service)
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "signals", "web_cache.json")
LOCK = os.path.join(BASE, "signals", "web_build.lock")
STATIC_DIR = os.path.join(BASE, "web")
PORT = int(os.environ.get("PORT", 8081))  # 测试可 PORT=18081 起临时实例
TRADING_DAYS = 244

_cache = {"mtime": 0, "data": None}
_cache_lock = threading.Lock()


# ---------- 缓存 ----------

def is_building():
    """lock 文件被 web_build.py 以 flock 持有 = 构建中。"""
    if not os.path.exists(LOCK):
        return False
    import fcntl
    fd = os.open(LOCK, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False  # 能抢到锁说明没人在构建
        except OSError:
            return True
    finally:
        os.close(fd)


def load_cache():
    """mtime 变化才重读; 缓存不存在返回 None。"""
    try:
        mtime = os.path.getmtime(CACHE)
    except OSError:
        return None
    with _cache_lock:
        if _cache["data"] is not None and _cache["mtime"] == mtime:
            return _cache["data"]
        with open(CACHE, encoding="utf-8") as f:
            data = json.load(f)
        _cache["data"] = data
        _cache["mtime"] = mtime
        return data


def maybe_trigger_build(force=False):
    """无缓存或强制刷新时后台子进程构建; 已在构建则跳过。返回是否新触发。"""
    if is_building():
        return False
    if not force and os.path.exists(CACHE):
        return False
    subprocess.Popen([sys.executable, os.path.join(BASE, "web_build.py")],
                     stdout=open(os.path.join(BASE, "signals", "web_build.log"), "ab"),
                     stderr=subprocess.STDOUT, cwd=BASE)
    return True


# ---------- 区间指标 ----------

def slice_daily(daily, start, end):
    return [row for row in daily if start <= row[0] <= end]


def range_metrics(daily):
    """daily: [[date, nav, ...]...] 已切片; 以首日为基期算区间指标。"""
    n = len(daily)
    if n < 2:
        return None
    navs = [row[1] for row in daily]
    base = navs[0]
    total = navs[-1] / base - 1.0
    years = n / TRADING_DAYS
    ann = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 and total > -1 else -1.0
    peak, mdd = base, 0.0
    for v in navs:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)
    rets = [navs[i] / navs[i - 1] - 1.0 for i in range(1, n)]
    mean = sum(rets) / len(rets)
    std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
    sharpe = mean / std * TRADING_DAYS ** 0.5 if std else 0.0
    return {
        "total_ret": round(total * 100, 2),
        "ann": round(ann * 100, 2),
        "max_dd": round(mdd * 100, 2),
        "sharpe": round(sharpe, 2),
        "calmar": round(ann / abs(mdd), 2) if mdd else None,
        "days": n,
    }


def yearly(daily):
    """逐年收益%: 每年末净值/上年末净值(首年用区间首日净值)。"""
    last = {}
    for row in daily:
        last[row[0][:4]] = row[1]
    out, base = [], daily[0][1]
    for y in sorted(last):
        out.append([y, round((last[y] / base - 1.0) * 100, 2)])
        base = last[y]
    return out


def trade_kind(trade, crash_set):
    d, frm, to = trade[0], trade[1], trade[2]
    if frm is None:
        return "建仓"
    if (d, to) in crash_set:
        return "恐慌抄底"
    if to == "511880":
        return "避险停靠"
    return "轮动"


def holding_info(ver):
    """当前持仓: 代码/名称/起始日/已持交易日数/持仓期收益(费后)。"""
    daily, trades = ver["daily"], ver["trades"]
    if not daily:
        return None
    last_d, last_nav, code = daily[-1]
    since = daily[0][0]
    entry_nav = daily[0][1]
    if trades:
        since = trades[-1][0]
        entry_nav = trades[-1][3]
    days = sum(1 for row in daily if row[0] >= since)
    return {
        "code": code, "since": since, "days": days,
        "pnl": round((last_nav / entry_nav - 1.0) * 100, 2),
        "date": last_d,
    }


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                self._send(200, f.read(), ctype)
        except OSError:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path == "/api/refresh":
            triggered = maybe_trigger_build(force=True)
            self._json({"ok": True, "triggered": triggered, "building": is_building()})
        else:
            self._json({"error": "unknown"}, 404)

    def do_GET(self):
        path, _, qs = self.path.partition("?")
        if path == "/" or path == "/index.html":
            return self._file(os.path.join(STATIC_DIR, "index.html"),
                              "text/html; charset=utf-8")
        if path == "/static/echarts.min.js":
            return self._file(os.path.join(STATIC_DIR, "echarts.min.js"),
                              "application/javascript; charset=utf-8")
        if path == "/api/meta":
            return self.api_meta()
        if path == "/api/series":
            return self.api_series(urllib.parse.parse_qs(qs))
        self._json({"error": "unknown"}, 404)

    def api_meta(self):
        data = load_cache()
        building = is_building()
        if data is None:
            maybe_trigger_build()
            return self._json({"ready": False, "building": True}, 503)
        self._json({
            "ready": True,
            "building": building,
            "built_at": data["built_at"],
            "data_range": data["data_range"],
            "versions": [{"id": k, "label": v["label"], "metrics": v["metrics"]}
                         for k, v in data["versions"].items()],
            "benchmarks": [{"id": k, "label": v["label"]}
                           for k, v in data["benchmarks"].items()],
        })

    def api_series(self, qs):
        data = load_cache()
        if data is None:
            maybe_trigger_build()
            return self._json({"ready": False, "building": True}, 503)
        vid = qs.get("version", ["v9.1"])[0]
        ver = data["versions"].get(vid)
        if ver is None:
            return self._json({"error": "unknown version"}, 400)
        start = qs.get("start", [data["start"]])[0] or data["start"]
        end = qs.get("end", ["9999"])[0] or "9999"
        daily = slice_daily(ver["daily"], start, end)
        if not daily:
            return self._json({"error": "empty range"}, 400)
        trades = [t for t in ver["trades"] if daily[0][0] <= t[0] <= daily[-1][0]]
        crash_set = {(d, c) for d, c in ver["crash_buys"]}
        names = data["names"]
        resp = {
            "version": vid,
            "label": ver["label"],
            "range": [daily[0][0], daily[-1][0]],
            "daily": daily,
            "trades": [t + [trade_kind(t, crash_set), names.get(t[1] or "", ""),
                            names.get(t[2], "")] for t in trades],
            "metrics": range_metrics(daily),
            "yearly": yearly(daily),
            "holding": holding_info(ver),
            "names": names,
        }
        cmp_id = qs.get("compare", [""])[0]
        cmp_daily, cmp_label = None, None
        if cmp_id in data["versions"]:
            cmp_label = data["versions"][cmp_id]["label"]
            cmp_daily = slice_daily(data["versions"][cmp_id]["daily"], start, end)
        elif cmp_id in data["benchmarks"]:
            cmp_label = data["benchmarks"][cmp_id]["label"]
            cmp_daily = slice_daily(data["benchmarks"][cmp_id]["daily"], start, end)
        if cmp_daily:
            resp["compare"] = {"id": cmp_id, "label": cmp_label,
                               "daily": [[r[0], r[1]] for r in cmp_daily],
                               "metrics": range_metrics(cmp_daily),
                               "yearly": yearly(cmp_daily)}
        self._json(resp)


if __name__ == "__main__":
    os.makedirs(STATIC_DIR, exist_ok=True)
    if load_cache() is None:
        print("无缓存, 触发首次构建(约3分钟)...")
        maybe_trigger_build()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("动量轮动看板: http://0.0.0.0:%d" % PORT)
    server.serve_forever()
