# -*- coding: utf-8 -*-
"""行情数据模块：腾讯财经公开接口，纯标准库，本地 CSV 缓存 + 增量更新。

日K: web.ifzq.gtimg.cn/appstock/app/fqkline/get  前复权日线, 单页最多640条, 分页拉全历史
报价: qt.gtimg.cn/q=...                          盘中实时价 / 盘后收盘价
"""
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 标的池: code -> (名称, 腾讯前缀, 角色)
# 角色: stock=A股竞赛池(受牛熊开关约束), global=跨境池(纯动量, 不受开关约束),
#       gold=黄金备胎, cash=货币停靠
UNIVERSE = {
    "159915": ("创业板ETF",    "sz", "stock"),
    "588080": ("科创50ETF",    "sh", "stock"),
    "510300": ("沪深300ETF",   "sh", "stock"),
    "510500": ("中证500ETF",   "sh", "stock"),
    "563300": ("中证2000ETF",  "sh", "stock"),
    "512400": ("有色金属ETF",  "sh", "stock"),
    "512890": ("红利低波ETF",  "sh", "stock"),
    "513100": ("纳指ETF",      "sh", "global"),
    "513120": ("港股创新药ETF","sh", "global"),
    "518880": ("黄金ETF",      "sh", "gold"),
    "511880": ("货币ETF",      "sh", "cash"),
}
STOCK_POOL = [c for c, v in UNIVERSE.items() if v[2] == "stock"]
GLOBAL_POOL = [c for c, v in UNIVERSE.items() if v[2] == "global"]
GOLD = "518880"
CASH = "511880"
FULL_START = "2010-01-01"  # 全量拉取的起点

_UA = {"User-Agent": "Mozilla/5.0"}


def _get(url, timeout=20, retries=4, binary=False):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            return raw if binary else raw.decode("utf-8")
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def _kline_page(symbol, start, end):
    """拉 [start, end] 区间内最近 640 根日K，返回 [[date, open, close, volume], ...] 升序。"""
    url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=%s,day,%s,%s,640,qfq"
           % (symbol, start, end))
    d = json.loads(_get(url))
    node = (d.get("data") or {}).get(symbol) or {}
    rows = node.get("qfqday") or node.get("day") or []
    out = []
    for r in rows:
        vol = float(r[5]) if len(r) > 5 and r[5] not in ("", None) else 0.0
        out.append([r[0], float(r[1]), float(r[2]), vol])
    return out


def fetch_history(code, full=False):
    """前复权日K [(date, open, close, volume), ...] 升序。默认缓存+增量，full=True 强制全量。
    旧版3列缓存(无volume)自动全量重拉。"""
    symbol = UNIVERSE[code][1] + code
    cache = os.path.join(DATA_DIR, "%s.csv" % code)
    rows = []
    if not full and os.path.exists(cache):
        with open(cache, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f) if r]
    if rows and len(rows[0]) < 4:
        rows = []  # 旧格式, 强制全量重拉

    if rows:
        # 增量: 从缓存最后一天重拉(覆盖盘中未完结的当日K);
        # 只保留 >= start 的新行, 防接口数据延迟时缓存回退丢最近一天
        start = rows[-1][0]
        new = _kline_page(symbol, start, _today())
        new = [r for r in new if r[0] >= start]
        rows = [r for r in rows if r[0] < new[0][0]] + new if new else rows
    else:
        # 全量: 向前分页
        end, start_limit = _today(), FULL_START
        while True:
            page = _kline_page(symbol, start_limit, end)
            if not page:
                break
            rows = page + rows
            if len(page) < 640 or page[0][0] <= start_limit:
                break
            end = (datetime.strptime(page[0][0], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            time.sleep(1.2)  # 礼貌间隔, 防频控

    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)
    with open(cache, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in rows]


def fetch_realtime(codes=None):
    """盘中实时价(盘后=收盘价)。返回 {code: (name, price)}。"""
    codes = codes or [c for c in UNIVERSE if c != CASH]
    q = ",".join(UNIVERSE[c][1] + c for c in codes)
    raw = _get("https://qt.gtimg.cn/q=%s" % q, timeout=10, binary=True).decode("gbk")
    out = {}
    for line in raw.strip().split(";"):
        if "=" not in line:
            continue
        f = line.split("=", 1)[1].strip('"').split("~")
        if len(f) > 3 and f[3]:
            out[f[2]] = (f[1], float(f[3]))
    return out


def _today():
    return datetime.now().strftime("%Y-%m-%d")


if __name__ == "__main__":
    for code in UNIVERSE:
        rows = fetch_history(code)
        print("%s %s: %d 行, %s ~ %s, 最新收盘 %.3f"
              % (code, UNIVERSE[code][0], len(rows), rows[0][0], rows[-1][0], rows[-1][2]))
        time.sleep(1.0)
    print("实时:", fetch_realtime())
