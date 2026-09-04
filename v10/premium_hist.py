# -*- coding: utf-8 -*-
"""QDII 历史溢价序列构建(v10 溢价修正实验的数据准备)。

东财 f10/lsjz 分页拉全历史单位净值 -> v10/data_extra/nav_{code}.csv
溢价口径(严格无前视): premium_t = 收盘价_t / 单位净值_{t-1} - 1
  (T日14:50信号时刻只能看到 T-1 净值, 与 premium.py 实时口径一致)

存 v10/data_extra/ 不走实盘 data/ 缓存(第三轮的目录约定)。
用法: python3.8 v10/premium_hist.py
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from market_data import fetch_history  # noqa: E402

OUT_DIR = os.path.join(BASE, "v10", "data_extra")
QDII = ["513100", "513120"]
# 份额分拆日(东财 fhsp 核实): 分拆日前"前复权价 vs 原始净值"错位, 溢价序列不可用;
# 513100 仅 2022-01-13 一次 1拆5, 之前段溢价恒-80%(方向安全, 永不误触发)——仍显式置空。
# 拆分不影响单位净值收益率连续性, P3 净值伪价全程可用。513120 无分红拆分。
SPLIT_DATE = {"513100": "2022-01-13"}
_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://fund.eastmoney.com/"}


def fetch_nav_history(code, page_size=100):
    """分页拉全历史单位净值: {日期: 净值}。
    注意 lsjz 接口: TotalCount/PageSize 在响应顶层, 且 PageSize 可能被强制为20。"""
    out = {}
    page = 1
    while True:
        url = ("https://api.fund.eastmoney.com/f10/lsjz?fundCode=%s&pageIndex=%d&pageSize=%d"
               % (code, page, page_size))
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode("utf-8"))
        rows = (d.get("Data") or {}).get("LSJZList") or []
        if not rows:
            break
        for row in rows:
            if row.get("DWJZ"):
                out[row["FSRQ"]] = float(row["DWJZ"])
        total = d.get("TotalCount") or 0          # 顶层字段
        page_size = d.get("PageSize") or page_size  # 服务端实际页大小(可能被强制)
        if page * page_size >= total:
            break
        page += 1
        time.sleep(0.2)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for code in QDII:
        nav_cache = os.path.join(OUT_DIR, "nav_%s.csv" % code)
        if os.path.exists(nav_cache):
            navs = {}
            with open(nav_cache) as f:
                next(f)
                for ln in f:
                    d, v = ln.strip().split(",")
                    navs[d] = float(v)
        else:
            navs = fetch_nav_history(code)
            with open(nav_cache, "w") as f:
                f.write("date,nav\n")
                for d in sorted(navs):
                    f.write("%s,%s\n" % (d, navs[d]))
        print("%s 净值 %d 条: %s ~ %s"
              % (code, len(navs), min(navs), max(navs)))
        # 市价序列
        hist = {r[0]: r[2] for r in fetch_history(code)}
        nav_dates = sorted(navs)
        split = SPLIT_DATE.get(code)
        rows = []
        for d in sorted(hist):
            # T-1 净值: 严格小于 d 的最近净值
            prev = [nd for nd in nav_dates if nd < d]
            if not prev:
                continue
            nav = navs[prev[-1]]
            prem = hist[d] / nav - 1.0
            if split and d <= split:
                prem = None  # 分拆前口径错位, 置空不参与
            rows.append((d, nav, round(hist[d], 4),
                         round(prem * 100, 3) if prem is not None else ""))
        path = os.path.join(OUT_DIR, "premium_%s.csv" % code)
        with open(path, "w") as f:
            f.write("date,nav,close,premium_pct\n")
            for r in rows:
                f.write("%s,%s,%s,%s\n" % r)
        valid = [r for r in rows if r[3] != ""]
        over2 = [r for r in valid if r[3] > 2]
        over5 = [r for r in valid if r[3] > 5]
        print("  写入 %s: %d 条(有效溢价 %d) | 溢价>2%%: %d 天 | >5%%: %d 天"
              % (path, len(rows), len(valid), len(over2), len(over5)))
        # 溢价>2% 的年份分布(评估是否单事件依赖)
        by_year = {}
        for r in over2:
            by_year[r[0][:4]] = by_year.get(r[0][:4], 0) + 1
        print("  >2%% 年份分布: %s"
              % " ".join("%s:%d" % kv for kv in sorted(by_year.items())))


if __name__ == "__main__":
    main()
