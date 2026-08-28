# -*- coding: utf-8 -*-
"""v10 新增标的数据拉取: 红利类/国债 ETF 全量前复权日K → v10/data_extra/*.csv。
只写本目录, 不碰实盘 data/ 缓存; 复用 market_data._kline_page(分页640根)。"""
import csv
import os
import sys
import time
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from market_data import _kline_page, _today

EXTRA = {
    "510880": ("红利ETF(上证红利)", "sh"),
    "515080": ("中证红利ETF", "sh"),
    "515100": ("红利低波100ETF", "sh"),
    "511010": ("国债ETF", "sh"),
    "513030": ("德国ETF", "sh"),
    "513520": ("日经ETF", "sh"),
    "159985": ("豆粕ETF", "sz"),
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_extra")
START_LIMIT = "2010-01-01"


def fetch_full(code, prefix):
    symbol = prefix + code
    rows, end = [], _today()
    while True:
        page = _kline_page(symbol, START_LIMIT, end)
        if not page:
            break
        rows = page + rows
        if len(page) < 640 or page[0][0] <= START_LIMIT:
            break
        end = (datetime.strptime(page[0][0], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(1.2)
    return rows


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    for code, (name, prefix) in EXTRA.items():
        rows = fetch_full(code, prefix)
        path = os.path.join(OUT, "%s.csv" % code)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        if rows:
            print("%s %s: %d 行, %s ~ %s, 最新收盘 %.3f"
                  % (code, name, len(rows), rows[0][0], rows[-1][0], float(rows[-1][2])), flush=True)
        else:
            print("%s %s: 无数据!" % (code, name), flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
