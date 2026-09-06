# -*- coding: utf-8 -*-
"""v10 第八轮数据管道: 行业ETF池扩容候选 → data_extra/ (实盘 data/ 不动)。

预注册纳入规则(防事后挑池): 上市 <= 2017-09 且 近20日均量 >= 100万手 的中证行业ETF,
每行业取最老一只。探测结果(2026-09-06): 消费159928/医药512010/非银512070(约2013起) +
证券512880(2016-08)/银行512800(2017-08)/地产512200(2017-09)。
落选: 159938医药卫生(与512010重复+量低) 159939信息技术(30万) 159940金融地产(2万) 512580环保(5万)。
"""
import csv
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_data import _kline_page

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_extra")
SECTOR = {
    "159928": ("消费ETF", "sz"),
    "512010": ("医药ETF", "sh"),
    "512070": ("非银ETF", "sh"),
    "512880": ("证券ETF", "sh"),
    "512800": ("银行ETF", "sh"),
    "512200": ("房地产ETF", "sh"),
}


def fetch_full(code, pre):
    """向前分页拉全历史 [(date, open, close, volume)] 升序。"""
    rows, end = [], time.strftime("%Y-%m-%d")
    while True:
        page = _kline_page(pre + code, "2010-01-01", end)
        if not page:
            break
        rows = page + rows
        if len(page) < 640 or page[0][0] <= "2010-01-01":
            break
        end = (datetime.strptime(page[0][0], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(1.0)
    return rows


if __name__ == "__main__":
    for code, (name, pre) in SECTOR.items():
        rows = fetch_full(code, pre)
        with open(os.path.join(OUT, "%s.csv" % code), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print("%s %s: %d 行 %s ~ %s" % (code, name, len(rows), rows[0][0], rows[-1][0]))
        time.sleep(1.0)
