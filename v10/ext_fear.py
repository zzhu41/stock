# -*- coding: utf-8 -*-
"""v10 第七轮数据管道: 恐慌情绪外部数据 → data_extra/ (实盘 data/ 不动)。

qvix50.csv  50ETF期权 QVIX 日线 (期权论坛 1.optbbs.com k.csv, 2015-02起):
            date,open,high,low,close —— 全市场恐慌温度计(隐含波动率指数)
rzrq.csv    沪深两融融资余额 日线 (东财 datacenter RPTA_RZRQ_LSHJ, 2010-03起):
            date,rzye(融资余额,元) —— 去杠杆恐慌标记; 发布滞后T+1早晨, 使用时须取 <d 的最后一根
"""
import csv
import json
import os
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data_extra")
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
       "Referer": "https://data.eastmoney.com/"}


def _get(url, timeout=25, retries=4):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def fetch_qvix():
    raw = _get("http://1.optbbs.com/d/csv/d/k.csv")
    rows = []
    for line in raw.splitlines()[1:]:
        f = line.split(",")
        if len(f) < 5 or not f[0]:
            continue
        try:
            y, m, d = f[0].split("/")
            date = "%s-%02d-%02d" % (y, int(m), int(d))
            o, h, l, c = (float(f[1]), float(f[2]), float(f[3]), float(f[4]))
        except (ValueError, IndexError):
            continue
        rows.append([date, o, h, l, c])
    rows.sort(key=lambda r: r[0])
    with open(os.path.join(OUT, "qvix50.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return rows


def fetch_rzrq():
    rows = []
    page = 1
    while True:
        url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?"
               "reportName=RPTA_RZRQ_LSHJ&columns=ALL&sortColumns=DIM_DATE&"
               "sortTypes=1&pageSize=500&pageNumber=%d" % page)
        d = json.loads(_get(url))
        data = (d.get("result") or {}).get("data") or []
        if not data:
            break
        for x in data:
            rows.append([x["DIM_DATE"][:10], "%.0f" % float(x["RZYE"])])
        if len(data) < 500:
            break
        page += 1
        time.sleep(0.6)
    rows.sort(key=lambda r: r[0])
    with open(os.path.join(OUT, "rzrq.csv"), "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return rows


if __name__ == "__main__":
    q = fetch_qvix()
    print("QVIX50: %d 行 %s ~ %s 最新 %.2f" % (len(q), q[0][0], q[-1][0], q[-1][4]))
    r = fetch_rzrq()
    print("融资余额: %d 行 %s ~ %s 最新 %.0f亿" % (len(r), r[0][0], r[-1][0], float(r[-1][1]) / 1e8))
