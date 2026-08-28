# -*- coding: utf-8 -*-
"""v8 外部数据管道: QQQ(东财) + USDCNY(腾讯) 日线缓存, 构造 513100 混合信号序列。

无前视对齐: A股 T 日的 QQQ 取值 = 美东日期严格 < T 的最后一根(美股T-1日 K 线于北京时间T日凌晨完结);
            汇率取值 = 日期 <= T 的最后一根。
混合价 = etf_close^(1-w) * (QQQ_close * USDCNY)^w  (几何加权 => 对数动量的线性加权, 与平台同口径)
"""
import bisect
import csv
import json
import os
import time
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QQQ_CSV = os.path.join(DATA_DIR, "usQQQ.csv")
FX_CSV = os.path.join(DATA_DIR, "whUSDCNY.csv")
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
       "Referer": "https://quote.eastmoney.com/"}


def _get(url, timeout=20, retries=4):
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


def fetch_qqq(full=False):
    """新浪美股 QQQ 日线 [(date, close)] 升序, 缓存+增量(单次全量返回)。"""
    rows = []
    if not full and os.path.exists(QQQ_CSV):
        with open(QQQ_CSV, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f) if r]
    url = ("https://stock.finance.sina.com.cn/usstock/api/jsonp.php/var%20t=/"
           "US_MinKService.getDailyK?symbol=qqq")
    raw = _get(url)
    m = raw[raw.find("([") + 1:raw.rfind("]") + 1]  # 剥 jsonp 包装, 留纯数组
    data = json.loads(m)
    new = [[x["d"], x["c"]] for x in data]
    rows = [r for r in rows if r[0] < new[0][0]] + new if new else rows
    with open(QQQ_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return [(r[0], float(r[1])) for r in rows]


def fetch_usdcny(full=False):
    """腾讯 whUSDCNY 日线 [(date, close)] 升序, 缓存+增量(向前分页640)。"""
    rows = []
    if not full and os.path.exists(FX_CSV):
        with open(FX_CSV, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f) if r]

    def page(s, e):
        url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=whUSDCNY,day,%s,%s,640,qfq" % (s, e))
        d = json.loads(_get(url))
        node = (d.get("data") or {}).get("whUSDCNY") or {}
        raw = node.get("qfqday") or node.get("day") or []
        return [[p[0], p[2]] for p in raw]

    if rows:
        start = rows[-1][0]
        new = page(start, time.strftime("%Y-%m-%d"))
        rows = [r for r in rows if r[0] < new[0][0]] + new if new else rows
    else:
        end, start_limit = time.strftime("%Y-%m-%d"), "2010-01-01"
        while True:
            pg = page(start_limit, end)
            if not pg:
                break
            rows = pg + rows
            if len(pg) < 640 or pg[0][0] <= start_limit:
                break
            end = (time.strptime(pg[0][0], "%Y-%m-%d") and
                   __import__("datetime").datetime.strptime(pg[0][0], "%Y-%m-%d") -
                   __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")
            time.sleep(1.0)
    with open(FX_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return [(r[0], float(r[1])) for r in rows]


def build_mixed_close(etf_rows, w, qqq=None, fx=None):
    """构造 513100 混合信号价 {date: mixed_close}。etf_rows=[(date,open,close)]。
    返回 dict; 无 QQQ/汇率覆盖的日期丢弃。"""
    if w <= 0:
        return {d: c for d, _, c in etf_rows}
    qqq = qqq if qqq is not None else fetch_qqq()
    fx = fx if fx is not None else fetch_usdcny()
    qqq_dates = sorted(d for d, _ in qqq)
    qqq_map = dict(qqq)
    fx_dates = sorted(d for d, _ in fx)
    fx_map = dict(fx)
    out = {}
    for d, _, c in etf_rows:
        i = bisect.bisect_left(qqq_dates, d)          # 严格 < d 的最近一根美股K(无前视)
        j = bisect.bisect_right(fx_dates, d) - 1      # <= d 的最近汇率
        if i <= 0 or j < 0:
            out[d] = c                                # 无覆盖期(汇率2015-03前): 退回ETF自身价
            continue
        qqqc = qqq_map[qqq_dates[i - 1]] * fx_map[fx_dates[j]]
        out[d] = (c ** (1.0 - w)) * (qqqc ** w)
    return out


if __name__ == "__main__":
    q = fetch_qqq()
    print("QQQ: %d 行 %s ~ %s 最新 %.2f" % (len(q), q[0][0], q[-1][0], q[-1][1]))
    f = fetch_usdcny()
    print("USDCNY: %d 行 %s ~ %s 最新 %.4f" % (len(f), f[0][0], f[-1][0], f[-1][1]))
