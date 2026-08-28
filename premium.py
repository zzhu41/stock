# -*- coding: utf-8 -*-
"""QDII 溢价监控: 实时价 vs 最新公布净值 → 溢价率。供 14:50 信号卡片调用。

数据源: 东财基金历史净值接口(api.fund.eastmoney.com/f10/lsjz)。
注意: 513100(纳指)为 QDII, 净值滞后一个交易日(T 日交易时只能看到 T-1 净值), 溢价为近似口径;
     513120(港股创新药)净值当日晚间公布, 尾盘时最新净值也是 T-1 —— 溢价同样是近似。
     口径与集思录/雪球的"实时溢价"一致(都是现价/最新已知净值), 足够用于 >2% 勿追的纪律。
纪律阈值: 溢价 >2% 不追(README 规则)。
"""
import json
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from market_data import UNIVERSE, fetch_realtime, fetch_history

QDII = ["513100", "513120"]           # 跨境池两只 QDII
WARN_THR = 0.02                       # 溢价纪律阈值
_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://fund.eastmoney.com/"}


def _latest_nav(code):
    """东财基金净值: 返回 (净值日期, 单位净值); 失败返回 (None, None)。"""
    url = ("https://api.fund.eastmoney.com/f10/lsjz?fundCode=%s&pageIndex=1&pageSize=2"
           % code)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read().decode("utf-8"))
    rows = (d.get("Data") or {}).get("LSJZList") or []
    for row in rows:
        if row.get("DWJZ"):
            return row["FSRQ"], float(row["DWJZ"])
    return None, None


def get_premium(code):
    """返回 dict(price, nav, nav_date, premium, warn); 任一环失败则 premium=None。"""
    try:
        price = fetch_realtime([code]).get(code, (None, None))[1]
        if not price:
            price = fetch_history(code)[-1][2]
    except Exception:
        try:
            price = fetch_history(code)[-1][2]
        except Exception:
            price = None
    try:
        nav_date, nav = _latest_nav(code)
    except Exception:
        nav_date, nav = None, None
    prem = (price / nav - 1.0) if (price and nav) else None
    return {"code": code, "name": UNIVERSE.get(code, (code,))[0],
            "price": price, "nav": nav, "nav_date": nav_date,
            "premium": prem, "warn": prem is not None and prem > WARN_THR}


def signal_block(codes=None):
    """信号卡片文本行(list); 全部失败返回 []。绝不抛异常(主链路保护)。"""
    out = []
    for code in (codes or QDII):
        try:
            p = get_premium(code)
            if p["premium"] is None:
                out.append("  %s %s 溢价: 获取失败(不影响信号)" % (code, p["name"]))
            else:
                flag = " ⚠️ >2% 勿追!" if p["warn"] else ""
                out.append("  %s %s 溢价 %+.2f%% (现价 %.3f / 净值 %.4f@%s)%s"
                           % (code, p["name"], p["premium"] * 100, p["price"],
                              p["nav"], p["nav_date"], flag))
        except Exception:
            continue
    return out


if __name__ == "__main__":
    print("== QDII 溢价监控 ==")
    for line in signal_block():
        print(line)
