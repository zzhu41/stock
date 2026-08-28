# -*- coding: utf-8 -*-
"""前复权漂移防护(每周六 10:00 cron): ETF 分红后前复权历史整体重定基,
而增量更新只覆盖最后一天 → 旧缓存静默漂移。每周比对最近45天重叠窗口,
偏差 >0.05% 即自动全量重拉修复并钉钉告警; 正常则记一行日志。
"""
import csv
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import live_utils
from market_data import UNIVERSE, _kline_page, fetch_history

DRIFT_THR = 0.0005                     # 0.05% 漂移阈值
LOOKBACK_DAYS = 45


def check_one(code):
    """返回 (最大偏差, 漂移日)。"""
    cache = {}
    with open(os.path.join(BASE, "data", "%s.csv" % code), newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if r:
                cache[r[0]] = float(r[2])
    last = sorted(cache)[-1]
    start = time.strftime("%Y-%m-%d",
                          time.localtime(time.mktime(time.strptime(last, "%Y-%m-%d"))
                                         - LOOKBACK_DAYS * 86400))
    fresh = _kline_page(UNIVERSE[code][1] + code, start, last)
    diffs = [(d, abs(p / cache[d] - 1.0)) for d, o, p, v in fresh
             if d in cache and cache[d] > 0]
    return max(diffs, key=lambda x: x[1]) if diffs else (None, 0.0)


def main():
    drifted, checked = [], 0
    for code in UNIVERSE:
        try:
            d, dev = check_one(code)
            checked += 1
            if dev > DRIFT_THR:
                drifted.append((code, d, dev))
        except Exception as e:
            live_utils.append_log("drift.log", "%s 检查失败: %r" % (code, e))
        time.sleep(0.8)
    if drifted:
        fixed = []
        for code, d, dev in drifted:
            try:
                fetch_history(code, full=True)     # 全量重拉修复
                fixed.append("%s(漂移日%s %.3f%%)" % (code, d, dev * 100))
            except Exception as e:
                fixed.append("%s 修复失败: %r" % (code, e))
        msg = "### 🔧 前复权漂移已修复\n\n" + "\n\n".join("- " + s for s in fixed)
        live_utils.send_dingtalk(msg, title="动量系统数据修复")
        live_utils.append_log("drift.log", "DRIFT fixed: " + "; ".join(fixed))
        print(msg)
    else:
        live_utils.append_log("drift.log", "OK %d codes checked, no drift" % checked)
        print("drift_guard OK: %d 只全部无漂移" % checked)


if __name__ == "__main__":
    main()
