# -*- coding: utf-8 -*-
"""看门狗(15:20 工作日 cron): 防系统静默死亡。

检查项:
  1. 数据新鲜度: 交易日收盘后, 510300 本地缓存应有当日K线; 缺失且东财侧有 → 腾讯源异常告警
  2. 信号生成: 交易日应有 signals/<今日>.txt 且 mtime 为今天; 缺失 → 告警(cron 挂了)
  3. 双源校验: 东财 kline vs 腾讯缓存收盘价, 重叠日偏差 >0.5% → 告警(数据源污染检测)
非交易日(周末/节假日)自动跳过。告警走钉钉(live_utils.send_dingtalk), 日志 signals/watchdog.log。
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import live_utils
from market_data import UNIVERSE

_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
DIVERGE_THR = 0.005                    # 双源收盘价偏差阈值 0.5%


def em_last_klines(code, n=5):
    """东财日K(前复权): 返回 [(date, close)] 最近 n 根; 失败返回 []。"""
    pref = "1" if UNIVERSE[code][1] == "sh" else "0"
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=%s.%s"
           "&fields1=f1,f2,f3&fields2=f51,f53&klt=101&fqt=1&beg=20260101&end=20990101"
           % (pref, code))
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read().decode("utf-8"))
    kl = (d.get("data") or {}).get("klines") or []
    return [(k.split(",")[0], float(k.split(",")[1])) for k in kl][-n:]


def tencent_tail(code, n=5):
    import csv
    path = os.path.join(BASE, "data", "%s.csv" % code)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if r:
                rows.append((r[0], float(r[2])))
    return rows[-n:]


def main():
    today = time.strftime("%Y-%m-%d")
    alerts = []

    # --- 1+2: 数据与信号新鲜度 ---
    tx_last = live_utils.read_last_date("510300")
    if tx_last == today:
        sig = os.path.join(BASE, "signals", "%s.txt" % today)
        if not (os.path.exists(sig) and
                time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(sig))) == today):
            alerts.append("交易日 %s 信号文件缺失或非今日生成(cron 异常?)" % today)
    else:
        try:
            em = em_last_klines("510300", 1)
            if em and em[0][0] == today:
                alerts.append("今日 %s 东财有K线但腾讯缓存最新仅 %s —— 腾讯数据源/更新异常"
                              % (today, tx_last))
            else:
                live_utils.append_log("watchdog.log", "非交易日(%s), 跳过" % today)
        except Exception as e:
            live_utils.append_log("watchdog.log", "东财探测失败: %r, 跳过" % e)
            _dual_source(alerts)           # 数据校验照做
            _finish(alerts)
            return
    _dual_source(alerts)
    _finish(alerts)


def _dual_source(alerts):
    """3: 全池双源收盘价比对。"""
    bad = []
    for code in UNIVERSE:
        try:
            tx = dict(tencent_tail(code, 5))
            for d, c in em_last_klines(code, 5):
                if d in tx and tx[d] > 0:
                    dev = abs(c / tx[d] - 1.0)
                    if dev > DIVERGE_THR:
                        bad.append("%s %s %s 偏差 %.2f%%" % (code, UNIVERSE[code][0], d, dev * 100))
                        break
        except Exception:
            continue
        time.sleep(0.5)
    if bad:
        alerts.append("双源收盘价偏差>0.5%%: %s" % "; ".join(bad))


def _finish(alerts):
    if alerts:
        text = "### ⚠️ 动量系统看门狗告警\n\n" + "\n\n".join("- " + a for a in alerts)
        live_utils.send_dingtalk(text)
        live_utils.append_log("watchdog.log", "ALERT: " + " | ".join(alerts))
    else:
        live_utils.append_log("watchdog.log", "OK")
        print("watchdog OK")


if __name__ == "__main__":
    main()
