# -*- coding: utf-8 -*-
"""看板影子版本曲线导出: 独立子进程跑 v10 lab 钩子引擎(fz25_m4 / fz25_cv), stdout 打印 JSON。
web_build.py 以子进程调用——隔离 lab 对 strategy 的 monkey-patch, 主构建进程零污染。
QVIX 用实盘日刷新的 data/qvix50.csv(2列) 重建 lab.FEAR(与 lab._load_fear 同口径),
data_extra 研究快照止于拉取日, 看板曲线尾部需要最新 QVIX。"""
import csv
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lab  # noqa: E402

SHADOWS = [
    ("v9.1-0906", "fz25_m4", "v9.1-0906 影子 +QVIX恐慌抄底(虚拟跟踪)"),
    ("v9.2",      "fz25_cv", "v9.2 影子 +QVIX∪量能恐慌抄底(虚拟跟踪)"),
]
START = "2014-01-01"


def _refresh_fear_from_live():
    p = os.path.join(lab.BASE, "data", "qvix50.csv")
    if not os.path.exists(p):
        return
    with open(p, newline="", encoding="utf-8") as f:
        q = [(r[0], float(r[1])) for r in csv.reader(f) if r and len(r) >= 2]
    if len(q) < 130:
        return
    lab.FEAR["qd"], lab.FEAR["qz"], lab.FEAR["qv"] = [], [], []
    for i, (d, v) in enumerate(q):
        if i < 120:
            continue
        win = [x for _, x in q[max(0, i - 250):i]]   # 严格截至前一日(同 lab 口径)
        m = sum(win) / len(win)
        s = (sum((x - m) ** 2 for x in win) / len(win)) ** 0.5
        if s > 0:
            lab.FEAR["qd"].append(d)
            lab.FEAR["qz"].append((v - m) / s)
            lab.FEAR["qv"].append(v)


def run(vid, label, variant):
    cfg, kw = lab.VARIANTS[variant]
    assert kw.get("engine"), "影子曲线只支持引擎副本变体"
    lab.CFG.clear()
    lab.CFG.update(cfg)
    lab.STATE.update(last_date=None, holding=None, pending=None, count=0,
                     ne_fb=False, ne_fb_bull=True, ne_fb_raw=False, ne_fb_cand=None)
    r = lab.backtest_v10(lab.H, lab.CAL, start=START)
    trades = [[d, frm, to, round(nav, 4)] for d, frm, to, nav in r["trades"]]
    if r["daily"]:  # 看板补一条建仓记录(同 web_build 口径)
        d0, nav0, h0 = r["daily"][0]
        trades.insert(0, [d0, None, h0, round(nav0, 4)])
    return vid, {
        "label": label,
        "daily": [[d, round(nav, 4), h] for d, nav, h in r["daily"]],
        "trades": trades,
        "crash_buys": [[d, c] for d, c in r["crash_buys"]],
        "metrics": {"nav": round(r["nav"], 4), "ann": round(r["ann"] * 100, 2),
                    "max_dd": round(r["max_dd"] * 100, 2), "sharpe": round(r["sharpe"], 2),
                    "switches": r["switches"]},
    }


def main():
    lab.init_data()
    _refresh_fear_from_live()
    out = {}
    for vid, variant, label in SHADOWS:
        k, v = run(vid, label, variant)
        out[k] = v
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
