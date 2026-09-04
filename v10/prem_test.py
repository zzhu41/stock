# -*- coding: utf-8 -*-
"""v10 第五轮: QDII 溢价修正实验(执行层, 文献+本土实践支持, README 人工纪律的自动化)。

假设: QDII 高溢价时买入 = 支付确定性回落的虚高水分(premium.py 上线首日抓到 +11.5%);
     溢价>2% 天数近年密集(513100: 2022-2026 共509天, 非单事件)。

变体:
  P1 溢价禁追(引擎新开关 premium_guard, 切换目标溢价超限顺延下一名; 只挡买入侧):
     premG1/G2/G3/G5 = 阈值 1%/2%/3%/5%(阈值平台检验)
  P3 净值信号(零引擎改动, 复用 signal_histories 信号/收益分离机制):
     QDII 信号用 T-1 净值伪价(无前视), 收益仍按市价 —— 剔除溢价噪声还原真实动量
  组合 premG2+navsig

协议(同第三/四轮): 6 起点终值比 + 2014起点逐年差; 结果 results/round5_prem.jsonl。
用法: python3.8 v10/prem_test.py
"""
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from market_data import UNIVERSE, fetch_history  # noqa: E402
import backtest  # noqa: E402
from backtest import buy_and_hold  # noqa: E402

EXTRA = os.path.join(BASE, "v10", "data_extra")
OUT = os.path.join(BASE, "v10", "results", "round5_prem.jsonl")
QDII = ["513100", "513120"]
STARTS = ("2014-01-01", "2016-01-01", "2018-01-01",
          "2020-01-01", "2022-01-01", "2024-01-01")


def load_premium():
    """premium_{code}.csv -> {code: {date: 溢价小数}}(拆分错位段已置空, 跳过)。"""
    out = {}
    for code in QDII:
        m = {}
        with open(os.path.join(EXTRA, "premium_%s.csv" % code)) as f:
            next(f)
            for ln in f:
                d, _, _, p = ln.strip().split(",")
                if p != "":
                    m[d] = float(p) / 100.0
        out[code] = m
    return out


def load_nav_pseudo(histories):
    """P3: QDII 净值伪价序列(T-1 净值对齐市价日), 其余标的原样。"""
    sig = dict(histories)
    for code in QDII:
        rows = []
        with open(os.path.join(EXTRA, "premium_%s.csv" % code)) as f:
            next(f)
            for ln in f:
                d, nav, _, _ = ln.strip().split(",")
                rows.append((d, float(nav), float(nav), 0.0))
        sig[code] = rows
    return sig


def yearly(daily):
    last = {}
    for d, nav, _ in daily:
        last[d[:4]] = nav
    res, base = [], 1.0
    for y in sorted(last):
        res.append((y, last[y] / base - 1))
        base = last[y]
    return dict(res)


def run(histories, calendar, name, start, **kw):
    res = backtest.backtest(histories, calendar, start=start, **kw)
    return {
        "variant": name, "start": start,
        "nav": round(res["nav"], 4), "ann": round(res["ann"] * 100, 2),
        "dd": round(res["max_dd"] * 100, 2), "sharpe": round(res["sharpe"], 3),
        "switches": res["switches"],
        "yearly": {y: round(r * 100, 1) for y, r in yearly(res["daily"]).items()},
    }


def main():
    histories = {c: fetch_history(c) for c in UNIVERSE}
    calendar = [r[0] for r in histories["510300"]]
    premium = load_premium()
    sig_nav = load_nav_pseudo(histories)
    print("溢价序列: %s" % {c: len(m) for c, m in premium.items()})

    variants = [
        ("baseline", {}),
        ("premG1", {"premium_data": premium, "premium_guard": 0.01}),
        ("premG2", {"premium_data": premium, "premium_guard": 0.02}),
        ("premG3", {"premium_data": premium, "premium_guard": 0.03}),
        ("premG5", {"premium_data": premium, "premium_guard": 0.05}),
        ("navsig", {"signal_histories": sig_nav}),
        ("premG2+navsig", {"premium_data": premium, "premium_guard": 0.02,
                           "signal_histories": sig_nav}),
    ]
    results = []
    base_nav = {}
    for name, kw in variants:
        for start in STARTS:
            t0 = time.time()
            r = run(histories, calendar, name, start, **kw)
            results.append(r)
            if name == "baseline":
                base_nav[start] = r["nav"]
            ratio = r["nav"] / base_nav.get(start, r["nav"])
            print("%-14s %s | nav %8.2f | 基线比 %.3f | 年化 %+5.1f%% 回撤 %5.1f%% | %ds"
                  % (name, start[:4], r["nav"], ratio, r["ann"], r["dd"],
                     time.time() - t0), flush=True)
    with open(OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 协议摘要: 6起点全赢? + 2014起点逐年差
    print("\n== 协议摘要(终值比, >1 为赢) ==")
    print("%-14s %s | 判定" % ("变体", " ".join("%4s" % s[:4] for s in STARTS)))
    base14 = {r["start"]: r for r in results if r["variant"] == "baseline"}
    for name, _ in variants[1:]:
        rs = {r["start"]: r for r in results if r["variant"] == name}
        ratios = [rs[s]["nav"] / base14[s]["nav"] for s in STARTS]
        win = all(x > 1.0 for x in ratios)
        print("%-14s %s | %s" % (name, " ".join("%.3f" % x for x in ratios),
                                 "✅ 6/6" if win else "❌"))
        yb = base14["2014-01-01"]["yearly"]
        yv = rs["2014-01-01"]["yearly"]
        diff = {y: round(yv.get(y, 0) - yb.get(y, 0), 1) for y in yb}
        print("  逐年差(pp): %s" % " ".join("%s:%+.1f" % (y[2:], v)
                                          for y, v in sorted(diff.items())))


if __name__ == "__main__":
    main()
