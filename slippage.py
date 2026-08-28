# -*- coding: utf-8 -*-
"""滑点核算: trades.csv 真实成交价 vs 当日收盘(本地缓存) → 逐笔/累计滑点与年化拖累估计。

口径: 买入不利滑点 = 成交价/收盘-1; 卖出不利滑点 = 收盘/成交-1(正数=吃亏)。
年化拖累粗估 = 平均每笔不利滑点 × 2 × 年换手(回测口径 ~17.7 次换仓/年, 每次换仓=买+卖两笔)。
README 经验值: 单边每 0.1% ≈ 年化 -3pp —— 本工具用实盘成交验证该假设。
账本为空时优雅退出(验证期未开始交易也正常运行)。
"""
import csv
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from market_data import UNIVERSE

TRADES = os.path.join(BASE, "trades.csv")
SW_PER_YEAR = 17.7                     # 回测口径年换仓次数(用于年化拖累粗估)


def _close_map(code):
    out = {}
    with open(os.path.join(BASE, "data", "%s.csv" % code), newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if r:
                out[r[0]] = float(r[2])
    return out


def load_trades():
    if not os.path.exists(TRADES):
        return []
    with open(TRADES, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("action") in ("buy", "sell")]


def report():
    trades = load_trades()
    if not trades:
        return ["滑点核算: 暂无成交记录, 首笔成交后自动启用"]
    cache = {}
    lines = ["滑点核算(成交价 vs 当日收盘, 正=吃亏):"]
    tot, n = 0.0, 0
    for t in trades:
        code = t["code"]
        if code not in cache:
            cache[code] = _close_map(code)
        d = t["date"][:10]
        close = cache[code].get(d)
        if not close:
            lines.append("  %s %s %s %s: 无当日收盘数据, 跳过"
                         % (t["date"], t["action"], code, UNIVERSE.get(code, ("?",))[0]))
            continue
        price = float(t["price"])
        slip = (price / close - 1.0) if t["action"] == "buy" else (close / price - 1.0)
        tot += slip
        n += 1
        lines.append("  %s %s %s %s: 成交 %.3f 收盘 %.3f 滑点 %+.2f%%"
                     % (t["date"], t["action"], code, UNIVERSE.get(code, ("?",))[0],
                        price, close, slip * 100))
    if n:
        avg = tot / n
        drag = avg * 2 * SW_PER_YEAR
        lines.append("  共 %d 笔 | 平均每笔 %+.3f%% | 粗估年化拖累 %+.1fpp(×2边×%.1f换/年)"
                     % (n, avg * 100, drag * 100, SW_PER_YEAR))
        lines.append("  参考: 平均滑点若稳定 >0.1%, 说明尾盘贴价执行有改进空间")
    return lines


if __name__ == "__main__":
    print("\n".join(report()))
