# -*- coding: utf-8 -*-
"""实盘跟踪误差日报: 实盘账本净值 vs 策略模型净值, 同一窗口对比。

模型净值: backtest 引擎从 MODEL_START(首个信号日/系统建成日 2026-08-13 起)跑 v9.1 默认口径。
实盘净值: trades.csv 成交序列在日常收盘价上重放(满仓语义, 买入日按当日收盘持有新标的;
          首笔买入金额视为初始资金, 零头现金逐笔跟踪)。
输出: signals/track.jsonl 追加一行 + stdout 摘要 + 滑点核算(顺带)。
账本空仓/无成交时只记录模型侧, 优雅运行 —— 验证期未开仓也每天积累模型基线。

这是过拟合的终极探测器: 实盘累计持续跑输模型 → 策略样本外退化或执行漏阿尔法。
"""
import csv
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import backtest
import slippage
from market_data import UNIVERSE, CASH, fetch_history

MODEL_START_FALLBACK = "2026-08-13"          # 系统建成日
TRACK_LOG = os.path.join(BASE, "signals", "track.jsonl")


def _model_start():
    """最早信号存档的数据日期; 无则回退到系统建成日。"""
    sdir = os.path.join(BASE, "signals")
    dates = []
    if os.path.isdir(sdir):
        for fn in os.listdir(sdir):
            if fn[:4].isdigit() and fn.endswith(".txt") and fn != "latest.txt":
                dates.append(fn[:-4])
    return min(dates) if dates else MODEL_START_FALLBACK


def model_nav_series(histories, calendar, start):
    r = backtest.backtest(histories, calendar, start=start)
    return [(d, nav, h) for d, nav, h in r["daily"]], r


def live_nav_series(trades, close_of, calendar):
    """从首笔成交日起重放实盘净值 [(date, nav)], 初始资金=首笔买入金额。"""
    if not trades:
        return []
    t0 = trades[0]["date"][:10]
    cash, holding, shares = 0.0, None, 0
    events = {}
    for t in trades:
        events.setdefault(t["date"][:10], []).append(t)
    first = trades[0]
    capital = float(first["amount"]) if first["action"] == "buy" else float(first["amount"])
    days = [d for d in calendar if d >= t0]
    out = []
    ti = 0
    ordered = [t for t in trades]
    for d in days:
        # 当日成交(14:50尾盘): 按当日收盘起持有新标的
        for t in [x for x in ordered if x["date"][:10] == d]:
            if t["action"] == "buy":
                amt = float(t["amount"])
                if holding is None and cash == 0.0:
                    cash = capital
                price = float(t["price"])
                sh = int(amt / price / 100) * 100
                cash -= sh * price
                holding, shares = t["code"], sh
            elif t["action"] == "sell" and holding == t["code"]:
                cash += shares * float(t["price"])
                holding, shares = None, 0
        if holding:
            c = close_of.get(holding, {}).get(d)
            nav = (shares * c + cash) if c else None
        else:
            nav = cash if cash > 0 else None
        if nav:
            out.append((d, nav))
    return out


def main():
    histories = {c: fetch_history(c) for c in UNIVERSE}
    calendar = [r[0] for r in histories["510300"]]
    close_of = {c: {r[0]: r[2] for r in rows} for c, rows in histories.items()}
    start = _model_start()

    trades = slippage.load_trades()
    model_daily, res = model_nav_series(histories, calendar, start)
    model_ret = model_daily[-1][1] / model_daily[0][1] - 1.0 if len(model_daily) > 1 else 0.0

    lines = ["== 实盘 vs 模型 (%s 起) ==" % start,
             "模型: 累计 %+6.2f%% | 年化 %+5.1f%% | 回撤 %5.1f%% | 期末持仓 %s"
             % (model_ret * 100, res["ann"] * 100, res["max_dd"] * 100,
                UNIVERSE.get(model_daily[-1][2], (model_daily[-1][2],))[0]
                if model_daily[-1][2] else "空仓")]

    live = live_nav_series(trades, close_of, calendar)
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "date": calendar[-1],
           "model_start": start, "model_cum_pct": round(model_ret * 100, 2),
           "model_holding": model_daily[-1][2], "n_trades": len(trades)}
    if live:
        live_ret = live[-1][1] / live[0][1] - 1.0
        # 同窗口模型收益
        m0 = next((nav for d, nav, _ in model_daily if d >= live[0][0]), None)
        m1 = model_daily[-1][1]
        m_ret = (m1 / m0 - 1.0) if m0 else 0.0
        diff_pp = (live_ret - m_ret) * 100
        lines.append("实盘: 累计 %+6.2f%% (%s 起) | 同窗口模型 %+6.2f%% | 跟踪差 %+.2fpp"
                     % (live_ret * 100, live[0][0], m_ret * 100, diff_pp))
        lines.append("判定: %s" % ("跟踪正常" if abs(diff_pp) < 3 else
                     "⚠️ 偏离超3pp, 检查滑点/纪律执行" if diff_pp < 0 else "实盘跑赢模型"))
        rec.update(live_start=live[0][0], live_cum_pct=round(live_ret * 100, 2),
                   model_same_window_pct=round(m_ret * 100, 2), diff_pp=round(diff_pp, 2))
    else:
        lines.append("实盘: 账本尚无成交(验证期未开仓), 仅记录模型基线")
    lines.append("")
    lines += slippage.report()
    text = "\n".join(lines)
    print(text)
    with open(TRACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
