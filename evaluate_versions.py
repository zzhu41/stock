# -*- coding: utf-8 -*-
"""Offline comparison of fixed v9/v9.1/v9.2 configurations after correctness fixes.

Uses local CSVs only, records input hashes, and never writes trading state.
Next-open replay is valid for these fixed presets: no stops or position sizing
depend on execution price. It is not an intraday reconstruction or an OOS test.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import backtest
from market_data import CASH
from strategy_versions import BASE_CODES, STOCK_CODES, GLOBAL_CODES, backtest_kwargs, load_local_histories

BASE = Path(__file__).resolve().parent
STARTS = ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01")


def metrics(daily):
    if len(daily) < 2:
        raise ValueError("At least two trading sessions are required")
    peak, drawdown = 1.0, 0.0
    for _, value, _ in daily:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1)
    returns = [daily[i][1] / daily[i - 1][1] - 1 for i in range(1, len(daily))]
    mean = sum(returns) / len(returns)
    std = (sum((r - mean) ** 2 for r in returns) / len(returns)) ** .5
    return dict(nav=daily[-1][1], ann=daily[-1][1] ** (244 / len(daily)) - 1,
                max_dd=drawdown, sharpe=mean / std * 244 ** .5 if std else 0,
                yearly=dict(backtest.yearly(daily)))


def next_open_replay(signal_daily, histories, fee=.0001, slippage=0.0):
    """Execute yesterday's closing target at today's open; mark at today's close.

    Old holding receives the overnight return, the new holding the intraday
    return. Charge each actual buy/sell and adverse slippage on both sides.
    Initial uninvested money is cash, not a free first ETF purchase.
    Missing bars defer the whole switch and carry the previous valuation until
    trading resumes. The desired path stays fixed: this is execution sensitivity,
    not a new simulation of decisions conditioned on a deferred portfolio.
    """
    if not 0 <= fee < 1 or not 0 <= slippage < 1:
        raise ValueError("Invalid execution costs")
    prices = {c: {r[0]: r for r in rows} for c, rows in histories.items()}
    nav, holding, previous_close = 1.0, None, None
    daily, trades, deferred = [], [], []
    for i, (date, _, _) in enumerate(signal_daily):
        def price(code, column):
            row = prices.get(code, {}).get(date)
            value = row[column] if row else None
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError("Missing/invalid execution price: %s %s" % (code, date))
            return value
        holding_available = holding is None or date in prices.get(holding, {})
        if holding is not None and holding_available:
            nav *= price(holding, 1) / previous_close
        target = signal_daily[i - 1][2] if i else None
        if target != holding and (not holding_available or (
                target is not None and date not in prices.get(target, {}))):
            deferred.append((date, holding, target))
            target = holding
        if target != holding:
            if holding is not None:
                nav *= (1 - fee) * (1 - slippage)
            if target is not None:
                price(target, 1)  # validate before accepting a fill
                nav *= (1 - fee) / (1 + slippage)
            trades.append((date, holding, target))
            holding = target
        if holding is not None and date in prices.get(holding, {}):
            previous_close = price(holding, 2)
            nav *= previous_close / price(holding, 1)
        daily.append((date, nav, holding))
    out = metrics(daily)
    out.update(daily=daily, trades=trades, switches=len(trades),
               deferred_trades=deferred, deferred_count=len(deferred))
    return out


def _reset_lab(lab, cfg):
    lab.CFG.clear()
    lab.CFG.update(cfg)
    lab.STATE.clear()
    lab.STATE.update(last_date=None, holding=None, pending=None, count=0,
                     ne_fb=False, ne_fb_bull=True, ne_fb_raw=False, ne_fb_cand=None)


def compare(end=None, starts=STARTS):
    import strategy
    sys.path.insert(0, str(BASE / "v10"))
    import lab
    strategy.STOCK_POOL = list(STOCK_CODES)
    strategy.GLOBAL_POOL = list(GLOBAL_CODES)
    histories = load_local_histories()
    qvix_path = BASE / "data" / "qvix50.csv"
    with qvix_path.open(newline="", encoding="utf-8") as f:
        qvix = [(r[0], float(r[1])) for r in csv.reader(f) if r]
    common_end = min([rows[-1][0] for rows in histories.values()] + [qvix[-1][0]])
    end = end or common_end
    if end > common_end:
        raise ValueError("Requested end exceeds common price/QVIX coverage: " + common_end)
    histories = {c: [r for r in rows if r[0] <= end] for c, rows in histories.items()}
    calendar = [r[0] for r in histories["510300"]]
    lab.H, lab.CAL = histories, calendar
    lab.FEAR.update(qd=[], qz=[], qv=[], rd=[], r5=[])
    for i, (date, value) in enumerate(qvix):
        if i < 120 or date > end:
            continue
        window = [v for _, v in qvix[max(0, i - 250):i]]
        mean = sum(window) / len(window)
        std = (sum((v - mean) ** 2 for v in window) / len(window)) ** .5
        if std:
            lab.FEAR["qd"].append(date)
            lab.FEAR["qz"].append((value - mean) / std)
            lab.FEAR["qv"].append(value)

    # These three presets share identical price indicators. Cache only those;
    # keep decisions, portfolio/lock state and QVIX checks independent per run.
    original_rank, cache = strategy.rank, {}
    def cached_rank(h, on_date=None, live_prices=None):
        lab.STATE["last_date"] = on_date
        if on_date not in cache:
            saved = lab.CFG.get("crash_volu")
            lab.CFG["crash_volu"] = 2.0
            try:
                cache[on_date] = original_rank(h, on_date=on_date, live_prices=live_prices)
            finally:
                if saved is None:
                    lab.CFG.pop("crash_volu", None)
                else:
                    lab.CFG["crash_volu"] = saved
        return cache[on_date]
    strategy.rank = cached_rank
    results, fidelity = [], {}
    try:
        for start in starts:
            for version in ("v9", "v9.1", "v9.2"):
                cfg = dict(lab.VARIANTS["fz25_cv"][0]) if version == "v9.2" else {}
                _reset_lab(lab, cfg)
                if version == "v9.2":
                    r = lab.backtest_v10(histories, calendar, start=start, end=end)
                else:
                    r = backtest.backtest(histories, calendar, start=start, end=end,
                                          **backtest_kwargs(version))
                scenarios = {"same_close_ideal": dict(metrics(r["daily"]), switches=r["switches"])}
                for bps in (0, 5, 10):
                    replay = next_open_replay(r["daily"], histories, slippage=bps / 10000)
                    scenarios["next_open_%dbps" % bps] = {
                        k: v for k, v in replay.items() if k not in ("daily", "trades")}
                results.append(dict(version=version, start=start, end=end,
                                    crash_buys=r["crash_buys"], scenarios=scenarios))
                print("%s %s: close %.2f%%, next-open+10bp %.2f%%" % (
                    version, start, r["ann"] * 100,
                    scenarios["next_open_10bps"]["ann"] * 100), flush=True)
                if version == "v9.1" and start == starts[0]:
                    for check in ("copy_check", "engine_check", "top1_check"):
                        _reset_lab(lab, {"use_copies": True})
                        if check == "copy_check":
                            check_r = backtest.backtest(histories, calendar, start=start, end=end,
                                                        **backtest_kwargs("v9.1"))
                        elif check == "engine_check":
                            check_r = lab.backtest_v10(histories, calendar, start=start, end=end)
                        else:
                            check_r = lab.backtest_top2(histories, calendar, start=start, end=end, max_legs=1)
                        fidelity[check] = all(check_r[k] == r[k]
                                              for k in ("daily", "trades", "crash_buys"))
                        if not fidelity[check]:
                            raise AssertionError("Corrected engine parity failed: " + check)
    finally:
        strategy.rank = original_rank
        _reset_lab(lab, {})
    inputs = [BASE / "data" / (c + ".csv") for c in BASE_CODES] + [qvix_path]
    code = [BASE / name for name in ("strategy.py", "backtest.py", "v10/lab.py",
                                    "strategy_versions.py", "evaluate_versions.py")]
    return dict(end=end, starts=list(starts), fidelity=fidelity,
                data_sha256={str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                code_sha256={str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
                caveats=["All historical windows have been used for research; none is a clean OOS test.",
                         "Nested starts are sensitivity checks, not independent confirmations.",
                         "Next-open quotes approximate fills; no historical order book, limits, or 14:50 bars.",
                         "Missing bars defer switches. Replay keeps original signal targets instead of recalculating from deferred holdings.",
                         "Slippage is per side, in addition to 1bp per-side commission.",
                         "Ideal close engine omits first-entry fee; next-open replay charges it.",
                         "Daily QVIX publication timestamps are unavailable; next-open assumes available by then."],
                results=results)


def markdown(report):
    out = ["# v9 系列正确性修复后复评", "", "固定截止日：%s；本地数据只读，未重新挑选参数。" % report["end"], "",
           "这些区间已用于策略研究，全部属于样本内复算。六起点重叠，不是六次独立样本外验证。", "",
           "| 版本/起点 | 原收盘成交年化 | 次日开盘年化 | 开盘+单边5bp滑点 | 开盘+单边10bp滑点 | 10bp情景最大回撤 |",
           "|---|---:|---:|---:|---:|---:|"]
    for row in report["results"]:
        s = row["scenarios"]
        out.append("| %s / %s | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% |" % (
            row["version"], row["start"][:4], s["same_close_ideal"]["ann"] * 100,
            s["next_open_0bps"]["ann"] * 100, s["next_open_5bps"]["ann"] * 100,
            s["next_open_10bps"]["ann"] * 100, s["next_open_10bps"]["max_dd"] * 100))
    out += ["", "各情景均含单边万一佣金；滑点是额外成本。次日开盘按照昨日收盘信号成交，计入旧持仓隔夜收益、新持仓日内收益和首笔建仓费用。",
            "", "## 逐年收益：次日开盘 + 单边 10bp 滑点", "", "| 年份 | v9 | v9.1 | v9.2 |", "|---|---:|---:|---:|"]
    first = {r["version"]: r["scenarios"]["next_open_10bps"]["yearly"]
             for r in report["results"] if r["start"] == report["starts"][0]}
    for year in sorted(first["v9"]):
        out.append("| %s | %.2f%% | %.2f%% | %.2f%% |" % (
            year, *(first[v][year] * 100 for v in ("v9", "v9.1", "v9.2"))))
    out += ["", "## 验证范围", "", "修正后的副本/引擎/top1与v9.1主引擎逐日持仓、净值、交易和抄底记录一致：%s。" % report["fidelity"], "",
            "使用日线开盘价近似成交，缺行时保留持仓并延后换仓，复牌时补计累计涨跌。该压力测试沿用原收盘目标序列，不根据延期成交后的持仓重新决策。尚未模拟涨跌停无法成交、盘口流动性和历史14:50快照。QVIX只有日期，无法验证每个历史值的实际发布时间。",
            "", "原有影子净值存在日期/价格错位，不能拼接为修复后版本的前向业绩。新的影子账本从修复后的首个有效报价重新计量，并保留旧记录说明。",
            "", "配套JSON记录完整逐年结果、参数所用代码哈希和输入数据哈希。结果仅供比较固定规则，对未来收益和过拟合概率没有统计保证。", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end")
    parser.add_argument("--starts", default=",".join(STARTS))
    parser.add_argument("--output", default="reports/correctness_review.json")
    args = parser.parse_args()
    report = compare(end=args.end, starts=tuple(args.starts.split(",")))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print("Written " + str(path))


if __name__ == "__main__":
    main()
