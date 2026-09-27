# -*- coding: utf-8 -*-
"""Offline, fixed-grid v9/v9.1 parameter sensitivity; no parameter selection.

Run: python3 -B research_parameter_stability.py
Only reports/parameter_stability.json and its Markdown companion are written.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import backtest
import strategy
from evaluate_versions import next_open_replay
from strategy_versions import (BASE_CODES, STOCK_CODES, GLOBAL_CODES,
                               backtest_kwargs, load_local_histories)
from v9_robustness import wls_window

BASE = Path(__file__).resolve().parent
START, END = "2014-01-01", "2026-09-24"
FEE, SLIPPAGE = .0001, .001
PERIODS = (
    ("2014-2017", "2014-01-01", "2017-12-31"),
    ("2018-2021", "2018-01-01", "2021-12-31"),
    ("2022-2025", "2022-01-01", "2025-12-31"),
    ("2026 YTD", "2026-01-01", END),
)
# Fixed before inspecting results. Centers are the frozen deployed parameters;
# global and gold buffers vary separately to keep every comparison one-factor.
AXES = (
    ("panic_drop", (.03, .04, .05), .04),
    ("wls_window", (20, 25, 30), 25),
    ("crash_mom5", (-.06, -.08, -.10), -.08),
    ("crash_below_ma", (.15, .20, .25), .20),
    ("crash_lock", (3, 5, 7), 5),
)
# These parameters affect decisions/engine state only, not strategy.indicators.
# Every other engine argument remains in the cache key, including disabled knobs.
DECISION_ONLY = frozenset(("panic_drop", "buffer", "pool_buffer", "crash_mom5",
                           "crash_below_ma", "crash_lock"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


@contextmanager
def score_window(window):
    if window == 25:
        yield
    else:
        with wls_window(window):
            yield


def scenarios():
    result = []
    for version in ("v9", "v9.1"):
        result.append(dict(id=version + ":baseline", version=version, axis="baseline",
                           value=None, center=None, window=25,
                           params=backtest_kwargs(version)))
        axes = list(AXES)
        if version == "v9":
            axes.append(("buffer", (.01, .02, .03), .02))
        else:
            axes.extend((("global_buffer", (.02, .03, .04), .03),
                         ("gold_buffer", (.02, .03, .04), .03)))
        for axis, values, center in axes:
            for value in values:
                if value == center:
                    continue
                params = backtest_kwargs(version)
                window = value if axis == "wls_window" else 25
                if axis in ("global_buffer", "gold_buffer"):
                    params["pool_buffer"][axis.split("_")[0]] = value
                elif axis != "wls_window":
                    params[axis] = value
                result.append(dict(id="%s:%s=%s" % (version, axis, value),
                                   version=version, axis=axis, value=value,
                                   center=center, window=window, params=params))
    return result


class RankCache:
    """Cache price-derived indicators, never decisions or portfolio state."""
    def __init__(self, histories):
        self.histories = histories
        self.original = strategy.rank
        self.entries = {}
        self.hits = self.misses = 0
        self.key = None

    def select(self, case):
        indicator_config = {k: v for k, v in case["params"].items() if k not in DECISION_ONLY}
        # WLS is patched separately from the engine signature, so it must be
        # explicit in the key. The source histories object is fixed for this run.
        self.key = (case["window"], digest(indicator_config))

    def rank(self, histories, on_date=None, live_prices=None):
        if histories is not self.histories or live_prices is not None or self.key is None:
            raise AssertionError("Cache received unregistered data or a live quote")
        key = self.key + (on_date,)
        if key not in self.entries:
            self.entries[key] = self.original(histories, on_date=on_date)
            self.misses += 1
        else:
            self.hits += 1
        return self.entries[key]


def signal_run(case, histories, calendar, cache=None, start=START, end=END):
    original_rank = strategy.rank
    if cache is not None:
        cache.select(case)
        strategy.rank = cache.rank
    try:
        # The native 25-day center is left untouched. Modified windows only
        # change score; all window-sensitive feature switches are frozen off.
        with score_window(case["window"]):
            return backtest.backtest(histories, calendar, start=start, end=end,
                                      **case["params"])
    finally:
        strategy.rank = original_rank


def segment(daily, trades, start, end):
    """Slice one continuous NAV path; carry NAV/holding across the boundary."""
    indices = [i for i, (date, _, _) in enumerate(daily) if start <= date <= end]
    if not indices:
        raise ValueError("No observations in " + start)
    first, last = indices[0], indices[-1]
    carry = daily[first - 1][1] if first else 1.0
    values = [daily[i][1] / carry for i in indices]
    peak, max_dd = 1.0, 0.0
    for value in values:
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1)
    return dict(start=daily[first][0], end=daily[last][0], sessions=len(indices),
                total_return=values[-1] - 1,
                ann=values[-1] ** (244.0 / len(indices)) - 1, max_dd=max_dd,
                trades=sum(start <= trade[0] <= end for trade in trades),
                inherited_holding=daily[first - 1][2] if first else None,
                carry_nav_in=carry, carry_nav_out=daily[last][1])


def collect(case, signal, histories):
    replay = next_open_replay(signal["daily"], histories, fee=FEE, slippage=SLIPPAGE)
    daily, trades = replay["daily"], replay["trades"]
    result = {k: case[k] for k in ("id", "version", "axis", "value", "center", "window", "params")}
    result.update(
        nav=replay["nav"], ann=replay["ann"], max_dd=replay["max_dd"],
        trades=len(trades), trade_legs=sum((old is not None) + (new is not None)
                                         for _, old, new in trades),
        signal_switches=signal["switches"], crash_buys=signal["crash_buys"],
        deferred_count=replay["deferred_count"], yearly=replay["yearly"],
        signal_path_sha256=digest(signal["daily"]), execution_path_sha256=digest(daily),
        periods={label: segment(daily, trades, start, end) for label, start, end in PERIODS})
    # The complete path's final NAV must equal the product of the segment factors.
    product = 1.0
    for period in result["periods"].values():
        product *= 1 + period["total_return"]
    if abs(product / result["nav"] - 1) > 1e-12:
        raise AssertionError("Segment boundaries lost or double-counted returns")
    return result


def parity_checks(cases, histories, calendar, cache):
    checks = []
    # Chosen before seeing performance: baseline, changed indicator window, and
    # changed decision threshold, over the 2024 crisis/rebound interval.
    selected = ("v9.1:baseline", "v9.1:wls_window=20", "v9:buffer=0.03")
    by_id = {case["id"]: case for case in cases}
    for name in selected:
        case = by_id[name]
        cached = signal_run(case, histories, calendar, cache, "2024-01-02", "2024-03-29")
        fresh = signal_run(case, histories, calendar, None, "2024-01-02", "2024-03-29")
        matched = all(cached[k] == fresh[k] for k in ("daily", "trades", "crash_buys"))
        cached_replay = next_open_replay(cached["daily"], histories, fee=FEE, slippage=SLIPPAGE)
        fresh_replay = next_open_replay(fresh["daily"], histories, fee=FEE, slippage=SLIPPAGE)
        matched = matched and cached_replay["daily"] == fresh_replay["daily"]
        if not matched:
            raise AssertionError("Cached/uncached mismatch: " + name)
        checks.append(dict(id=name, start="2024-01-02", end="2024-03-29",
                           sessions=len(fresh["daily"]), exact_daily_trade_crash_match=matched))
    return checks


def summarize(results):
    summaries = {}
    for version in ("v9", "v9.1"):
        base = next(r for r in results if r["id"] == version + ":baseline")
        neighbors = [r for r in results if r["version"] == version and r["axis"] != "baseline"]
        summaries[version] = dict(
            baseline_ann=base["ann"], baseline_max_dd=base["max_dd"],
            count=len(neighbors), positive_cagr=sum(r["ann"] > 0 for r in neighbors),
            above_baseline=sum(r["ann"] > base["ann"] + 1e-12 for r in neighbors),
            equal_baseline=sum(abs(r["ann"] - base["ann"]) <= 1e-12 for r in neighbors),
            cagr_range=[min(r["ann"] for r in neighbors), max(r["ann"] for r in neighbors)],
            max_dd_range=[min(r["max_dd"] for r in neighbors), max(r["max_dd"] for r in neighbors)],
            terminal_ratio_range=[min(r["nav"] / base["nav"] for r in neighbors),
                                  max(r["nav"] / base["nav"] for r in neighbors)],
            axes={})
        for axis in sorted({r["axis"] for r in neighbors}):
            rows = [r for r in neighbors if r["axis"] == axis]
            all_rows = sorted(rows + [dict(base, value=rows[0]["center"])], key=lambda r: r["value"])
            summaries[version]["axes"][axis] = [
                dict(value=r["value"], ann=r["ann"], max_dd=r["max_dd"],
                     nav_ratio_to_baseline=r["nav"] / base["nav"],
                     period_cagr_delta_pp={p: 100 * (r["periods"][p]["ann"] - base["periods"][p]["ann"])
                                           for p, _, _ in PERIODS}) for r in all_rows]
    return summaries


def research():
    started = time.monotonic()
    cases = scenarios()
    inputs = [BASE / "data" / (code + ".csv") for code in BASE_CODES]
    sources = [BASE / name for name in ("research_parameter_stability.py", "strategy_versions.py",
                "strategy.py", "backtest.py", "market_data.py", "v9_robustness.py", "evaluate_versions.py")]
    hashes = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs + sources}
    histories = {code: [row for row in rows if row[0] <= END]
                 for code, rows in load_local_histories().items()}
    if any(not rows or rows[-1][0] < END for rows in histories.values()):
        raise ValueError("All baseline assets must have local coverage through " + END)
    calendar = [row[0] for row in histories["510300"]]
    original_stock, original_global = strategy.STOCK_POOL, strategy.GLOBAL_POOL
    strategy.STOCK_POOL, strategy.GLOBAL_POOL = list(STOCK_CODES), list(GLOBAL_CODES)
    cache, results = RankCache(histories), []
    try:
        for i, case in enumerate(cases, 1):
            signal = signal_run(case, histories, calendar, cache)
            row = collect(case, signal, histories)
            results.append(row)
            print("%2d/%d %-30s CAGR %+6.2f%% DD %6.2f%% trades %d" % (
                i, len(cases), case["id"], 100 * row["ann"], 100 * row["max_dd"], row["trades"]), flush=True)
        parity = parity_checks(cases, histories, calendar, cache)
    finally:
        strategy.STOCK_POOL, strategy.GLOBAL_POOL = original_stock, original_global
    final_hashes = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs + sources}
    if final_hashes != hashes:
        raise AssertionError("Source code or data changed during the run; rerun on a fixed snapshot")
    return dict(
        start=START, end=END, fee_per_side=FEE, slippage_per_side=SLIPPAGE,
        protocol="Fixed one-factor neighborhoods; no new parameters selected from outcomes.",
        data_and_code_sha256=hashes,
        used_history_sha256={code: digest(rows) for code, rows in histories.items()},
        cases_registered_before_execution=cases,
        cache=dict(hits=cache.hits, misses=cache.misses,
                   entries_by_wls_window=dict(Counter(key[0] for key in cache.entries)),
                   key="WLS window + all indicator-relevant frozen engine args + date; fixed histories identity",
                   excluded_decision_only_params=sorted(DECISION_ONLY), parity=parity),
        elapsed_seconds=round(time.monotonic() - started, 2),
        caveats=[
            "All periods were already used for strategy research; this is in-sample sensitivity, not OOS validation.",
            "Neighbors are deliberately limited, not an exhaustive or probabilistic overfitting test.",
            "All performance segments inherit positions and wealth from one 2014-start path; no segment restarts.",
            "Each scenario's fixed closing target path is repriced at next open with 1bp commission plus 10bp adverse slippage per side.",
            "Missing quotes defer fills without re-solving targets from actual deferred holdings; limit-lock and intraday liquidity are not modeled.",
            "WLS window changes the score only; other window-sensitive feature switches are disabled in the frozen configurations.",
            "Annualization uses 244 sessions per year, including the initial uninvested signal day; trade counts include first entry.",
            "Point-in-time asset-selection and adjusted-history biases remain; no future return guarantee follows from a smooth neighborhood.",
        ], results=results, summaries=summarize(results))


def markdown(report):
    out = ["# v9 / v9.1 参数邻域稳定性", "",
           "固定区间：%s 至 %s；次日开盘重放，单边佣金 1bp + 滑点 10bp。" % (START, END),
           "预先限定单因素邻域，不据收益挑选新参数。全部历史均已参与过研究，以下是样本内敏感性，不是样本外验证。", "",
           "| 版本/场景 | 年化 | 最大回撤 | 成交切换次数 | 2014–17年化 | 2018–21年化 | 2022–25年化 | 2026年化至截止日 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["results"]:
        out.append("| %s | %.2f%% | %.2f%% | %d | %s |" % (
            r["id"], r["ann"] * 100, r["max_dd"] * 100, r["trades"],
            " | ".join("%.2f%%" % (r["periods"][label]["ann"] * 100) for label, _, _ in PERIODS)))
    out += ["", "## 数值摘要", ""]
    for version, s in report["summaries"].items():
        out.append("- %s 基线年化 %.2f%%、回撤 %.2f%%；%d 个邻居年化范围 %.2f%%～%.2f%%，终值为基线 %.3fx～%.3fx；%d 个高于基线、%d 个持平。" % (
            version, s["baseline_ann"] * 100, s["baseline_max_dd"] * 100, s["count"],
            s["cagr_range"][0] * 100, s["cagr_range"][1] * 100,
            s["terminal_ratio_range"][0], s["terminal_ratio_range"][1],
            s["above_baseline"], s["equal_baseline"]))
    out += ["", "## 分段与缓存核验", "",
            "分段保留连续净值与跨年持仓。每段收益乘积已断言等于全段终值；分段回撤从该段期初净值重新计算，不代表全程峰值下的回撤。JSON 同时保存分段累计收益、回撤和成交次数。",
            "WLS 20/25/30 独立缓存，其他影响指标的配置纳入缓存键；缓存只保存指标，不缓存仓位、锁仓或决策状态。三个预先指定的短区间情景与未缓存引擎逐日净值、交易、抄底及开盘重放结果完全一致。",
            "", "## 方法限制", ""]
    out += ["- " + text for text in report["caveats"]]
    out += ["", "复现：`python3 -B research_parameter_stability.py`。仅写本报告及配套 JSON，不修改生产策略或实盘状态。", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/parameter_stability.json")
    args = parser.parse_args()
    report = research()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print("Written %s (%.1fs)" % (output, report["elapsed_seconds"]), flush=True)


if __name__ == "__main__":
    main()
