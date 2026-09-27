"""Fixed, offline crash-channel ablation; no parameter selection or live state I/O.

Reproduce with: python3 research_crash_ablation.py
The first run builds a shared cache of price indicators, never decisions or
crash triggers. All variants rerun portfolio/lock state and threshold checks.
"""
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import backtest
import strategy
from evaluate_versions import _reset_lab, metrics, next_open_replay
from strategy_versions import BASE_CODES, GLOBAL_CODES, STOCK_CODES, backtest_kwargs, load_local_histories

BASE = Path(__file__).resolve().parent
START, END = "2014-01-01", "2026-09-24"
FEE, SLIPPAGE = .0001, .001
QVIX = {"fear_qz": 2.5, "fear_m5": -.04}
VOLUME = {"crash_volu": 2.0, "cv_m5": -.04, "cv_dep": .10}
PRESETS = {
    "v9": {},
    "v9.1": {},
    "v9.2": dict(QVIX, **VOLUME),
    "qvix_only": dict(QVIX),
    "volume_only": dict(VOLUME),
    "no_crash": {},
    "v9.2_qvix_lag1": dict(QVIX, **VOLUME, fear_lag=1),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fear(lab):
    with (BASE / "data/qvix50.csv").open(newline="", encoding="utf-8") as f:
        rows = [(r[0], float(r[1])) for r in csv.reader(f) if r and r[0] <= END]
    if not rows or rows[-1][0] != END:
        raise ValueError("QVIX does not cover the fixed end date")
    lab.FEAR.update(qd=[], qz=[], qv=[], rd=[], r5=[])
    for i, (date, value) in enumerate(rows):
        if i < 120:
            continue
        previous = [v for _, v in rows[max(0, i - 250):i]]
        mean = sum(previous) / len(previous)
        std = (sum((v - mean) ** 2 for v in previous) / len(previous)) ** .5
        if std:
            lab.FEAR["qd"].append(date)
            lab.FEAR["qz"].append((value - mean) / std)
            lab.FEAR["qv"].append(value)


def channel_labels(lab, cfg, date, ind):
    channels = []
    if ind["mom5"] <= -.08 and ind["dist_ma250"] < -.20:
        channels.append("deep_drop")
    if cfg.get("fear_qz") and lab._fear_active(date) and ind["mom5"] <= cfg["fear_m5"] \
            and ind["dist_ma250"] < -cfg.get("fear_depth", .20):
        channels.append("qvix")
    if cfg.get("crash_volu") and ind["vol_ratio20"] >= cfg["crash_volu"] \
            and ind["mom5"] <= cfg["cv_m5"] and ind["dist_ma250"] < -cfg["cv_dep"]:
        channels.append("volume")
    return channels


def event_rows(lab, cfg, result, cache, histories, baseline_events):
    """Label actual executed engine triggers, not a census of potential signals."""
    events = []
    for date, code in result["crash_buys"]:
        ind = dict(cache[date])[code]
        rows = [r for r in histories[code] if r[0] <= date]
        avg = sum(r[3] for r in rows[-21:-1]) / 20
        volume_ratio = rows[-1][3] / avg if avg > 0 else 0
        if not math.isclose(volume_ratio, ind["vol_ratio20"], rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("Cached volume ratio differs from raw history")
        channels = channel_labels(lab, cfg, date, ind)
        if not channels:
            raise AssertionError("An actual crash trigger has no enabled qualifying channel")
        events.append(dict(date=date, code=code, channels=channels, mom5=ind["mom5"],
                           dist_ma250=ind["dist_ma250"], volume_ratio20=volume_ratio,
                           same_event_as_v91=(date, code) in baseline_events))
    return events


def divergence_episodes(candidate, baseline, candidate_events, baseline_events):
    """Exact additive log-return attribution to contiguous differing path episodes.

    Episodes include reconvergence-day costs. These are descriptive path
    contributions, not independent events or causal leave-one-trade-out tests.
    """
    if [r[0] for r in candidate] != [r[0] for r in baseline]:
        raise ValueError("Daily dates must match for paired attribution")
    episodes, active = [], None
    prior_c = prior_b = 1.0
    for i, (c, b) in enumerate(zip(candidate, baseline)):
        date, cn, ch = c
        _, bn, bh = b
        excess = math.log(cn / prior_c) - math.log(bn / prior_b)
        differs = ch != bh or abs(excess) > 1e-12
        if differs and active is None:
            active = dict(start=date, end=date, sessions=0, log_excess=0.0,
                          trigger_window_start=candidate[max(0, i - 1)][0])
        if active is not None and differs:
            active["end"] = date
            active["sessions"] += 1
            active["log_excess"] += excess
        elif active is not None:
            episodes.append(active)
            active = None
        prior_c, prior_b = cn, bn
    if active is not None:
        episodes.append(active)
    for ep in episodes:
        ep["relative_gain"] = math.expm1(ep["log_excess"])
        lo, hi = ep.pop("trigger_window_start"), ep["end"]
        ep["candidate_crash_events"] = [e for e in candidate_events if lo <= e["date"] <= hi]
        ep["v91_crash_events"] = [e for e in baseline_events if lo <= e["date"] <= hi]
    total = math.log(candidate[-1][1] / baseline[-1][1])
    if not math.isclose(sum(e["log_excess"] for e in episodes), total, abs_tol=1e-9):
        raise AssertionError("Episode contributions do not reconcile to total excess")
    positive = sorted((e for e in episodes if e["log_excess"] > 0),
                      key=lambda e: e["log_excess"], reverse=True)
    positive_sum = sum(e["log_excess"] for e in positive)
    top = {}
    for count in (1, 3, 5):
        contribution = sum(e["log_excess"] for e in positive[:count])
        top[str(count)] = dict(log_contribution=contribution,
                              fraction_of_gross_positive_log_excess=contribution / positive_sum if positive_sum else None,
                              ratio_if_omitting_contribution=math.exp(total - contribution))
    return dict(relative_terminal_nav=math.exp(total), total_log_excess=total,
                positive_episode_count=len(positive), negative_episode_count=sum(e["log_excess"] < 0 for e in episodes),
                gross_positive_log_excess=positive_sum,
                gross_negative_log_excess=sum(e["log_excess"] for e in episodes if e["log_excess"] < 0),
                top_positive=top, episodes=episodes)


def run():
    sys.path.insert(0, str(BASE / "v10"))
    import lab
    strategy.STOCK_POOL = list(STOCK_CODES)
    strategy.GLOBAL_POOL = list(GLOBAL_CODES)
    histories = {c: [r for r in rows if r[0] <= END] for c, rows in load_local_histories().items()}
    if any(not rows or rows[-1][0] != END for rows in histories.values()):
        raise ValueError("Every baseline instrument must cover the fixed end date")
    calendar = [r[0] for r in histories["510300"]]
    lab.H, lab.CAL = histories, calendar
    load_fear(lab)
    original_rank, cache = strategy.rank, {}

    def cached_rank(h, on_date=None, live_prices=None):
        if live_prices is not None or h is not histories or on_date is None:
            raise ValueError("Cache is restricted to one dated local history snapshot")
        lab.STATE["last_date"] = on_date
        if on_date not in cache:
            saved = lab.CFG.get("crash_volu")
            lab.CFG["crash_volu"] = 2.0  # Only asks indicators to compute volume ratio.
            try:
                cache[on_date] = original_rank(h, on_date=on_date)
            finally:
                if saved is None:
                    lab.CFG.pop("crash_volu", None)
                else:
                    lab.CFG["crash_volu"] = saved
            if any("_crash_buy" in ind for _, ind in cache[on_date]):
                raise AssertionError("A decision marker must never be stored in the indicator cache")
        # Independent per-run indicator dictionaries prevent incidental mutation.
        return [(code, dict(ind)) for code, ind in cache[on_date]]

    strategy.rank = cached_rank
    raw_results, results, fidelity = {}, {}, {}
    try:
        # Prime the common indicators on explicit fixed v9.1 defaults.
        for name, cfg in PRESETS.items():
            _reset_lab(lab, cfg)
            if name in ("v9", "v9.1", "no_crash"):
                params = backtest_kwargs("v9" if name == "v9" else "v9.1")
                if name == "no_crash":
                    params["crash_mom5"] = 0.0
                result = backtest.backtest(histories, calendar, start=START, end=END, **params)
            else:
                result = lab.backtest_v10(histories, calendar, start=START, end=END)
            raw_results[name] = result
            replay = next_open_replay(result["daily"], histories, fee=FEE, slippage=SLIPPAGE)
            results[name] = dict(cfg=cfg, crash_count=len(result["crash_buys"]),
                                 crash_year_counts=dict(sorted(Counter(d[:4] for d, _ in result["crash_buys"]).items())),
                                 scenarios={"same_close_ideal": dict(metrics(result["daily"]),
                                            daily=result["daily"], switches=result["switches"]),
                                            "next_open_10bps": replay})
            print("%s: crashes %d; ideal-close CAGR %.2f%%; next-open+10bp CAGR %.2f%%" % (
                name, len(result["crash_buys"]), result["ann"] * 100, replay["ann"] * 100), flush=True)

        _reset_lab(lab, {})
        anchor = lab.backtest_v10(histories, calendar, start=START, end=END)
        fidelity["lab_empty_cfg_matches_v91"] = all(anchor[k] == raw_results["v9.1"][k]
                                                     for k in ("daily", "trades", "crash_buys"))
        # A disabled volume channel must not reuse buys from a cached lower threshold.
        _reset_lab(lab, dict(VOLUME, crash_volu=1e9))
        disabled_volume = lab.backtest_v10(histories, calendar, start=START, end=END)
        fidelity["impossible_volume_threshold_matches_v91"] = all(
            disabled_volume[k] == raw_results["v9.1"][k] for k in ("daily", "trades", "crash_buys"))
        fidelity["no_crash_has_zero_crash_events"] = not raw_results["no_crash"]["crash_buys"]
        if not all(fidelity.values()):
            raise AssertionError("Ablation engine/cache validation failed: %r" % fidelity)
        baseline_events = set(map(tuple, raw_results["v9.1"]["crash_buys"]))
        for name, cfg in PRESETS.items():
            _reset_lab(lab, cfg)
            events = event_rows(lab, cfg, raw_results[name], cache, histories, baseline_events)
            results[name]["crash_events"] = events
            results[name]["new_event_count_vs_v91"] = sum(not e["same_event_as_v91"] for e in events)
            results[name]["channel_counts"] = dict(Counter(ch for e in events for ch in e["channels"]))
    finally:
        strategy.rank = original_rank
        _reset_lab(lab, {})

    comparisons = {}
    for name in results:
        if name == "v9.1":
            continue
        comparisons[name] = divergence_episodes(
            results[name]["scenarios"]["next_open_10bps"]["daily"],
            results["v9.1"]["scenarios"]["next_open_10bps"]["daily"],
            results[name]["crash_events"], results["v9.1"]["crash_events"])
    data_paths = [BASE / "data" / (c + ".csv") for c in BASE_CODES] + [BASE / "data/qvix50.csv"]
    code_paths = [BASE / p for p in ("research_crash_ablation.py", "strategy.py", "backtest.py",
                                    "v10/lab.py", "strategy_versions.py", "evaluate_versions.py")]
    return dict(start=START, end=END, costs={"per_side_commission": FEE, "per_side_slippage": SLIPPAGE},
                fidelity=fidelity,
                data_sha256={str(p.relative_to(BASE)): digest(p) for p in data_paths},
                code_sha256={str(p.relative_to(BASE)): digest(p) for p in code_paths},
                results=results, comparisons=comparisons,
                caveats=[
                    "Fixed historical ablations, no fresh OOS data or parameter optimization.",
                    "qvix_only and volume_only retain the original v9.1 deep-drop channel; no_crash disables every crash buy.",
                    "Same-close ideal uses end-of-day close/QVIX/full-day volume and cannot establish 14:50 executability.",
                    "Next-open execution assumes previous day's QVIX is published before the next open; timestamps are unavailable.",
                    "v9.2_qvix_lag1 additionally uses the previous available QVIX day at signal generation as availability sensitivity.",
                    "Next-open replay keeps the original closing target path; it does not recalculate signals from deferred holdings.",
                    "Next-open charges the first ETF buy, each sell and buy, and adverse 10bp per-side slippage plus 1bp commission.",
                    "Adjusted open prices approximate fills; no order book, price-limit or suspension liquidity reconstruction.",
                    "Trigger counts are endogenous to holdings and lock state; added event sets are not independent trades.",
                    "Divergence episodes exactly decompose paired log returns but do not identify causal per-trigger effects.",
                    "Omitting top positive episode contributions is descriptive accounting, not a simulated alternative tradable strategy.",
                    "No extensive neighborhood scan was conducted; the only extra configuration is QVIX publication-lag sensitivity.",
                ])


def markdown(report):
    lines = ["# 修复后的抄底通道消融", "", "区间：%s — %s；固定配置、本地行情只读。全部历史已用于研究，不是新的样本外验证。" % (START, END),
             "", "次开盘情景：收盘信号在次日开盘执行，单边佣金1bp、额外单边滑点10bp。",
             "", "| 配置 | 抄底次数 | 与v9.1不同事件 | 同收盘理想年化 | 次开盘成本后年化 | 次开盘最大回撤 | 次开盘终值/v9.1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    base = report["results"]["v9.1"]["scenarios"]["next_open_10bps"]["nav"]
    for name, row in report["results"].items():
        ideal, executed = row["scenarios"]["same_close_ideal"], row["scenarios"]["next_open_10bps"]
        lines.append("| %s | %d | %d | %.2f%% | %.2f%% | %.2f%% | %.3fx |" % (
            name, row["crash_count"], row["new_event_count_vs_v91"], ideal["ann"] * 100,
            executed["ann"] * 100, executed["max_dd"] * 100, executed["nav"] / base))
    lines += ["", "qvix_only/volume_only 保留 v9.1 原深跌口；no_crash 关闭全部抄底。",
              "", "## v9.2 的实际抄底事件", "", "| 年份 | v9.1 | v9.2 | 仅加QVIX | 仅加量能 |", "|---|---:|---:|---:|---:|"]
    years = sorted({year for row in report["results"].values() for year in row["crash_year_counts"]})
    for year in years:
        lines.append("| %s | %d | %d | %d | %d |" % (year, *(report["results"][name]["crash_year_counts"].get(year, 0)
                                                           for name in ("v9.1", "v9.2", "qvix_only", "volume_only"))))
    cmp = report["comparisons"]["v9.2"]
    lines += ["", "## v9.2 与 v9.1 的成本后差异集中度", "",
              "按两策略执行持仓和日收益发生差异的连续区间分段，并将每日相对对数收益精确相加。此分段会包含锁仓导致的后续路径变化，不能视作独立交易或单一触发的因果贡献。", "",
              "相对终值 %.3fx；正贡献区间 %d 个，负贡献区间 %d 个。" % (
                  cmp["relative_terminal_nav"], cmp["positive_episode_count"], cmp["negative_episode_count"])]
    for count, summary in cmp["top_positive"].items():
        share = summary["fraction_of_gross_positive_log_excess"]
        lines.append("- 最大%s个正贡献区间占全部正对数贡献 %.1f%%；仅作账面剔除后，相对终值为 %.3fx。" % (
            count, (share or 0) * 100, summary["ratio_if_omitting_contribution"]))
    lines += ["", "| 差异区间 | 相对收益 | 期间候选抄底触发 |", "|---|---:|---|"]
    for ep in sorted(cmp["episodes"], key=lambda e: abs(e["log_excess"]), reverse=True)[:12]:
        desc = "; ".join(e["date"] + " " + e["code"] + " " + "+".join(e["channels"])
                         for e in ep["candidate_crash_events"])
        lines.append("| %s — %s | %+.2f%% | %s |" % (ep["start"], ep["end"], ep["relative_gain"] * 100, desc))
    lines += ["", "## 数据可得性与验证边界", "",
              "同收盘情景读取最终日收盘、全日量和当日QVIX，不代表14:50可以用相同信息成交。次开盘改善时序假设，但QVIX历史缺发布时间，仍须假定前日值在开盘前已公开；额外lag1情景用于检验再滞后一日的影响。",
              "", "缓存仅存储价格指标和连续量比，所有触发门槛及持仓、锁仓均逐次重算。量比直接按原CSV复算核对；超高量比门槛必须退化为v9.1。检查结果：%s。" % report["fidelity"],
              "", "JSON保存每一配置两个成交情景的完整每日净值和持仓、所有抄底事件、差异区间及代码/数据哈希，可用于配对区块bootstrap。原始行情、生产状态和策略参数均未被修改。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    report = run()
    output = BASE / "reports/crash_ablation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print("Written " + str(output), flush=True)
