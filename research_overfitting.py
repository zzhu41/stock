# -*- coding: utf-8 -*-
"""Conditional uncertainty and return concentration for frozen strategy paths.

Reads the offline crash-ablation report; never fits a strategy or accesses the
network. Block bootstrap intervals describe the supplied, already selected
historical paths. They do NOT account for prior strategy/universe selection and
are not probabilities of overfitting or of future outperformance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TRADING_DAYS = 244
SEED = 20260927


def path_returns(daily):
    dates = [row[0] for row in daily]
    nav = np.asarray([row[1] for row in daily], dtype=float)
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Dates must be nonempty, unique and sorted")
    if not np.all(np.isfinite(nav)) or np.any(nav <= 0):
        raise ValueError("NAV must be finite and positive")
    return dates, np.diff(np.log(np.concatenate(([1.0], nav))))


def annualized(log_returns):
    if len(log_returns) == 0:
        raise ValueError("No returns")
    return float(np.expm1(np.sum(log_returns) * TRADING_DAYS / len(log_returns)))


def concentration(daily):
    """Attribution only: zero selected returns without recalculating decisions."""
    dates, logs = path_returns(daily)
    total = float(logs.sum())
    positive = np.flatnonzero(logs > 0)
    ordered = positive[np.argsort(-logs[positive], kind="stable")]
    best_days = [{"date": dates[i], "return": float(np.expm1(logs[i]))}
                 for i in ordered[:10]]
    neutralized = []
    for k in (5, 10, 20):
        indices = ordered[:k]
        reduced = logs.copy()
        reduced[indices] = 0
        neutralized.append(dict(days=len(indices), annualized=annualized(reduced),
                                share_net_log_growth=float(logs[indices].sum() / total)
                                if total != 0 else None))
    years = sorted(set(d[:4] for d in dates))
    yearly = {y: float(np.expm1(sum(r for d, r in zip(dates, logs) if d.startswith(y))))
              for y in years}
    leave_year_out = []
    for y in years:
        retained = np.asarray([r for d, r in zip(dates, logs) if not d.startswith(y)])
        if len(retained):
            leave_year_out.append(dict(removed_year=y, days=len(retained),
                                       annualized=annualized(retained)))
    return dict(annualized=annualized(logs), yearly=yearly, best_days=best_days,
                neutralize_best_days=neutralized, leave_one_year_out=leave_year_out)


def paired_block_bootstrap(candidate, baseline, block_size=60, draws=10000, seed=SEED):
    """Circular blocks, paired dates, exact original sample length in every draw.

    The last block is truncated to the remainder. Precomputed rolling sums make
    memory proportional to draws * number_of_blocks instead of draws * days.
    Returns CAGR DIFFERENCES (percentage-point units when multiplied by 100).
    """
    dates_a, a = path_returns(candidate)
    dates_b, b = path_returns(baseline)
    if dates_a != dates_b:
        raise ValueError("Paired paths must have exactly the same dates")
    n = len(a)
    if not 1 <= block_size <= n or draws < 2:
        raise ValueError("Invalid block size or draw count")
    rng = np.random.default_rng(seed)
    full, remainder = divmod(n, block_size)
    starts = rng.integers(0, n, size=(draws, full))
    tail_starts = rng.integers(0, n, size=draws) if remainder else None

    def resampled_annual(logs):
        padded = np.concatenate((logs, logs[:block_size]))
        sums = np.concatenate(([0.0], np.cumsum(padded)))
        rolling = sums[block_size:block_size + n] - sums[:n]
        total = rolling[starts].sum(axis=1)
        if remainder:
            total += sums[tail_starts + remainder] - sums[tail_starts]
        return np.expm1(total * TRADING_DAYS / n)

    differences = resampled_annual(a) - resampled_annual(b)
    low, high = np.quantile(differences, [.025, .975])
    return dict(block_size=block_size, draws=draws, seed=seed,
                observed_cagr_difference=annualized(a) - annualized(b),
                conditional_percentile_95=[float(low), float(high)])


def analyze(source, draws=10000):
    raw = source.read_bytes()
    report = json.loads(raw)
    paths = {name: result["scenarios"]["next_open_10bps"]["daily"]
             for name, result in report["results"].items()}
    pairs = [("v9.1", "v9"), ("v9.2", "v9.1"),
             ("qvix_only", "v9.1"), ("volume_only", "v9.1"),
             ("v9.1", "no_crash")]
    return dict(
        source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        start=report["start"], end=report["end"], costs=report["costs"],
        assumptions=[
            "All historical paths were selected using overlapping research data; no clean OOS evidence.",
            "Conditional paired circular-block bootstrap; assumes historical dependence/regimes are representative.",
            "95% intervals are NOT selection/multiple-testing adjusted, NOT PBO, and NOT future profit probabilities.",
            "20/60/120-session blocks are prespecified dependence sensitivities, not independent tests.",
            "Neutralizing best days or removing years keeps original decisions; attribution, not a tradable alternative.",
            "2026 is a partial year. Yearly returns follow one continuous path including boundary overnight returns.",
            "Next-open replay uses fixed original close targets, 1bp commission and 10bp slippage per side.",
        ],
        concentration={name: concentration(paths[name]) for name in ("v9", "v9.1", "v9.2", "no_crash")},
        paired_comparisons=[dict(candidate=a, baseline=b, intervals=[
            paired_block_bootstrap(paths[a], paths[b], block_size=k, draws=draws)
            for k in (20, 60, 120)]) for a, b in pairs],
    )


def markdown(report):
    out = ["# 固定策略路径：收益集中度与条件不确定性", "",
           "区间：%s 至 %s。次日开盘重放，单边 1bp 佣金 + 10bp 滑点。" % (report["start"], report["end"]), "",
           "## 版本差异", "",
           "以下为年化收益差（百分点），不是相对收益率。区间采用配对循环区块重采样，",
           "**只描述已选定历史路径的不确定性，未校正反复选规则/选ETF造成的偏差，不能解释为过拟合概率。**", "",
           "| 候选 − 对照 | 实际年化差 | 20日块条件95%区间 | 60日块条件95%区间 | 120日块条件95%区间 |",
           "|---|---:|---:|---:|---:|"]
    for pair in report["paired_comparisons"]:
        intervals = pair["intervals"]
        ranges = ["[%+.2f, %+.2f]" % tuple(v * 100 for v in i["conditional_percentile_95"])
                  for i in intervals]
        out.append("| %s − %s | %+.2f | %s |" % (
            pair["candidate"], pair["baseline"],
            intervals[0]["observed_cagr_difference"] * 100, " | ".join(ranges)))
    out += ["", "## 收益集中度", "",
            "将收益最高的交易日收益置零，其他日期和原决策保持不变。用于说明收益依赖程度，",
            "不是可交易回测，也不能单凭这一项认定过拟合。", "",
            "| 版本 | 原年化 | 最佳5日置零 | 最佳10日置零 | 最佳20日置零 | 最佳10日占净对数增长 |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, c in report["concentration"].items():
        zeroed = c["neutralize_best_days"]
        share = zeroed[1]["share_net_log_growth"]
        share_text = "—" if share is None else "%.2f%%" % (share * 100)
        out.append("| %s | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %s |" % (
            name, c["annualized"] * 100,
            *(x["annualized"] * 100 for x in zeroed),
            share_text))
    out += ["", "逐年收益、逐个移除年份的归因结果和最佳交易日保存在同名 JSON。",
            "2026 年为未结束年份；本报告不将它与完整年份等量比较。", "",
            "## 复现及边界", "", "```bash",
            "python3.8 -B research_crash_ablation.py",
            "python3.8 -B research_overfitting.py", "```", "",
            "所有历史区间已经参与策略研究。本次没有重新挑选正式参数，未修改生产信号或真实账本。",
            "成本重放仍缺盘口、涨跌停、历史14:50价格和可靠的QVIX发布时间；不等同实际可成交收益。", "",
            "本项目缺少完整且可重算的历次候选逐日收益矩阵及选择记录，因此不报告看似精确的PBO或选型校正显著性。",
            "方法背景：[回测过拟合概率](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)、",
            "[Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)。", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("reports/crash_ablation.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/overfitting_diagnostics.json"))
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    report = analyze(args.source, draws=args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
