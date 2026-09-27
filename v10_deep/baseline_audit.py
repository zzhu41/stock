"""Read-only v9.2 path anatomy on the frozen corrected close-return vintage.

Run: python3.8 -B -m v10_deep.baseline_audit
No scanner, production modules, network service or protected-file writer is
imported. Only this module's JSON/Markdown diagnostic reports are written.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from functools import lru_cache
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


BASE = Path(__file__).resolve().parent
SOURCE = BASE.parent / "v10_h_close"
PATHS = SOURCE / "corrected_results/selected_paths.json"
MANIFEST = SOURCE / "corrected_manifest.json"
FEE_FACTOR = .9998
YEAR = 244
EXPECTED_BASELINE_CAGR = .46317489386091726
STOCK = ("159915", "588080", "510300", "510500", "563300", "512400", "512890")
GLOBAL = ("513100", "513120")
GOLD, CASH = "518880", "511880"
NAMES = dict(zip(STOCK + GLOBAL + (GOLD, CASH),
                ("创业板", "科创50", "沪深300", "中证500", "中证2000", "有色", "红利低波",
                 "纳指", "港股创新药", "黄金", "货币")))
PERIODS = (("full", "2014-01-01", "2026-09-24"),
           ("selection_2014_2025", "2014-01-01", "2025-12-31"),
           ("recent_2022_2025", "2022-01-01", "2025-12-31"),
           ("2026_ytd", "2026-01-01", "2026-09-24"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role(code):
    return "stock" if code in STOCK else "global" if code in GLOBAL else "gold" if code == GOLD else "cash"


class Inputs:
    def __init__(self):
        manifest = json.loads(MANIFEST.read_text())
        self.rows, self.dates, self.prices = {}, {}, {}
        self.inputs = [PATHS, MANIFEST, Path(__file__).resolve()]
        for code in NAMES:
            path = SOURCE / "corrected_snapshots" / (code + ".csv")
            if sha(path) != manifest["assets"][code]["sha256"]:
                raise ValueError("Corrected snapshot fingerprint differs: " + code)
            with path.open(newline="") as stream:
                rows = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(stream) if r]
            if [r[0] for r in rows] != sorted({r[0] for r in rows}) or rows[-1][0] != "2026-09-24":
                raise ValueError("Unexpected snapshot dates")
            self.rows[code] = rows
            self.dates[code] = [r[0] for r in rows]
            self.prices[code] = {r[0]: r[2] for r in rows}
            self.inputs.append(path)
        qpath = SOURCE / "snapshots/qvix50.csv"
        with qpath.open(newline="") as stream:
            self.qvix = [(r[0], float(r[1])) for r in csv.reader(stream) if r]
        self.qdates = [r[0] for r in self.qvix]
        self.inputs.append(qpath)

    @lru_cache(maxsize=None)
    def features(self, code, day):
        ds = self.dates.get(code, ())
        i = bisect_left(ds, day)
        if i >= len(ds) or ds[i] != day or i < 269:
            return None
        rows = self.rows[code]
        closes = [r[2] for r in rows[i - 269:i + 1]]
        returns = [closes[j] / closes[j - 1] - 1 for j in range(len(closes) - 20, len(closes))]
        mean = sum(returns) / 20
        vol = (sum((r - mean) ** 2 for r in returns) / 20) ** .5
        values, weights = closes[-25:], list(range(1, 26))
        weight_sum = sum(weights)
        mx = sum(j * w for j, w in enumerate(weights)) / weight_sum
        my = sum(p * w for p, w in zip(values, weights)) / weight_sum
        xy = sum(w * (j - mx) * (p - my) for j, (p, w) in enumerate(zip(values, weights)))
        xx = sum(w * (j - mx) ** 2 for j, w in enumerate(weights))
        ma = sum(closes[-250:]) / 250
        average_volume = sum(r[3] for r in rows[i - 20:i]) / 20
        return dict(mom5=closes[-1] / closes[-6] - 1, mom20=closes[-1] / closes[-21] - 1,
                    ret1=returns[-1], vol20=vol, above_ma=closes[-1] > ma,
                    dist_ma250=closes[-1] / ma - 1,
                    wls25_score=xy / xx / my * 250 / vol if vol > 0 and my > 0 else 0.,
                    volume_ratio20=rows[i][3] / average_volume if average_volume else 0.)

    @lru_cache(maxsize=None)
    def fear(self, day):
        i = bisect_left(self.qdates, day)
        if i >= len(self.qdates) or self.qdates[i] != day or i < 120:
            return dict(available=False, active=False, z=None)
        previous = [v for _, v in self.qvix[max(0, i - 250):i]]
        mean = sum(previous) / len(previous)
        std = (sum((v - mean) ** 2 for v in previous) / len(previous)) ** .5
        z = (self.qvix[i][1] - mean) / std if std else 0.
        return dict(available=True, active=z >= 2.5, z=z)

    def forward_return(self, code, calendar, index, lag):
        if code not in self.prices or index + lag >= len(calendar):
            return None
        start, end = self.prices[code].get(calendar[index]), self.prices[code].get(calendar[index + lag])
        return end / start - 1 if start and end else None


def decompose(result, data):
    daily = result["daily"]
    events, mark = [], None
    for i, (day, nav, holding) in enumerate(daily):
        prior_nav = daily[i - 1][1] if i else 1.
        previous = daily[i - 1][2] if i else None
        factor = 1.
        if i and previous and day in data.prices[previous]:
            factor = data.prices[previous][day] / mark
            mark = data.prices[previous][day]
        changed = bool(i and holding != previous)
        expected = factor * (FEE_FACTOR if changed else 1.)
        if not math.isclose(nav / prior_nav, expected, rel_tol=1e-11, abs_tol=1e-12):
            raise AssertionError("Price/fee path does not reconcile: " + day)
        events.append(dict(date=day, nav=nav, prior_nav=prior_nav, earning_asset=previous,
                           prior_date=daily[i - 1][0] if i else None,
                           holding=holding, market_log=math.log(factor),
                           fee_log=math.log(FEE_FACTOR) if changed else 0.,
                           net_log=math.log(nav / prior_nav), net_return=nav / prior_nav - 1,
                           switched=changed))
        if holding != previous and holding:
            mark = data.prices[holding][day]
    if sum(e["switched"] for e in events) != result["switches"]:
        raise AssertionError("Actual switch count does not reconcile")
    return events


def period(events, start, end):
    rows = [e for e in events if start <= e["date"] <= end]
    if not rows:
        return None
    n, initial = len(rows), rows[0]["prior_nav"]
    log_growth = sum(e["net_log"] for e in rows)
    peak_date = rows[0]["prior_date"] or rows[0]["date"]
    peak, worst = initial, dict(value=0., peak_date=peak_date, trough_date=rows[0]["date"])
    assets, roles, exposure = defaultdict(float), defaultdict(float), Counter()
    for e in rows:
        if e["earning_asset"]:
            assets[e["earning_asset"]] += e["market_log"]
            roles[role(e["earning_asset"])] += e["market_log"]
        exposure["uninvested" if not e["holding"] else "cash_etf" if e["holding"] == CASH else "risk"] += 1
        if e["nav"] > peak:
            peak, peak_date = e["nav"], e["date"]
        if e["nav"] / peak - 1 < worst["value"]:
            worst = dict(value=e["nav"] / peak - 1, peak_date=peak_date, trough_date=e["date"])
    switches = sum(e["switched"] for e in rows)
    fee_log = switches * math.log(FEE_FACTOR)
    if not math.isclose(sum(assets.values()) + fee_log, log_growth, abs_tol=1e-10):
        raise AssertionError("Attribution mismatch")
    cagr = math.expm1(log_growth * YEAR / n)
    no_fee_cagr = math.expm1((log_growth - fee_log) * YEAR / n)
    top = sorted(rows, key=lambda e: e["net_log"], reverse=True)
    return dict(start=rows[0]["date"], end=rows[-1]["date"], sessions=n,
                total_return=math.expm1(log_growth), cagr=cagr, max_dd=worst["value"], drawdown=worst,
                switches=switches, switches_per_year=switches * YEAR / n,
                cost_multiplier=math.exp(fee_log), fixed_path_zero_fee_cagr=no_fee_cagr,
                commission_cagr_drag=cagr - no_fee_cagr,
                asset_log_contribution=dict(assets), role_log_contribution=dict(roles), fee_log_contribution=fee_log,
                exposure={k: v / n for k, v in exposure.items()},
                top_positive_log_share={str(k): sum(e["net_log"] for e in top[:k] if e["net_log"] > 0) / log_growth
                                        if log_growth else None for k in (1, 5, 10)})


def anatomy(result, data):
    daily = result["daily"]
    calendar = [r[0] for r in daily]
    indices = {day: i for i, day in enumerate(calendar)}
    crash_set = set(map(tuple, result["policy_metadata"]["crash_buys"]))
    switches = []
    for i in range(1, len(daily)):
        day, nav, target = daily[i]
        previous = daily[i - 1][2]
        if target == previous:
            continue
        old, new = data.features(previous, day), data.features(target, day)
        benchmark = data.features("510300", day)
        bull = benchmark["above_ma"] if benchmark else True
        crash = (day, target) in crash_set
        if crash:
            reason = "crash_override"
        elif previous != CASH and old and old["ret1"] <= -.04:
            reason = "panic_exit"
        elif not bull and previous in STOCK:
            reason = "bear_regime_exit"
        elif old and old["mom20"] <= 0:
            reason = "negative_momentum_exit"
        elif old and old["mom20"] > .40 and old["mom5"] <= 0:
            reason = "overheat_exit"
        elif previous in (None, CASH):
            reason = "cash_entry"
        else:
            reason = "rank_rotation_or_fallback"
        target_gate = .0 if bull else .07
        fallback = bool(not crash and new and new["mom20"] <= target_gate and target != CASH)
        switches.append(dict(date=day, index=i, previous=previous, target=target,
                             reason_proxy=reason, bull=bull, target_below_ordinary_momentum_gate=fallback,
                             previous_features=old, target_features=new,
                             previous_asset_forward5=data.forward_return(previous, calendar, i, 5),
                             target_asset_forward5=data.forward_return(target, calendar, i, 5)))
    by_entry = {t["date"]: t for t in switches}
    entries = [0] + [t["index"] for t in switches]
    episodes = []
    for j, begin in enumerate(entries):
        closed = j + 1 < len(entries)
        end = entries[j + 1] if closed else len(daily) - 1
        day, start_nav, code = daily[begin]
        gross = data.prices[code][calendar[end]] / data.prices[code][day] - 1
        episodes.append(dict(entry_date=day, exit_date=calendar[end], code=code, name=NAMES[code],
                             entry_index=begin, exit_index=end, sessions=end - begin,
                             closed=closed, gross_return=gross, net_return=daily[end][1] / start_nav - 1,
                             crash_entry=(day, code) in crash_set,
                             entry_reason_proxy=by_entry.get(day, {}).get("reason_proxy", "initial")))
    roundtrips = []
    for j in range(1, len(episodes) - 1):
        e = episodes[j]
        old = episodes[j - 1]["code"]
        if not e["closed"] or e["sessions"] > 5 or episodes[j + 1]["code"] != old:
            continue
        reference = data.forward_return(old, calendar, e["entry_index"], e["sessions"])
        if reference is not None:
            roundtrips.append(dict(e, previous_asset=old, stay_in_previous_gross_return=reference,
                                  relative_wealth_factor=(1 + e["gross_return"]) * FEE_FACTOR ** 2 / (1 + reference)))
    crashes = []
    for day, code in sorted(crash_set):
        i = indices[day]
        prior = daily[i - 1][2] if i else None
        feature, fear = data.features(code, day), data.fear(day)
        channels = []
        if feature["mom5"] <= -.08 and feature["dist_ma250"] < -.20:
            channels.append("deep")
        if fear["active"] and feature["mom5"] <= -.04 and feature["dist_ma250"] < -.20:
            channels.append("qvix")
        if feature["volume_ratio20"] >= 2 and feature["mom5"] <= -.04 and feature["dist_ma250"] < -.10:
            channels.append("volume")
        if not channels:
            raise AssertionError("Recorded crash has no reconstructed trigger: " + day)
        five = data.forward_return(code, calendar, i, 5)
        prior_five = data.forward_return(prior, calendar, i, 5)
        lock_navs = [daily[k][1] / daily[i][1] for k in range(i + 1, min(i + 6, len(daily)))]
        crashes.append(dict(date=day, code=code, previous_asset=prior, channels=channels,
                            entry_features=feature, fear=fear, five_session_asset_return=five,
                            previous_asset_five_session_return=prior_five,
                            relative_five_session_factor=(1 + five) * FEE_FACTOR ** 2 / (1 + prior_five)
                            if five is not None and prior_five is not None else None,
                            worst_portfolio_move_during_five_sessions=min([1.] + lock_navs) - 1,
                            portfolio_five_session_return=lock_navs[-1] - 1 if len(lock_navs) == 5 else None))
    return dict(switches=switches, episodes=episodes, roundtrips=roundtrips, crashes=crashes)


def distribution(values):
    return dict(count=len(values), positive=sum(x > 0 for x in values),
                mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None)


def execution_period(audit, start, end):
    trades = [t for t in audit["switches"] if start <= t["date"] <= end]
    episodes = [e for e in audit["episodes"] if e["closed"] and start <= e["exit_date"] <= end]
    trips = [t for t in audit["roundtrips"] if start <= t["entry_date"] <= end]
    crash = [c for c in audit["crashes"] if start <= c["date"] <= end]
    ordinary = [t for t in trips if not t["crash_entry"]]
    return dict(
        switch_reasons=dict(Counter(t["reason_proxy"] for t in trades)),
        entries_below_ordinary_gate=sum(t["target_below_ordinary_momentum_gate"] for t in trades),
        closed_episodes=len(episodes), short_episodes={str(n): distribution([e["net_return"] for e in episodes if e["sessions"] <= n])
                                                      for n in (1, 3, 5)},
        worst_closed_episodes=sorted(episodes, key=lambda e: e["net_return"])[:10],
        roundtrips_within_five_sessions=distribution([t["relative_wealth_factor"] - 1 for t in trips]),
        ordinary_roundtrips=distribution([t["relative_wealth_factor"] - 1 for t in ordinary]),
        ordinary_roundtrips_by_entry_reason={reason: distribution([t["relative_wealth_factor"] - 1 for t in ordinary
                                            if t["entry_reason_proxy"] == reason])
                                            for reason in sorted({t["entry_reason_proxy"] for t in ordinary})},
        worst_roundtrips=sorted(trips, key=lambda t: t["relative_wealth_factor"])[:10],
        best_roundtrips=sorted(trips, key=lambda t: t["relative_wealth_factor"], reverse=True)[:10],
        crash_entries=len(crash), crash_channel_combinations=dict(Counter("+".join(c["channels"]) for c in crash)),
        crash_five_session_gross=distribution([c["five_session_asset_return"] for c in crash if c["five_session_asset_return"] is not None]),
        crash_vs_displaced_asset=distribution([c["relative_five_session_factor"] - 1 for c in crash if c["relative_five_session_factor"] is not None]),
        worst_crash_entries=sorted(crash, key=lambda c: c["five_session_asset_return"] if c["five_session_asset_return"] is not None else 0)[:8])


def audit():
    data = Inputs()
    hashes = {str(path.relative_to(BASE.parent)): sha(path) for path in data.inputs}
    paths = json.loads(PATHS.read_text())
    baseline, reference = paths["c_v92"], paths["c_v91"]
    events = decompose(baseline, data)
    reference_events = decompose(reference, data)
    periods = {name: period(events, start, end) for name, start, end in PERIODS}
    if not math.isclose(periods["full"]["cagr"], EXPECTED_BASELINE_CAGR, abs_tol=1e-10):
        raise AssertionError("The baseline differs from the fixed corrected v9.2 target")
    anatomy_result = anatomy(baseline, data)
    execution = {name: execution_period(anatomy_result, start, end) for name, start, end in PERIODS}
    yearly = {}
    for year in sorted(baseline["yearly"]):
        lo, hi = year + "-01-01", year + "-12-31"
        b, ref = period(events, lo, hi), period(reference_events, lo, hi)
        if not math.isclose(b["total_return"], baseline["yearly"][year], abs_tol=1e-10):
            raise AssertionError("Year boundary mismatch")
        yearly[year] = dict(metrics=b, v91_total_return=ref["total_return"],
                            v92_vs_v91_factor=(1 + b["total_return"]) / (1 + ref["total_return"]),
                            execution=execution_period(anatomy_result, lo, hi))
    final_hashes = {str(path.relative_to(BASE.parent)): sha(path) for path in data.inputs}
    if hashes != final_hashes:
        raise AssertionError("An input changed during this read-only diagnostic")
    return dict(
        baseline="c_v92", reference="c_v91", start=events[0]["date"], end=events[-1]["date"],
        data_view="corrected close total-return indices", fee_per_side=.0001, slippage=0.,
        checks=dict(daily_price_fee_reconstruction=True, yearly_boundaries=True,
                    all_recorded_crash_channels_reconstructed=True, input_hashes_unchanged=True),
        input_sha256=hashes, periods=periods, execution=execution, yearly=yearly, detail=anatomy_result,
        diagnostic_findings=dict(
            qvix_only_recorded_entries=sum(c["channels"] == ["qvix"] for c in anatomy_result["crashes"]),
            qvix_redundancy="No exclusive QVIX trigger in the recorded path is a simplification-control hypothesis; verify by resimulation before claiming identical behavior.",
            fee_only_full_cagr_drag=periods["full"]["commission_cagr_drag"],
            recent_roundtrip_counterexample="Ordinary quick reversals are mostly adverse in 2022-2025 but mostly beneficial in 2026. A blanket holding floor is not supported.",
        ),
        mechanism_hypotheses=[
            dict(id="rotation_quality", idea="Add a fixed confidence/confirmation requirement only to healthy ordinary rank rotations, retaining panic and crash exceptions.",
                 economic_rationale="Short-lived leader changes can reflect estimation noise rather than a new trend.",
                 audit_test="Compare avoided ordinary whipsaws with missed positive roundtrips; test all dates, never selected losing dates.",
                 limitation="Generic minimum holding/cooldowns can suppress valuable fast switches; 2026 has both helpful and harmful quick reversals."),
            dict(id="crash_displacement", idea="Make event override conditional on whether the existing asset already offers comparable recovery/trend quality.",
                 economic_rationale="Buying a bouncing asset is insufficient if it displaces an even stronger rebound or hedge.",
                 audit_test="Use entry-time information only; report 5-session return relative to displaced holdings and the full resimulated path.",
                 limitation="Five-day stay-put returns are hindsight diagnostics, not inputs; event sample is small and clustered."),
            dict(id="exit_symmetry", idea="Test a small fixed volatility-normalized emergency threshold against the existing universal 4% exit, separately from ordinary rotation confirmation.",
                 economic_rationale="A 4% move represents different tail severity across Nasdaq, gold and high-volatility domestic ETFs; insurance and foregone rebounds must be measured together.",
                 audit_test="Report tails and foregone rebound gains together; retain the original fee and same-close execution assumptions.",
                 limitation="Do not improve mean return simply by removing tail protection or by tuning to isolated crisis dates."),
        ],
        caveats=[
            "This is attribution of existing paths, not a new parameter scan, a validated strategy improvement or a clean OOS test.",
            "Corrected dividends include inferred cash flows and immediate free ex-date reinvestment; the historical same-close fill remains idealized.",
            "Reason proxies are reconstructed from available indicators and known rule precedence, not original execution-reason logs.",
            "Episode net return assigns the next switch's whole legacy 2bp fee to the exiting episode; episodes are grouped by exit date and can cross calendar boundaries.",
            "Roundtrips can overlap. Their stay-put comparisons must not be added as attainable incremental profits.",
            "Crash five-session comparisons hold the displaced asset fixed and charge two switch factors; they do not rerun the strategy without the crash.",
            "2026 is partial and already observed; per-year CAGR annualizes this partial path, while total_return reports the actual observed gain.",
            "New mechanism tests require dated inputs, fixed budgets and explicit comparison with the same baseline; this audit does not authorize changing production.",
        ])


def markdown(report):
    out = ["# 修正后 v9.2 基线诊断", "", "只读已有路径；未重新选参或改变持仓。固定同日收盘、单边万一、无滑点，首次评估日免费。", "",
           "| 区间 | 累计收益 | 年化 | 最大回撤 | 换仓 | <=3日已平仓段 | 普通五日内往返相对原仓胜/总 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metric in report["periods"].items():
        e = report["execution"][name]
        trips = e["ordinary_roundtrips"]
        out.append("| %s | %.2f%% | %.2f%% | %.2f%% | %d | %d | %d/%d |" % (
            name, metric["total_return"] * 100, metric["cagr"] * 100, metric["max_dd"] * 100,
            metric["switches"], e["short_episodes"]["3"]["count"], trips["positive"], trips["count"]))
    full, recent, ytd = report["execution"]["full"], report["execution"]["recent_2022_2025"], report["execution"]["2026_ytd"]
    out += ["", "## 主要发现", "",
            "- 全程佣金对固定持仓路径的年化拖累约 %.2f 个百分点；主要改进空间若存在，应来自持仓选择和错误切换，而非把万一费用降到零。" %
            (-100 * report["periods"]["full"]["commission_cagr_drag"]),
            "- 2022–2025 普通五日内往返相对原仓 %d/%d 胜，平均 %.2f%%；2026 则 %d/%d 胜，平均 %.2f%%。这些往返可能重叠，不能相加当成可赚取收益。" %
            (recent["ordinary_roundtrips"]["positive"], recent["ordinary_roundtrips"]["count"], recent["ordinary_roundtrips"]["mean"] * 100,
             ytd["ordinary_roundtrips"]["positive"], ytd["ordinary_roundtrips"]["count"], ytd["ordinary_roundtrips"]["mean"] * 100),
            "- 26 次已记录危机入场中，触发组合为 %s；没有 QVIX 单独触发的记录。先验证删除冗余 QVIX 通道是否逐日同路径，不能把这一简化冒称收益提升。" % full["crash_channel_combinations"],
            "- 抄底标的五日正收益 %d/%d，但扣两次原费率后优于原持仓固定持有的只有 %d/%d；绝对反弹并不自动等于更好的资产配置。" %
            (full["crash_five_session_gross"]["positive"], full["crash_five_session_gross"]["count"],
             full["crash_vs_displaced_asset"]["positive"], full["crash_vs_displaced_asset"]["count"])]
    out += ["", "## 逐年与额外危机通道的增量", "", "v9.1 对照来自同一修正快照和已保存回测，不是重新拟合。", "",
            "| 年份 | v9.2收益 | v9.1收益 | v9.2/v9.1财富比 | 换仓 | 抄底 |", "|---|---:|---:|---:|---:|---:|"]
    for year, value in report["yearly"].items():
        m = value["metrics"]
        out.append("| %s | %.2f%% | %.2f%% | %.4fx | %d | %d |" % (
            year, m["total_return"] * 100, value["v91_total_return"] * 100, value["v92_vs_v91_factor"],
            m["switches"], value["execution"]["crash_entries"]))
    out += ["", "## 待检验的经济机制", ""]
    for item in report["mechanism_hypotheses"]:
        out += ["- **%s**：%s %s" % (item["id"], item["idea"], item["limitation"])]
    out += ["", "完整 JSON 保存逐笔持仓段、进出原因代理、危机事件条件、原持仓参照、资产/角色贡献和数据指纹。", "",
            "## 限制", ""]
    out += ["- " + text for text in report["caveats"]]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "results/baseline_audit.json")
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print("Written", args.output)
    for name, metric in report["periods"].items():
        e = report["execution"][name]
        print(name, "CAGR %.5f%% DD %.2f%% switches %d short<=3 %d ordinary-roundtrips %s crash-vs-old %s" % (
            metric["cagr"] * 100, metric["max_dd"] * 100, metric["switches"], e["short_episodes"]["3"]["count"],
            e["ordinary_roundtrips"], e["crash_vs_displaced_asset"]))


if __name__ == "__main__":
    main()
