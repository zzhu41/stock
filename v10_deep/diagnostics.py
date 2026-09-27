"""Conditional diagnostics of a complete frozen, adaptively constructed stage.

Read saved net return streams only: never simulate, shortlist, or select a new
strategy. All historical periods and this family were already researched. The
White-style statistics cannot account for the undisclosed/older search universe.
"""
import os

# Set before importing NumPy (including when launched with python -m).
for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from v10_search.statistics import white_style_max_test, walkforward_selection_diagnostic

BASE = Path(__file__).resolve().parent
TRADING_DAYS = 244
WARNINGS = [
    "The entire candidate family was adaptively designed using already known history, including knowledge of later outcomes; this is not clean out-of-sample evidence.",
    "The max test is conditional on this supplied complete stage; it does not correct all previous, discarded, or unrecorded searches, nor the adaptive construction of this family.",
    "White-style means paired, unstudentized, zero-centered circular-block maximum mean log-return differences; it is not Hansen SPA, PBO, or a probability of future profit.",
    "The four historical re-selection folds reuse a family designed with known future history; stitched model return streams omit transfers and model-switch transaction costs and are not a tradable strategy.",
    "The 2026 tail was already inspected. It is displayed separately and does not enter the bootstrap or historical re-selection calculations.",
    "Same-close decisions/fills, fee 1 bp per side with first evaluation day free, and reconstructed close total-return prices retain the original execution and dividend assumptions.",
]
PERIODS = {
    "selection_2014_2025": ("2014-01-01", "2025-12-31"),
    "early_2014_2017": ("2014-01-01", "2017-12-31"),
    "middle_2018_2021": ("2018-01-01", "2021-12-31"),
    "recent_2022_2025": ("2022-01-01", "2025-12-31"),
    "report_only_2026": ("2026-01-01", "2026-12-31"),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")


def _dates(values, label):
    dates = list(values)
    if not dates or dates != sorted(set(dates)):
        raise ValueError(label + " dates must be nonempty, unique and sorted")
    if any(not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) for day in dates):
        raise ValueError(label + " dates must be ISO calendar strings")
    return dates


def _expected_sha(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError("Frozen artifact SHA256 mismatch: " + str(path))


def _rows(rows, candidates, label):
    if len(rows) != len(candidates):
        raise ValueError(label + " must include every registered candidate")
    ordered = [None] * len(candidates)
    for row in rows:
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candidates):
            raise ValueError(label + " has an invalid matrix row index")
        if ordered[index] is not None or row["id"] != candidates[index]["id"]:
            raise ValueError(label + " row IDs/order differ from the registered matrix")
        if row.get("hash") != candidates[index].get("hash"):
            raise ValueError(label + " row configuration hash differs from registry")
        ordered[index] = row
    return ordered


def load_stage(stage, results_root=None):
    """Validate routing and frozen artifacts without importing scan/data/protect."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", stage):
        raise ValueError("Stage must be a simple directory name")
    directory = Path(results_root) / stage if results_root is not None else BASE / "results" / stage
    registry = read_json(directory / "registered_candidates.json")
    registration = read_json(directory / "registration.json")
    development = read_json(directory / "development.json")
    selection = read_json(directory / "selection.json")
    evaluation = read_json(directory / "evaluation.json")  # completion marker; never used to choose
    candidates = registry["candidates"]
    ids = [c["id"] for c in candidates]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Registered candidate IDs must be nonempty and unique")
    if registry.get("stage", stage) != stage:
        raise ValueError("Registry belongs to a different stage")
    if registration["candidate_count"] != len(ids) or registry.get("unique_count", len(ids)) != len(ids):
        raise ValueError("Frozen candidate counts disagree")
    _expected_sha(directory / "registered_candidates.json", registration["registry_sha256"])
    _expected_sha(directory / "registration.json", selection["registration_sha256"])
    _expected_sha(directory / "development.json", selection["development_sha256"])
    _expected_sha(directory / "selection_returns.npy", selection["selection_returns_sha256"])
    _expected_sha(directory / "full_returns.npy", evaluation["full_returns_sha256"])
    if evaluation["selection"] != selection:
        raise ValueError("Evaluation contains a different frozen selection")
    if evaluation["fingerprints"] != registration["fingerprints"]:
        raise ValueError("Selection and full evaluation fingerprints differ")
    rows = _rows(development["rows"], candidates, "development")
    full_rows = _rows(evaluation["rows"], candidates, "evaluation")
    dates, full_dates = _dates(development["dates"], "selection"), _dates(evaluation["dates"], "full")
    if dates[-1] > "2025-12-31" or dates[-1] > registration["train_end"]:
        raise ValueError("Selection matrix includes post-selection dates")
    if dates[0] < registration["start"] or full_dates[-1] > registration["end"]:
        raise ValueError("Matrix dates exceed frozen registration")
    if full_dates[:len(dates)] != dates:
        raise ValueError("Selection calendar must be the exact full-path prefix")
    values = np.load(directory / "selection_returns.npy", mmap_mode="r", allow_pickle=False)
    full = np.load(directory / "full_returns.npy", mmap_mode="r", allow_pickle=False)
    if values.shape != (len(ids), len(dates)) or full.shape != (len(ids), len(full_dates)):
        raise ValueError("Matrices must include exactly all registered rows and dates")
    if values.dtype != np.float64 or full.dtype != np.float64:
        raise ValueError("Frozen returns must use float64")
    if not np.array_equal(values, full[:, :len(dates)]):
        raise ValueError("Full matrix changed selection returns")
    baseline_id = selection["controls"]["v9.2"]
    selected_ids = list(dict.fromkeys(x for x in (selection["primary"], selection["guarded"]) if x))
    if baseline_id not in ids or not selected_ids or any(x not in ids for x in selected_ids):
        raise ValueError("Frozen baseline/selected ID missing from registry")
    paths = read_json(directory / "selected_paths.json")
    for candidate_id in set(selected_ids + [baseline_id]):
        path = paths[candidate_id]
        index = ids.index(candidate_id)
        if len(path["holdings"]) != len(full_dates) or not np.array_equal(path["returns"], full[index]):
            raise ValueError("Saved selected path disagrees with full matrix")
    sources = {name: sha(directory / name) for name in (
        "registered_candidates.json", "registration.json", "development.json", "selection.json",
        "evaluation.json", "selection_returns.npy", "full_returns.npy", "selected_paths.json")}
    return dict(directory=directory, registry=registry, registration=registration, candidates=candidates,
                ids=ids, dates=dates, full_dates=full_dates, values=values, full=full,
                baseline_id=baseline_id, baseline_index=ids.index(baseline_id), selected_ids=selected_ids,
                rows=rows, full_rows=full_rows, selection=selection, paths=paths, sources=sources)


def metric(returns):
    returns = np.asarray(returns, dtype=np.float64)
    if not len(returns) or not np.all(np.isfinite(returns)) or np.any(returns <= -1):
        raise ValueError("Metrics need finite, nonempty simple returns greater than -1")
    logs = np.log1p(returns)
    nav = np.exp(np.cumsum(logs))
    peaks = np.maximum.accumulate(np.r_[1., nav])[1:]
    return dict(sessions=len(returns), total_return=float(np.expm1(logs.sum())),
                cagr=float(np.expm1(logs.mean() * TRADING_DAYS)),
                max_dd=float(np.min(nav / peaks - 1)), net_log_growth=float(logs.sum()))


def paired_period(selected, baseline, dates):
    a, b = metric(selected), metric(baseline)
    delta = np.log1p(selected) - np.log1p(baseline)
    return dict(start=dates[0], end=dates[-1], selected=a, baseline=b,
                cagr_difference=a["cagr"] - b["cagr"],
                cumulative_relative_wealth=float(np.expm1(delta.sum())),
                paired_log_growth_difference=float(delta.sum()),
                positive_log_difference_days=int(np.sum(delta > 1e-15)),
                negative_log_difference_days=int(np.sum(delta < -1e-15)),
                same_return_days=int(np.sum(np.abs(delta) <= 1e-15)))


def _role(code, config):
    if code is None:
        return "uninvested_cash"
    if code == "511880":
        return "money_market_etf"
    if code == "518880":
        return "gold"
    if code in config["stock_pool"]:
        return "stock_pool"
    if code in config["global_pool"]:
        return "global_pool_exempt_from_A_share_gate"
    return "other"


def _exposure_attribution(returns, holdings, config, fee):
    logs = np.log1p(returns)
    charged = np.r_[False, np.asarray(holdings[1:], dtype=object) != np.asarray(holdings[:-1], dtype=object)]
    fees = charged.astype(np.float64) * np.log1p(-2 * fee)
    market = logs - fees
    # Closing holdings are after the trade; today's market P&L belongs to
    # yesterday's holding. Fees are separated so a bought ETF does not inherit
    # the selling ETF's close-to-close movement.
    by_asset, by_role = {}, {}
    for index, amount in enumerate(market):
        code = holdings[index - 1] if index else None
        key, role = code or "UNINVESTED", _role(code, config)
        for target, name in ((by_asset, key), (by_role, role)):
            row = target.setdefault(name, dict(market_log_growth=0., exposure_days=0))
            row["market_log_growth"] += float(amount)
            row["exposure_days"] += 1
    reconstructed = sum(r["market_log_growth"] for r in by_asset.values()) + fees.sum()
    if not np.isclose(reconstructed, logs.sum(), rtol=1e-12, atol=1e-12):
        raise AssertionError("Attribution does not sum to saved net log returns")
    return dict(by_asset=by_asset, by_role=by_role, fee_log_growth=float(fees.sum()),
                charged_switches=int(charged.sum()), total_net_log_growth=float(logs.sum()),
                convention="Market logs belong to previous close's holding; same-close switch fees shown separately; no causal rule attribution.")


def paired_diagnostics(stage):
    ids, dates, full = stage["ids"], stage["full_dates"], stage["full"]
    baseline_id, baseline_index = stage["baseline_id"], stage["baseline_index"]
    baseline = np.asarray(full[baseline_index])
    b_holdings = stage["paths"][baseline_id]["holdings"]
    calendar = np.asarray(dates)
    result = {}
    for candidate_id in stage["selected_ids"]:
        index = ids.index(candidate_id)
        selected, holdings = np.asarray(full[index]), stage["paths"][candidate_id]["holdings"]
        delta = np.log1p(selected) - np.log1p(baseline)
        total = float(delta.sum())
        periods = {"full": paired_period(selected, baseline, dates)}
        for name, (start, end) in PERIODS.items():
            mask = (calendar >= start) & (calendar <= end)
            if np.any(mask):
                periods[name] = paired_period(selected[mask], baseline[mask], calendar[mask].tolist())
        years = {}
        for year in sorted({d[:4] for d in dates}):
            mask = np.asarray([d.startswith(year) for d in dates])
            years[year] = paired_period(selected[mask], baseline[mask], calendar[mask].tolist())
        def daily(i):
            return dict(date=dates[i], selected_return=float(selected[i]), baseline_return=float(baseline[i]),
                        log_difference=float(delta[i]),
                        selected_exposure=holdings[i - 1] if i else None,
                        baseline_exposure=b_holdings[i - 1] if i else None,
                        selected_closing_holding=holdings[i], baseline_closing_holding=b_holdings[i])
        best = sorted(range(len(dates)), key=lambda i: (-delta[i], dates[i]))
        worst = sorted(range(len(dates)), key=lambda i: (delta[i], dates[i]))
        concentration = {}
        for count in (1, 5, 10, 20):
            contribution = float(delta[best[:count]].sum())
            concentration[str(count)] = dict(top_positive_log_difference=contribution,
                share_of_net_log_advantage=contribution / total if abs(total) > 1e-12 else None,
                fixed_path_advantage_excluding_these_days=total - contribution)
        keep = calendar >= "2016-01-01"
        fee = stage["registration"]["fee"]
        if fee != .0001:
            raise ValueError("This diagnostic's attribution expects legacy fee 1 bp per side")
        result[candidate_id] = dict(
            frozen_labels=[key for key in ("primary", "guarded") if stage["selection"][key] == candidate_id],
            baseline_id=baseline_id, configuration=stage["candidates"][index],
            periods=periods, yearly=years,
            fixed_path_excluding_2014_2015=paired_period(selected[keep], baseline[keep], calendar[keep].tolist()),
            exclusion_note="Geometric contribution check along the existing continuous paths; no reinitialization, re-selection or executable alternative.",
            top_advantage_days=[daily(i) for i in best[:20]], top_disadvantage_days=[daily(i) for i in worst[:20]],
            advantage_concentration=concentration,
            concentration_note="Signed additive log differences. Ratios may exceed 100% or be negative because other dates offset gains; deleting dates does not rerun a policy.",
            different_closing_holding_days=int(sum(a != b for a, b in zip(holdings, b_holdings))),
            selected_attribution=_exposure_attribution(selected, holdings, stage["candidates"][index], fee),
            baseline_attribution=_exposure_attribution(baseline, b_holdings, stage["candidates"][baseline_index], fee))
    return result


def analyze(stage, draws=1000, seed=20260927, block_sizes=(20, 60), batch_size=4, progress=None):
    values = stage["values"].T  # supplied storage is N by T; statistics takes T by M
    benchmark = stage["values"][stage["baseline_index"]]
    tests = {}
    for block in block_sizes:
        if progress:
            progress("White-style block=%d draws=%d candidates=%d" % (block, draws, len(stage["ids"])))
        tests[str(block)] = white_style_max_test(values, benchmark, block_size=block, draws=draws,
            seed=seed, candidate_ids=stage["ids"], batch_size=batch_size)
    walkforward = walkforward_selection_diagnostic(values, benchmark, stage["dates"], candidate_ids=stage["ids"])
    return dict(stage=stage["directory"].name, completed_at=datetime.now(timezone.utc).isoformat(),
        clean_oos=False, adaptive_family=True, diagnostic_only=True, replaces_frozen_selection=False,
        candidate_count=len(stage["ids"]), complete_family=True, shortlisted=False,
        selection_observations=len(stage["dates"]), selection_start=stage["dates"][0], selection_end=stage["dates"][-1],
        baseline_id=stage["baseline_id"], frozen_selected_ids=stage["selected_ids"],
        primary=stage["selection"]["primary"], guarded=stage["selection"]["guarded"],
        warnings=list(WARNINGS), input_sha256=stage["sources"],
        implementation_sha256={"diagnostics.py": sha(__file__),
            "v10_search/statistics.py": sha(BASE.parent / "v10_search/statistics.py")},
        statistical_parameters=dict(block_sizes=list(block_sizes), draws=draws, seed=seed,
                                    batch_size=batch_size, blas_threads=1, annualization_days=TRADING_DAYS),
        white_style=tests, historical_reselection=walkforward, paired=paired_diagnostics(stage))


def markdown(result):
    lines = ["# v10 深入研究：完整候选族条件诊断", "",
             "阶段 `%s`，全部 %d 个候选；基准 `%s`。" % (result["stage"], result["candidate_count"], result["baseline_id"]), "",
             "**整个候选族由已经看过的历史自适应构造；这些统计不是干净样本外检验，也没有校正所有旧试验。**", "",
             "|区块长度|候选数|重采样次数|White-style p|Monte Carlo SE|", "|---|---:|---:|---:|---:|"]
    for block, row in result["white_style"].items():
        lines.append("|%s|%d|%d|%.4f|%.4f|" % (block, row["tested_candidate_count"], row["draws"], row["p_value"], row["p_value_monte_carlo_se"]))
    walk = result["historical_reselection"]
    lines += ["", "固定四段历史重选（2018–2025 拼接）的年化 %.2f%%，基准 %.2f%%；没有重建策略切换时的持仓和成本，不能当可交易策略。" % (100 * walk["selected_stream_cagr"], 100 * walk["benchmark_stream_cagr"]), "",
              "|随后历史区间|训练时所选|重选流年化|基准年化|", "|---|---|---:|---:|"]
    for fold in walk["folds"]:
        lines.append("|%s–%s|%s|%.2f%%|%.2f%%|" % (fold["test_start"], fold["test_end"], fold["selected_id"] or "基准", 100 * fold["selected_test_cagr"], 100 * fold["benchmark_test_cagr"]))
    for candidate_id, pair in result["paired"].items():
        lines += ["", "冻结候选 `%s`（%s）：" % (candidate_id, "/".join(pair["frozen_labels"])), "",
                  "|连续路径分段|候选年化|基准年化|候选累计|基准累计|", "|---|---:|---:|---:|---:|"]
        for name, row in pair["periods"].items():
            a, b = row["selected"], row["baseline"]
            lines.append("|%s|%.2f%%|%.2f%%|%.2f%%|%.2f%%|" % (name, a["cagr"] * 100, b["cagr"] * 100, a["total_return"] * 100, b["total_return"] * 100))
        lines += ["", "2026 为不足一年的已知历史，表内年化仅机械换算，应同时看累计收益。年度、前 20 个优势/劣势日、收益贡献、费用和固定路径排除早期年份诊断详见 JSON。"]
    lines += ["", "方法限制：", ""] + ["- " + warning for warning in result["warnings"]]
    return "\n".join(lines) + "\n"


def run(stage, results_root=None, draws=1000, seed=20260927, batch_size=4, overwrite=False):
    loaded = load_stage(stage, results_root)
    destination = loaded["directory"] / "diagnostics.json"
    if destination.exists() and not overwrite:
        raise FileExistsError("Diagnostic exists; use --overwrite for an explicit rerun")
    result = analyze(loaded, draws=draws, seed=seed, batch_size=batch_size, progress=lambda message: print(message, flush=True))
    # Refuse to publish against any source changed while a long bootstrap ran.
    for name, expected in loaded["sources"].items():
        _expected_sha(loaded["directory"] / name, expected)
    dump(destination, result)
    destination.with_suffix(".md").write_text(markdown(result))
    print(destination, flush=True)
    return result


def self_test():
    """Small temporary stages exercise routing/integrity; no real scan or prices."""
    dates = ["%d-%02d-15" % (year, month) for year in range(2014, 2026) for month in range(1, 7)]
    full_dates = dates + ["2026-%02d-15" % month for month in range(1, 7)]
    n, t = 5, len(full_dates)
    rng = np.random.default_rng(712)
    full = rng.normal(.001, .006, (n, t)).astype(np.float64)
    full[:, 0] = 0
    full[1, -6:] = .2  # known tail must never enter the tests or re-selection
    candidates = [dict(id="id_%d" % i, hash="config_%d" % i, stock_pool=["510300"], global_pool=["513100"]) for i in range(n)]
    with tempfile.TemporaryDirectory(prefix="v10_deep_diagnostics_") as temporary:
        directory = Path(temporary) / "refinements"
        directory.mkdir()
        poison = Path(temporary) / "mechanisms"
        poison.mkdir()
        (poison / "selection.json").write_text("not valid JSON")
        registry = dict(stage="refinements", candidates=candidates, unique_count=n)
        dump(directory / "registered_candidates.json", registry)
        registration = dict(candidate_count=n, start="2014-01-01", train_end="2025-12-31", end="2026-09-24", fee=.0001,
                            fingerprints={"fixture": True}, registry_sha256=sha(directory / "registered_candidates.json"))
        dump(directory / "registration.json", registration)
        rows = [dict(id=c["id"], hash=c["hash"], index=i) for i, c in enumerate(candidates)]
        dump(directory / "development.json", dict(rows=rows, dates=dates))
        np.save(directory / "selection_returns.npy", full[:, :len(dates)])
        np.save(directory / "full_returns.npy", full)
        selection = dict(primary="id_2", guarded="id_2", controls={"v9.2": "id_0"}, top_ids=["id_2"],
                         registration_sha256=sha(directory / "registration.json"), development_sha256=sha(directory / "development.json"),
                         selection_returns_sha256=sha(directory / "selection_returns.npy"))
        dump(directory / "selection.json", selection)
        dump(directory / "evaluation.json", dict(selection=selection, rows=rows, dates=full_dates,
                fingerprints=registration["fingerprints"], hindsight_best="id_1", full_returns_sha256=sha(directory / "full_returns.npy")))
        paths = {c["id"]: dict(returns=full[i].tolist(), holdings=["510300"] * t) for i, c in enumerate(candidates)}
        dump(directory / "selected_paths.json", paths)
        loaded = load_stage("refinements", temporary)
        result = analyze(loaded, draws=7, batch_size=2)
        assert result["candidate_count"] == n and result["frozen_selected_ids"] == ["id_2"]
        assert all(row["tested_candidate_count"] == n for row in result["white_style"].values())
        assert result["selection_end"] < "2026" and len(result["historical_reselection"]["folds"]) == 4
        assert set(result["paired"]) == {"id_2"} and result["paired"]["id_2"]["frozen_labels"] == ["primary", "guarded"]
        # Routing cannot depend on a poisoned old stage, nor a Top1 list. A
        # changed known tail must not change any selection-only statistic.
        old_white, old_walk = result["white_style"], result["historical_reselection"]
        full[:, -6:] = -.15
        np.save(directory / "full_returns.npy", full)
        evaluation = read_json(directory / "evaluation.json")
        evaluation["full_returns_sha256"] = sha(directory / "full_returns.npy")
        dump(directory / "evaluation.json", evaluation)
        for i, candidate in enumerate(candidates):
            paths[candidate["id"]]["returns"] = full[i].tolist()
        dump(directory / "selected_paths.json", paths)
        changed = analyze(load_stage("refinements", temporary), draws=7, batch_size=2)
        assert changed["white_style"] == old_white and changed["historical_reselection"] == old_walk
        # Market contribution must be allocated to previous holding, not the
        # newly bought instrument. A full switch includes two-sided fees.
        attribution = _exposure_attribution(np.asarray([0., (1.1 * .9998) - 1, -.05]),
            ["510300", "513100", "513100"], candidates[0], .0001)
        assert np.isclose(attribution["by_asset"]["510300"]["market_log_growth"], np.log(1.1))
        assert np.isclose(attribution["by_asset"]["513100"]["market_log_growth"], np.log(.95))
        assert attribution["charged_switches"] == 1
        # Fail closed rather than accepting a cropped/reordered or corrupt family.
        development = read_json(directory / "development.json")
        development["rows"][0]["id"] = "id_2"
        dump(directory / "development.json", development)
        try:
            load_stage("refinements", temporary)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed frozen development was accepted")
    print("Synthetic diagnostics self-test passed: complete-family routing, all four folds, tail isolation, paired attribution and tamper rejection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="refinements")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.stage, draws=args.draws, seed=args.seed, batch_size=args.batch_size, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
