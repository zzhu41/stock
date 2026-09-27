"""Original-close-target H search with explicit pre-2026 choices and risks."""
import argparse
import bisect
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
import time

import numpy as np

from .data import BASE, START, CASH, load_histories, load_fear, protect, sha
from .registry import build_registry, controls, candidate_hash
from .engine import prepare, run
from .policy import ClosePolicy
from v10_search.features import FeatureBank
from v10_search.scan import performance


OUT = BASE / "results"
TRAIN_END = "2025-12-31"
FEE, STRESS_FEE = .0001, .0005
OWN_SOURCES = ("data.py", "engine.py", "policy.py", "registry.py", "scan.py", "refresh_snapshots.py", "PROTOCOL.md")
DEPENDENCIES = ("v10_search/data.py", "v10_search/features.py", "v10_search/policy.py", "v10_search/registry.py",
                "v10_search/fast_execution.py", "v10_search/scan.py", "v10_next/data.py", "v10_next/frozen/strategy.py",
                "v10_next/frozen/metadata.py", "v10_next/frozen/presets.json", "v10_round2/v92.py", "v10_search/statistics.py")
_H = _BANK = _FRAME = _FEAR = _CONFIGS = _END = _RUN_END = None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fingerprints():
    return dict(source_sha256={p: sha(BASE / p) for p in OWN_SOURCES},
                dependency_sha256={p: sha(BASE.parent / p) for p in DEPENDENCIES},
                snapshot_manifest_sha256=sha(BASE / "snapshot_manifest.json"),
                qvix_manifest_sha256=sha(BASE / "qvix_manifest.json"),
                protected_manifest_sha256=sha(BASE / "protected_manifest.json"))


def setup(configs):
    global _H, _BANK, _FRAME, _FEAR, _CONFIGS, _END
    _H, _END = load_histories()
    calendar = [r[0] for r in _H["510300"]]
    _FRAME = prepare(_H, calendar)
    _BANK = FeatureBank(_H, calendar)
    _FEAR = load_fear(calendar)
    _CONFIGS = configs
    for mode, windows in sorted({(c["score_mode"], tuple(c["score_windows"])) for c in configs}):
        _BANK.ranked(mode, windows)
    print("Prepared", len(configs), "configs,", len(_H), "assets; common end", _END, flush=True)


def simulate(config, end=None, fee=FEE, details=False):
    policy = ClosePolicy(config, _BANK, _FRAME, _FEAR, trace=details)
    result = run(_FRAME, policy, START, end or _END, fee=fee,
                 capture_daily=True if details else "arrays", capture_trades=details)
    if details:
        result["policy_metadata"] = policy.metadata
    return result


def fee_reprice(result, fee_from=FEE, fee_to=STRESS_FEE):
    """Exact fee convention on a cost-independent holding path.

    Policy inputs contain no NAV, cash balance, or fee parameter. Registered
    variants have no NAV/entry-price-dependent risk overlays, and use full
    allocation with no lot rounding. Fees therefore cannot change their targets.
    """
    nav = np.asarray(result["navs"], dtype=np.float64)
    holdings = result["holdings"]
    charged = np.asarray([0] + [int(a != b) for a, b in zip(holdings[:-1], holdings[1:])], dtype=np.int64)
    if int(charged.sum()) != result["switches"]:
        raise AssertionError("Actual switch count disagrees with original fee convention")
    return nav * ((1 - 2 * fee_to) / (1 - 2 * fee_from)) ** np.cumsum(charged)


@lru_cache(maxsize=4)
def date_slices(dates):
    periods = {"full": (START, dates[-1]), "selection": (START, TRAIN_END),
               "audit_2014_2021": (START, "2021-12-31"), "recent_2022_2025": ("2022-01-01", TRAIN_END),
               "report_only_2026": ("2026-01-01", dates[-1])}
    slices = {name: (bisect.bisect_left(dates, lo), bisect.bisect_right(dates, hi))
              for name, (lo, hi) in periods.items()}
    years = {y: (bisect.bisect_left(dates, y + "-01-01"), bisect.bisect_right(dates, y + "-12-31"))
             for y in sorted({x[:4] for x in dates})}
    return slices, years


def summarize_nav(nav, dates):
    nav = np.asarray(nav, dtype=np.float64)
    returns = nav / np.concatenate(([1.0], nav[:-1])) - 1
    slices, years = date_slices(tuple(dates))
    out = {}
    for name, (lo, hi) in slices.items():
        if hi > lo:
            out[name] = performance(returns[lo:hi])
    out["yearly"] = {y: float(np.expm1(np.log1p(returns[lo:hi]).sum())) for y, (lo, hi) in years.items()}
    return out, returns


def collect(config, result, dates):
    metrics, returns = summarize_nav(result["navs"], dates)
    stress, stress_returns = summarize_nav(fee_reprice(result), dates)
    selection_switches = sum(a != b and dates[i] <= TRAIN_END
                             for i, (a, b) in enumerate(zip(result["holdings"][:-1], result["holdings"][1:]), 1))
    return dict(id=config["id"], candidate_hash=config["candidate_hash"], stages=config.get("stages", []),
                metrics=metrics, fee5=stress, switches=result["switches"],
                selection_switches=selection_switches,
                switches_per_year=result["switches"] * 244 / len(dates),
                blocked_switch_days=result["diagnostics"]["blocked_switch_days"],
                missing_held_days=result["diagnostics"]["missing_held_days"],
                nonmoney_exposure=sum(c not in (None, CASH) for c in result["holdings"]) / len(dates),
                path_hash=hashlib.sha256(np.round(returns, 12).tobytes()).hexdigest()), returns, stress_returns


def worker(index):
    try:
        c = _CONFIGS[index]
        r = simulate(c, _RUN_END)
        dates = [d for d in _FRAME.dates if START <= d <= _RUN_END]
        row, returns, stress = collect(c, r, dates)
        row["index"] = index
        return index, row, returns, stress
    except Exception as exc:
        return index, dict(index=index, id=_CONFIGS[index]["id"], error=repr(exc)), None, None


def batch(configs, end, label, workers):
    global _RUN_END
    _RUN_END = end
    dates = [d for d in _FRAME.dates if START <= d <= end]
    matrices = [np.lib.format.open_memmap(str(OUT / (label + suffix)), mode="w+", dtype=np.float64,
                                         shape=(len(dates), len(configs)), fortran_order=True)
                for suffix in ("_returns.npy", "_fee5_returns.npy")]
    rows = [None] * len(configs)
    started = time.monotonic()
    with (OUT / (label + "_ledger.jsonl")).open("w", encoding="utf-8") as ledger, mp.get_context("fork").Pool(workers) as pool:
        for done, (index, row, returns, stress) in enumerate(pool.imap_unordered(worker, range(len(configs)), chunksize=16), 1):
            ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if returns is None:
                ledger.flush()
                raise RuntimeError("Candidate failed: " + str(row))
            matrices[0][:, index], matrices[1][:, index] = returns, stress
            rows[index] = row
            if done % 400 == 0 or done == len(configs):
                ledger.flush()
                print(label, done, "/", len(configs), "%.1fs" % (time.monotonic() - started), flush=True)
    for matrix in matrices:
        matrix.flush()
    dump(OUT / (label + "_dates.json"), dates)
    return rows, dates


def rank_key(row, config):
    return (-row["metrics"]["selection"]["cagr"], len(set(config["channels"]) & {"qvix", "volume"}),
            config["complexity"]["active_optional_rule_count"], row["selection_switches"], row["id"])


def select(rows, configs, baseline):
    by_id = {c["id"]: c for c in configs}
    ordered = sorted(rows, key=lambda row: rank_key(row, by_id[row["id"]]))
    bm, bf = baseline["metrics"], baseline["fee5"]
    qualified, checks = [], {}
    for row in rows:
        m, f = row["metrics"], row["fee5"]
        tests = dict(above_v92=m["selection"]["cagr"] > bm["selection"]["cagr"] + 1e-12,
                     drawdown=abs(m["selection"]["max_dd"]) <= abs(bm["selection"]["max_dd"]) + .05 + 1e-12,
                     recent=m["recent_2022_2025"]["cagr"] >= bm["recent_2022_2025"]["cagr"] - .05 - 1e-12,
                     fee_retention=f["selection"]["cagr"] >= .8 * m["selection"]["cagr"] - 1e-12,
                     fee_relative=f["selection"]["cagr"] >= bf["selection"]["cagr"] - .02 - 1e-12)
        checks[row["id"]] = tests
        if all(tests.values()):
            qualified.append(row)
    qualified.sort(key=lambda row: rank_key(row, by_id[row["id"]]))
    top = ordered[0]
    return dict(primary_return_champion=top["id"], primary_above_v92=checks[top["id"]]["above_v92"],
                risk_guarded_candidate=qualified[0]["id"] if qualified else None,
                risk_eligible_count=len(qualified), risk_checks=checks,
                top20_return_ids=[r["id"] for r in ordered[:20]],
                selected_from="2014-2025 original-close known history; 2026 not used by this ranking")


def verify_fee_transform(configs):
    cases = controls() + [configs[i] for i in (0, 800, 2400, 4300, len(configs) - 1)]
    checks = []
    for c in cases:
        base = simulate(c, "2025-12-31")
        higher = simulate(c, "2025-12-31", fee=STRESS_FEE)
        expected = fee_reprice(base)
        actual = np.asarray(higher["navs"])
        if base["holdings"] != higher["holdings"] or not np.allclose(expected, actual, rtol=1e-11, atol=1e-10):
            raise AssertionError("Fee path transform failed: " + c["id"])
        checks.append(dict(id=c["id"], daily_holdings_equal=True, maximum_absolute_nav_error=float(np.max(np.abs(expected - actual)))))
    return checks


def develop(workers):
    protect()
    OUT.mkdir(exist_ok=True)
    if (OUT / "selection.json").exists():
        raise RuntimeError("Selection is already frozen; do not overwrite after revealing 2026")
    reg = build_registry()
    dump(OUT / "registry.json", reg)
    fp = fingerprints()
    manifest = json.loads((BASE / "snapshot_manifest.json").read_text())
    dump(OUT / "registration.json", dict(registered_at=stamp(), fingerprints=fp, end=manifest["end"],
                                         registry_sha256=sha(OUT / "registry.json"), fee=FEE, stress_fee=STRESS_FEE,
                                         primary_clock="original_same_close", candidate_count=len(reg["candidates"])))
    configs = reg["candidates"]
    setup(configs)
    fee_checks = verify_fee_transform(configs)
    control_paths, baseline_metrics = {}, {}
    dates = [d for d in _FRAME.dates if START <= d <= TRAIN_END]
    for c in controls():
        result = simulate(c, TRAIN_END, details=True)
        control_paths[c["id"]] = result
        baseline_metrics[c["id"]] = collect(c, result, dates)[0]
        print("CONTROL", c["id"], baseline_metrics[c["id"]]["metrics"]["selection"], flush=True)
    rows, dates = batch(configs, TRAIN_END, "selection", workers)
    if fp != fingerprints():
        raise AssertionError("Source changed during selection")
    dump(OUT / "development.json", dict(results=rows, controls=baseline_metrics, fee_transform_checks=fee_checks))
    dump(OUT / "development_controls.json", control_paths)
    chosen = select(rows, configs, baseline_metrics["c_v92"])
    chosen.update(selected_at=stamp(), registration_sha256=sha(OUT / "registration.json"),
                  development_sha256=sha(OUT / "development.json"),
                  return_matrix_sha256=sha(OUT / "selection_returns.npy"),
                  fee5_matrix_sha256=sha(OUT / "selection_fee5_returns.npy"),
                  candidate_ids=[c["id"] for c in configs])
    dump(OUT / "selection.json", chosen)
    protect()
    print("FROZEN", chosen["primary_return_champion"], chosen["risk_guarded_candidate"], "risk eligible", chosen["risk_eligible_count"], flush=True)


def next_open(config, slippage=.001):
    from v10_search.policy import SearchPolicy
    from v10_search.fast_execution import run as open_run
    policy = SearchPolicy(config, _BANK, _FRAME, _FEAR, trace=True)
    return open_run(_FRAME, policy, START, _END, fee=FEE, slippage=slippage, capture_daily=True, capture_trades=True)


def evaluate(workers):
    protect()
    registration = json.loads((OUT / "registration.json").read_text())
    chosen = json.loads((OUT / "selection.json").read_text())
    assert fingerprints() == registration["fingerprints"]
    assert sha(OUT / "registry.json") == registration["registry_sha256"]
    assert sha(OUT / "registration.json") == chosen["registration_sha256"]
    assert sha(OUT / "development.json") == chosen["development_sha256"]
    assert sha(OUT / "selection_returns.npy") == chosen["return_matrix_sha256"]
    assert sha(OUT / "selection_fee5_returns.npy") == chosen["fee5_matrix_sha256"]
    configs = json.loads((OUT / "registry.json").read_text())["candidates"]
    by_id = {c["id"]: c for c in configs}
    setup(configs)
    rows, dates = batch(configs, _END, "full", workers)
    for suffix in ("_returns.npy", "_fee5_returns.npy"):
        before = np.load(OUT / ("selection" + suffix), mmap_mode="r")
        after = np.load(OUT / ("full" + suffix), mmap_mode="r")
        if not np.allclose(before, after[:len(before)], rtol=1e-11, atol=1e-12):
            raise AssertionError("2026 altered the pre-2026 path")
    hindsight = sorted(rows, key=lambda r: (-r["metrics"]["full"]["cagr"], r["id"]))[0]
    primary, guarded = chosen["primary_return_champion"], chosen["risk_guarded_candidate"]
    report_ids = list(dict.fromkeys(n for n in (primary, guarded, hindsight["id"]) if n))
    report_configs = controls() + [by_id[n] for n in report_ids]
    paths, summarized, pressures = {}, {}, {}
    for c in report_configs:
        result = simulate(c, _END, details=True)
        paths[c["id"]] = result
        summarized[c["id"]] = collect(c, result, dates)[0]
        pressures[c["id"]] = {}
        for bp in (10, 20):
            opened = next_open(c, slippage=bp / 10000)
            from v10_next.metrics import summarize
            pressures[c["id"]][str(bp)] = dict(full=summarize(opened),
                                               selection=summarize(opened, START, TRAIN_END),
                                               recent_2022_2025=summarize(opened, "2022-01-01", TRAIN_END),
                                               report_only_2026=summarize(opened, "2026-01-01", _END))
    neighbors = []
    for name in dict.fromkeys(n for n in (primary, guarded) if n):
        c = by_id[name]
        if c["score_windows"]:
            for scale in (.9, 1.1):
                altered = deepcopy(c)
                altered["score_windows"] = [max(2, round(w * scale)) for w in c["score_windows"]]
                neighbors.append(dict(parent=name, scale=scale, windows=altered["score_windows"],
                                      metrics=collect(altered, simulate(altered, _END), dates)[0]))
    full_baseline = summarized["c_v92"]["metrics"]["full"]["cagr"]
    assessment = {name: dict(full_cagr=summarized[name]["metrics"]["full"]["cagr"],
                             full_above_v92=summarized[name]["metrics"]["full"]["cagr"] > full_baseline,
                             difference_to_v92=summarized[name]["metrics"]["full"]["cagr"] - full_baseline)
                  for name in report_ids}
    report = dict(completed_at=stamp(), end=_END, registration=registration, selection=chosen,
                  selection_sha256=sha(OUT / "selection.json"), fingerprints=fingerprints(),
                  results=rows, detailed_summaries=summarized, hindsight_winner=hindsight,
                  hindsight_label="Full-history hindsight maximum, not used to replace frozen selections",
                  next_open_pressure=pressures, window_neighbors=neighbors, assessment=assessment,
                  full_return_matrix_sha256=sha(OUT / "full_returns.npy"),
                  full_fee5_matrix_sha256=sha(OUT / "full_fee5_returns.npy"),
                  pre2026_prefixes_unchanged=True, protected_files_verified=protect())
    assert report["fingerprints"] == registration["fingerprints"]
    dump(OUT / "evaluation.json", report)
    dump(OUT / "selected_paths.json", paths)
    print("Final old-close assessment:", assessment, flush=True)


def statistics():
    from v10_search.statistics import white_style_max_test
    r = json.loads((OUT / "evaluation.json").read_text())
    control = json.loads((OUT / "development_controls.json").read_text())["c_v92"]
    matrix = np.load(OUT / "selection_returns.npy", mmap_mode="r")
    reference = np.asarray(control["returns"])
    out = []
    for block in (20, 60):
        t = white_style_max_test(matrix, reference, block_size=block, draws=1000, seed=20260927,
                                candidate_ids=r["selection"]["candidate_ids"], batch_size=4)
        out.append(t)
        print("Original-close family test", block, t["p_value"], t["tested_candidate_count"], flush=True)
    dump(OUT / "search_diagnostics.json", dict(white_style=out, evaluation_sha256=sha(OUT / "evaluation.json"),
                                               selection_matrix_sha256=sha(OUT / "selection_returns.npy"),
                                               statistical_scope="2014-2025 known-history close-clock family; not all historic searches or clean OOS",
                                               protected_files_verified=protect()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("develop", "evaluate", "statistics"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.stage == "statistics":
        statistics()
    else:
        (develop if args.stage == "develop" else evaluate)(args.workers)


if __name__ == "__main__":
    main()
