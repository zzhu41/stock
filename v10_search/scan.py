"""Large, auditable search: develop/freeze, reveal all, then search-aware stats.

Only v10_search is writable. The caller can report both a development-selected
champion and a labelled full-history hindsight maximum, never interchange them.
"""
import argparse
import bisect
from datetime import datetime, timezone
import gc
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .data import BASE, START, END, CASH, load_histories, protect, sha
from .registry import build_registry, controls, candidate_hash
from .features import FeatureBank
from .policy import SearchPolicy
from .fast_execution import prepare, run
from v10_round2.v92 import load_qvix


OUT = BASE / "results"
TRAIN_END = "2021-12-31"
FEE, SLIPPAGE = .0001, .001
SOURCES = ("data.py", "registry.py", "features.py", "policy.py", "fast_execution.py", "statistics.py", "scan.py", "same_close.py", "PROTOCOL.md")
DEPENDENCIES = ("v10_next/data.py", "v10_next/frozen/strategy.py", "v10_next/frozen/metadata.py", "v10_next/frozen/presets.json",
                "v10_round2/v92.py", "v10_round2/qvix_manifest.json", "v10_round2/data/qvix50.csv",
                "v10_round2/timing.py", "v10_next/execution.py", "v10_next/metrics.py")
PERIODS = {"development": (START, TRAIN_END), "early": (START, "2017-12-31"),
           "late": ("2018-01-01", TRAIN_END), "validation": ("2022-01-01", "2025-12-31"),
           "report_only_2026": ("2026-01-01", END)}
_BANK = _FRAME = _FEAR = _CONFIGS = _END = None


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def fingerprints():
    return dict(source_sha256={p: sha(BASE / p) for p in SOURCES},
                dependencies_sha256={p: sha(BASE.parent / p) for p in DEPENDENCIES},
                prices_manifest_sha256=sha(BASE / "data_manifest.json"),
                protected_manifest_sha256=sha(BASE / "protected_manifest.json"))


def setup(configs):
    global _BANK, _FRAME, _FEAR, _CONFIGS
    h = load_histories()
    calendar = [r[0] for r in h["510300"]]
    _FRAME = prepare(h, calendar)
    _BANK = FeatureBank(h, calendar)
    q = load_qvix()
    _FEAR = tuple(q.state(d)["active"] for d in calendar)
    _CONFIGS = configs
    keys = {(c["score_mode"], tuple(c["score_windows"])) for c in configs}
    for mode, windows in sorted(keys):
        _BANK.ranked(mode, windows)
    print("Prepared %d dated feature families across %d assets" % (len(keys), len(h)), flush=True)


def performance(returns):
    r = np.asarray(returns, dtype=np.float64)
    if not len(r) or np.any(r <= -1) or not np.all(np.isfinite(r)):
        raise ValueError("Invalid performance returns")
    log_growth = np.log1p(r).sum()
    wealth = np.exp(np.cumsum(np.log1p(r)))
    peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    sd = float(r.std())
    cagr = float(np.expm1(log_growth * 244 / len(r)))
    dd = float(np.min(wealth / peak - 1))
    return dict(sessions=len(r), cagr=cagr, nav_factor=float(np.exp(log_growth)), total_return=float(np.expm1(log_growth)),
                max_dd=dd, volatility=sd * 244 ** .5,
                sharpe=float(r.mean()) / sd * 244 ** .5 if sd else 0.0,
                calmar=cagr / abs(dd) if dd < 0 else None)


def summaries(result, dates):
    nav = np.asarray(result["navs"], dtype=np.float64)
    previous = np.concatenate(([1.0], nav[:-1]))
    ret = nav / previous - 1
    out = dict(full=performance(ret), trade_days=result["trade_count"], fill_legs=result["fill_count"],
               trade_days_per_year=result["trade_count"] * 244 / len(ret),
               turnover=result["diagnostics"]["total_turnover"],
               turnover_per_year=result["diagnostics"]["total_turnover"] * 244 / len(ret),
               deferred_count=result["deferred_count"], missing_held_days=result["diagnostics"]["missing_held_days"],
               nonmoney_exposure=sum(c not in (None, CASH) for c in result["holdings"]) / len(ret),
               yearly=result["yearly"],
               holdings_hash=hashlib.sha256(",".join(c or "" for c in result["holdings"]).encode()).hexdigest(),
               return_path_hash=hashlib.sha256(np.round(ret, 12).tobytes()).hexdigest())
    for name, (lo, hi) in PERIODS.items():
        a, b = bisect.bisect_left(dates, lo), bisect.bisect_right(dates, hi)
        if b > a:
            out[name] = performance(ret[a:b])
    return out, ret


def simulate(config, end, slippage=SLIPPAGE, details=False):
    policy = SearchPolicy(config, _BANK, _FRAME, _FEAR, trace=details)
    r = run(_FRAME, policy, START, end, fee=FEE, slippage=slippage,
            capture_daily=True if details else "arrays", capture_trades=details)
    if details:
        r["policy_metadata"] = policy.metadata
    return r


def worker(index):
    try:
        c = _CONFIGS[index]
        r = simulate(c, _END)
        dates = [d for d in _FRAME.dates if START <= d <= _END]
        row, ret = summaries(r, dates)
        row.update(index=index, id=c["id"], candidate_hash=c["candidate_hash"], family=c["family"], regime_mode=c.get("regime_mode", "ma250"))
        return index, row, ret
    except Exception as exc:
        return index, {"index": index, "id": _CONFIGS[index]["id"], "error": repr(exc)}, None


def scan_all(configs, end, label, workers=2):
    global _END
    _END = end
    dates = [d for d in _FRAME.dates if START <= d <= end]
    matrix_path = OUT / (label + "_returns.npy")
    mat = np.lib.format.open_memmap(str(matrix_path), mode="w+", dtype=np.float64,
                                  shape=(len(dates), len(configs)), fortran_order=True)
    rows = [None] * len(configs)
    started = time.monotonic()
    context = mp.get_context("fork")
    with (OUT / (label + "_ledger.jsonl")).open("w", encoding="utf-8") as ledger, context.Pool(workers) as pool:
        for done, (index, row, ret) in enumerate(pool.imap_unordered(worker, range(len(configs)), chunksize=16), 1):
            ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if ret is None:
                ledger.flush()
                raise RuntimeError("Candidate failed; retain error and repair before claiming a complete matrix: " + str(row))
            mat[:, index] = ret
            rows[index] = row
            if done % 250 == 0 or done == len(configs):
                ledger.flush()
                print("%s %d/%d, %.1fs" % (label, done, len(configs), time.monotonic() - started), flush=True)
    mat.flush()
    del mat
    dump(OUT / (label + "_dates.json"), dates)
    return rows, dates


def choose(rows, benchmark, top_k=20):
    """Only precomputed development/early/late fields are accessed."""
    eligible = []
    for row in rows:
        if all(row[p]["cagr"] > benchmark[p]["cagr"] + 1e-12 for p in ("early", "late", "development")):
            eligible.append(row)
    eligible.sort(key=lambda r: (-r["development"]["cagr"], r["id"]))
    return dict(champion=eligible[0]["id"] if eligible else None, top_ids=[r["id"] for r in eligible[:top_k]],
                eligible_count=len(eligible), top_k=top_k,
                gates={p: benchmark[p]["cagr"] for p in ("early", "late", "development")},
                selection_uses_validation=False)


def develop(workers):
    protect()
    if (OUT / "selection.json").exists():
        raise RuntimeError("Development choice is frozen; no overwrite after observing later results")
    registration = build_registry()
    dump(OUT / "registry.json", registration)
    fp = fingerprints()
    dump(OUT / "registration.json", dict(registered_at=stamp(), fingerprints=fp,
                                         registry_sha256=sha(OUT / "registry.json"), raw=registration["raw_count"],
                                         unique=registration["unique_count"], workers=workers,
                                         primary_clock="next_open_actual_holdings", fee=FEE, slippage=SLIPPAGE))
    configs = registration["candidates"]
    setup(configs)
    control_results, control_metrics = {}, {}
    dates = [d for d in _FRAME.dates if START <= d <= TRAIN_END]
    for c in controls():
        r = simulate(c, TRAIN_END, details=True)
        control_results[c["id"]] = r
        control_metrics[c["id"]], _ = summaries(r, dates)
        print("CONTROL", c["id"], control_metrics[c["id"]]["full"], flush=True)
    rows, dates = scan_all(configs, TRAIN_END, "development", workers)
    if fp != fingerprints():
        raise AssertionError("Source/protocol changed during development")
    dump(OUT / "development.json", dict(rows=rows, controls=control_metrics))
    dump(OUT / "development_controls.json", control_results)
    picked = choose(rows, control_metrics["c_v92"])
    picked.update(selected_at=stamp(), registration_sha256=sha(OUT / "registration.json"),
                  development_sha256=sha(OUT / "development.json"),
                  development_matrix_sha256=sha(OUT / "development_returns.npy"),
                  candidate_ids=[c["id"] for c in configs], known_history=True)
    dump(OUT / "selection.json", picked)
    protect()
    print("FROZEN CHAMPION", picked["champion"], "eligible", picked["eligible_count"], flush=True)


def evaluate(workers):
    protect()
    registration = json.loads((OUT / "registration.json").read_text())
    picked = json.loads((OUT / "selection.json").read_text())
    assert registration["fingerprints"] == fingerprints()
    assert sha(OUT / "registry.json") == registration["registry_sha256"]
    assert sha(OUT / "registration.json") == picked["registration_sha256"]
    assert sha(OUT / "development.json") == picked["development_sha256"]
    assert sha(OUT / "development_returns.npy") == picked["development_matrix_sha256"]
    configs = json.loads((OUT / "registry.json").read_text())["candidates"]
    setup(configs)
    dates = [d for d in _FRAME.dates if START <= d <= END]
    control_results, control_metrics = {}, {}
    for c in controls():
        r = simulate(c, END, details=True)
        control_results[c["id"]] = r
        control_metrics[c["id"]], _ = summaries(r, dates)
    rows, dates = scan_all(configs, END, "full", workers)
    training = np.load(OUT / "development_returns.npy", mmap_mode="r")
    all_returns = np.load(OUT / "full_returns.npy", mmap_mode="r")
    if not np.allclose(all_returns[:len(training)], training, rtol=1e-11, atol=1e-12):
        raise AssertionError("Later observations changed development returns")
    hindsight = sorted(rows, key=lambda r: (-r["full"]["cagr"], r["id"]))[0]
    selected_ids = list(dict.fromkeys(picked["top_ids"] + [hindsight["id"]]))
    by_id = {c["id"]: c for c in configs}
    extra_paths, costs, neighbors, ideals = {}, {}, [], {}
    for name in selected_ids:
        c = by_id[name]
        result = simulate(c, END, details=True)
        extra_paths[name] = result
        costs[name] = {}
        for bp in (20, 50):
            costs[name][str(bp)] = summaries(simulate(c, END, slippage=bp / 10000), dates)[0]
    for c in controls():
        name = c["id"]
        costs[name] = {str(bp): summaries(simulate(c, END, slippage=bp / 10000), dates)[0] for bp in (20, 50)}
    if picked["champion"]:
        winner = by_id[picked["champion"]]
        if winner["score_windows"]:
            for scale in (.9, 1.1):
                c = json.loads(json.dumps(winner))
                c["score_windows"] = [max(2, round(x * scale)) for x in c["score_windows"]]
                neighbors.append(dict(scale=scale, windows=c["score_windows"], metrics=summaries(simulate(c, END), dates)[0]))
    # Counterfactual same-close diagnostics reuse the already tested clock with
    # a small adapter, never the next-open search matrix or a favorable clock.
    from .same_close import simulate_same_close
    ideal_configs = controls() + [by_id[name] for name in dict.fromkeys([picked["champion"], hindsight["id"]]) if name]
    for c in ideal_configs:
        r = simulate_same_close(c, _BANK, _FEAR)
        from v10_next.metrics import summarize
        ideals[c["id"]] = summarize(r)
    family_best = {}
    for row in rows:
        key = row["family"]
        if key not in family_best or row["full"]["cagr"] > family_best[key]["full"]["cagr"]:
            family_best[key] = row
    no_regime = {mode: max((r for r in rows if r["regime_mode"] == mode), key=lambda r: r["full"]["cagr"])
                 for mode in ("ma250", "open_stock", "always_bull", "all_assets")}
    report = dict(completed_at=stamp(), selection=picked, selection_sha256=sha(OUT / "selection.json"),
                  registration_sha256=sha(OUT / "registration.json"), fingerprints=fingerprints(),
                  results=rows, controls=control_metrics, hindsight_winner=hindsight,
                  hindsight_label="exploratory hindsight winner; not development selected or clean OOS",
                  costs=costs, champion_neighbors=neighbors, ideal_close=ideals,
                  family_best=family_best, regime_best=no_regime,
                  unique_return_paths_development=len({r["return_path_hash"] for r in json.loads((OUT / "development.json").read_text())["rows"]}),
                  unique_return_paths_full=len({r["return_path_hash"] for r in rows}),
                  matrix_sha256=sha(OUT / "full_returns.npy"), development_prefixes_unchanged=True,
                  protected_files_verified=protect(), candidate_count=len(configs))
    if report["fingerprints"] != registration["fingerprints"]:
        raise AssertionError("Source changed during evaluation")
    dump(OUT / "evaluation.json", report)
    dump(OUT / "selected_paths.json", dict(controls=control_results, candidates=extra_paths))
    print("HINDSIGHT MAX (not champion):", hindsight["id"], hindsight["full"], flush=True)


def statistics():
    from .statistics import white_style_max_test, walkforward_selection_diagnostic
    r = json.loads((OUT / "evaluation.json").read_text())
    dev_controls = json.loads((OUT / "development_controls.json").read_text())
    paths = json.loads((OUT / "selected_paths.json").read_text())
    picked = r["selection"]
    matrix = np.load(OUT / "development_returns.npy", mmap_mode="r")
    nav = np.asarray(dev_controls["c_v92"]["navs"])
    benchmark = nav / np.concatenate(([1.0], nav[:-1])) - 1
    tests = []
    for block in (20, 60):
        test = white_style_max_test(matrix, benchmark, block_size=block, draws=1000, seed=20260927,
                                    candidate_ids=picked["candidate_ids"], batch_size=8)
        tests.append(test)
        print("WHITE-STYLE block", block, "p", test["p_value"], "columns", test["tested_candidate_count"], flush=True)
        gc.collect()
    full = np.load(OUT / "full_returns.npy", mmap_mode="r")
    dates = json.loads((OUT / "full_dates.json").read_text())
    nav = np.asarray(paths["controls"]["c_v92"]["navs"])
    benchmark_full = nav / np.concatenate(([1.0], nav[:-1])) - 1
    folds = walkforward_selection_diagnostic(full, benchmark_full, dates, picked["candidate_ids"])
    dump(OUT / "search_diagnostics.json", dict(white_style=tests, walkforward=folds,
                                               evaluation_sha256=sha(OUT / "evaluation.json"),
                                               full_matrix_sha256=sha(OUT / "full_returns.npy"),
                                               development_matrix_sha256=sha(OUT / "development_returns.npy"),
                                               protected_files_verified=protect()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("develop", "evaluate", "statistics"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        parser.error("This host permits 1 or 2 research workers")
    if args.stage == "statistics":
        statistics()
    else:
        (develop if args.stage == "develop" else evaluate)(args.workers)


if __name__ == "__main__":
    main()
