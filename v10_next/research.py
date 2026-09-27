"""Bounded v10 research. Develop/select first; reveal validation afterwards.

python3.8 -B -m v10_next.research develop
python3.8 -B -m v10_next.research validate
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .candidates import CANDIDATES, get_candidate, build_policy, build_from_config, registry
from .data import BASE, START, END, Features, load_histories, verify_protected
from .execution import run
from .legacy import factory, LegacyPolicy
from .metrics import summarize, attribution


OUT = BASE / "results"
TRAIN_END = "2021-12-31"
FEE, SLIPPAGE = .0001, .001
PERIODS = {"development": (START, TRAIN_END),
           "development_early": (START, "2017-12-31"),
           "development_late": ("2018-01-01", TRAIN_END),
           "validation": ("2022-01-01", "2025-12-31"),
           "report_only_2026": ("2026-01-01", END)}
IMPLEMENTATION = ("candidates.py", "data.py", "legacy.py", "execution.py", "metrics.py", "research.py",
                  "frozen/strategy.py", "frozen/metadata.py", "frozen/presets.json", "PROTOCOL.md")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def registered():
    value = dict(protocol_sha256=sha(BASE / "PROTOCOL.md"), registry=registry(),
                source_sha256={p: sha(BASE / p) for p in IMPLEMENTATION},
                data_manifest_sha256=sha(BASE / "data_manifest.json"),
                protected_manifest_sha256=sha(BASE / "protected_manifest.json"),
                start=START, end=END, development_end=TRAIN_END,
                commission_per_side=FEE, main_slippage_per_side=SLIPPAGE,
                costs_to_report_bps=[5, 10, 20, 50], neighborhood_scales=[.9, 1.1])
    return json.loads(json.dumps(value))


def complexity(candidate):
    if candidate["kind"] == "monthly_momentum":
        return len(candidate["lookbacks"]) + (1 if candidate["selection"] == "broad" else 2)
    # Predefined tie-break only; not a claim to measure statistical degrees of freedom.
    return 8 - (candidate["id"] in ("h_no_crash", "h_no_bear_gate")) + (len(candidate["score_windows"]) > 1)


def build(candidate, features, calendar):
    if isinstance(candidate, str):
        return build_policy(candidate, factory(features, calendar), features)
    if candidate["kind"] == "legacy":
        return LegacyPolicy(candidate, features, calendar)
    # For prespecified diagnostic neighbors, use the same implementation with
    # an explicit independent config, never replacing a registered candidate.
    return build_from_config(candidate, factory(features, calendar), features)


def simulate(candidate, histories, calendar, features, end=END, slippage=SLIPPAGE):
    policy = build(candidate, features, calendar)
    result = run(histories, calendar, policy, START, end, fee=FEE, slippage=slippage)
    result["policy_metadata"] = getattr(policy, "metadata", {})
    return result


def cash_reference(histories, calendar, end):
    entered = [False]
    def policy(date, observed, state):
        if entered[0]:
            return None
        entered[0] = True
        return {"511880": 1.0}
    return run(histories, calendar, policy, START, end, fee=FEE, slippage=SLIPPAGE)


def selection(results, cash):
    base = summarize(results["control_v91"], START, TRAIN_END)
    high_rows, robust_rows = [], []
    for c in CANDIDATES:
        name = c["id"]
        r = results[name]
        m = summarize(r, START, TRAIN_END)
        if c["family"] == "high_return" or name == "control_v91":
            checks = dict(drawdown=abs(m["max_dd"]) <= abs(base["max_dd"]) + .05 + 1e-12,
                          trading_days=m["rebalances_per_year"] <= 1.25 * base["rebalances_per_year"] + 1e-12)
            high_rows.append(dict(id=name, eligible=all(checks.values()), checks=checks, metrics=m,
                                  simplicity_tiebreak=complexity(c)))
        elif c["family"] == "robust":
            segments = {p: summarize(r, *PERIODS[p]) for p in ("development_early", "development_late")}
            cash_segments = {p: summarize(cash, *PERIODS[p]) for p in segments}
            checks = dict(above_cash_both_segments=all(segments[p]["cagr"] > cash_segments[p]["cagr"] for p in segments),
                          drawdown=abs(m["max_dd"]) <= abs(base["max_dd"]) + 1e-12,
                          trading_days=m["rebalances_per_year"] <= 30,
                          nonzero_drawdowns=all(segments[p]["calmar"] is not None for p in segments))
            robust_rows.append(dict(id=name, eligible=all(checks.values()), checks=checks, metrics=m,
                                    segments=segments, cash_segments=cash_segments,
                                    score=min(s["calmar"] for s in segments.values()) if checks["nonzero_drawdowns"] else None,
                                    simplicity_tiebreak=complexity(c)))
    high = sorted((r for r in high_rows if r["eligible"]),
                  key=lambda r: (-r["metrics"]["cagr"], r["simplicity_tiebreak"], r["metrics"]["turnover"], r["id"]))
    robust = sorted((r for r in robust_rows if r["eligible"]),
                    key=lambda r: (-r["score"], r["simplicity_tiebreak"], r["metrics"]["turnover"], r["id"]))
    return dict(high_return=high[0]["id"] if high else None,
                robust=robust[0]["id"] if robust else None,
                high_return_constraints=high_rows, robust_constraints=robust_rows,
                high_return_improved_development=bool(high and high[0]["metrics"]["cagr"] > base["cagr"]))


def develop():
    protected = verify_protected()
    registration = registered()
    OUT.mkdir(exist_ok=True)
    if (OUT / "selection.json").exists():
        raise RuntimeError("Selection already exists. Do not silently rerun selection after validation; use a new audited batch.")
    dump(OUT / "registration.json", dict(registration, registered_at=stamp()))
    histories = load_histories()
    calendar = [r[0] for r in histories["510300"]]
    features = Features(histories)
    results = {}
    for c in CANDIDATES:
        r = simulate(c["id"], histories, calendar, features, end=TRAIN_END)
        results[c["id"]] = r
        m = summarize(r)
        print("DEV %-20s CAGR %6.2f%% DD %6.2f%% turnover %.1f/y" %
              (c["id"], m["cagr"] * 100, m["max_dd"] * 100, m["turnover_per_year"]), flush=True)
    cash = cash_reference(histories, calendar, TRAIN_END)
    if registered() != registration:
        raise AssertionError("Protocol, implementation or snapshots changed while computing development")
    dump(OUT / "development.json", dict(registration=registration, results=results, cash_reference=cash,
                                       protected_files_verified=protected))
    picked = selection(results, cash)
    picked.update(selected_at=stamp(), development_sha256=sha(OUT / "development.json"),
                  registration_sha256=sha(OUT / "registration.json"), validation_not_yet_computed=True,
                  caveat="This split is already-known historical data, not a clean OOS holdout.")
    dump(OUT / "selection.json", picked)
    verify_protected()
    print("FROZEN SELECTION:", picked["high_return"], picked["robust"], flush=True)


def interval_summary(result):
    return dict(full=summarize(result), **{name: summarize(result, lo, hi)
                                         for name, (lo, hi) in PERIODS.items()})


def validate():
    verify_protected()
    dev = json.loads((OUT / "development.json").read_text())
    picked = json.loads((OUT / "selection.json").read_text())
    if sha(OUT / "development.json") != picked["development_sha256"]:
        raise AssertionError("Development results changed after selection")
    if sha(OUT / "registration.json") != picked["registration_sha256"]:
        raise AssertionError("Registration changed after selection")
    if registered() != dev["registration"]:
        raise AssertionError("Implementation/protocol changed since development; record a correction before rerunning")
    histories = load_histories()
    calendar = [r[0] for r in histories["510300"]]
    features = Features(histories)
    full_results, summaries, cost_results = {}, {}, {}
    for c in CANDIDATES:
        name = c["id"]
        result = simulate(name, histories, calendar, features)
        prefix = [r for r in result["daily"] if r[0] <= TRAIN_END]
        if json.loads(json.dumps(prefix)) != dev["results"][name]["daily"]:
            raise AssertionError("Future data/execution changed development prefix: " + name)
        full_results[name] = result
        summaries[name] = interval_summary(result)
        summaries[name]["attribution"] = attribution(result, histories)
        m = summaries[name]["full"]
        print("FULL %-20s CAGR %6.2f%% DD %6.2f%% validation %6.2f%%" %
              (name, m["cagr"] * 100, m["max_dd"] * 100, summaries[name]["validation"]["cagr"] * 100), flush=True)
        cost_results[name] = {"10": interval_summary(result)}
        for bp in (5, 20, 50):
            other = simulate(name, histories, calendar, features, slippage=bp / 10000)
            cost_results[name][str(bp)] = interval_summary(other)
    neighbors = []
    for route in ("high_return", "robust"):
        name = picked[route]
        if not name:
            continue
        for scale in (.9, 1.1):
            c = get_candidate(name)
            field = "score_windows" if c["kind"] == "legacy" else "lookbacks"
            c[field] = tuple(max(2, round(n * scale)) for n in c[field])
            result = simulate(c, histories, calendar, features)
            neighbors.append(dict(route=route, selected_id=name, scale=scale,
                                  config=c, metrics=interval_summary(result)))
    cash = cash_reference(histories, calendar, END)
    coverage = {}
    for c, rows in histories.items():
        coverage[c] = dict(first=rows[0][0], last=rows[-1][0], bars=len(rows),
                           first_244_return=rows[244][0] if len(rows) > 244 else None,
                           first_270_row_legacy_feature=rows[269][0] if len(rows) > 269 else None)
    report = dict(selection_sha256=sha(OUT / "selection.json"), registered=dev["registration"],
                  completed_at=stamp(), selection=picked, summaries=summaries,
                  costs=cost_results, neighbors=neighbors,
                  cash_reference=interval_summary(cash), coverage=coverage,
                  development_prefixes_unchanged=True, protected_files_verified=verify_protected(),
                  total_registered_candidates=len(CANDIDATES), additional_cost_runs=30,
                  neighborhood_runs=len(neighbors),
                  caveats=["All historical periods were previously inspected; none is a clean OOS sample.",
                           "Actual units/cash next-open engine; commissions charged on executed notional.",
                           "End 2026-09-11 is the last common data date including the bond ETF.",
                           "No leverage; no QVIX, volume fear or learned weights in new candidates.",
                           "Historical adjusted prices, ETF universe selection, price-limit fills and market impact remain limitations.",
                           "Window/cost diagnostics cannot replace the frozen development winners."])
    dump(OUT / "paths.json", dict(results=full_results, cash_reference=cash))
    dump(OUT / "evaluation.json", report)
    if registered() != dev["registration"]:
        raise AssertionError("Source changed during validation")
    print("Written", OUT / "evaluation.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("develop", "validate"))
    args = parser.parse_args()
    (develop if args.stage == "develop" else validate)()


if __name__ == "__main__":
    main()
