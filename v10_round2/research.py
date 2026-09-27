"""Second, bounded batch with explicit H > v9.2 and S >= 30% objectives.

Run develop once to freeze choices, then validate to disclose later known data.
The earlier study and all production files remain byte-for-byte unchanged.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from v10_next.data import Features, load_histories
from v10_next.execution import run
from v10_next.legacy import LegacyPolicy
from v10_next.metrics import summarize, attribution
from .registry import CANDIDATES, HIGH_RETURN_IDS, STABILITY_IDS, START, END, FEE, SLIPPAGE, get_candidate, registry
from .v92 import factory as v92_factory, load_qvix
from .portfolios import run_accounts
from .timing import run_same_close, for_same_close


BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
TRAIN_END = "2021-12-31"
PERIODS = dict(full=(START, END), development=(START, TRAIN_END),
               validation=("2022-01-01", "2025-12-31"), report_only_2026=("2026-01-01", END))
OWN_SOURCES = ("registry.py", "research.py", "v92.py", "portfolios.py", "timing.py", "PROTOCOL.md")
PARENT_SOURCES = ("data.py", "execution.py", "legacy.py", "metrics.py", "candidates.py", "frozen/strategy.py", "frozen/metadata.py", "frozen/presets.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def protect():
    files = json.loads((BASE / "protected_manifest.json").read_text())["sha256"]
    bad = [name for name, expected in files.items()
           if not (BASE.parent / name).is_file() or sha(BASE.parent / name) != expected]
    if bad:
        raise AssertionError("Prior work changed: " + ", ".join(bad))
    return len(files)


def registered():
    value = dict(registry=registry(), source_sha256={p: sha(BASE / p) for p in OWN_SOURCES},
                 parent_source_sha256={p: sha(BASE.parent / "v10_next" / p) for p in PARENT_SOURCES},
                 prices_manifest_sha256=sha(BASE.parent / "v10_next/data_manifest.json"),
                 qvix_manifest_sha256=sha(BASE / "qvix_manifest.json"),
                 bootstrap_source_sha256=sha(BASE.parent / "research_overfitting.py"),
                 protocol_sha256=sha(BASE / "PROTOCOL.md"),
                 primary_clock="next_open", primary_slippage_bps=10, commission_bps=1,
                 costs_bps=[10, 20, 50], ideal_close_selection=False, neighborhood_scales=[.9, 1.1],
                 history_status="Previously examined history, including first-round results; not clean OOS")
    return json.loads(json.dumps(value))


class Simulator:
    def __init__(self, histories=None, qvix=None):
        self.histories = load_histories() if histories is None else histories
        self.calendar = [r[0] for r in self.histories["510300"]]
        self.features = Features(self.histories)
        self.qvix = load_qvix() if qvix is None else qvix
        self.cache = {}
        self.close_histories = {c: [(d, close, close, vol) for d, op, close, vol in rows]
                                for c, rows in self.histories.items()}

    def simulate(self, config, end=END, slippage=SLIPPAGE, clock="next_open"):
        config = get_candidate(config) if isinstance(config, str) else deepcopy(config)
        if clock not in ("next_open", "ideal_close"):
            raise ValueError("Unknown clock")
        semantic = {k: config[k] for k in ("kind", "base_version", "overrides", "score_windows", "stock_pool",
                                         "global_pool", "risk_weight", "channels") if k in config}
        if config["kind"] == "accounts":
            h = self.close_histories if clock == "ideal_close" else self.histories
            result = run_accounts(config, h, self.calendar,
                                  lambda c: self.simulate(c, end=end, slippage=slippage, clock=clock))
            result["execution_clock"] = clock
            return result
        key = json.dumps([semantic, end, slippage, clock], sort_keys=True)
        if key in self.cache:
            return deepcopy(self.cache[key])
        effective = for_same_close(config) if clock == "ideal_close" else config
        policy = (v92_factory(effective, self.features, self.calendar, self.qvix)
                  if effective["kind"] == "v92" else LegacyPolicy(effective, self.features, self.calendar))
        executor = run_same_close if clock == "ideal_close" else run
        result = executor(self.histories, self.calendar, policy, START, end, fee=FEE, slippage=slippage)
        result["policy_metadata"] = deepcopy(policy.metadata)
        result["execution_clock"] = clock
        self.cache[key] = deepcopy(result)
        return result


def select(results):
    baseline = summarize(results["c_v92"], START, TRAIN_END)
    hrows, srows = [], []
    for name in HIGH_RETURN_IDS:
        m = summarize(results[name], START, TRAIN_END)
        cfg = get_candidate(name)
        external = len(set(cfg.get("channels", ())) & {"qvix", "volume"})
        checks = dict(cagr_above_v92=m["cagr"] > baseline["cagr"] + 1e-12,
                      drawdown=abs(m["max_dd"]) <= abs(baseline["max_dd"]) + .02 + 1e-12)
        hrows.append(dict(id=name, eligible=all(checks.values()), checks=checks,
                          metrics=m, external_channels=external))
    for name in STABILITY_IDS:
        m = summarize(results[name], START, TRAIN_END)
        checks = dict(cagr_at_least_30=m["cagr"] >= .30,
                      drawdown=abs(m["max_dd"]) <= abs(baseline["max_dd"]) + 1e-12)
        srows.append(dict(id=name, eligible=all(checks.values()), checks=checks, metrics=m))
    high = sorted([r for r in hrows if r["eligible"]],
                  key=lambda r: (-r["metrics"]["cagr"], r["external_channels"], r["metrics"]["turnover"], r["id"]))
    robust = sorted([r for r in srows if r["eligible"]],
                    key=lambda r: (abs(r["metrics"]["max_dd"]), r["metrics"]["turnover"], r["id"]))
    return dict(high_return=high[0]["id"] if high else None, robust=robust[0]["id"] if robust else None,
                benchmark=baseline, high_rows=hrows, robust_rows=srows)


def develop():
    protect()
    OUT.mkdir(exist_ok=True)
    if (OUT / "selection.json").exists():
        raise RuntimeError("Selection is frozen; do not overwrite it after viewing validation")
    reg = registered()
    dump(OUT / "registration.json", dict(reg, registered_at=stamp()))
    simulator = Simulator()
    results = {}
    for c in CANDIDATES:
        name = c["id"]
        results[name] = simulator.simulate(name, end=TRAIN_END)
        m = summarize(results[name])
        print("DEV %-25s CAGR %6.2f%% DD %6.2f%%" % (name, m["cagr"] * 100, m["max_dd"] * 100), flush=True)
    if registered() != reg:
        raise AssertionError("Sources changed during development")
    dump(OUT / "development.json", dict(registration=reg, results=results))
    selection = select(results)
    selection.update(selected_at=stamp(), development_sha256=sha(OUT / "development.json"),
                     registration_sha256=sha(OUT / "registration.json"), validation_not_yet_computed=True)
    dump(OUT / "selection.json", selection)
    protect()
    print("FROZEN:", selection["high_return"], selection["robust"], flush=True)


def periods(result):
    return {name: summarize(result, lo, hi) for name, (lo, hi) in PERIODS.items()}


def scaled_windows(config, scale):
    c = deepcopy(config)
    if c["kind"] == "accounts":
        c["components"] = [scaled_windows(x, scale) for x in c["components"]]
    else:
        c["score_windows"] = tuple(max(2, round(w * scale)) for w in c["score_windows"])
    return c


def gates(summaries, picked):
    result = {}
    for period in ("full", "validation"):
        ref = summaries["c_v92"][period]["cagr"]
        h = summaries[picked["high_return"]][period]["cagr"] if picked["high_return"] else None
        s = summaries[picked["robust"]][period]["cagr"] if picked["robust"] else None
        result[period] = dict(v92_cagr=ref, high_cagr=h, high_difference=h - ref if h is not None else None,
                              high_pass=h is not None and h > ref + 1e-12, robust_cagr=s,
                              robust_pass=s is not None and s >= .30)
    return result


def validate():
    protect()
    dev = json.loads((OUT / "development.json").read_text())
    picked = json.loads((OUT / "selection.json").read_text())
    assert sha(OUT / "development.json") == picked["development_sha256"]
    assert sha(OUT / "registration.json") == picked["registration_sha256"]
    assert registered() == dev["registration"], "Implementation changed after selection"
    simulator = Simulator()
    paths, summary, costs, ideal = {}, {}, {}, {}
    for c in CANDIDATES:
        name = c["id"]
        r = simulator.simulate(name)
        prefix = json.loads(json.dumps([x for x in r["daily"] if x[0] <= TRAIN_END]))
        if prefix != dev["results"][name]["daily"]:
            raise AssertionError("Future observations changed development path: " + name)
        paths[name] = r
        summary[name] = periods(r)
        summary[name]["attribution"] = attribution(r, simulator.histories)
        costs[name] = {"10": periods(r)}
        for bp in (20, 50):
            costs[name][str(bp)] = periods(simulator.simulate(name, slippage=bp / 10000))
        ic = simulator.simulate(name, slippage=0, clock="ideal_close")
        ideal[name] = periods(ic)
        print("FULL %-25s next-open %6.2f%% DD %6.2f%% ideal-close %6.2f%%" % (
            name, summary[name]["full"]["cagr"] * 100, summary[name]["full"]["max_dd"] * 100,
            ideal[name]["full"]["cagr"] * 100), flush=True)
    neighbors = []
    for route in ("high_return", "robust"):
        selected = picked[route]
        if not selected:
            continue
        for scale in (.9, 1.1):
            c = scaled_windows(get_candidate(selected), scale)
            neighbors.append(dict(route=route, selected=selected, scale=scale, config=c,
                                  metrics=periods(simulator.simulate(c))))
    bootstrap = []
    if picked["high_return"]:
        from research_overfitting import paired_block_bootstrap
        a = paths[picked["high_return"]]["daily"]
        b = paths["c_v92"]["daily"]
        bootstrap = [paired_block_bootstrap(a, b, block_size=k, draws=10000, seed=20260928)
                     for k in (20, 60, 120)]
    report = dict(registration=dev["registration"], selection=picked, selection_sha256=sha(OUT / "selection.json"),
                  completed_at=stamp(), summaries=summary, costs=costs, ideal_close=ideal, neighbors=neighbors,
                  primary_acceptance=gates(summary, picked), ideal_close_diagnostic=gates(ideal, picked),
                  high_paired_bootstrap=bootstrap,
                  primary_clock="next_open_actual_positions", ideal_close_used_for_selection=False,
                  qvix_manifest=json.loads((BASE / "qvix_manifest.json").read_text()),
                  original_and_round1_files_verified=protect(), development_prefixes_unchanged=True,
                  additional_cost_runs=18, ideal_close_runs=9, neighborhood_runs=len(neighbors),
                  caveats=["Known, repeatedly researched history; no claim of clean OOS or eliminated overfitting.",
                           "Targets are historical net CAGR acceptance tests, not future return promises.",
                           "The same-close diagnostic assumes final close information is tradable at that close; not practical execution.",
                           "QVIX publication timestamps are unverified; missing same-day observations disable that channel.",
                           "Independent accounts start equally funded and drift without transfers or trade netting.",
                           "No leverage or hidden funding; all costs charged per actual fill.",
                           "Block bootstrap is conditional on already-selected paths and not multiple-testing adjusted."])
    if registered() != dev["registration"]:
        raise AssertionError("Source changed during validation")
    dump(OUT / "paths.json", paths)
    dump(OUT / "evaluation.json", report)
    print("Primary acceptance:", report["primary_acceptance"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("develop", "validate"))
    args = parser.parse_args()
    (develop if args.stage == "develop" else validate)()


if __name__ == "__main__":
    main()
