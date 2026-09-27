"""Unified registered V12 evaluation. This module never changes old caches.

Registration and input preparation are separate explicit steps. `evaluate`
requires their receipts, seals all source/input hashes before simulation, and
freezes selection only after all four complete path matrices are saved.
"""
import argparse
from bisect import bisect_left, bisect_right
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import BASE, ROOT, START, END, sha, dump, stamp, load_inputs, protect
from .native import Simulator
from .allocation import run_allocation
from v10_deep.portfolios import CASH, prices_from_features, run_weights
from v10_deep.scan import metrics


SCENARIOS = (("close_1bp", 0, .0001), ("close_11bp", 0, .0011),
             ("lag1_1bp", 1, .0001), ("lag1_11bp", 1, .0011))
BLOCKS = (("early", START, "2017-12-31"), ("middle", "2018-01-01", "2021-12-31"),
          ("recent", "2022-01-01", "2025-12-31"))
OUTPUT = BASE / "results/main"
NUMERIC_FIELDS = ("returns", "holdings", "summary", "turnover_equivalent", "turnover_by_day")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _scenarios(values):
    result = tuple(tuple(value) for value in values)
    require(result and all(len(v) == 3 for v in result), "Nonempty (name,lag,fee) scenarios are required")
    require(len({v[0] for v in result}) == len(result), "Scenario names must be unique")
    for name, lag, fee in result:
        require(isinstance(name, str) and name and type(lag) is int and lag in (0, 1)
                and np.isfinite(fee) and 0 <= fee < .5, "Invalid scenario")
    return result


def _strict_weights(prices, dates, desired, start, end, fee):
    """Reuse the units engine, rejecting the entire nonzero rebalance on gaps."""
    blocked = []

    def policy(i, state):
        target = desired(i, state)
        if target is None:
            return None
        turnover = sum(abs(target.get(c, 0.) - state["weights"].get(c, 0.))
                       for c in set(target) | set(state["weights"]))
        missing = sorted(c for c in set(target) | set(state["weights"])
                         if state["current_quotes"][c] is None and (target.get(c, 0.) > 0 or state["weights"].get(c, 0.) > 0))
        if turnover > 1e-12 and missing:
            blocked.append(dict(date=dates[i], target=deepcopy(target), missing_codes=missing,
                                reason="strict_whole_rebalance_missing_leg"))
            return None
        return target

    result = run_weights(prices, dates, policy, start, end, fee=fee)
    combined = sorted(blocked + result["diagnostics"]["blocked_rebalances"], key=lambda row: row["date"])
    result["diagnostics"].update(blocked_rebalance_count=len(combined), blocked_rebalances=combined)
    return result


def _fixed_exposure(prices, dates, tape, lo, hi, weight, fee):
    require(weight in (.5, .75), "Fixed exposure controls use 50% or 75%")

    def desired(i, state):
        code = tape[i]
        if code is None:
            return None
        return {CASH: 1.} if code == CASH else {code: weight, CASH: 1. - weight}

    return _strict_weights(prices, dates, desired, lo, hi, fee)


def _benchmark(record, arrays, meta, prices, lo, hi, lag, fee):
    spec = record["benchmark"]
    assets = list(spec["assets"])
    require(assets and len(set(assets)) == len(assets) and CASH not in assets
            and all(c in prices for c in assets), "Invalid benchmark assets")
    mode = spec["mode"]
    require(mode in ("buy_hold", "monthly_equal") and (mode != "buy_hold" or len(assets) == 1), "Invalid benchmark mode")
    fields = {name: j for j, name in enumerate(meta["feature_names"])}
    indices = {code: j for j, code in enumerate(meta["assets"])}
    dates = meta["dates"]

    def mature(code, q):
        if q < 0:
            return False
        # Frozen valid already includes 270 ACTUAL own quotes, including
        # pre-calendar history. Counting only benchmark-aligned rows is wrong.
        row = arrays["features"][q, indices[code]]
        return row[fields["valid"]] > .5 and np.isfinite(row[fields["close"]]) and row[fields["close"]] > 0

    def desired(i, state):
        q = i - lag
        if mode == "buy_hold":
            if state["units"].get(assets[0], 0.) > 0:
                return None  # Missing held bars carry marks, never reset to cash.
            return {assets[0]: 1.} if mature(assets[0], q) else {CASH: 1.}
        first_signal_month_day = q >= 0 and (q == 0 or dates[q][:7] != dates[q - 1][:7])
        if i != lo and not first_signal_month_day:
            return None
        targets = {code: 1. / len(assets) for code in assets if mature(code, q)}
        cash_weight = 1. - sum(targets.values())
        if cash_weight > 1e-12:
            targets[CASH] = cash_weight
        return targets

    return _strict_weights(prices, dates, desired, lo, hi, fee)


def _portfolio_summary(result, days):
    daily = np.asarray(result["returns"], dtype=np.float64)
    require(daily.shape == (days,) and np.isfinite(daily).all() and np.all(daily > -1), "Invalid portfolio returns")
    diagnostic = result["diagnostics"]
    # For portfolio rows columns 3/4 count rebalance events, not the number of
    # single-security changes; scalar/per-day L1/2 is stored separately.
    return np.asarray([result["nav"], result["ann"], result["max_dd"], result["switches"], len(result["trades"]),
                       diagnostic["blocked_rebalance_count"], len(diagnostic["missing_held_bars"]), 0.,
                       float(daily.sum()), float(np.dot(daily, daily))], dtype=np.float64)


def _capture_portfolio(payload, index, record, result, dates, extra=None):
    payload["returns"][index] = result["returns"]
    payload["summary"][index] = _portfolio_summary(result, len(dates))
    lookup = {date: i for i, date in enumerate(dates)}
    for trade in result["trades"]:
        if not trade["initial_session_free"]:
            payload["turnover_by_day"][index, lookup[trade["date"]]] += trade["turnover"] / 2.
    payload["turnover_equivalent"][index] = payload["turnover_by_day"][index].sum()
    metadata = dict(kind=record["kind"], diagnostics=deepcopy(result["diagnostics"]),
                    first_fill=result["trades"][0] if result["trades"] else None,
                    summary_columns_3_4="Non-initial rebalance events / all successful rebalance events",
                    holdings_sentinel=-1, holdings_sentinel_means="Portfolio, not an empty position",
                    turnover_definition="sum(noninitial actual security L1 turnover)/2")
    if "overlay_metadata" in result:
        metadata["overlay"] = result["overlay_metadata"]
    metadata.update(extra or {})
    payload["metadata"][record["id"]] = metadata


def run_candidates(records, arrays, meta, start=START, end=END, scenarios=SCENARIOS, workers=1, cache_dir=None):
    """All rows retain registry order; no performance filtering or selection.

    Allocation base_id must resolve to a single record in this same input set.
    Portfolio holdings are -1 by contract; daily turnover and metadata prevent
    that sentinel from being mistaken for uninvested cash. Only V12 compiles.
    """
    records, scenarios = list(records), _scenarios(scenarios)
    require(records and len({r["id"] for r in records}) == len(records), "Unique nonempty candidate records required")
    require(workers in (1, 2), "Use one or two native workers")
    dates_all = list(meta["dates"])
    require(dates_all == sorted(set(dates_all)), "Invalid feature calendar")
    lo, stop = bisect_left(dates_all, start), bisect_right(dates_all, end)
    dates = dates_all[lo:stop]
    require(dates, "Empty evaluation interval")
    # Expose only the evaluation prefix, including pre-start warmup, to every
    # engine and the volatility allocator. Even malformed future prices must
    # not affect whether a historical training prefix can be evaluated.
    if stop < len(dates_all):
        arrays = {name: np.ascontiguousarray(value[:, :stop] if name in ("scores", "orders") else value[:stop])
                  for name, value in arrays.items()}
        meta = deepcopy(meta)
        meta["dates"] = dates_all[:stop]
        if isinstance(meta.get("shape"), list) and meta["shape"]:
            meta["shape"][0] = stop
        dates_all = meta["dates"]
    hi, n, days = stop - 1, len(records), len(dates)
    index = {r["id"]: i for i, r in enumerate(records)}
    require(all(r["kind"] in ("single", "allocation", "benchmark") for r in records), "Unknown candidate kind")
    singles = [i for i, row in enumerate(records) if row["kind"] == "single"]
    allocation_bases = sorted({row["allocation"]["base_id"] for row in records if row["kind"] == "allocation"})
    for cid in allocation_bases:
        require(cid in index and records[index[cid]]["kind"] == "single", "Allocation base_id needs its explicit single record")
        require(not records[index[cid]].get("selectable", False), "Allocation anchors must be fixed controls, not new winners")
    results = {name: dict(returns=np.full((n, days), np.nan), holdings=np.full((n, days), -1, dtype=np.int32),
                         summary=np.full((n, 10), np.nan), turnover_equivalent=np.zeros(n),
                         turnover_by_day=np.zeros((n, days)), metadata={}) for name, unused, fee in scenarios}
    simulator = Simulator(arrays, meta, cache_dir=cache_dir) if singles else None
    for name, lag, fee in scenarios:
        payload = results[name]
        for offset in range(0, len(singles), 128):
            positions = singles[offset:offset + 128]
            configs = [dict(deepcopy(records[i]["config"]), lag=lag) for i in positions]
            result = simulator.run(configs, start=start, end=end, fee=fee, workers=workers)
            require(result["dates"] == dates, "Native date axis changed")
            for field in ("returns", "holdings", "summary"):
                payload[field][positions] = result[field]
        for i in singles:
            switches = np.r_[0., payload["holdings"][i, 1:] != payload["holdings"][i, :-1]]
            require(float(switches.sum()) == payload["summary"][i, 3], "Native switch count differs from actual holdings")
            payload["turnover_by_day"][i] = switches
            payload["turnover_equivalent"][i] = switches.sum()
            payload["metadata"][records[i]["id"]] = dict(kind="single", blocked_rebalances=int(payload["summary"][i, 5]),
                missing_held_days=int(payload["summary"][i, 6]), full_switches=int(switches.sum()),
                configured_lag=lag, turnover_definition="Noninitial charged full switches")
    for lag in {scenario[1] for scenario in scenarios}:
        matching = [name for name, current_lag, unused in scenarios if current_lag == lag]
        for name in matching[1:]:
            require(np.array_equal(results[name]["holdings"][singles], results[matching[0]]["holdings"][singles]),
                    "Fees changed a price-only single/base model holding path")
    prices = prices_from_features(arrays, meta) if any(r["kind"] != "single" for r in records) else None
    for name, lag, fee in scenarios:
        payload, base_tapes, proofs = results[name], {}, {}
        for cid in allocation_bases:
            tape = [None] * len(dates_all)
            tape[lo:stop] = [meta["assets"][h] if h >= 0 else None for h in payload["holdings"][index[cid]]]
            provenance = dict(base_id=cid, base_config_sha256=digest(records[index[cid]]["config"]),
                              base_lag=lag, tape_scope="Native actual filled holdings from the same execution start/lag")
            clone = run_allocation(prices, dates_all, tape, lo, hi, mode="full_exposure", lag=lag, fee=fee,
                                   target_alignment="execution_day", provenance=provenance)
            require(clone["overlay_metadata"]["native_baseline_fidelity_check_eligible"],
                    "Allocation/native startup is incompatible; do not move the registered start: " + cid)
            errors = np.abs(np.asarray(clone["returns"]) - payload["returns"][index[cid]])
            require(float(errors.max()) <= 1e-12, "Full-exposure base replay differs from native: " + cid)
            held = [next(iter(row[2])) if len(row[2]) == 1 else None for row in clone["daily"]]
            require(held == tape[lo:stop], "Full-exposure base holdings differ from native")
            base_tapes[cid] = (tape, provenance, clone)
            proofs[cid] = dict(compatible=True, max_return_error=float(errors.max()), first_entry_free=True,
                              target_alignment="execution_day", base_lag=lag, observations=days)
        for i, record in enumerate(records):
            if record["kind"] == "single":
                continue
            if record["kind"] == "allocation":
                spec = record["allocation"]
                cid, mode = spec["base_id"], spec["mode"]
                tape, provenance, clone = base_tapes[cid]
                if mode == "full_exposure":
                    result = clone
                elif mode == "fixed_exposure":
                    result = _fixed_exposure(prices, dates_all, tape, lo, hi, spec["weight"], fee)
                else:
                    require(mode in ("vol_target", "progressive"), "Unknown registered allocation mode")
                    result = run_allocation(prices, dates_all, tape, lo, hi, mode=mode,
                        target_vol=spec.get("target_vol"), risk_context=spec.get("risk_context", "prior20"),
                        speed=spec.get("speed", 1.), lag=lag, fee=fee, target_alignment="execution_day", provenance=provenance)
                extra = dict(base_id=cid, base_replay_fidelity=proofs[cid],
                             allocation_spec=deepcopy(spec), base_model_state_independent=True)
            else:
                result = _benchmark(record, arrays, meta, prices, lo, hi, lag, fee)
                extra = dict(benchmark_spec=deepcopy(record["benchmark"]), maturity_rule="Frozen own-quote valid>=270",
                             rebalance_clock="Signal-day first observed calendar session of month; first evaluation session initializes")
            require([row[0] for row in result["daily"]] == dates, "Portfolio date axis changed")
            _capture_portfolio(payload, i, record, result, dates, extra)
        require(np.isfinite(payload["returns"]).all() and np.all(payload["returns"] > -1)
                and np.isfinite(payload["summary"]).all(), "Incomplete/invalid candidate results")
        print("evaluated", name, n, "candidates", days, "observations", flush=True)
    return results, dates


def summary_rows(records, results, dates, period_end=None):
    """Slice continuous returns and actual turnover; never restart a subperiod."""
    dates = list(dates)
    require(dates and dates == sorted(set(dates)), "Invalid result calendar")
    period_end = period_end or dates[-1]
    require(dates[-1] <= period_end and dates[-1][:4] == period_end[:4], "Result cutoff does not cover the declared tail year")
    rows = []
    for i, record in enumerate(records):
        row = dict(id=record["id"], kind=record["kind"], families=record.get("families", []),
                   period_end=period_end, scenarios={})
        for name, payload in results.items():
            daily = payload["returns"][i]
            turnover = payload["turnover_by_day"][i]

            def sliced(begin, end):
                a, b = bisect_left(dates, begin), bisect_right(dates, end)
                if b <= a:
                    return None
                value = metrics(daily[a:b])
                value.update(start=dates[a], end=dates[b - 1], turnover_equivalent=float(turnover[a:b].sum()))
                return value

            blocks = {label: value for label, begin, end in BLOCKS
                      for value in [sliced(begin, min(end, period_end))] if value is not None}
            years = sorted({date[:4] for date in dates})
            tail = sliced(period_end[:4] + "-01-01", period_end)
            require(tail is not None, "Declared tail has no observations")
            row["scenarios"][name] = dict(full=sliced(dates[0], period_end), blocks=blocks, tail=tail,
                yearly={year: sliced(year + "-01-01", min(year + "-12-31", period_end)) for year in years},
                turnover_equivalent=float(payload["turnover_equivalent"][i]),
                noninitial_rebalance_count=int(payload["summary"][i, 3]),
                blocked_rebalances=int(payload["summary"][i, 5]), crashes=int(payload["summary"][i, 7]))
        rows.append(row)
    return rows


summarize_rows = summary_rows


def fingerprints():
    sources = sorted(set(BASE.glob("*.py")) | set(BASE.glob("*.cpp")) | set(BASE.glob("*PROTOCOL.md"))
                     | set((BASE / "tests").rglob("*.py")))
    return dict(sources={str(path.relative_to(BASE)): sha(path) for path in sources},
        inputs={name: sha(BASE / name) for name in ("registered_candidates.json", "registration.json", "PROTOCOL.md",
            "protected_manifest.json", "inputs/manifest.json", "inputs/features.json", "inputs/features.npz.gz")})


def evaluate(output=OUTPUT, workers=1):
    """Execute only an already registered/prepared family; never auto-register."""
    from .registry import register
    from .selection import select
    output = Path(output)
    require((BASE / "registration.json").is_file() and (BASE / "registered_candidates.json").is_file(),
            "Explicitly register the protocol and candidate family before evaluation")
    require((BASE / "inputs/manifest.json").is_file(), "Prepare the pinned V12-only inputs before evaluation")
    require(not (output / "selection.json").exists(), "Selection is frozen; refusing overwrite")
    protect()
    registry = register()
    require(registry["unique_count"] == len(registry["candidates"]), "Registry count differs")
    expected = fingerprints()
    arrays, meta, unused_profiles = load_inputs()
    require(fingerprints() == expected, "Registered inputs/source changed during loading")
    design = dict(fingerprints=expected, candidate_count=registry["unique_count"], period=[START, END],
        scenarios=[list(s) for s in SCENARIOS], blocks=[list(b) for b in BLOCKS],
        known_history_including_2026=True, clean_oos=False,
        turnover_definition="Noninitial native full switches or actual portfolio L1 turnover/2",
        allocation_tape_alignment="execution_day; each lag's base is rerun, never shifted twice")
    registration_path = output / "registration.json"
    if registration_path.exists():
        require(json.loads(registration_path.read_text())["design"] == design, "This evaluation was registered with different inputs/source")
    else:
        dump(registration_path, dict(registered_at=stamp(), design=design))
    registration_hash = sha(registration_path)
    results, dates = run_candidates(registry["candidates"], arrays, meta, workers=workers)
    require(fingerprints() == expected and sha(registration_path) == registration_hash, "Inputs/source changed during evaluation")
    np.savez_compressed(output / "paths.npz", **{name + "__" + field: payload[field]
        for name, payload in results.items() for field in NUMERIC_FIELDS})
    dump(output / "path_metadata.json", dict(dates=dates, ids=[c["id"] for c in registry["candidates"]],
         sha256=sha(output / "paths.npz"), fields=list(NUMERIC_FIELDS),
         non_single_holdings=-1, non_single_holdings_meaning="Portfolio, not empty cash"))
    dump(output / "execution_metadata.json", dict(registration_sha256=registration_hash,
         scenarios={name: payload["metadata"] for name, payload in results.items()}))
    rows = summary_rows(registry["candidates"], results, dates, period_end=END)
    dump(output / "evaluation.json", dict(period=[START, END], observed_period=[dates[0], dates[-1]], rows=rows,
         registration_sha256=registration_hash, paths_sha256=sha(output / "paths.npz"), clean_oos=False))
    require(fingerprints() == expected and sha(registration_path) == registration_hash, "Inputs/source changed before selection")
    protect()
    selection = select(rows, registry["candidates"], registry["controls"], period_end=END)
    require(fingerprints() == expected and sha(registration_path) == registration_hash, "Inputs/source changed while selecting")
    selection.update(frozen_at=stamp(), provenance=dict(registration_sha256=registration_hash,
        registry_sha256=sha(BASE / "registered_candidates.json"), evaluation_sha256=sha(output / "evaluation.json"),
        paths_sha256=sha(output / "paths.npz"), path_metadata_sha256=sha(output / "path_metadata.json"),
        execution_metadata_sha256=sha(output / "execution_metadata.json"), fingerprints=expected))
    dump(output / "selection.json", selection)
    print(json.dumps({key: selection[key] for key in ("primary", "qualified_count", "status", "top_return", "top_tail", "least_drawdown")}, indent=2), flush=True)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate",))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    evaluate(workers=args.workers)
