"""Registered diagnostics for frozen finalists; never select candidates here.

Run only after the requested stage has frozen its choices. Costs and one-close
delay use the existing Simulator. Precision features are rebuilt in memory by
the original feature implementation; its real .npz/.json cache is never saved.
One-close delay is NOT next-open execution: no placeholder open is consumed.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import features
from .data import BASE, ROOT, START, END, dump, sha, protect
from .native import Simulator
from .precision import SCENARIOS, build_scenario
from .scan import summarize
from .schema import identifier


FEES = (.0001, .0005, .0011, .0021)


def _stamp():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text())


def scenario_specs():
    """Finite diagnostics fixed independently of their observed performance."""
    result = []
    for clock, lag in (("registered_lag", None), ("lag1", 1)):
        for fee in FEES:
            result.append(dict(name="%s_fee%dbp" % (clock, round(fee * 10000)),
                               fee=fee, force_lag=lag, data_scenario="base"))
    result.extend(dict(name=name, fee=.0001, force_lag=None, data_scenario=name)
                  for name in SCENARIOS)
    return result


def _cache_hashes():
    cache = features.BASE / "cache"
    return {name: sha(cache / name) if (cache / name).is_file() else None
            for name in ("features.npz", "features.json")}


def _assert_cache(expected):
    if _cache_hashes() != expected:
        raise AssertionError("Real feature cache changed during sensitivity construction")


def _reject_cache_write(*args, **kwargs):
    raise RuntimeError("Base feature cache is missing/stale; prepare it explicitly before stress")


def cached_base_features():
    """Use original build/cache validation, while forbidding an implicit save."""
    before = _cache_hashes()
    if any(value is None for value in before.values()):
        raise RuntimeError("Base feature cache must already exist before stress")
    try:
        with patch.object(features, "dump", _reject_cache_write), \
                patch.object(features.np, "savez_compressed", _reject_cache_write):
            return features.build()
    finally:
        _assert_cache(before)


def precision_features(histories, scenario_metadata, fear, expected_calendar):
    """Single-process, in-memory rebuild; restore all patched functions first.

    This temporarily patches a module-global NumPy writer. Do not call from
    parallel threads. Native simulations run only after this context has ended.
    """
    calendar = list(scenario_metadata["calendar"])
    if calendar != list(expected_calendar) or len(fear) != len(calendar):
        raise ValueError("Precision scenario changed the registered calendar/fear alignment")
    before = _cache_hashes()
    try:
        with patch.object(features, "inputs", lambda: (histories, calendar, fear)), \
                patch.object(features, "dump", lambda *args, **kwargs: None), \
                patch.object(features.np, "savez_compressed", lambda *args, **kwargs: None):
            arrays, returned_meta = features.build(force=True)
    finally:
        _assert_cache(before)
    meta = deepcopy(returned_meta)
    if list(meta["dates"]) != calendar:
        raise AssertionError("Feature builder returned another date axis")
    # The original builder fingerprints the frozen base manifest before reading
    # its inputs. Replace the effective data identity, retaining base provenance.
    fp = meta["fingerprints"]
    fp["base_data_manifest_sha256"] = fp.pop("data")
    fp.update(data=scenario_metadata["histories_content_sha256"],
              data_hash_kind="canonical_precision_scenario_histories",
              precision_scenario=scenario_metadata["scenario"],
              precision_input_manifest_content_sha256=scenario_metadata["input_manifest_content_sha256"],
              precision_source_file_sha256=deepcopy(scenario_metadata["source_file_sha256"]),
              precision_constructor_source_sha256=scenario_metadata["constructor_source_sha256"])
    meta.update(data_view=scenario_metadata["data_view"],
                precision_scenario=scenario_metadata["scenario"], cache_persisted=False,
                usable_for_next_open=False)
    return arrays, meta


def effective_configs(configs, spec):
    """Keep source IDs for pairing; separately identify altered lag semantics."""
    result = []
    for config in configs:
        c = deepcopy(config)
        if spec["force_lag"] is not None:
            c["lag"] = spec["force_lag"]
        h = identifier(c)
        c.update(id="vd_" + h[:20], hash=h)
        result.append(c)
    return result


def finalist_plan(stage):
    if not stage or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in stage):
        raise ValueError("Invalid registered stage name")
    stage_path = BASE / "results" / stage
    first_path = BASE / "results" / "mechanisms"
    selected = _read(stage_path / "selection.json")
    first = _read(first_path / "selection.json")
    registration = _read(stage_path / "registration.json")
    if selected["registration_sha256"] != sha(stage_path / "registration.json"):
        raise ValueError("Final-stage selection registration hash mismatch")
    if first["registration_sha256"] != sha(first_path / "registration.json"):
        raise ValueError("First-stage selection registration hash mismatch")
    references, by_id = {}, {}
    for folder in dict.fromkeys((stage_path, first_path)):
        reg = _read(folder / "registration.json")
        registry_path = folder / "registered_candidates.json"
        if sha(registry_path) != reg["registry_sha256"]:
            raise ValueError("Frozen candidate registry hash mismatch")
        for filename in ("registration.json", "registered_candidates.json", "selection.json"):
            relative = str((folder / filename).relative_to(BASE))
            references[relative] = sha(folder / filename)
        for c in _read(registry_path)["candidates"]:
            h = identifier(c)
            if c["hash"] != h or c["id"] != "vd_" + h[:20]:
                raise ValueError("Candidate semantic identity mismatch: " + c["id"])
            if c["id"] in by_id and identifier(by_id[c["id"]]) != h:
                raise ValueError("Different semantics share a source candidate ID")
            by_id[c["id"]] = c
    roles = dict(primary=selected["primary"], guarded=selected["guarded"],
                 control_v9=selected["controls"]["v9"], control_v91=selected["controls"]["v9.1"],
                 control_v92=selected["controls"]["v9.2"], stage1_primary=first["primary"])
    ids = list(dict.fromkeys(c for c in roles.values() if c))
    if any(c not in by_id for c in ids):
        raise ValueError("A frozen finalist/control is absent from its registered sources")
    return dict(stage=stage, roles=roles, configs=[deepcopy(by_id[c]) for c in ids],
                source_artifact_sha256=references, parent_registration=registration)


def _source_fingerprints():
    names = ("stress.py", "precision.py", "features.py", "native.py", "native.cpp",
             "schema.py", "data.py", "scan.py")
    return {name: sha(BASE / name) for name in names}


def _assert_registration_inputs(plan, source_hashes, cache_hashes):
    if _source_fingerprints() != source_hashes:
        raise AssertionError("Stress implementation changed after registration")
    if any(sha(BASE / name) != value for name, value in plan["source_artifact_sha256"].items()):
        raise AssertionError("Frozen stage selection or registry changed during stress")
    _assert_cache(cache_hashes)


def _evaluate(simulator, original_configs, effective, roles, spec, workers):
    result = simulator.run(effective, start=START, end=END, fee=spec["fee"], workers=workers, capture=True)
    dates = list(result["dates"])
    if not dates or dates[0] != START or dates[-1] != END:
        raise AssertionError("Stress result does not cover the registered research period")
    returns = np.asarray(result["returns"])
    if returns.shape != (len(original_configs), len(dates)) or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ArithmeticError("Invalid stress return matrix")
    rows = []
    for i, (original, c) in enumerate(zip(original_configs, effective)):
        summary = result["summary"][i]
        rows.append(dict(id=original["id"], hash=original["hash"],
                         effective_id=c["id"], effective_hash=c["hash"], lag=c["lag"],
                         roles=[role for role, source_id in roles.items() if source_id == original["id"]],
                         metrics=summarize(returns[i], dates), switches=int(summary[3]),
                         blocked=int(summary[5]), missing_held=int(summary[6]), crashes=int(summary[7])))
    return dict(spec=deepcopy(spec), rows=rows, sessions=len(dates),
                first_date=dates[0], last_date=dates[-1],
                feature_fingerprints=deepcopy(simulator.meta["fingerprints"]),
                native_build_fingerprint=deepcopy(simulator.build_fingerprint))


def run_stress(stage="refinements", workers=2):
    """Register then evaluate fixed diagnostics; no chooser, matrix export or retuning."""
    if workers not in (1, 2):
        raise ValueError("Use one or two native workers")
    protect()
    plan = finalist_plan(stage)
    output = BASE / "results" / stage / "stress"
    if (output / "registration.json").exists() or (output / "evaluation.json").exists():
        raise RuntimeError("Stress already registered; do not overwrite diagnostics")
    arrays, meta = cached_base_features()
    parent_fp = plan["parent_registration"]["fingerprints"]
    if meta["fingerprints"] != parent_fp["features"]:
        raise ValueError("Feature inputs differ from frozen final-stage registration")
    cache_hashes = _cache_hashes()
    if cache_hashes["features.npz"] != parent_fp["feature_cache_sha256"]:
        raise ValueError("Feature cache differs from frozen final-stage registration")
    source_hashes = _source_fingerprints()
    base_calendar, base_fear = list(meta["dates"]), arrays["fear"].copy()
    precision_metadata = {}
    # Data preparation only: no feature bank or candidate returns evaluated yet.
    for name in SCENARIOS:
        histories, detail = build_scenario(name)
        if detail["calendar"] != base_calendar:
            raise ValueError("Scenario calendar differs before stress registration")
        precision_metadata[name] = detail
        del histories
    specs = scenario_specs()
    cases = [dict(spec=spec, effective_configs=effective_configs(plan["configs"], spec)) for spec in specs]
    registration = dict(registered_at=_stamp(), stage=stage, roles=plan["roles"],
                        source_configs=plan["configs"], distinct_source_config_count=len(plan["configs"]),
                        cases=cases, scenario_count=len(cases),
                        source_artifact_sha256=plan["source_artifact_sha256"], source_sha256=source_hashes,
                        base_feature_fingerprints=meta["fingerprints"], base_feature_cache_sha256=cache_hashes,
                        precision_scenarios=precision_metadata, start=START, end=END, workers=workers,
                        execution="Original total-return close engine; lag=1 recomputes decisions from previous-session features with current actual holdings, then uses the current close valuation; never next open",
                        fee_convention="Single-side parameter; charged switch factor 1-2*fee, initial session free",
                        clean_oos=False, selection_feedback_permitted=False,
                        interpretation="Frozen final candidates and controls under preset diagnostics; never replace a selected candidate based on these outcomes")
    _assert_registration_inputs(plan, source_hashes, cache_hashes)
    dump(output / "registration.json", registration)
    registration_hash = sha(output / "registration.json")
    scenarios = []
    try:
        simulator = Simulator(arrays, meta)
        for case in cases:
            if case["spec"]["data_scenario"] != "base":
                continue
            scenarios.append(_evaluate(simulator, plan["configs"], case["effective_configs"],
                                       plan["roles"], case["spec"], workers))
            print("stress", stage, case["spec"]["name"], "complete", flush=True)
        del simulator, arrays, meta
        gc.collect()
        for case in cases:
            name = case["spec"]["data_scenario"]
            if name == "base":
                continue
            histories, detail = build_scenario(name)
            if detail != precision_metadata[name]:
                raise AssertionError("Precision inputs changed after stress registration")
            arrays, meta = precision_features(histories, detail, base_fear, base_calendar)
            del histories
            simulator = Simulator(arrays, meta)
            scenarios.append(_evaluate(simulator, plan["configs"], case["effective_configs"],
                                       plan["roles"], case["spec"], workers))
            del simulator, arrays, meta
            gc.collect()
            print("stress", stage, case["spec"]["name"], "complete", flush=True)
        _assert_registration_inputs(plan, source_hashes, cache_hashes)
        if sha(output / "registration.json") != registration_hash:
            raise AssertionError("Stress registration changed during evaluation")
        result = dict(completed_at=_stamp(), registration_sha256=registration_hash,
                      stage=stage, roles=plan["roles"], scenarios=scenarios,
                      base_feature_cache_unchanged=True, protected_files_verified=protect(),
                      selection_feedback_permitted=False, clean_oos=False,
                      limitations=["Lag1 is a one-close-delay diagnostic, not opening-price or 14:50 execution.",
                                   "Cost levels include no order-book impact or fill assurance.",
                                   "Cash bounds apply to detected event amounts, not to strategy performance.",
                                   "2026 figures are report-only cumulative returns on previously examined history."])
        dump(output / "evaluation.json", result)
        return result
    finally:
        _assert_cache(cache_hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="refinements")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run_stress(args.stage, args.workers)
