"""Development-only independent integration audit of registered V11 candidates.

Select cases from configuration IDs/families, never returns or the champion.
No strategy is executed past 2021-12-31. The Python oracle shares the verified
feature inputs but sorts/decides/accounts independently of the native engine.
"""
import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import BASE, ROOT, START, DEV_END, sha, dump
from .features import build, risk_view
from .scan import SCENARIOS, select
from v10_deep.reference import run_reference

CONTEXTS = ("current20", "prior20", "prior60", "prior_max20_60")
SUMMARY_FIELDS = ("nav", "cagr", "max_dd", "switches", "entries", "blocked", "missing_held", "crashes", "sum_returns", "sum_squared_returns")


def choose_cases(registry, meta):
    """Fixed family-first sampling, plus missing context/estimator/rank coverage."""
    candidates = sorted(registry["candidates"], key=lambda candidate: candidate["id"])
    specifications = {spec["name"]: spec for spec in registry["score_specs"]}
    reasons = defaultdict(list)
    coverage = {}
    def include(label, matches, count=1):
        chosen = [candidate["id"] for candidate in matches[:count]]
        coverage[label] = chosen
        for candidate_id in chosen:
            reasons[candidate_id].append(label)
    families = sorted({name for candidate in candidates for name in candidate["families"]})
    for family in families:
        include("family:" + family, [c for c in candidates if family in c["families"]], 2)
    for context in CONTEXTS:
        matches = [c for c in candidates if c["risk_context"] == context]
        if not matches:
            raise ValueError("Registered family lacks expected risk context: " + context)
        if not any(c["id"] in reasons for c in matches):
            include("supplement_context:" + context, matches)
    estimators = sorted({specifications[c["score"]]["estimator"] for c in candidates if c["score"] in specifications})
    for estimator in estimators:
        matches = [c for c in candidates if specifications.get(c["score"], {}).get("estimator") == estimator]
        if not any(c["id"] in reasons for c in matches):
            include("supplement_estimator:" + estimator, matches)
    rank_warmup = [c for c in candidates if specifications.get(c["score"], {}).get("aggregate") == "rank"
                   and meta["required_observations_by_score"][c["score"]] > 270]
    if not rank_warmup:
        raise ValueError("Expected at least one fully warmed smoothed rank candidate")
    if not any(c["id"] in reasons for c in rank_warmup):
        include("supplement_rank_smoothing_warmup", rank_warmup)
    chosen = [c for c in candidates if c["id"] in reasons]
    return chosen, dict(reasons), coverage


def feature_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def audit_grouping(candidates, arrays, meta):
    """Verify the scanner's context+valid-mask key determines the whole cube."""
    valid_index = meta["feature_names"].index("valid")
    groups, visited = {}, set()
    original_hashes = {key: feature_digest(value) for key, value in arrays.items()}
    for candidate in sorted(candidates, key=lambda c: c["id"]):
        lookup = candidate["risk_context"], candidate["score"]
        if lookup in visited:
            continue
        visited.add(lookup)
        view = risk_view(arrays, meta, lookup[0], score=lookup[1])
        key = lookup[0], feature_digest(view["features"][:, :, valid_index])
        complete = feature_digest(view["features"])
        if key in groups and groups[key]["feature_sha256"] != complete:
            raise AssertionError("Scanner grouping key aliases different decision features: " + str(lookup))
        for field in ("scores", "orders"):
            if view[field] is not arrays[field] and not np.array_equal(view[field], arrays[field]):
                raise AssertionError("Risk context modified shared scores or ranking order")
        if key not in groups:
            groups[key] = dict(context=lookup[0], valid_sha256=key[1], feature_sha256=complete, scores=[])
        groups[key]["scores"].append(lookup[1])
    if any(feature_digest(arrays[key]) != expected for key, expected in original_hashes.items()):
        raise AssertionError("Risk-view grouping mutated its input arrays")
    return dict(passed=True, checked_candidate_count=len(candidates), checked_score_context_pairs=len(visited),
                group_count=len(groups), groups=list(groups.values()),
                invariant="Within each risk context, a matching validity-mask SHA implies identical complete decision features; scores/orders remain shared and unchanged.")


def development_prefix(arrays, meta):
    """Give the oracle no dated feature/score rows from confirmation or 2026."""
    dates = meta["dates"]
    stop = sum(day <= DEV_END for day in dates)
    if not stop:
        raise ValueError("No development observations in feature cache")
    shortened = {}
    for key, value in arrays.items():
        if key in ("scores", "orders"):
            shortened[key] = np.array(value[:, :stop], copy=True, order="C")
        elif value.ndim and value.shape[0] == len(dates):
            shortened[key] = np.array(value[:stop], copy=True, order="C")
        else:
            shortened[key] = np.array(value, copy=True)
    view_meta = deepcopy(meta)
    view_meta["dates"] = dates[:stop]
    view_meta["shape"] = list(shortened["features"].shape)
    return shortened, view_meta


def compare_path(reference, expected_returns, expected_holdings, expected_summary, dates):
    if reference["dates"] != dates:
        raise AssertionError("Independent oracle and stored development calendars differ")
    if reference["returns"].shape != expected_returns.shape or reference["holdings"].shape != expected_holdings.shape:
        raise AssertionError("Independent path shape differs")
    returns_same = bool(np.array_equal(reference["returns"], expected_returns))
    holdings_same = bool(np.array_equal(reference["holdings"], expected_holdings))
    summary_same = bool(np.array_equal(reference["summary"], expected_summary))
    detail = dict(exact_returns=returns_same, exact_holdings=holdings_same, exact_summary=summary_same,
                  max_return_error=float(np.max(np.abs(reference["returns"] - expected_returns))),
                  max_summary_error=float(np.max(np.abs(reference["summary"] - expected_summary))),
                  holding_mismatch_count=int(np.count_nonzero(reference["holdings"] != expected_holdings)))
    if not (returns_same and holdings_same and summary_same):
        wrong = np.flatnonzero((reference["returns"] != expected_returns) | (reference["holdings"] != expected_holdings))
        if len(wrong):
            index = int(wrong[0])
            detail["first_mismatch"] = dict(date=dates[index], reference_return=float(reference["returns"][index]),
                stored_return=float(expected_returns[index]), reference_holding=int(reference["holdings"][index]),
                stored_holding=int(expected_holdings[index]))
        detail["different_summary_fields"] = [name for name, a, b in zip(SUMMARY_FIELDS, reference["summary"], expected_summary) if a != b]
        raise AssertionError(json.dumps(detail, sort_keys=True))
    return detail


def validate_calendar(dates):
    if not dates or dates != sorted(set(dates)) or dates[0] < START or dates[-1] > DEV_END:
        raise ValueError("This audit accepts only sorted, unique development dates in 2014-2021")


def _expect(path, expected):
    if sha(path) != expected:
        raise ValueError("Frozen audit input changed: " + str(path))


def verify(overwrite=False):
    directory = BASE / "results/development"
    destination = BASE / "results/development_fidelity.json"
    if destination.exists() and not overwrite:
        raise FileExistsError("Fidelity receipt already exists; use --overwrite for an explicit rerun")
    registration = json.loads((directory / "registration.json").read_text())
    path_meta = json.loads((directory / "path_metadata.json").read_text())
    registry_path = BASE / "registered_candidates.json"
    registry = json.loads(registry_path.read_text())
    selection = json.loads((directory / "selection.json").read_text())
    evaluation = json.loads((directory / "evaluation.json").read_text())
    dates = path_meta["dates"]
    validate_calendar(dates)
    if registration["only_evaluated_until"] != DEV_END or evaluation["period"] != [dates[0], dates[-1]]:
        raise ValueError("Development artifact period differs from its registration")
    if path_meta["ids"] != [c["id"] for c in registry["candidates"]] or len(set(path_meta["ids"])) != len(path_meta["ids"]):
        raise ValueError("Stored path rows do not match the full registered candidate order")
    if registration["candidate_count"] != len(path_meta["ids"]):
        raise ValueError("Candidate count changed")
    _expect(registry_path, registration["registry_sha256"])
    _expect(directory / "paths.npz", path_meta["sha256"])
    for name, expected in registration["sources"].items():
        _expect(BASE / name, expected)
    protected = json.loads((BASE / "protected_manifest.json").read_text())["sha256"]
    for name in ("v10_deep/reference.py", "v10_deep/native.cpp", "v10_deep/native.py"):
        _expect(ROOT / name, protected[name])
    provenance = selection["provenance"]
    for name, key in (("registration.json", "registration_sha256"), ("evaluation.json", "evaluation_sha256"), ("paths.npz", "paths_sha256")):
        _expect(directory / name, provenance[key])
    if provenance["registry_sha256"] != registration["registry_sha256"] or provenance["source_sha256"] != registration["sources"]:
        raise ValueError("Selection provenance differs from the original registration")
    if [list(row) for row in SCENARIOS] != registration["scenarios"]:
        raise ValueError("Scenario definitions changed")
    arrays, meta = build(registry["score_specs"])
    if meta["fingerprints"] != registration["feature_fingerprints"]:
        raise ValueError("Feature construction differs from development")
    candidates, reasons, coverage = choose_cases(registry, meta)
    # Configuration sampling is frozen above before opening candidate returns.
    watched_paths = [registry_path, BASE / "protected_manifest.json", directory / "registration.json", directory / "path_metadata.json",
        directory / "paths.npz", directory / "selection.json", directory / "evaluation.json",
        Path(meta["cache_paths"]["arrays"]), Path(meta["cache_paths"]["metadata"]), Path(__file__),
        ROOT / "v10_deep/reference.py", ROOT / "v10_deep/native.cpp", ROOT / "v10_deep/native.py"]
    watched_paths += [BASE / name for name in registration["sources"]]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in watched_paths}
    result = dict(passed=False, period=[dates[0], dates[-1]], observations=len(dates),
        tested_candidate_ids=[c["id"] for c in candidates], sampling_reasons=reasons, sampling_coverage=coverage,
        sampling_rule="Per family first two IDs sorted lexically; supplement first-ID missing risk contexts, estimators and smoothed-rank warmup; no returns/selected IDs used.",
        independent_decision_and_accounting=True, shared_feature_inputs=True,
        confirmation_or_2026_evaluated=False, input_sha256=source_hashes, cases=[],
        limitations=["This verifies implementation fidelity, not future performance or resistance to overfitting.",
                     "All periods remain known history. The oracle shares precomputed, separately tested features."])
    try:
        result["grouping_audit"] = audit_grouping(registry["candidates"], arrays, meta)
        # Reproduce frozen selection, dropping only timestamp/provenance fields.
        replayed = select(evaluation["rows"], registry["candidates"], registry["controls"])
        checked_selection_fields = [name for name in replayed if name != "frozen_at"]
        if any(replayed[name] != selection[name] for name in checked_selection_fields):
            raise AssertionError("Frozen selection does not reproduce from development rows")
        result["selection_audit"] = dict(passed=True, reproduced_fields=checked_selection_fields,
            protocol_source_sha256=registration["sources"]["PROTOCOL.md"],
            note="Current selector matches registered 90%/30% CAGR floor, two-percentage-point DD gates, both high-cost scenarios and worst/median development-block tie order; direct boundary/unused-field tests are separate.")
        truncated, truncated_meta = development_prefix(arrays, meta)
        del arrays
        result["oracle_last_visible_date"] = truncated_meta["dates"][-1]
        with np.load(directory / "paths.npz", allow_pickle=False) as archive:
            expected = {scenario: {field: archive[scenario + "__" + field] for field in ("returns", "holdings", "summary")}
                        for scenario, unused_lag, unused_fee in SCENARIOS}
        row_index = {candidate_id: i for i, candidate_id in enumerate(path_meta["ids"])}
        for number, candidate in enumerate(candidates, 1):
            view = risk_view(truncated, truncated_meta, candidate["risk_context"], score=candidate["score"])
            index = row_index[candidate["id"]]
            for scenario, lag, fee in SCENARIOS:
                config = deepcopy(candidate)
                config["lag"] = lag
                reference = run_reference(view, truncated_meta, config, start=START, end=DEV_END, fee=fee)
                detail = compare_path(reference, expected[scenario]["returns"][index], expected[scenario]["holdings"][index],
                                      expected[scenario]["summary"][index], dates)
                detail.update(id=candidate["id"], scenario=scenario, risk_context=candidate["risk_context"], score=candidate["score"],
                              required_observations=meta["required_observations_by_score"][candidate["score"]])
                result["cases"].append(detail)
            print("V11 independent fidelity %d/%d: %s, all four development scenarios exact" % (number, len(candidates), candidate["id"]), flush=True)
        for name, expected_hash in source_hashes.items():
            _expect(ROOT / name, expected_hash)
        result.update(passed=True, completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        result.update(failure=type(error).__name__ + ": " + str(error), completed_at=datetime.now(timezone.utc).isoformat())
        dump(destination, result)
        raise
    dump(destination, result)
    print(destination, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    verify(overwrite=args.overwrite)
