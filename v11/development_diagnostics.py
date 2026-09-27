"""Registered, development-only full-family diagnostics; never select a model.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.8 -B -m
v11.development_diagnostics. Only the completed 2014-2021 path archive is read.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .diagnostics import development_family_tests, advantage_concentration


BASE = Path(__file__).resolve().parent
DEVELOPMENT = BASE / "results/development"
OUTPUT = BASE / "results/development_diagnostics"
EXPECTED_COUNT = 586
SCENARIOS = ("close_1bp", "close_11bp", "lag1_1bp", "lag1_11bp")
BLOCKS = (20, 60)
DRAWS = 2000
SEED = 20260927
PERIOD = ["2014-01-02", "2021-12-31"]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_verified_inputs(base=BASE):
    """Verify the frozen selection's hash chain before opening its return paths."""
    base = Path(base)
    development = base / "results/development"
    files = {"registry": base / "registered_candidates.json"}
    files.update({name: development / (name + ".json") for name in
                  ("registration", "path_metadata", "evaluation", "selection", "control_fidelity")})
    files["paths"] = development / "paths.npz"
    hashes = {name: sha(path) for name, path in files.items()}
    payloads = {name: read(path) for name, path in files.items() if name != "paths"}
    registry, registration = payloads["registry"], payloads["registration"]
    metadata, selection = payloads["path_metadata"], payloads["selection"]
    evaluation, fidelity = payloads["evaluation"], payloads["control_fidelity"]
    require(registry["unique_count"] == registration["candidate_count"] == EXPECTED_COUNT,
            "Unexpected registered family size; no TopK substitution is allowed")
    require(registry["development"] == selection["selection_period"] == evaluation["period"] == PERIOD,
            "The registry, selection and evaluation must use only development 2014-2021")
    require(registration["only_evaluated_until"] == PERIOD[1], "Nondevelopment registration")
    require(registration["registry_sha256"] == hashes["registry"], "Registry receipt mismatch")
    require(metadata["sha256"] == hashes["paths"], "Return archive hash mismatch")
    provenance = selection["provenance"]
    for name in ("registry", "registration", "evaluation", "paths"):
        require(provenance[name + "_sha256"] == hashes[name], "Selection receipt mismatch: " + name)
    require(provenance["source_sha256"] == registration["sources"], "Source receipts disagree")
    sources = {name: sha(base / name) for name in registration["sources"]}
    require(sources == registration["sources"], "A registered development source has changed")
    require(fidelity["passed"] is True and fidelity["period"] == PERIOD,
            "Missing or mismatched control fidelity audit")
    require(all(c["exact_returns"] and c["exact_holdings"] for c in fidelity["cases"])
            and len(fidelity["cases"]) == 12, "Control fidelity is incomplete")
    ids = metadata["ids"]
    require(ids == [c["id"] for c in registry["candidates"]] and len(set(ids)) == EXPECTED_COUNT,
            "Path rows must contain every unique registered candidate in registered order")
    require(ids == [row["id"] for row in evaluation["rows"]], "Evaluation row order mismatch")
    dates = metadata["dates"]
    require(dates and dates == sorted(set(dates)) and [dates[0], dates[-1]] == PERIOD,
            "Invalid or nondevelopment date axis")
    require([s[0] for s in registration["scenarios"]] == list(SCENARIOS), "Scenario order mismatch")
    require(selection["controls"] == registry["controls"], "Control IDs changed")
    for candidate in [selection["primary"]] + selection["parents"]:
        require(candidate is None or candidate in ids, "Unknown frozen finalist")
    require(datetime.fromisoformat(registration["registered_at"]) <=
            datetime.fromisoformat(selection["frozen_at"]), "Selection predates registration")
    require(sum(bool(row["previously_seen"]) for row in registry["audit"].values()) == 17,
            "The disclosed prior-signature overlap has changed")
    matrices = {}
    with np.load(str(files["paths"]), allow_pickle=False) as archive:
        for name in SCENARIOS:
            values = archive[name + "__returns"]
            require(values.shape == (EXPECTED_COUNT, len(dates)) and values.dtype == np.float64,
                    "Invalid full-family matrix shape or precision: " + name)
            require(np.isfinite(values).all() and np.all(values > -1), "Invalid daily returns: " + name)
            # Verify the receipt's CAGR/terminal NAV against ALL path rows. This
            # does not rank or select candidates or access any later period.
            nav = np.prod(1 + values, axis=1)
            cagr = np.expm1(np.log1p(values).sum(axis=1) * 244 / len(dates))
            for i, row in enumerate(evaluation["rows"]):
                metrics = row["scenarios"][name]["full"]
                require(metrics["sessions"] == len(dates)
                        and np.isclose(nav[i], metrics["nav"], rtol=1e-12, atol=1e-12)
                        and np.isclose(cagr[i], metrics["cagr"], rtol=1e-12, atol=1e-12),
                        "Evaluation does not describe its path: " + name + " " + ids[i])
            values.flags.writeable = False
            matrices[name] = values
    return dict(files=files, hashes=hashes, sources=sources, payloads=payloads,
                ids=ids, dates=dates, matrices=matrices)


def limitations():
    return [
        "The full tested family is exactly 586 registered configurations including three frozen controls; no TopK filter was used.",
        "This conditional family count is not the entire research history: V10 had 9558 unique single-position configurations, and 17 V11 signatures match prior configurations.",
        "All historical years were previously examined. Only 2014-2021 returns are used here; this is not clean out-of-sample evidence.",
        "Four clock/cost scenarios and both mean block lengths are reported in full. Their p-values are not adjusted for choosing the smallest across these eight tests; none is used to reselect a model.",
        "Stationary paired block resampling and least-favorable zero-centering test maximum mean net log excess versus the matching H path. This is White-style, not SPA, PBO or DSR.",
        "Annual log excess is not a CAGR difference. A family p-value does not validate the particular multi-criterion finalist, show causality, or give the probability of future profit.",
        "Concentration shares describe relative net/positive log excess along frozen paths, not shares of strategy profit or an executable strategy after excluding days.",
    ]


def main():
    require(os.environ.get("OPENBLAS_NUM_THREADS") == "1", "Set OPENBLAS_NUM_THREADS=1")
    require(not (OUTPUT / "diagnostics.json").exists(), "Completed diagnostics are frozen; refusing to overwrite")
    verified = load_verified_inputs()
    payloads = verified["payloads"]
    selection, registry = payloads["selection"], payloads["registry"]
    ids, dates = verified["ids"], verified["dates"]
    source_files = {"development_diagnostics.py": sha(BASE / "development_diagnostics.py"),
                    "diagnostics.py": sha(BASE / "diagnostics.py")}
    design = dict(schema=1, input_sha256=verified["hashes"], development_source_sha256=verified["sources"],
        diagnostic_source_sha256=source_files, period=PERIOD, observations=len(dates),
        expected_candidate_count=EXPECTED_COUNT, includes_controls=True, topk_filter=False,
        benchmark_id=selection["controls"]["h"], block_lengths=list(BLOCKS), draws_per_test=DRAWS,
        method="stationary", batch_size=32, base_seed=SEED,
        scenario_seeds={name: SEED + i for i, name in enumerate(SCENARIOS)},
        seed_rule="Declared scenario order offset 0/1/2/3; same scenario seed for both block lengths; no seed selection",
        scenarios=payloads["registration"]["scenarios"],
        concentration_ids=list(dict.fromkeys(i for i in [selection["primary"]] + selection["parents"] if i)),
        concentration_top_k=[1, 5, 10, 20], family_memberships=registry["family_memberships"],
        prior_v10_unique_count=9558, previously_seen_signatures=17,
        frozen_primary=selection["primary"], candidate_selection_performed=False,
        confirmation_or_2026_returns_read=False, clean_oos=False, limitations=limitations())
    registration_path = OUTPUT / "registration.json"
    if registration_path.exists():
        require(read(registration_path)["design"] == design, "A different diagnostics design is already registered")
    else:
        dump(registration_path, dict(registered_at=stamp(), design=design))
    registration_hash = sha(registration_path)
    index = {candidate: i for i, candidate in enumerate(ids)}
    benchmark_index = index[selection["controls"]["h"]]
    tests, concentrations = {}, {}
    print("Registered 586-row development family, four scenarios, stationary20/60 x2000", flush=True)
    for name in SCENARIOS:
        matrix = verified["matrices"][name]
        benchmark = matrix[benchmark_index]
        tests[name] = development_family_tests(matrix, benchmark, dates, candidate_ids=ids,
            block_lengths=BLOCKS, draws=DRAWS, seed=design["scenario_seeds"][name],
            method="stationary", batch_size=32, expected_candidate_count=EXPECTED_COUNT)
        dump(OUTPUT / (name + ".json"), dict(registration_sha256=registration_hash,
                                              completed_at=stamp(), **tests[name]))
        concentrations[name] = {candidate: advantage_concentration(matrix[index[candidate]], benchmark,
            dates=dates, top_k=tuple(design["concentration_top_k"])) for candidate in design["concentration_ids"]}
        print(name + ": " + json.dumps({block: dict(p_value=result["p_value"],
             mc_standard_error=result["mc_standard_error"]) for block, result in tests[name]["tests"].items()}), flush=True)
    # Seal all inputs and sources after calculation, including the selection
    # receipt itself. Other research stages may run, but these files cannot drift.
    require(verified["hashes"] == {name: sha(path) for name, path in verified["files"].items()},
            "A frozen diagnostic input changed during calculation")
    require(verified["sources"] == {name: sha(BASE / name) for name in verified["sources"]},
            "A development source changed during diagnostics")
    require(source_files == {name: sha(BASE / name) for name in source_files}, "Diagnostic source changed")
    require(sha(registration_path) == registration_hash, "Diagnostics registration changed")
    dump(OUTPUT / "concentration.json", dict(registration_sha256=registration_hash, period=PERIOD,
         benchmark_id=selection["controls"]["h"], frozen_primary=selection["primary"],
         scenarios=concentrations, interpretation=limitations()[-1], candidate_selection_performed=False))
    result = dict(registration_sha256=registration_hash, completed_at=stamp(), period=PERIOD,
        observations=len(dates), scope=dict(registered_candidates=EXPECTED_COUNT, includes_controls=True,
        family_memberships=registry["family_memberships"], earlier_v10_unique_configurations=9558,
        previously_seen_signatures=17, scope_is_entire_research_history=False),
        benchmark_id=selection["controls"]["h"], scenarios=tests,
        input_sha256=verified["hashes"], source_sha256=source_files, all_receipts_verified=True,
        concentration_sha256=sha(OUTPUT / "concentration.json"),
        candidate_selection_performed=False, clean_oos=False, limitations=limitations())
    dump(OUTPUT / "diagnostics.json", result)
    dump(OUTPUT / "receipt.json", dict(completed_at=stamp(),
         artifacts_sha256={p.name: sha(p) for p in sorted(OUTPUT.glob("*.json")) if p.name != "receipt.json"},
         input_sha256=verified["hashes"], diagnostic_source_sha256=source_files,
         result="completed_development_only_no_reselection"))


if __name__ == "__main__":
    main()
