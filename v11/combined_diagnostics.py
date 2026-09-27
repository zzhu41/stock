"""Conditional A-union-B development statistics, with no reselection or later returns."""
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from .development_diagnostics import (
    BASE, PERIOD, SCENARIOS, BLOCKS, DRAWS, SEED,
    load_verified_inputs, sha, read, dump, stamp, require,
)
from .diagnostics import development_family_tests, advantage_concentration
from v10_deep.schema import semantic, identifier


OUTPUT = BASE / "results/combined_development_diagnostics"
UNION_COUNT = 714
B_PRIMARY = "v11_e2a10e8c6af1c0803ba4"


def verify_metrics(matrix, rows, scenario, days):
    nav = np.prod(1 + matrix, axis=1)
    cagr = np.expm1(np.log1p(matrix).sum(axis=1) * 244 / days)
    for i, row in enumerate(rows):
        m = row["scenarios"][scenario]["full"]
        require(m["sessions"] == days and np.isclose(nav[i], m["nav"], rtol=1e-12, atol=1e-12)
                and np.isclose(cagr[i], m["cagr"], rtol=1e-12, atol=1e-12),
                "B path/evaluation mismatch: " + scenario + " " + row["id"])


def verify_semantics(candidates):
    for row in candidates:
        digest = identifier(row)
        require(row["hash"] == digest and row["id"] == "v11_" + digest[:20],
                "Candidate semantic hash mismatch: " + row["id"])


def load_union(base=BASE):
    base = Path(base)
    a = load_verified_inputs(base)
    directory = base / "results/combinations"
    bfiles = {name: directory / (name + ".json") for name in
              ("registration", "registered_candidates", "path_metadata", "evaluation", "selection")}
    bfiles["paths"] = directory / "paths.npz"
    hashes = {name: sha(path) for name, path in bfiles.items()}
    b = {name: read(path) for name, path in bfiles.items() if name != "paths"}
    reg, selected, candidates = b["registration"], b["selection"], b["registered_candidates"]
    metadata, evaluation = b["path_metadata"], b["evaluation"]
    ar = a["payloads"]["registry"]
    require(candidates["raw_count"] == candidates["combination_unique_count"] == 150
            and candidates["unique_count"] == 152, "Unexpected B family")
    require(reg["period"] == selected["selection_period"] == evaluation["period"] == PERIOD,
            "B must use development 2014-2021 only")
    require(metadata["dates"] == a["dates"], "A/B date axes differ")
    require(selected["primary"] == B_PRIMARY, "The frozen B primary changed")
    require(candidates["controls"] == selected["controls"] == ar["controls"], "A/B controls differ")
    require(reg["a_selection_sha256"] == a["hashes"]["selection"]
            and reg["a_evaluation_sha256"] == a["hashes"]["evaluation"], "B's A-input receipts differ")
    require(reg["registry_sha256"] == hashes["registered_candidates"]
            and metadata["sha256"] == hashes["paths"], "B registry/path receipt mismatch")
    proof = selected["provenance"]
    for field, name in (("registration", "registration"), ("registry", "registered_candidates"),
                        ("evaluation", "evaluation"), ("paths", "paths")):
        require(proof[field + "_sha256"] == hashes[name], "B selection receipt mismatch: " + field)
    sources = {name: sha(base / name) for name in reg["sources"]}
    require(proof["source_sha256"] == reg["sources"] == sources, "B registered source changed")
    require(reg["feature_fingerprints"] == a["payloads"]["registration"]["feature_fingerprints"],
            "A and B feature inputs differ")
    require(reg["scenarios"] == a["payloads"]["registration"]["scenarios"], "A/B scenarios differ")
    require(datetime.fromisoformat(a["payloads"]["selection"]["frozen_at"]) <=
            datetime.fromisoformat(reg["registered_at"]) <= datetime.fromisoformat(selected["frozen_at"]),
            "A-selection/B-registration/B-selection ordering is invalid")
    aids, bids = a["ids"], metadata["ids"]
    require(bids == [c["id"] for c in candidates["candidates"]]
            == [row["id"] for row in evaluation["rows"]] and len(set(bids)) == 152,
            "B archive row order or count differs from registry/evaluation")
    verify_semantics(ar["candidates"])
    verify_semantics(candidates["candidates"])
    ac = {c["id"]: c for c in ar["candidates"]}
    bc = {c["id"]: c for c in candidates["candidates"]}
    common = sorted(set(aids) & set(bids))
    union_ids = sorted(set(aids) | set(bids))
    new_ids = sorted(set(bids) - set(aids))
    require(len(common) == 24 and len(new_ids) == 128 and len(union_ids) == UNION_COUNT,
            "Unexpected semantic-ID overlap or union size")
    for cid in common:
        require(semantic(ac[cid]) == semantic(bc[cid]) and ac[cid]["hash"] == bc[cid]["hash"],
                "A/B same ID has different parameters: " + cid)
    ai, bi = {cid: i for i, cid in enumerate(aids)}, {cid: i for i, cid in enumerate(bids)}
    apos, bpos = [ai[cid] for cid in common], [bi[cid] for cid in common]
    matrices, overlap_checks = {}, {}
    with np.load(str(a["files"]["paths"]), allow_pickle=False) as az, \
            np.load(str(bfiles["paths"]), allow_pickle=False) as bz:
        for scenario in SCENARIOS:
            checks = {}
            for field in ("returns", "holdings", "summary"):
                key = scenario + "__" + field
                av, bv = az[key], bz[key]
                expected_shape = (152, 10 if field == "summary" else len(a["dates"]))
                require(bv.shape == expected_shape and np.isfinite(bv).all(), "Invalid B archive: " + key)
                require(np.array_equal(av[apos], bv[bpos]), "Shared candidate path mismatch: " + key)
                checks[field] = dict(exact_equal=True, checked_values=int(av[apos].size),
                                     shared_candidate_count=len(common))
                if field == "returns":
                    require(bv.dtype == np.float64 and np.all(bv > -1), "Invalid B returns")
                    verify_metrics(bv, evaluation["rows"], scenario, len(a["dates"]))
                    matrix = np.empty((UNION_COUNT, len(a["dates"])), dtype=np.float64)
                    for i, cid in enumerate(union_ids):
                        matrix[i] = av[ai[cid]] if cid in ai else bv[bi[cid]]
                    matrix.flags.writeable = False
                    matrices[scenario] = matrix
            overlap_checks[scenario] = checks
    # Verify the manifest fingerprints without opening any full-history price
    # matrix or a post-2021 strategy path.
    manifests = {"corrected": base.parent / "v10_h_close/corrected_manifest.json",
                 "qvix": base.parent / "v10_h_close/qvix_manifest.json",
                 "frozen_profiles": base.parent / "v10_deep/profiles.json"}
    require({name: sha(path) for name, path in manifests.items()} ==
            reg["feature_fingerprints"]["input_fingerprints"], "Frozen input manifest changed")
    files = {"a_" + name: path for name, path in a["files"].items()}
    files.update({"b_" + name: path for name, path in bfiles.items()})
    files.update({"manifest_" + name: path for name, path in manifests.items()})
    return dict(a=a, b=b, files=files, hashes={name: sha(path) for name, path in files.items()},
                sources=dict(a["sources"], **sources), ids=union_ids, new_ids=new_ids, common_ids=common,
                matrices=matrices, dates=a["dates"], overlap_checks=overlap_checks)


def supplemental_counts(base, union_ids):
    """Metadata-only disclosure, not additional return inputs to the 714 test."""
    files = {}
    neighborhood = base / "results/neighborhood"
    nr = read(neighborhood / "registered_candidates.json")
    nm = read(neighborhood / "path_metadata.json")
    require(nr["development"] == PERIOD and nm["dates"][0] == PERIOD[0]
            and nm["dates"][-1] == PERIOD[1], "Nondevelopment neighborhood metadata")
    require(len(nm["ids"]) == nr["candidate_count"] == 58, "Unexpected neighborhood count")
    trend = base / "results/long_trend_controls"
    tr, tm = read(trend / "registration.json"), read(trend / "path_metadata.json")
    require(tr["period"] == PERIOD and tr["controls"]["count"] == 8
            and tm["dates"][0] == PERIOD[0] and tm["dates"][-1] == PERIOD[1],
            "Unexpected long-trend metadata")
    for prefix, directory, names in (("neighborhood", neighborhood, ("registered_candidates", "path_metadata")),
                                    ("long_trend", trend, ("registration", "path_metadata"))):
        for name in names:
            files[prefix + "_" + name] = directory / (name + ".json")
    return dict(neighborhood=dict(evaluated_rows=58, shared_with_union=len(set(nm["ids"]) & set(union_ids)),
                outside_tested_union=len(set(nm["ids"]) - set(union_ids)), return_paths_read=False),
                long_trend_controls=dict(evaluated_models=8, return_paths_read=False),
                counting_note="These supplemental diagnostics were also observed. They are not included in the 714-row test; counts do not exhaust later diagnostics or all adaptive research."), files


def limitations():
    return [
        "The complete tested union is A586 plus B152 minus 24 common semantic IDs = 714 configurations, including three controls; B contributes 128 new IDs. No TopK filtering occurred.",
        "B components were chosen adaptively using A development results and diagnostics. Zero-centered resampling of the resulting fixed union does not reproduce or correct this data-dependent family construction.",
        "The 714 count excludes prior V10's 9558 unique single-position configurations, A's 17 previously seen signatures, supplemental neighborhoods, and eight long-trend controls. It is not the total number of observed research attempts.",
        "Only 2014-2021 paths are read. All years have previously been seen in earlier research; this is not clean OOS or new validation evidence.",
        "All four cost/lag scenarios and both stationary mean block lengths are reported. There is no adjustment for choosing the smallest across eight p-values, and none is used for reselection.",
        "This conditional White-style maximum mean net log-excess test is not SPA/PBO/DSR, does not jointly test the selector's risk criteria, and does not give a probability of future profit or overfitting.",
        "Concentration is relative log-excess accounting for each frozen primary versus H, with separate net and positive denominators; it is not total strategy profit or an executable day-exclusion strategy.",
    ]


def main():
    require(os.environ.get("OPENBLAS_NUM_THREADS") == "1", "Set OPENBLAS_NUM_THREADS=1")
    require(not (OUTPUT / "diagnostics.json").exists(), "Completed union diagnostics are frozen")
    verified = load_union()
    supplemental, metadata_files = supplemental_counts(BASE, verified["ids"])
    verified["files"].update(metadata_files)
    verified["hashes"].update({name: sha(path) for name, path in metadata_files.items()})
    source_files = {name: sha(BASE / name) for name in
                    ("combined_diagnostics.py", "development_diagnostics.py", "diagnostics.py")}
    source_files["../v10_deep/schema.py"] = sha(BASE.parent / "v10_deep/schema.py")
    aselected = verified["a"]["payloads"]["selection"]
    benchmark_id = aselected["controls"]["h"]
    primaries = {"a": aselected["primary"], "b": B_PRIMARY}
    design = dict(schema=1, period=PERIOD, observations=len(verified["dates"]),
        family=dict(a=586, b=152, common=24, b_new=128, union=UNION_COUNT, ids=verified["ids"],
                    common_ids=verified["common_ids"], new_b_ids=verified["new_ids"]),
        expected_candidate_count=UNION_COUNT, includes_controls=True, benchmark_id=benchmark_id,
        method="stationary", draws_per_test=DRAWS, block_lengths=list(BLOCKS), batch_size=32,
        scenario_order=list(SCENARIOS), scenario_seeds={s: SEED + i for i, s in enumerate(SCENARIOS)},
        seed_rule="Fixed offsets 0/1/2/3 in declared scenario order; same seed for both block lengths",
        overlap_checks=verified["overlap_checks"], input_sha256=verified["hashes"],
        development_source_sha256=verified["sources"], diagnostic_source_sha256=source_files,
        frozen_primaries=primaries, concentration_top_k=[1, 5, 10, 20],
        supplemental_observed_diagnostics=supplemental, all_old_a_diagnostics_preserved=True,
        candidate_selection_performed=False, clean_oos=False, limitations=limitations())
    registration = OUTPUT / "registration.json"
    if registration.exists():
        require(read(registration)["design"] == design, "A different union diagnostic design is registered")
    else:
        dump(registration, dict(registered_at=stamp(), design=design))
    registration_hash = sha(registration)
    index = {cid: i for i, cid in enumerate(verified["ids"])}
    tests, concentration = {}, {}
    print("Registered union714; all 24 shared IDs agree exactly in all four scenarios and three fields", flush=True)
    for name in SCENARIOS:
        matrix = verified["matrices"][name]
        benchmark = matrix[index[benchmark_id]]
        tests[name] = development_family_tests(matrix, benchmark, verified["dates"],
            candidate_ids=verified["ids"], block_lengths=BLOCKS, draws=DRAWS,
            seed=design["scenario_seeds"][name], method="stationary", batch_size=32,
            expected_candidate_count=UNION_COUNT)
        dump(OUTPUT / (name + ".json"), dict(registration_sha256=registration_hash,
             completed_at=stamp(), **tests[name]))
        concentration[name] = {stage: dict(candidate_id=cid, **advantage_concentration(
            matrix[index[cid]], benchmark, dates=verified["dates"], top_k=(1, 5, 10, 20)))
            for stage, cid in primaries.items()}
        print(name + ": " + str({block: result["p_value"] for block, result in tests[name]["tests"].items()}), flush=True)
    require(verified["hashes"] == {name: sha(path) for name, path in verified["files"].items()},
            "An input changed during union diagnostics")
    require(verified["sources"] == {name: sha(BASE / name) for name in verified["sources"]},
            "A registered development source changed")
    require(source_files == {name: sha(BASE / name) for name in source_files}, "A diagnostic source changed")
    require(sha(registration) == registration_hash, "Union registration changed")
    dump(OUTPUT / "concentration.json", dict(registration_sha256=registration_hash,
         period=PERIOD, benchmark_id=benchmark_id, frozen_primaries=primaries,
         scenarios=concentration, interpretation=limitations()[-1], candidate_selection_performed=False))
    dump(OUTPUT / "diagnostics.json", dict(registration_sha256=registration_hash, completed_at=stamp(),
         period=PERIOD, observations=len(verified["dates"]), candidates=UNION_COUNT,
         input_sha256=verified["hashes"], source_sha256=source_files, all_receipts_verified=True,
         overlap_checks=verified["overlap_checks"], benchmark_id=benchmark_id,
         scenarios=tests, supplemental_observed_diagnostics=supplemental,
         concentration_sha256=sha(OUTPUT / "concentration.json"),
         candidate_selection_performed=False, clean_oos=False, limitations=limitations()))
    dump(OUTPUT / "receipt.json", dict(completed_at=stamp(),
         artifacts_sha256={p.name: sha(p) for p in sorted(OUTPUT.glob("*.json")) if p.name != "receipt.json"},
         input_sha256=verified["hashes"], diagnostic_source_sha256=source_files,
         result="completed_conditional_union714_development_only_no_reselection"))


if __name__ == "__main__":
    main()
