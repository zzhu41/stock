"""Receipt-verified complete-family union statistics, without model selection.

Stage descriptors contain name, directory, registry (paths relative to v12).
Each completed stage uses the scan.py registration/design/fingerprints and
selection/provenance contract. Source maps are verified entry by entry: a new
extension file must not invalidate an older stage's frozen source inventory.
Only `run` writes diagnostic artifacts; `verify` never bootstraps or writes.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from v11.diagnostics import white_style_test, advantage_concentration


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "results/family_diagnostics"
SCENARIOS = ("close_1bp", "close_11bp", "lag1_1bp", "lag1_11bp")
SCENARIO_DEFINITIONS = [["close_1bp", 0, .0001], ["close_11bp", 0, .0011],
                        ["lag1_1bp", 1, .0001], ["lag1_11bp", 1, .0011]]
BLOCKS, DRAWS, SEED = (20, 60), 2000, 20260928
FIELDS = ("returns", "holdings", "summary")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def _path(base, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Stage/source paths must be relative to the V12 directory")
    target = base / path
    require(base.resolve() in target.resolve().parents, "Stage/source path escapes V12")
    return target


def verify_fingerprints(base, fingerprint):
    """Do not compare a historic map against today's expanded file glob."""
    files = {}
    for section in ("sources", "inputs"):
        values = fingerprint.get(section)
        require(isinstance(values, dict) and values, "Missing registered " + section)
        for name, expected in values.items():
            path = _path(base, name)
            require(sha(path) == expected, "Registered %s changed: %s" % (section, name))
            files[section + ":" + name] = path
    return files


def candidate_semantics(record):
    kind = record.get("kind")
    require(kind in ("single", "allocation", "benchmark"), "Unknown candidate kind")
    value = record[{"single": "config", "allocation": "allocation", "benchmark": "benchmark"}[kind]]
    if kind == "single":
        value = {k: v for k, v in value.items() if k not in ("id", "hash", "families", "parents", "stage")}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    require(record.get("hash") == digest and record.get("id") == "v12_" + digest[:20], "Canonical candidate id/hash differs")
    return dict(kind=kind, payload=value, hash=digest)


def load_stage(descriptor, base=BASE):
    base = Path(base)
    require(isinstance(descriptor.get("name"), str) and descriptor["name"], "Stage needs a name")
    directory = _path(base, descriptor["directory"])
    files = {name: directory / (name + ".json") for name in
             ("registration", "path_metadata", "evaluation", "selection", "execution_metadata")}
    files.update(registry=_path(base, descriptor["registry"]), paths=directory / "paths.npz")
    hashes = {name: sha(path) for name, path in files.items()}
    registry, registration = read(files["registry"]), read(files["registration"])
    metadata, evaluation, selection = read(files["path_metadata"]), read(files["evaluation"]), read(files["selection"])
    design, proof = registration["design"], selection["provenance"]
    for name in ("registry", "registration", "evaluation", "paths", "path_metadata", "execution_metadata"):
        require(proof[name + "_sha256"] == hashes[name], "Stage selection receipt mismatch: " + name)
    require(evaluation["registration_sha256"] == hashes["registration"]
            and evaluation["paths_sha256"] == metadata["sha256"] == hashes["paths"], "Stage evaluation/path receipts differ")
    require(read(files["execution_metadata"])["registration_sha256"] == hashes["registration"], "Execution metadata receipt differs")
    require(proof["fingerprints"] == design["fingerprints"], "Stage source/input receipts disagree")
    watched = verify_fingerprints(base, design["fingerprints"])
    records = registry["candidates"]
    ids = [row["id"] for row in records]
    count = registry.get("unique_count", registry.get("count"))
    require(len(ids) == len(set(ids)) == count == design["candidate_count"], "Stage family count/IDs differ")
    require(ids == metadata["ids"] == [row["id"] for row in evaluation["rows"]], "Stage registry/path/evaluation row order differs")
    dates = metadata["dates"]
    require(len(dates) >= 2 and dates == sorted(set(dates))
            and all(date.fromisoformat(day).isoformat() == day for day in dates), "Invalid stage calendar")
    require(evaluation["period"] == selection["selection_period"] == design["period"], "Stage selection periods differ")
    # Older/adaptive stage writers may omit this redundant display field.
    # Actual dates remain pinned by selection.path_metadata_sha256 and NPZ shape.
    require(evaluation.get("observed_period", [dates[0], dates[-1]]) == [dates[0], dates[-1]]
            and design["period"][0] <= dates[0] <= dates[-1] <= design["period"][1], "Stage observed interval differs")
    require(design["scenarios"] == SCENARIO_DEFINITIONS, "All four fixed scenarios are required")
    require(selection["controls"] == registry["controls"] and registry["controls"]["h"] in ids,
            "Stage benchmark/control receipts differ")
    require(datetime.fromisoformat(registration["registered_at"]) <= datetime.fromisoformat(selection["frozen_at"]),
            "Selection predates stage registration")
    semantics = {record["id"]: candidate_semantics(record) for record in records}
    matrices = {}
    with np.load(files["paths"], allow_pickle=False) as archive:
        for scenario in SCENARIOS:
            matrices[scenario] = {}
            for field in FIELDS:
                values = archive[scenario + "__" + field]
                shape = (count, 10 if field == "summary" else len(dates))
                require(values.shape == shape and np.isfinite(values).all(), "Invalid stage matrix: " + scenario + "/" + field)
                if field == "returns":
                    require(values.dtype == np.float64 and np.all(values > -1), "Returns must be finite float64 simple net returns > -1")
                elif field == "summary":
                    require(values.dtype == np.float64, "Summary precision differs")
                else:
                    require(np.issubdtype(values.dtype, np.integer), "Holding indices must be integers")
                values.flags.writeable = False
                matrices[scenario][field] = values
    watched.update({"artifact:" + name: path for name, path in files.items()})
    expected_hashes = {section + ":" + name: expected for section in ("sources", "inputs")
                       for name, expected in design["fingerprints"][section].items()}
    expected_hashes.update({"artifact:" + name: value for name, value in hashes.items()})
    require(expected_hashes == {name: sha(path) for name, path in watched.items()},
            "Evidence/source changed during stage verification")
    return dict(name=descriptor["name"], descriptor=dict(descriptor), ids=ids, dates=dates,
        period=design["period"], controls=registry["controls"], records=records, semantics=semantics, matrices=matrices,
        files=watched, input_sha256=expected_hashes,
        frozen_selection={key: selection.get(key) for key in ("primary", "status", "qualified_count", "top_return", "top_tail", "least_drawdown")})


def load_union(descriptors, base=BASE):
    descriptors = list(descriptors)
    require(descriptors and len({d["name"] for d in descriptors}) == len(descriptors), "Distinct nonempty stage descriptors required")
    stages = [load_stage(descriptor, base) for descriptor in descriptors]
    first = stages[0]
    for stage in stages[1:]:
        require(stage["dates"] == first["dates"] and stage["period"] == first["period"], "Stage calendars/intervals must agree exactly; no intersections")
        require(stage["controls"]["h"] == first["controls"]["h"], "Stages must use the same canonical H benchmark")
    seen, members, records, comparisons = {}, {}, {}, []
    for si, stage in enumerate(stages):
        for index, cid in enumerate(stage["ids"]):
            members.setdefault(cid, []).append(stage["name"])
            records.setdefault(cid, stage["records"][index])
            if cid not in seen:
                seen[cid] = (si, index)
                continue
            old_stage, old_index = seen[cid]
            earlier = stages[old_stage]
            require(stage["semantics"][cid] == earlier["semantics"][cid], "Repeated ID has different semantics: " + cid)
            checks = {}
            for scenario in SCENARIOS:
                checks[scenario] = {}
                for field in FIELDS:
                    left, right = earlier["matrices"][scenario][field][old_index], stage["matrices"][scenario][field][index]
                    require(np.array_equal(left, right), "Repeated ID path differs exactly: %s %s %s" % (cid, scenario, field))
                    checks[scenario][field] = dict(exact=True, compared_values=int(left.size))
            comparisons.append(dict(id=cid, first_stage=earlier["name"], repeated_stage=stage["name"], checks=checks))
    ids = sorted(seen)
    returns = {scenario: np.asarray([stages[seen[cid][0]]["matrices"][scenario]["returns"][seen[cid][1]]
                                    for cid in ids], dtype=np.float64) for scenario in SCENARIOS}
    for values in returns.values():
        values.flags.writeable = False
    watched = {stage["name"] + "/" + name: path for stage in stages for name, path in stage["files"].items()}
    expected_hashes = {stage["name"] + "/" + name: value for stage in stages for name, value in stage["input_sha256"].items()}
    require(expected_hashes == {name: sha(path) for name, path in watched.items()}, "Stage inputs changed while merging the family")
    scope = dict(stage_counts={stage["name"]: len(stage["ids"]) for stage in stages},
        raw_stage_rows=sum(len(stage["ids"]) for stage in stages), union_candidates=len(ids),
        repeated_occurrences=len(comparisons), shared_ids=[cid for cid in ids if len(members[cid]) > 1],
        kind_counts=dict(Counter(records[cid]["kind"] for cid in ids)),
        includes_all_controls_and_rejected_candidates=True, topk_filter=False,
        stage_additions={stage["name"]: sum(seen[cid][0] == si for cid in stage["ids"]) for si, stage in enumerate(stages)})
    return dict(ids=ids, dates=first["dates"], period=first["period"], benchmark_id=first["controls"]["h"],
        returns=returns, scope=scope, membership=members, overlap_checks=comparisons,
        stages=[dict(name=stage["name"], descriptor=stage["descriptor"], frozen_selection=stage["frozen_selection"],
                     source_input_receipts=stage["input_sha256"]) for stage in stages],
        files=watched, input_sha256=expected_hashes)


@lru_cache(maxsize=8)
def _calendar_groups(dates):
    require(dates and tuple(sorted(set(dates))) == dates
            and all(date.fromisoformat(day).isoformat() == day for day in dates), "Invalid concentration dates")
    groups = []
    for i, day in enumerate(dates):
        if not groups or groups[-1][0] != day[:4]:
            groups.append([day[:4], i, i + 1])
        else:
            groups[-1][2] = i + 1
    return tuple(tuple(group) for group in groups)


def concentration_by_year(selected, benchmark, dates):
    dates = tuple(dates)
    groups = _calendar_groups(dates)
    require(len(selected) == len(dates), "Concentration path/date length differs")
    # Validate the shared calendar once, not once per candidate via millions of
    # repeated strptime calls. Numeric attribution still uses the frozen helper.
    result = advantage_concentration(selected, benchmark, dates=None, top_k=(10,))
    result["top"]["10"]["dates"] = [dates[i] for i in result["top"]["10"]["indices"]]
    for item in result["top_positive_days"]:
        item["date"] = dates[item["index"]]
    selected, reference = np.log1p(selected), np.log1p(benchmark)
    difference = selected - reference
    net, positive = result["net_log_excess"], result["positive_log_excess"]
    years = {}
    for year, begin, end in groups:
        delta = difference[begin:end]
        amount, positive_amount = float(delta.sum()), float(delta[delta > 0].sum())
        years[year] = dict(start=dates[begin], end=dates[end - 1], observations=end - begin,
            selected_total_return=float(np.expm1(selected[begin:end].sum())),
            benchmark_total_return=float(np.expm1(reference[begin:end].sum())), net_log_excess=amount,
            positive_log_excess=positive_amount, negative_log_excess=float(delta[delta < 0].sum()),
            share_of_net_log_excess=amount / net if net > 1e-12 else None,
            share_of_positive_log_excess=positive_amount / positive if positive > 0 else None)
    result.update(years=years, annualization_of_partial_year=False,
                  annual_interpretation="Signed yearly contributions to net log excess and positive-mass contributions are separate; they are not shares of total strategy profit or yearly CAGR differences.")
    return result


def _write_concentration(output, scenario, union, benchmark_index, registration_hash):
    """One compact, lossless gzip per scenario; at most one candidate in RAM."""
    path = Path(output) / ("concentration_" + scenario + ".json.gz")
    temporary = path.with_name(path.name + ".tmp")
    header = dict(registration_sha256=registration_hash, scenario=scenario, period=union["period"],
                  benchmark_id=union["benchmark_id"], candidate_ids=union["ids"],
                  candidate_selection_performed=False, interpretation=limitations()[-1])
    digest, size = hashlib.sha256(), 0
    values = union["returns"][scenario]
    dates = tuple(union["dates"])
    try:
        with temporary.open("wb") as destination, gzip.GzipFile(filename="", mode="wb", fileobj=destination,
                                                               mtime=0, compresslevel=6) as compressed:
            def write(text):
                nonlocal size
                raw = text.encode("utf-8")
                digest.update(raw); size += len(raw); compressed.write(raw)
            write(json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False)[:-1] + ',"candidates":{')
            for i, cid in enumerate(union["ids"]):
                if i:
                    write(",")
                value = concentration_by_year(values[i], values[benchmark_index], dates)
                write(json.dumps(cid) + ":" + json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
            write("}}\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(archive=path.name, archive_sha256=sha(path), archive_bytes=path.stat().st_size,
                raw_json_sha256=digest.hexdigest(), raw_json_bytes=size, candidates=len(union["ids"]))


def load_concentration(output, scenario):
    """Verified reader for a completed scenario shard; never rewrites it."""
    output = Path(output)
    index_path = output / "concentration.json"
    index, diagnostics = read(index_path), read(output / "diagnostics.json")
    require(sha(index_path) == diagnostics["concentration_sha256"], "Concentration index hash differs")
    require(index["format"] == "gzip_json_by_scenario" and scenario in index["scenarios"], "Unknown concentration shard")
    item = index["scenarios"][scenario]
    path = _path(output, item["archive"])
    require(path.stat().st_size == item["archive_bytes"] and sha(path) == item["archive_sha256"], "Concentration archive differs")
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    require(len(raw) == item["raw_json_bytes"] and hashlib.sha256(raw).hexdigest() == item["raw_json_sha256"],
            "Concentration decompressed JSON differs")
    value = json.loads(raw.decode("utf-8"))
    require(value["scenario"] == scenario and value["candidate_ids"] == index["candidate_ids"]
            and set(value["candidates"]) == set(index["candidate_ids"])
            and value["registration_sha256"] == index["registration_sha256"], "Concentration shard scope differs")
    return value


def limitations():
    return [
        "The tested family is the complete canonical-ID union of supplied completed stage registries, including all controls and failed candidates; it is not a TopK shortlist.",
        "Later adaptive batches (including B/C when supplied) were proposed after earlier outcomes. Bootstrapping the final fixed union does not reproduce or correct that data-dependent family construction.",
        "V10's prior 9558 unique single-position configurations, V11's 714-row union, neighborhoods, controls and historical reselection attempts remain part of the research history; this conditional trial count does not cover them.",
        "The complete 2014-2026 interval is previously examined history and 2026 explicitly participates in V12 optimization; no independent OOS claim is available.",
        "Unstudentized least-favorable paired stationary-block maximum mean net log excess versus H: White-style, not SPA, PBO or DSR. It does not jointly test the selector's CAGR/drawdown/tail constraints.",
        "All four scenarios and both mean block lengths are reported. The eight p-values are not an additional correction for choosing whichever is smallest; no result changes the frozen candidate selection.",
        "A p-value is not a probability of overfitting, future failure or future profit. Yearly/top10 relative log-excess accounting is not a tradable strategy after removing favorable days.",
    ]


def run(descriptors, output=OUTPUT, base=BASE):
    output, base = Path(output), Path(base)
    descriptors = list(descriptors)
    require(not (output / "diagnostics.json").exists(), "Completed family diagnostics are frozen; use an explicitly new output for a different union")
    union = load_union(descriptors, base)
    source_files = {"family_diagnostics.py": Path(__file__),
                    "v11/diagnostics.py": Path(__file__).resolve().parents[1] / "v11/diagnostics.py"}
    sources = {name: sha(path) for name, path in source_files.items()}
    design = dict(schema=1, stage_descriptors=list(descriptors), period=union["period"],
        observed_period=[union["dates"][0], union["dates"][-1]], observations=len(union["dates"]),
        candidate_ids=union["ids"], expected_candidate_count=len(union["ids"]), scope=union["scope"],
        benchmark_id=union["benchmark_id"], stage_frozen_selections=union["stages"],
        input_sha256=union["input_sha256"], source_sha256=sources, method="stationary",
        numpy_version=np.__version__,
        block_lengths=list(BLOCKS), draws_per_test=DRAWS, seed=SEED, batch_size=32,
        seed_rule="Same fixed seed for every scenario and block length; equal block lengths share paired sample counts across scenarios",
        concentration_scope="Every canonical union candidate, all scenarios, every observed calendar year, and top10 positive advantage days",
        concentration_storage="Four compact gzip JSON shards, one per scenario, with compressed and raw-JSON SHA256/size; one-candidate streaming",
        candidate_selection_performed=False, clean_oos=False, limitations=limitations())
    registration_path = output / "registration.json"
    if registration_path.exists():
        require(read(registration_path)["design"] == design, "Different family inputs/design are already registered")
    else:
        dump(registration_path, dict(registered_at=stamp(), design=design))
    registration_hash = sha(registration_path)
    dump(output / "overlap_checks.json", dict(registration_sha256=registration_hash,
         scope=union["scope"], comparisons=union["overlap_checks"], membership=union["membership"]))
    benchmark_index = union["ids"].index(union["benchmark_id"])
    tests, concentration_archives = {}, {}
    for scenario in SCENARIOS:
        values = union["returns"][scenario];benchmark = values[benchmark_index]
        tests[scenario] = {str(block): white_style_test(values, benchmark, block_length=block,
            draws=DRAWS, seed=SEED, method="stationary", candidate_ids=union["ids"], batch_size=32,
            expected_candidate_count=len(union["ids"])) for block in BLOCKS}
        dump(output / (scenario + ".json"), dict(registration_sha256=registration_hash, tests=tests[scenario]))
        concentration_archives[scenario] = _write_concentration(output, scenario, union, benchmark_index, registration_hash)
        print(scenario, {block: value["p_value"] for block, value in tests[scenario].items()}, flush=True)
    require(union["input_sha256"] == {name: sha(path) for name, path in union["files"].items()}, "Stage evidence/source changed during statistics")
    require(sources == {name: sha(path) for name, path in source_files.items()}
            and sha(registration_path) == registration_hash, "Diagnostic source/registration changed during statistics")
    dump(output / "concentration.json", dict(format="gzip_json_by_scenario", registration_sha256=registration_hash, period=union["period"],
        benchmark_id=union["benchmark_id"], candidate_ids=union["ids"], scenarios=concentration_archives,
        candidate_selection_performed=False, interpretation=limitations()[-1]))
    dump(output / "diagnostics.json", dict(registration_sha256=registration_hash, completed_at=stamp(),
        period=union["period"], observed_period=[union["dates"][0], union["dates"][-1]], scope=union["scope"],
        benchmark_id=union["benchmark_id"], scenarios=tests, input_sha256=union["input_sha256"], source_sha256=sources,
        concentration_sha256=sha(output / "concentration.json"), no_reselection=True, clean_oos=False, limitations=limitations()))
    dump(output / "receipt.json", dict(completed_at=stamp(), input_sha256=union["input_sha256"], source_sha256=sources,
        artifacts_sha256={path.name: sha(path) for path in sorted(output.iterdir())
                          if path.is_file() and path.name.endswith((".json", ".json.gz")) and path.name != "receipt.json"},
        no_reselection=True))
    return read(output / "diagnostics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "run"))
    parser.add_argument("--stage", nargs=3, action="append", required=True, metavar=("NAME", "DIRECTORY", "REGISTRY"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    descriptors = [dict(name=name, directory=directory, registry=registry) for name, directory, registry in args.stage]
    if args.command == "verify":
        value = load_union(descriptors)
        print(json.dumps(dict(scope=value["scope"], period=value["period"], receipts_verified=True, statistics_run=False), indent=2))
    else:
        run(descriptors, args.output)
