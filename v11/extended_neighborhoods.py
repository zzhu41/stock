"""Separate B-parent and fixed-H neighborhood diagnostics, development only.

Reuses the frozen A neighborhood generator and scanner without changing their
source or module constants. Both complete neighbor sets and a predetermined
common-operation comparison are registered before either new batch is run.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from . import neighborhood as original_neighborhood
from .data import BASE, ROOT, START, DEV_END, dump, sha, protect
from .features import build
from .scan import SCENARIOS, DEV_BLOCKS, run_candidates, summarize_rows, select, stamp


A_DIR = BASE / "results/development"
B_DIR = BASE / "results/combinations"
A_NEIGHBOR_DIR = BASE / "results/neighborhood"
OUTS = {"b": BASE / "results/b_neighborhood", "h": BASE / "results/h_neighborhood"}
SOURCE_NAMES = tuple(original_neighborhood.SOURCE_NAMES) + (
    "v11/extended_neighborhoods.py", "v11/tests/test_extended_neighborhoods.py",
    "v11/COMBINATION_PROTOCOL.md", "v11/combinations.py")
PRESSURE_KEYS = ("close_11bp_growth", "close_11bp_drawdown", "lag1_11bp_growth", "lag1_11bp_drawdown")


def _read_stage(directory, registry_path):
    selection = json.loads((directory / "selection.json").read_text())
    proof = selection["provenance"]
    if selection["selection_period"] != [START, DEV_END] or DEV_END != "2021-12-31":
        raise ValueError("Only frozen development choices through 2021 are permitted")
    if sha(registry_path) != proof["registry_sha256"]:
        raise ValueError("Source candidate registry changed")
    for filename, key in (("registration.json", "registration_sha256"), ("evaluation.json", "evaluation_sha256"),
                          ("paths.npz", "paths_sha256")):
        if sha(directory / filename) != proof[key]:
            raise ValueError("Source development artifact changed: " + str(directory / filename))
    for name, digest in proof["source_sha256"].items():
        if sha(BASE / name) != digest:
            raise ValueError("Frozen source changed: " + name)
    registry = json.loads(registry_path.read_text())
    metadata = json.loads((directory / "path_metadata.json").read_text())
    evaluation = json.loads((directory / "evaluation.json").read_text())
    if (metadata["sha256"] != proof["paths_sha256"] or metadata["dates"][-1] > DEV_END
            or metadata["ids"] != [c["id"] for c in registry["candidates"]]
            or evaluation["period"] != [metadata["dates"][0], metadata["dates"][-1]]):
        raise ValueError("Source path axis/identities are inconsistent")
    return dict(directory=directory, registry=registry, selection=selection,
                path_metadata=metadata, evaluation=evaluation)


def load_sources():
    a = _read_stage(A_DIR, BASE / "registered_candidates.json")
    b = _read_stage(B_DIR, B_DIR / "registered_candidates.json")
    b_registration = json.loads((B_DIR / "registration.json").read_text())
    if (b_registration["a_selection_sha256"] != sha(A_DIR / "selection.json")
            or b_registration["a_evaluation_sha256"] != sha(A_DIR / "evaluation.json")):
        raise ValueError("B component provenance no longer matches the frozen A development")
    a_registration = json.loads((A_NEIGHBOR_DIR / "registration.json").read_text())
    a_neighbors = json.loads((A_NEIGHBOR_DIR / "registered_candidates.json").read_text())
    a_evaluation = json.loads((A_NEIGHBOR_DIR / "evaluation.json").read_text())
    if (a_registration["fingerprints"] != original_neighborhood.fingerprints()
            or a_registration["registry_sha256"] != sha(A_NEIGHBOR_DIR / "registered_candidates.json")
            or a_evaluation["registry_sha256"] != a_registration["registry_sha256"]
            or a_evaluation["registration_sha256"] != sha(A_NEIGHBOR_DIR / "registration.json")
            or a_evaluation["paths_sha256"] != sha(A_NEIGHBOR_DIR / "paths.npz")
            or a_evaluation["primary_unchanged"] != a["selection"]["primary"]):
        raise ValueError("Frozen A neighborhood proof changed")
    return dict(a=a, b=b, a_neighbors=a_neighbors, a_neighborhood_evaluation=a_evaluation)


def prepare_records(sources):
    a, b = sources["a"], sources["b"]
    b_selection = b["selection"]
    if b_selection["parents"] != [b_selection["primary"]]:
        raise ValueError("This extension is registered for the one frozen B primary")
    b_record = original_neighborhood.generate(b["registry"], b_selection)
    b_record.update(stage="b_development_neighborhood_diagnostic", subject_kind="frozen_B_candidate",
                    adaptive_B_scope=b["registry"]["scope"], selection_performed_here=False)
    h_id = a["registry"]["controls"]["h"]
    # This is a generator argument, not a replacement selection receipt.
    h_arguments = dict(primary=h_id, parents=[h_id], controls=deepcopy(a["registry"]["controls"]),
                       selection_period=[START, DEV_END])
    h_record = original_neighborhood.generate(a["registry"], h_arguments)
    h_record.pop("primary_unchanged")
    h_record.update(stage="fixed_H_development_neighborhood_diagnostic", subject_kind="fixed_H_control",
                    fixed_control_parent=h_id, selection_performed_here=False,
                    A_primary_unchanged=a["selection"]["primary"], B_primary_unchanged=b_selection["primary"])
    override = dict(created_by="explicit fixed-control diagnostic; no ranking or model selection",
        parent=h_id, parent_role="pre-existing frozen H benchmark", generator_arguments=h_arguments,
        selected_as_new_A_or_B_candidate=False, adaptive_selection=False,
        A_primary_unchanged=a["selection"]["primary"], B_primary_unchanged=b_selection["primary"],
        reason="Compare candidate neighborhoods with the inherited H neighborhood, not just with one H point")
    records = {"b": b_record, "h": h_record}
    subject_records = {"A": (sources["a_neighbors"], a["selection"]["primary"]),
                       "B": (b_record, b_selection["primary"]), "H": (h_record, h_id)}
    operation_maps = {}
    for label, (record, parent) in subject_records.items():
        operation_maps[label] = {p["label"]: p["candidate_id"]
            for p in record["parent_map"][parent]["proposals"]
            if p["role"] == "neighbor" and not p["no_op"] and not p["label"].startswith("score_window_")}
    common = sorted(set.intersection(*(set(value) for value in operation_maps.values())))
    matched = dict(operation_labels=common,
        subjects={label: dict(parent=subject_records[label][1],
                              operation_to_id={name: value[name] for name in common})
                  for label, value in operation_maps.items()},
        predefined_rule="Intersect effective whole-window/smooth/panic/buffer/hold/confirmation operators across A/B/H before new results",
        role="Supplementary only; full neighborhoods remain primary and disclose all additional active axes")
    for record in records.values():
        record.update(complete_neighborhood_is_primary=True, matched_comparison=deepcopy(matched),
                      A_primary_unchanged=a["selection"]["primary"], B_primary_unchanged=b_selection["primary"],
                      read_confirmation_results=False, read_2026_results=False)
    return records, override, matched


def fingerprints():
    files = {"A_registry": BASE / "registered_candidates.json"}
    for stage, directory in (("A", A_DIR), ("B", B_DIR)):
        for name in ("registration.json", "selection.json", "evaluation.json", "path_metadata.json", "paths.npz"):
            files[stage + "/" + name] = directory / name
    files["B/registered_candidates.json"] = B_DIR / "registered_candidates.json"
    for name in ("registered_candidates.json", "registration.json", "feature_receipt.json", "evaluation.json",
                 "path_metadata.json", "paths.npz", "REPORT.md"):
        files["A_neighborhood/" + name] = A_NEIGHBOR_DIR / name
    return dict(sources={name: sha(ROOT / name) for name in SOURCE_NAMES},
                original_artifacts={key: sha(path) for key, path in files.items()},
                corrected_manifest_sha256=sha(ROOT / "v10_h_close/corrected_manifest.json"),
                qvix_manifest_sha256=sha(ROOT / "v10_h_close/qvix_manifest.json"),
                protected_manifest_sha256=sha(BASE / "protected_manifest.json"))


def _freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError("Do not overwrite frozen extended-neighborhood definition: " + str(path))
    else:
        dump(path, value)


def register_all():
    sources = load_sources()
    records, override, matched = prepare_records(sources)
    current = fingerprints()
    protect()
    registrations = {}
    # Both sets and the fixed common-operation table precede either simulation.
    for key in ("b", "h"):
        out, record = OUTS[key], records[key]
        _freeze_json(out / "registered_candidates.json", record)
        _freeze_json(out / "matched_comparison_registration.json", matched)
        if key == "h":
            receipt = dict(override, source_A_registry_sha256=current["original_artifacts"]["A_registry"],
                           source_A_selection_sha256=current["original_artifacts"]["A/selection.json"],
                           source_B_selection_sha256=current["original_artifacts"]["B/selection.json"])
            _freeze_json(out / "parent_override.json", receipt)
        path = out / "registration.json"
        record_hash = sha(out / "registered_candidates.json")
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["fingerprints"] != current or previous["registry_sha256"] != record_hash:
                raise ValueError("Registered extended-neighborhood source/input changed")
            registrations[key] = previous
        else:
            registrations[key] = dict(registered_at=stamp(), subject=key, subject_kind=record["subject_kind"],
                registry_sha256=record_hash, fingerprints=current, candidate_count=record["candidate_count"],
                parent=record["parents"][0], period=[START, DEV_END], scenarios=SCENARIOS,
                matched_registration_sha256=sha(out / "matched_comparison_registration.json"),
                parent_override_sha256=sha(out / "parent_override.json") if key == "h" else None,
                selection_performed=False, complete_and_matched_tables_both_required=True,
                center_relative_definition="CAGR differences are neighbor minus its own center; worst pressure delta is min of four paired block differences, not difference of minima",
                gate_definition="Existing full development gates versus H, plus separately labeled four pressure-only checks; no new selection gate",
                clean_oos=False)
            dump(path, registrations[key])
        print("REGISTERED", key, record["candidate_count"], "configs; center", record["parents"][0], flush=True)
    return sources, records, registrations, matched


def _distribution(values):
    return dict(count=len(values), median=float(np.median(values)), q25=float(np.quantile(values, .25)),
                q75=float(np.quantile(values, .75)), minimum=min(values), maximum=max(values)) if values else None


def compare_to_center(parent, ids, rows, checks):
    mapping = {row["id"]: row for row in rows}
    center = mapping[parent]
    passed = [cid for cid in ids if all(checks[cid].values())]
    pressure_passed = [cid for cid in ids if all(checks[cid][key] for key in PRESSURE_KEYS)]
    scenario_stats = {}
    for name, unused_lag, unused_fee in SCENARIOS:
        base = center["scenarios"][name]["full"]
        cagr_delta = [mapping[cid]["scenarios"][name]["full"]["cagr"] - base["cagr"] for cid in ids]
        dd_worsening = [base["max_dd"] - mapping[cid]["scenarios"][name]["full"]["max_dd"] for cid in ids]
        scenario_stats[name] = dict(center=base, cagr_delta_to_center=_distribution(cagr_delta),
                                   cagr_drop_from_center=_distribution([-value for value in cagr_delta]),
                                   drawdown_worsening=_distribution(dd_worsening))
    worst = {}
    for cid in ids:
        differences = [mapping[cid]["scenarios"][name]["blocks"][block]["cagr"]
                       - center["scenarios"][name]["blocks"][block]["cagr"]
                       for name in ("close_11bp", "lag1_11bp") for block, unused_lo, unused_hi in DEV_BLOCKS]
        worst[cid] = min(differences)
    return dict(center=parent, neighbor_ids=ids, neighbor_count=len(ids),
                existing_H_gate_passed=len(passed), existing_H_gate_rate=len(passed) / len(ids) if ids else None,
                existing_H_gate_passed_ids=passed,
                pressure_only_passed=len(pressure_passed), pressure_only_rate=len(pressure_passed) / len(ids) if ids else None,
                scenarios=scenario_stats, worst_paired_pressure_block_delta_to_center=_distribution(list(worst.values())),
                per_neighbor_worst_paired_delta=worst, no_selection_or_new_gate=True)


def fidelity(record, results, dates, source):
    metadata = source["path_metadata"]
    if dates != metadata["dates"]:
        raise AssertionError("Extended neighborhood has a different development calendar")
    previous_index = {cid: i for i, cid in enumerate(metadata["ids"])}
    shared = [(i, previous_index[c["id"]], c["id"]) for i, c in enumerate(record["candidates"])
              if c["id"] in previous_index]
    with np.load(source["directory"] / "paths.npz", allow_pickle=False) as previous:
        for name, unused_lag, unused_fee in SCENARIOS:
            for field in ("returns", "holdings", "summary"):
                before = previous[name + "__" + field]
                for i, old, cid in shared:
                    if not np.array_equal(results[name][field][i], before[old]):
                        raise AssertionError("Known development path changed: " + cid + " " + name + " " + field)
    required = set(record["parents"]) | set(record["controls"].values())
    if not required <= {cid for unused_i, unused_old, cid in shared}:
        raise AssertionError("A diagnostic center/control has no frozen source comparison")
    return dict(passed=True, exact_ids=[cid for unused_i, unused_old, cid in shared],
                scenarios=[row[0] for row in SCENARIOS], fields=["returns", "holdings", "summary"])


def _run_one(key, sources, record, registration, workers):
    out = OUTS[key]
    if (out / "evaluation.json").exists():
        value = json.loads((out / "evaluation.json").read_text())
        if value["registration_sha256"] != sha(out / "registration.json") or value["fingerprints"] != fingerprints():
            raise ValueError("Existing diagnostic evaluation has mismatched provenance")
        for name, field in (("registered_candidates.json", "registry_sha256"),
                            ("feature_receipt.json", "feature_receipt_sha256"), ("paths.npz", "paths_sha256")):
            if sha(out / name) != value[field]:
                raise ValueError("Existing diagnostic artifact changed: " + name)
        return value
    arrays, meta = build(record["score_specs"])
    arrays, meta = original_neighborhood.development_arrays(arrays, meta)
    dump(out / "feature_receipt.json", dict(prepared_at=stamp(), registration_sha256=sha(out / "registration.json"),
        feature_fingerprints=meta["fingerprints"], cache_array_sha256=meta["cache_array_sha256"],
        cache_paths=meta["cache_paths"], execution_last_date=meta["dates"][-1], execution_observations=len(meta["dates"])))
    if fingerprints() != registration["fingerprints"]:
        raise AssertionError("Registered inputs/source changed before execution")
    results, dates = run_candidates(record["candidates"], arrays, meta, start=START, end=DEV_END,
                                    scenarios=SCENARIOS, workers=workers)
    if dates[-1] > "2021-12-31":
        raise AssertionError("Confirmation/report-only firewall violated")
    verified = fidelity(record, results, dates, sources["b" if key == "b" else "a"])
    rows = summarize_rows(record["candidates"], results, dates)
    diagnosed = select(rows, record["candidates"], record["controls"])
    flatness = original_neighborhood.stability(record, diagnosed["checks"], diagnosed["scores"])
    parent = record["parents"][0]
    relative = compare_to_center(parent, record["parent_map"][parent]["neighbors"], rows, diagnosed["checks"])
    np.savez_compressed(out / "paths.npz", **{name + "__" + field: value for name, payload in results.items()
                                              for field, value in payload.items()})
    dump(out / "path_metadata.json", dict(dates=dates, ids=[c["id"] for c in record["candidates"]],
                                          sha256=sha(out / "paths.npz"), only_until=DEV_END))
    if fingerprints() != registration["fingerprints"]:
        raise AssertionError("Registered inputs/source changed during execution")
    value = dict(completed_at=stamp(), period=[dates[0], dates[-1]], subject_kind=record["subject_kind"],
        diagnostic_parent=parent, A_primary_unchanged=sources["a"]["selection"]["primary"],
        B_primary_unchanged=sources["b"]["selection"]["primary"], rows=rows,
        checks=diagnosed["checks"], scores=diagnosed["scores"], parent_flatness=flatness,
        relative_to_own_center=relative, fidelity=verified, fingerprints=registration["fingerprints"],
        registration_sha256=sha(out / "registration.json"), registry_sha256=sha(out / "registered_candidates.json"),
        feature_receipt_sha256=sha(out / "feature_receipt.json"), paths_sha256=sha(out / "paths.npz"),
        protected_files_verified=protect(), selected_here=False, new_gate_added=False,
        heldout_performance_accessed=False, clean_oos=False)
    dump(out / "evaluation.json", value)
    print("COMPLETED", key, "parent", parent, "gate", relative["existing_H_gate_passed"], "/", relative["neighbor_count"], flush=True)
    return value


def _table(subjects):
    lines = ["| 中心 | 邻居数 | 原H门槛通过 | 主口径相对中心CAGR中位数/Q25 | 最差压力子段相对中心中位数/Q25 |",
             "|---|---:|---:|---:|---:|"]
    for label in ("A", "B", "H"):
        result = subjects[label]
        main = result["scenarios"]["close_1bp"]["cagr_delta_to_center"]
        worst = result["worst_paired_pressure_block_delta_to_center"]
        lines.append("| %s | %d | %d/%d (%.1f%%) | %+.3f / %+.3f pp | %+.3f / %+.3f pp |" % (
            label, result["neighbor_count"], result["existing_H_gate_passed"], result["neighbor_count"],
            100 * result["existing_H_gate_rate"], 100 * main["median"], 100 * main["q25"],
            100 * worst["median"], 100 * worst["q25"]))
    return lines


def write_comparison(sources, records, evaluations, matched):
    prior = sources["a_neighborhood_evaluation"]
    all_evaluations = dict(A=prior, B=evaluations["b"], H=evaluations["h"])
    all_records = dict(A=sources["a_neighbors"], B=records["b"], H=records["h"])
    full, common = {}, {}
    for label, evaluation in all_evaluations.items():
        parent = matched["subjects"][label]["parent"]
        ids = all_records[label]["parent_map"][parent]["neighbors"]
        full[label] = compare_to_center(parent, ids, evaluation["rows"], evaluation["checks"])
        common_ids = list(dict.fromkeys(matched["subjects"][label]["operation_to_id"].values()))
        common[label] = compare_to_center(parent, common_ids, evaluation["rows"], evaluation["checks"])
    value = dict(completed_at=stamp(), period=[START, DEV_END], full_neighborhoods=full,
                 matched_operations=matched, matched_comparison=common,
                 provenance=dict(A_evaluation_sha256=sha(A_NEIGHBOR_DIR / "evaluation.json"),
                    B_evaluation_sha256=sha(OUTS["b"] / "evaluation.json"),
                    H_evaluation_sha256=sha(OUTS["h"] / "evaluation.json"),
                    B_registration_sha256=sha(OUTS["b"] / "registration.json"),
                    H_registration_sha256=sha(OUTS["h"] / "registration.json")),
                 complete_neighborhood_is_primary=True, matched_is_supplementary=True,
                 no_new_selection=True, no_new_gate=True, heldout_performance_accessed=False, clean_oos=False)
    path = OUTS["b"] / "comparison.json"
    if path.exists():
        raise ValueError("Extended comparison is already frozen; do not overwrite")
    dump(path, value)
    lines = ["# A/B与固定H的开发期邻域比较", "", "只使用2014–2021；B为独立开发自适应扩展，A原结果完整保留。",
             "H是原有固定基准的参数扰动诊断，未经选型，也不是新A/B候选。A、B的冻结主配置均不改变。", "",
             "## 全部登记邻域（主表）", ""] + _table(full)
    lines += ["", "## 共同有效操作（预先登记的辅助表）", "",
              "共同操作在本轮绩效运行前取固定交集；该表不能替代全部邻域，也不隐藏B/A的持有期和多窗口等新增轴。", ""] + _table(common)
    lines += ["", "表中差值均为邻居减自身中心，单位为年化百分点；负值表示性能下降。最差压力差先逐格与自身中心比较，再取四格最小值，未错误使用两个最小值之差。",
              "完整四场景主指标/回撤变化、相对H的原全部门槛及独立标注的压力四项通过比例见comparison.json。相关参数扰动不是独立证据，且只在已知开发历史上评估。",
              "未据邻域择优、未添加晋级门槛、未访问确认段或2026绩效。", ""]
    (OUTS["b"] / "REPORT.md").write_text("\n".join(lines))
    (OUTS["h"] / "REPORT.md").write_text("# 固定H匹配邻域\n\n这是基准诊断，不是新的候选选择。参见[完整A/B/H比较](../b_neighborhood/REPORT.md)。\n")
    return value


def run(workers=1):
    sources, records, registrations, matched = register_all()
    if (OUTS["b"] / "comparison.json").exists():
        raise ValueError("Extended neighborhood comparison already completed")
    evaluations = {key: _run_one(key, sources, records[key], registrations[key], workers) for key in ("b", "h")}
    comparison = write_comparison(sources, records, evaluations, matched)
    for label, result in comparison["full_neighborhoods"].items():
        print(label, "full gate", result["existing_H_gate_passed"], "/", result["neighbor_count"],
              "own worst-block Q25", result["worst_paired_pressure_block_delta_to_center"]["q25"], flush=True)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    register_all() if args.command == "register" else run(args.workers)
