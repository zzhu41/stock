"""Preregistered development-only neighborhoods of the four frozen parents.

This diagnostic never chooses a replacement primary, never reads confirmation
results, and truncates execution arrays at 2021 before calling the batch engine.
"""
import argparse
from bisect import bisect_right
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import numpy as np

from .data import BASE, ROOT, START, DEV_END, dump, sha, protect
from .features import build, canonical_specs
from .registry import canonical, score_name
from .scan import SCENARIOS, DEV_BLOCKS, run_candidates, summarize_rows, select, stamp
from v10_deep.schema import identifier, semantic


OUT = BASE / "results/neighborhood"
PARENT_REGISTRY = BASE / "registered_candidates.json"
DEVELOPMENT = BASE / "results/development"
FROZEN_SCORES = {
    "wls20_smooth3": dict(estimator="wls", windows=[20], aggregate="mean", smooth=3),
    "wls25_v20": dict(estimator="wls", windows=[25], aggregate="mean", smooth=1),
}
SOURCE_NAMES = ("v11/neighborhood.py", "v11/tests/test_neighborhood.py", "v11/PROTOCOL.md",
                "v11/data.py", "v11/features.py", "v11/registry.py", "v11/scan.py",
                "v10_deep/native.py", "v10_deep/native.cpp", "v10_deep/schema.py",
                "v10_deep/reference.py", "v10_deep/scan.py")


def normalize(config):
    value = canonical(config)
    # One decision per observed day: ordinary hold 0 and 1 have identical exits.
    if value["min_hold"] == 1:
        value["min_hold"] = 0
    return value


def spec_for(name, specs):
    if name in FROZEN_SCORES:
        return dict(name=name, **deepcopy(FROZEN_SCORES[name]))
    if name not in specs:
        raise ValueError("No registered definition for parent score: " + name)
    return deepcopy(specs[name])


def materialize_spec(spec, catalog):
    value = {key: deepcopy(spec[key]) for key in ("estimator", "windows", "aggregate", "smooth")}
    value["windows"] = sorted(set(value["windows"]))
    for name, frozen in FROZEN_SCORES.items():
        if value == frozen:
            return name  # Never replace an exact frozen control by a fresh center calculation.
    name = score_name(value["estimator"], value["windows"], value["aggregate"], value["smooth"])
    value["name"] = name
    if name in catalog and catalog[name] != value:
        raise ValueError("Conflicting neighborhood score name")
    catalog[name] = value
    return name


def _decimal_shift(value, delta):
    return float(Decimal(str(value)) + Decimal(str(delta)))


def _decimal_scale(value, scale):
    return float(Decimal(str(value)) * Decimal(str(scale)))


def generate(original, selection):
    """Pure registration generator. Its inputs contain only already frozen choices."""
    centers = {candidate["id"]: candidate for candidate in original["candidates"]}
    parent_ids = list(selection["parents"])
    if not parent_ids or len(parent_ids) > 4 or selection["primary"] not in parent_ids:
        raise ValueError("Expected one to four frozen parents including the existing primary")
    if selection["selection_period"] != [START, DEV_END] or DEV_END != "2021-12-31":
        raise ValueError("Only the frozen 2014–2021 development selection is permitted")
    h = centers[selection["controls"]["h"]]
    catalog = {spec["name"]: deepcopy(spec) for spec in original["score_specs"]}
    original_spec_names = set(catalog)
    candidates, links = {}, {}
    attempts = 0

    def add(config, role, parent=None):
        c = normalize(config)
        digest = identifier(c)
        cid = "v11_" + digest[:20]
        if cid not in candidates:
            prior = centers.get(cid)
            families = list(prior["families"]) if prior else []
            candidates[cid] = dict(c, id=cid, hash=digest, families=families, parents=[], stage="neighborhood")
        candidate = candidates[cid]
        candidate["families"] = sorted(set(candidate["families"]) | {"diagnostic_" + role})
        if parent:
            candidate["parents"] = sorted(set(candidate["parents"]) | {parent})
        assert identifier(candidate) == digest
        return cid

    for cid in selection["controls"].values():
        if add(centers[cid], "control") != cid:
            raise ValueError("Normalization unexpectedly changed a frozen control")
    for pid in parent_ids:
        parent = centers[pid]
        if add(parent, "center", pid) != pid:
            raise ValueError("Normalization unexpectedly changed a frozen parent")
        link = dict(center=pid, neighbors=[], ablations=[], proposals=[])
        links[pid] = link

        def proposal(role, label, changes=None, score_spec=None, specification=None):
            nonlocal attempts
            attempts += 1
            c = deepcopy(parent)
            c.update(changes or {})
            if score_spec is not None:
                c["score"] = materialize_spec(score_spec, catalog)
            cid = add(c, role, pid)
            bucket = link["neighbors" if role == "neighbor" else "ablations"]
            duplicate = cid in bucket
            if cid != pid and not duplicate:
                bucket.append(cid)
            link["proposals"].append(dict(role=role, label=label, candidate_id=cid,
                no_op=cid == pid, deduplicated=duplicate, specification=specification,
                changed_parameters={key: value for key, value in semantic(candidates[cid]).items()
                                    if value != normalize(parent).get(key)}))

        score = spec_for(parent["score"], catalog)
        for index, window in enumerate(score["windows"]):
            for scale in (.8, 1.2):
                changed = deepcopy(score)
                changed["windows"][index] = max(2, int(round(window * scale)))
                proposal("neighbor", "score_window_%d_x%.1f" % (index, scale), score_spec=changed,
                         specification=dict(window_index=index, scale=scale, windows=changed["windows"]))
        for scale in (.8, 1.2):
            changed = deepcopy(score)
            changed["windows"] = [max(2, int(round(window * scale))) for window in score["windows"]]
            proposal("neighbor", "score_all_windows_x%.1f" % scale, score_spec=changed,
                     specification=dict(scale=scale, windows=changed["windows"]))
        for step in (-1, 1):
            changed = deepcopy(score)
            changed["smooth"] = max(1, score["smooth"] + step)
            proposal("neighbor", "smooth_%+d" % step, score_spec=changed)
        for scale in (.9, 1.1):
            proposal("neighbor", "panic_x%.1f" % scale,
                     changes=dict(panic=_decimal_scale(parent["panic"], scale)),
                     specification=dict(panic_mode=parent["panic_mode"], relative_scale=scale))
        if parent["buffer_mode"] == "momentum":
            for shift in (-.005, .005):
                proposal("neighbor", "momentum_buffers_%+.3f" % shift,
                         changes={field: _decimal_shift(parent[field], shift)
                                  for field in ("buffer", "global_buffer", "gold_buffer")})
        for step in (-1, 1):
            proposal("neighbor", "minimum_hold_%+d" % step,
                     changes=dict(min_hold=max(0, parent["min_hold"] + step)))
            proposal("neighbor", "switch_confirmation_%+d" % step,
                     changes=dict(switch_confirm=max(1, parent["switch_confirm"] + step)))

        if any(parent[field] != h[field] for field in ("buffer", "global_buffer", "gold_buffer")):
            proposal("ablation", "restore_H_buffer_group",
                     changes={field: h[field] for field in ("buffer", "global_buffer", "gold_buffer")})
        if parent["min_hold"] != h["min_hold"]:
            proposal("ablation", "restore_H_minimum_hold", changes=dict(min_hold=h["min_hold"]))
        if any(parent[field] != h[field] for field in ("risk_context", "panic_mode", "panic")):
            proposal("ablation", "restore_H_risk_context", changes=dict(risk_context=h["risk_context"]))
            proposal("ablation", "restore_H_panic_magnitude",
                     changes={field: h[field] for field in ("panic_mode", "panic")})
            proposal("ablation", "restore_H_whole_risk",
                     changes={field: h[field] for field in ("risk_context", "panic_mode", "panic")})
        if parent["score"] != h["score"]:
            proposal("ablation", "restore_H_whole_score", changes=dict(score=h["score"]))
            changed = deepcopy(score)
            changed["estimator"] = "wls"
            proposal("ablation", "restore_WLS_estimator", score_spec=changed)
            changed = deepcopy(score)
            changed["smooth"] = 3
            proposal("ablation", "restore_three_quote_smoothing", score_spec=changed)
            if len(score["windows"]) > 1:
                changed = deepcopy(score)
                changed["windows"] = [20]
                proposal("ablation", "restore_single_20_window", score_spec=changed)

    configs = sorted(candidates.values(), key=lambda c: c["id"])
    all_specs = canonical_specs(list(catalog.values()))
    return dict(schema=1, stage="development_neighborhood_diagnostic", parents=parent_ids,
                primary_unchanged=selection["primary"], controls=deepcopy(selection["controls"]),
                candidates=configs, candidate_count=len(configs), raw_proposal_count=attempts,
                parent_map=links, score_specs=all_specs,
                extra_score_specs=[spec for spec in all_specs if spec["name"] not in original_spec_names],
                original_score_spec_count=len(original_spec_names),
                known_stage1_ids=sorted(set(candidates) & set(centers)),
                development=[START, DEV_END], scenarios=SCENARIOS, development_blocks=DEV_BLOCKS,
                heldout_performance_accessed=False, clean_oos=False, reselect_primary=False,
                conventions=dict(windows="Each window separately ±20%, and whole group ±20%; round, floor2, distinct windows",
                    smoothing="±1, floor1", panic="Relative ±10%, retaining parent's fixed/volatility mode",
                    buffers="Common ±0.005 shift of three momentum buffers, preserving their differences; no fitted weights",
                    min_hold="±1, floor0, then normalize1→0 for once-daily decisions",
                    switch_confirm="±1, floor1", frozen_scores=deepcopy(FROZEN_SCORES),
                    flatness="Distinct non-center neighbor configs only; ablations reported separately",
                    no_new_gate="Report existing development gates and median/Q25 scores; never change the frozen primary"))


def _read_inputs():
    selection = json.loads((DEVELOPMENT / "selection.json").read_text())
    provenance = selection["provenance"]
    if sha(PARENT_REGISTRY) != provenance["registry_sha256"]:
        raise ValueError("Original registered candidates changed")
    for name, key in (("registration.json", "registration_sha256"), ("evaluation.json", "evaluation_sha256"),
                      ("paths.npz", "paths_sha256")):
        if sha(DEVELOPMENT / name) != provenance[key]:
            raise ValueError("Frozen development artifact changed: " + name)
    for name, digest in provenance["source_sha256"].items():
        if sha(BASE / name) != digest:
            raise ValueError("Frozen stage-one source changed: " + name)
    original = json.loads(PARENT_REGISTRY.read_text())
    metadata = json.loads((DEVELOPMENT / "path_metadata.json").read_text())
    if not metadata["dates"] or metadata["dates"][-1] > DEV_END or metadata["sha256"] != provenance["paths_sha256"]:
        raise ValueError("Development path metadata is inconsistent or exceeds 2021")
    return original, selection, metadata


def fingerprints():
    return dict(sources={name: sha(ROOT / name) for name in SOURCE_NAMES},
                original_registry_sha256=sha(PARENT_REGISTRY),
                development={name: sha(DEVELOPMENT / name) for name in
                    ("selection.json", "registration.json", "evaluation.json", "path_metadata.json", "paths.npz")},
                corrected_manifest_sha256=sha(ROOT / "v10_h_close/corrected_manifest.json"),
                qvix_manifest_sha256=sha(ROOT / "v10_h_close/qvix_manifest.json"),
                protected_manifest_sha256=sha(BASE / "protected_manifest.json"))


def register():
    original, selection, unused = _read_inputs()
    record = generate(original, selection)
    registry_path = OUT / "registered_candidates.json"
    registration_path = OUT / "registration.json"
    current = fingerprints()
    if registry_path.exists() or registration_path.exists():
        if not registry_path.exists() or not registration_path.exists():
            raise ValueError("Incomplete neighborhood registration; do not overwrite it")
        prior = json.loads(registration_path.read_text())
        if json.loads(registry_path.read_text()) != json.loads(json.dumps(record)) or prior["fingerprints"] != current:
            raise ValueError("Neighborhood definition is already frozen and differs")
        if sha(registry_path) != prior["registry_sha256"]:
            raise ValueError("Neighborhood registry hash changed")
        return record, prior
    protect()
    dump(registry_path, record)
    registered = dict(registered_at=stamp(), registry_sha256=sha(registry_path), fingerprints=current,
                      development_only_until=DEV_END, primary_unchanged=selection["primary"],
                      parent_selection_sha256=sha(DEVELOPMENT / "selection.json"),
                      candidate_count=record["candidate_count"], extra_score_count=len(record["extra_score_specs"]),
                      reselect_primary=False, clean_oos=False)
    dump(registration_path, registered)
    print("REGISTERED neighborhood", record["candidate_count"], "configs;", len(record["extra_score_specs"]), "new scores", flush=True)
    return record, registered


def development_arrays(arrays, meta):
    """Remove all 2022+ observations before the execution engine can see them."""
    cut = bisect_right(meta["dates"], DEV_END)
    if not cut:
        raise ValueError("No development observations")
    out = {}
    for key, values in arrays.items():
        out[key] = np.ascontiguousarray(values[:, :cut] if key in ("scores", "orders") else values[:cut])
    metadata = deepcopy(meta)
    metadata["dates"] = meta["dates"][:cut]
    if "shape" in metadata:
        metadata["shape"][0] = cut
    return out, metadata


def stability(record, checks, scores):
    result = {}
    for pid in record["parents"]:
        ids = record["parent_map"][pid]["neighbors"]
        values = [scores[cid]["worst_block_excess"] for cid in ids]
        passed = [cid for cid in ids if all(checks[cid].values())]
        result[pid] = dict(center=pid, center_checks=checks[pid], center_score=scores[pid],
            neighbor_count=len(ids), passed_count=len(passed), pass_rate=len(passed) / len(ids) if ids else None,
            passed_ids=passed, worst_block_excess_median=float(np.median(values)) if values else None,
            worst_block_excess_q25=float(np.quantile(values, .25)) if values else None,
            worst_block_excess_min=min(values) if values else None, worst_block_excess_max=max(values) if values else None,
            ablations=[dict(id=cid, checks=checks[cid], score=scores[cid])
                       for cid in record["parent_map"][pid]["ablations"]],
            center_in_neighbor_statistics=False, ablations_in_neighbor_statistics=False)
    return result


def _fidelity(record, results, dates, original_metadata):
    original_index = {cid: i for i, cid in enumerate(original_metadata["ids"])}
    shared = [(i, original_index[c["id"]], c["id"]) for i, c in enumerate(record["candidates"])
              if c["id"] in original_index]
    cases = [cid for unused_i, unused_old, cid in shared]
    if dates != original_metadata["dates"]:
        raise AssertionError("Neighborhood development dates changed")
    with np.load(DEVELOPMENT / "paths.npz", allow_pickle=False) as previous:
        for scenario, unused_lag, unused_fee in SCENARIOS:
            for field in ("returns", "holdings", "summary"):
                # npz entries are decompressed on indexing: read each matrix
                # once, not once for every already-known configuration.
                matrix = previous[scenario + "__" + field]
                for i, old, cid in shared:
                    if not np.array_equal(results[scenario][field][i], matrix[old]):
                        raise AssertionError("Frozen development path changed: " + cid + " " + scenario + " " + field)
    required = set(record["parents"]) | set(record["controls"].values())
    if not required <= set(cases):
        raise AssertionError("Missing frozen parent/control verification")
    return dict(passed=True, exact_original_ids=cases, all_four_scenarios_exact=True,
                verified_fields=["returns", "holdings", "summary"], cutoff=DEV_END)


def run(workers=1):
    if (OUT / "evaluation.json").exists():
        raise ValueError("Neighborhood diagnostics already exist; never overwrite after viewing results")
    record, registration = register()
    arrays, meta = build(record["score_specs"])
    arrays, meta = development_arrays(arrays, meta)
    dump(OUT / "feature_receipt.json", dict(prepared_at=stamp(),
        registration_sha256=sha(OUT / "registration.json"), feature_fingerprints=meta["fingerprints"],
        cache_array_sha256=meta["cache_array_sha256"], cache_paths=meta["cache_paths"],
        execution_last_date=meta["dates"][-1], execution_observations=len(meta["dates"])))
    if fingerprints() != registration["fingerprints"]:
        raise AssertionError("Inputs/source changed after neighborhood registration")
    results, dates = run_candidates(record["candidates"], arrays, meta, start=START, end=DEV_END,
                                    scenarios=SCENARIOS, workers=workers)
    if dates[-1] > DEV_END:
        raise AssertionError("Holdout firewall violated")
    original, selection, original_metadata = _read_inputs()
    audit = _fidelity(record, results, dates, original_metadata)
    rows = summarize_rows(record["candidates"], results, dates)
    diagnostic = select(rows, record["candidates"], record["controls"])
    # Retain only existing gate tests and diagnostic scores. The selector's
    # neighborhood ranking/primary is intentionally neither persisted nor used.
    flatness = stability(record, diagnostic["checks"], diagnostic["scores"])
    packed = {scenario + "__" + field: values for scenario, payload in results.items()
              for field, values in payload.items()}
    np.savez_compressed(OUT / "paths.npz", **packed)
    dump(OUT / "path_metadata.json", dict(dates=dates, ids=[c["id"] for c in record["candidates"]],
                                          sha256=sha(OUT / "paths.npz"), only_until=DEV_END))
    if fingerprints() != registration["fingerprints"]:
        raise AssertionError("Inputs/source changed during neighborhood diagnostics")
    evaluation = dict(completed_at=stamp(), period=[dates[0], dates[-1]], rows=rows,
        primary_unchanged=selection["primary"], parent_flatness=flatness,
        checks=diagnostic["checks"], scores=diagnostic["scores"], fidelity=audit,
        registration_sha256=sha(OUT / "registration.json"), registry_sha256=sha(OUT / "registered_candidates.json"),
        feature_receipt_sha256=sha(OUT / "feature_receipt.json"), paths_sha256=sha(OUT / "paths.npz"),
        parent_selection_sha256=sha(DEVELOPMENT / "selection.json"), fingerprints=registration["fingerprints"],
        protected_files_verified=protect(), reselected=False, new_gate_added=False, clean_oos=False,
        limitation="Selected-parent, known-development-history diagnostics; correlated neighboring trials are not independent evidence.")
    dump(OUT / "evaluation.json", evaluation)
    lines = ["# V11开发期邻域诊断", "", "范围仅2014–2021。先登记后运行；未访问2022–2025确认结果或2026结果。",
             "冻结主候选仍为 `%s`；本表不重新选型、不增加临时晋级门槛。" % selection["primary"], "",
             "| 父候选 | 非中心独立参数邻居 | 原开发门槛通过 | 最差压力子段超额中位数 | Q25 |",
             "|---|---:|---:|---:|---:|"]
    for pid in record["parents"]:
        row = flatness[pid]
        lines.append("| %s | %d | %d/%d (%.1f%%) | %+.3f pp | %+.3f pp |" %
            (pid, row["neighbor_count"], row["passed_count"], row["neighbor_count"], 100 * row["pass_rate"],
             100 * row["worst_block_excess_median"], 100 * row["worst_block_excess_q25"]))
    lines += ["", "统计以不同的非中心参数配置计数；单项消融单独保存在evaluation.json，未混入邻域通过率。",
              "最差压力子段超额是2014–2017/2018–2021、同收盘11bp/滞后1日11bp四格中相对H年化超额的最小值。",
              "同一数据上的邻近试验高度相关，不构成独立样本外证据。所有已在第一阶段出现的配置均逐日复核四场景收益、持仓和汇总完全一致。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print("DIAGNOSTICS COMPLETE; unchanged primary", selection["primary"], flush=True)
    for pid, row in flatness.items():
        print(pid, row["passed_count"], "/", row["neighbor_count"],
              "worst-excess median", row["worst_block_excess_median"], "Q25", row["worst_block_excess_q25"], flush=True)
    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    register() if args.command == "register" else run(args.workers)
