"""Registered paired trading-pool exclusions for frozen A/B/H; never select.

Run `python3.8 -B -m v11.leave_one_out register` only after report_2026 is
complete, then `... run`. The embedded protocol is saved before evaluation.
Prices, scores, asset order and the benchmark's regime information stay intact.
Only stock_pool/global_pool trading permissions change in copied configs.
"""
import argparse
from bisect import bisect_left, bisect_right
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .data import BASE, ROOT, START, DEV_END, CONFIRM_END, END
from .development_diagnostics import sha, read, dump, stamp, require
from .features import build, risk_view
from .scan import SCENARIOS, run_candidates
from v10_deep.data import ASSET_ORDER, CASH, GOLD, BENCHMARK
from v10_deep.schema import STOCK, GLOBAL, semantic, identifier
from v10_deep.scan import metrics
from v10_deep.reference import run_reference


OUTPUT = BASE / "results/leave_one_out"
ROLES = ("a", "b", "h")
PERIODS = (("full", START, END), ("development", START, DEV_END),
           ("confirmation", "2022-01-01", CONFIRM_END), ("report_only_2026", "2026-01-01", END))
SOURCE_NAMES = ("v11/leave_one_out.py", "v11/tests/test_leave_one_out.py",
    "v11/development_diagnostics.py", "v11/data.py", "v11/features.py", "v11/scan.py",
    "v11/PROTOCOL.md", "v11/COMBINATION_PROTOCOL.md", "v11/LITERATURE.md",
    "v10_deep/native.cpp", "v10_deep/native.py", "v10_deep/schema.py",
    "v10_deep/reference.py", "v10_deep/features.py", "v10_deep/data.py", "v10_deep/scan.py")

PROTOCOL = {
    "purpose": "Diagnostic trading-eligibility sensitivity of already frozen A, B and H, not asset-pool selection.",
    "cases": "Full pool plus one exclusion for each of the seven original stock ETFs and two global ETFs; paired identical exclusions for A/B/H.",
    "preserved": "Gold and cash remain available. Original price/features/scores/ranks, asset order, QVIX and benchmark MA information are retained, including the excluded asset's information.",
    "interpretation": "This removes permission to trade an ETF, not all information attributable to it; no subset is promoted and no future-result-based parameter changes are allowed.",
    "execution": "Use the existing native single-position engine with actual state/fees under all four registered SCENARIOS; no splice of target-path returns.",
    "periods": "Run once continuously from 2014-01-02 to 2026-09-24, then slice full/development/2022-2025/2026. Keep boundary-day returns and switch fees; no fresh free entry at a period boundary.",
    "comparisons": "Each reduced-pool path versus its own full-pool path, versus H under the identical exclusion, and change in excess versus the full-pool A/B-to-H comparison. CAGR and DD differences are percentage points; also report paired net log-growth differences.",
    "tail": "2026 is frozen reporting only. Report its cumulative return and DD, not a short-period annualized CAGR or any reselection verdict.",
    "reference_cases": "For A remove benchmark 510300; for B remove first registered global ETF; for H remove first stock ETF. Check close_1bp and lag1_11bp against independent Python reference, all daily returns/holdings/summary exactly.",
    "receipts": "Require completed report_2026 registration/evaluation/path receipts. Full-pool A/B/H must equal the report's complete paths, including every summary field, before interpreting exclusions.",
    "limitations": "All historical periods and frozen candidates were already researched; exclusions are dependent diagnostics, not independent OOS or causal asset attribution. This diagnostic does not overturn failed confirmation or correct earlier adaptive searches.",
    "literature_notes": [
        {"paper": "Hurst, Ooi, Pedersen (2017), A Century of Evidence on Trend-Following Investing",
         "url": "https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/AQR-JPM-Fall-2017.pdf",
         "use": "Broad multi-market evidence motivates checking dependence on one tradable market; their diversified futures method is not this ETF strategy, and leave-one-out is an engineering diagnostic, not their performance guarantee."},
        {"paper": "Garleanu, Pedersen (2013), Dynamic Trading with Predictable Returns and Transaction Costs",
         "url": "https://www.nber.org/papers/w15205",
         "use": "Trading responsiveness must be assessed after costs; this does not establish the optimal holding period or pool here."},
        {"paper": "White (2000), A Reality Check for Data Snooping",
         "url": "https://doi.org/10.1111/1468-0262.00152",
         "use": "Selecting whichever exclusion looks best would add another search; report all fixed exclusions and retain both frozen failed candidates without replacement."},
    ],
}


def exclude_asset(config, code=None):
    """Copy a semantic config, changing only a single pool-membership entry."""
    result = semantic(config)
    pools = result["stock_pool"] + result["global_pool"]
    require(len(pools) == len(set(pools)) and not set(pools) & {GOLD, CASH}, "Invalid original asset roles")
    if code is not None:
        require(code not in (GOLD, CASH) and code in pools, "Exclusion must be one current stock/global member")
        key = "stock_pool" if code in result["stock_pool"] else "global_pool"
        result[key] = [asset for asset in result[key] if asset != code]
    digest = identifier(result)
    return dict(result, id="v11_" + digest[:20], hash=digest,
                families=["paired_pool_exclusion_diagnostic"], parents=[config["id"]], stage="leave_one_out")


def make_plan(configs):
    require(set(configs) == set(ROLES), "Exactly frozen A, B and H are required")
    reference = configs["h"]
    for role in ROLES:
        require(configs[role]["stock_pool"] == reference["stock_pool"]
                and configs[role]["global_pool"] == reference["global_pool"], "A/B/H starting pools differ")
        require(identifier(configs[role]) == configs[role]["hash"], "Frozen config hash differs")
    excluded = list(reference["stock_pool"] + reference["global_pool"])
    require(len(excluded) == len(set(excluded)), "Duplicate pool assets")
    cases, candidates = [], {}
    for code in [None] + excluded:
        roles = {}
        for role in ROLES:
            candidate = exclude_asset(configs[role], code)
            if candidate["id"] in candidates:
                require(semantic(candidate) == semantic(candidates[candidate["id"]]), "Semantic ID collision")
            candidates[candidate["id"]] = candidate
            roles[role] = candidate["id"]
        cases.append(dict(case="full_pool" if code is None else "without_" + code, excluded=code, roles=roles))
    return dict(cases=cases, candidates=[candidates[cid] for cid in sorted(candidates)],
                excluded_assets=excluded, preserved_assets=[GOLD, CASH],
                frozen_ids={role: configs[role]["id"] for role in ROLES}, no_reselection=True)


def _stage(folder, registry_path):
    selected, registration, registry = read(folder / "selection.json"), read(folder / "registration.json"), read(registry_path)
    proof = selected["provenance"]
    files = {"selection": folder / "selection.json", "registration": folder / "registration.json",
             "evaluation": folder / "evaluation.json", "paths": folder / "paths.npz",
             "path_metadata": folder / "path_metadata.json", "registry": registry_path}
    for name in ("registration", "evaluation", "paths", "registry"):
        require(sha(files[name]) == proof[name + "_sha256"], "Frozen stage changed: " + name)
    require(proof["source_sha256"] == registration["sources"], "Stage source receipts disagree")
    require({name: sha(BASE / name) for name in registration["sources"]} == registration["sources"],
            "Frozen stage source changed")
    require(selected["selection_period"] == [START, DEV_END], "Unexpected selection period")
    require(read(files["path_metadata"])["sha256"] == proof["paths_sha256"], "Stage path receipt mismatch")
    return registry, selected, registration, files


def frozen_inputs():
    a, sa, ra, afiles = _stage(BASE / "results/development", BASE / "registered_candidates.json")
    b, sb, rb, bfiles = _stage(BASE / "results/combinations", BASE / "results/combinations/registered_candidates.json")
    require(sa["primary"] and sb["primary"], "Both frozen primary IDs are required")
    require(rb["a_selection_sha256"] == sha(afiles["selection"])
            and rb["a_evaluation_sha256"] == sha(afiles["evaluation"]), "B's A-input receipts differ")
    require(ra["feature_fingerprints"] == rb["feature_fingerprints"], "A/B data fingerprints differ")
    combined = {c["id"]: c for c in a["candidates"] + b["candidates"]}
    ids = dict(a=sa["primary"], b=sb["primary"], h=a["controls"]["h"])
    configs = {role: deepcopy(combined[cid]) for role, cid in ids.items()}
    require(set(configs["h"]["stock_pool"]) == set(STOCK) and set(configs["h"]["global_pool"]) == set(GLOBAL),
            "This diagnostic is restricted to the original seven stock/two global ETFs")
    plan = make_plan(configs)
    folder = BASE / "results/report_2026"
    require((folder / "evaluation.json").exists(), "Completed frozen report_2026 is required before registration or execution")
    report, rr, rm = read(folder / "evaluation.json"), read(folder / "registration.json"), read(folder / "path_metadata.json")
    require(report["registration_sha256"] == sha(folder / "registration.json"), "Report registration mismatch")
    require(report["paths_sha256"] == rm["sha256"] == sha(folder / "paths.npz"), "Report path hash mismatch")
    require(rr["evaluation_end"] == END and rr["no_reselection"] and report["no_reselection"], "Report is incomplete or reselected")
    require(report["roles"] == rr["finalists"]["roles"], "Report roles differ from its registration")
    require(rr["finalists"]["a_selection_sha256"] == sha(afiles["selection"])
            and rr["finalists"]["b_selection_sha256"] == sha(bfiles["selection"]), "Report used different frozen selections")
    for role, name in (("a", "v11_a"), ("b", "v11_b"), ("h", "v10_h")):
        require(report["roles"][name] == ids[role], "Report/frozen primary mismatch: " + role)
    require(report["sources"] == rr["sources"] == {n: sha(BASE / n) for n in rr["sources"]}, "Report source changed")
    require(rm["ids"] == [c["id"] for c in rr["finalists"]["candidates"]]
            and rm["dates"] == sorted(set(rm["dates"]))
            and [rm["dates"][0], rm["dates"][-1]] == [START, END], "Report date/ID axis mismatch")
    require(all(list(s) in rr["scenarios"] for s in SCENARIOS), "Report is missing a required scenario")
    files = {"a_" + name: path for name, path in afiles.items()}
    files.update({"b_" + name: path for name, path in bfiles.items()})
    files.update({"report_" + name: folder / name for name in
                  ("registration.json", "evaluation.json", "path_metadata.json", "paths.npz")})
    manifests = {"corrected": ROOT / "v10_h_close/corrected_manifest.json",
                 "qvix": ROOT / "v10_h_close/qvix_manifest.json", "frozen_profiles": ROOT / "v10_deep/profiles.json"}
    require({name: sha(path) for name, path in manifests.items()} == ra["feature_fingerprints"]["input_fingerprints"],
            "Frozen data manifests changed")
    files.update({"manifest_" + name: path for name, path in manifests.items()})
    return dict(plan=plan, files=files, score_specs=a["score_specs"], feature_fingerprints=ra["feature_fingerprints"],
                feature_array_sha256=rb["feature_array_sha256"], report_metadata=rm)


def registration_design(inputs):
    return dict(schema=1, protocol=PROTOCOL, plan=inputs["plan"], scenarios=[list(s) for s in SCENARIOS],
        periods=[list(p) for p in PERIODS], evaluation_start=START, evaluation_end=END,
        score_specs=inputs["score_specs"], feature_fingerprints=inputs["feature_fingerprints"],
        feature_array_sha256=inputs["feature_array_sha256"],
        inputs={name: dict(path=str(path), sha256=sha(path))
                for name, path in inputs["files"].items()},
        sources={name: sha(ROOT / name) for name in SOURCE_NAMES},
        path_count=len(inputs["plan"]["cases"]) * len(ROLES) * len(SCENARIOS),
        candidate_selection_performed=False, confirmation_failure_overturned=False, clean_oos=False)


def register(output=OUTPUT):
    output = Path(output)
    inputs = frozen_inputs()
    design = registration_design(inputs)
    path = output / "registration.json"
    if path.exists():
        require(read(path)["design"] == design, "A different leave-one-out design is already registered")
    else:
        dump(path, dict(registered_at=stamp(), design=design))
    return design


def period_metrics(returns, holdings, dates, periods=PERIODS):
    require(len(returns) == len(holdings) == len(dates), "Path/date length mismatch")
    changes = np.r_[False, np.asarray(holdings)[1:] != np.asarray(holdings)[:-1]]
    out = {}
    for name, begin, end in periods:
        lo, hi = bisect_left(dates, begin), bisect_right(dates, end)
        if hi <= lo:
            continue
        values = metrics(returns[lo:hi])
        values.update(net_log_growth=float(np.log1p(returns[lo:hi]).sum()), switches=int(changes[lo:hi].sum()))
        if name == "report_only_2026":
            values["cagr"] = values["calmar"] = None
        out[name] = values
    return out


def metric_difference(current, reference):
    return dict(cagr_pp=None if current["cagr"] is None or reference["cagr"] is None
                else 100 * (current["cagr"] - reference["cagr"]),
                total_return_pp=100 * (current["total_return"] - reference["total_return"]),
                max_dd_pp=100 * (current["max_dd"] - reference["max_dd"]),
                net_log_growth=current["net_log_growth"] - reference["net_log_growth"],
                switches=current["switches"] - reference["switches"])


def summarize(plan, results, dates, periods=PERIODS):
    indices = {c["id"]: i for i, c in enumerate(plan["candidates"])}
    base = plan["cases"][0]
    require(base["excluded"] is None, "Full pool must be the first case")
    rows = []
    for case in plan["cases"]:
        row = dict(case=case["case"], excluded=case["excluded"], roles={})
        for role in ROLES:
            scenarios = {}
            for scenario, payload in results.items():
                def summary(cid):
                    i = indices[cid]
                    return period_metrics(payload["returns"][i], payload["holdings"][i], dates, periods)
                current, own_full = summary(case["roles"][role]), summary(base["roles"][role])
                paired_h, full_h = summary(case["roles"]["h"]), summary(base["roles"]["h"])
                scenarios[scenario] = {}
                for period, value in current.items():
                    h_difference = metric_difference(value, paired_h[period])
                    prior_difference = metric_difference(own_full[period], full_h[period])
                    scenarios[scenario][period] = dict(metrics=value,
                        versus_own_full_pool=metric_difference(value, own_full[period]),
                        versus_paired_h=h_difference,
                        change_in_excess_vs_full_pool={k: None if h_difference[k] is None or prior_difference[k] is None
                            else h_difference[k] - prior_difference[k] for k in h_difference})
            row["roles"][role] = dict(id=case["roles"][role], scenarios=scenarios)
        rows.append(row)
    return rows


def reference_cases(plan):
    full = plan["cases"][0]
    lookup = {c["id"]: c for c in plan["candidates"]}
    h = lookup[full["roles"]["h"]]
    required = (("a", BENCHMARK), ("b", h["global_pool"][0]), ("h", h["stock_pool"][0]))
    cases = {c["excluded"]: c for c in plan["cases"]}
    return [(role, code, cases[code]["roles"][role]) for role, code in required]


def validate_reference(plan, arrays, meta, results, dates):
    configs = {c["id"]: c for c in plan["candidates"]}
    indices = {c["id"]: i for i, c in enumerate(plan["candidates"])}
    proofs = []
    for role, code, cid in reference_cases(plan):
        config = configs[cid]
        view = risk_view(arrays, meta, config["risk_context"], score=config["score"])
        for name, lag, fee in (SCENARIOS[0], SCENARIOS[3]):
            oracle = run_reference(view, meta, dict(config, lag=lag), start=dates[0], end=dates[-1], fee=fee)
            require(oracle["dates"] == dates, "Independent reference date mismatch")
            for field in ("returns", "holdings", "summary"):
                require(np.array_equal(oracle[field], results[name][field][indices[cid]]),
                        "Leave-one-out reference mismatch: " + role + " " + code + " " + name + " " + field)
            proofs.append(dict(role=role, excluded=code, id=cid, scenario=name, observations=len(dates), exact=True))
    return proofs


def validate_full_pool(plan, results, dates, report_metadata, archive):
    require(dates == report_metadata["dates"], "Full-pool/report date axes differ")
    indices = {c["id"]: i for i, c in enumerate(plan["candidates"])}
    original = {cid: i for i, cid in enumerate(report_metadata["ids"])}
    proofs = []
    for role, cid in plan["cases"][0]["roles"].items():
        require(cid == plan["frozen_ids"][role], "Full-pool configuration changed")
        require(cid in original, "Frozen role missing from report paths")
        for name, _, _ in SCENARIOS:
            for field in ("returns", "holdings", "summary"):
                require(np.array_equal(results[name][field][indices[cid]], archive[name + "__" + field][original[cid]]),
                        "Full-pool/report path mismatch: " + role + " " + name + " " + field)
            proofs.append(dict(role=role, id=cid, scenario=name, observations=len(dates), exact=True))
    return proofs


def run(output=OUTPUT, workers=1):
    output = Path(output)
    require((output / "registration.json").exists(), "Register the diagnostic before running")
    require(not (output / "evaluation.json").exists(), "Completed leave-one-out results are frozen")
    inputs = frozen_inputs()
    design = registration_design(inputs)
    require(read(output / "registration.json")["design"] == design, "Diagnostic inputs/source changed after registration")
    registration_hash = sha(output / "registration.json")
    arrays, meta = build(inputs["score_specs"])
    require(meta["fingerprints"] == inputs["feature_fingerprints"]
            and meta["cache_array_sha256"] == inputs["feature_array_sha256"], "Feature cache differs from frozen stages")
    dump(output / "feature_receipt.json", dict(registration_sha256=registration_hash,
         array_sha256=meta["cache_array_sha256"], fingerprints=meta["fingerprints"],
         excluded_assets_keep_their_features=True))
    plan = inputs["plan"]
    results, dates = run_candidates(plan["candidates"], arrays, meta, start=START, end=END, scenarios=SCENARIOS, workers=workers)
    with np.load(str(inputs["files"]["report_paths.npz"]), allow_pickle=False) as old:
        full_fidelity = validate_full_pool(plan, results, dates, inputs["report_metadata"], old)
    independent_fidelity = validate_reference(plan, arrays, meta, results, dates)
    # Deleted assets must be absent from all executed holdings, including the
    # crash override path; their feature rows remain available throughout.
    positions = {c["id"]: i for i, c in enumerate(plan["candidates"])}
    for case in plan["cases"][1:]:
        forbidden = ASSET_ORDER.index(case["excluded"])
        for name, payload in results.items():
            for cid in case["roles"].values():
                require(not np.any(payload["holdings"][positions[cid]] == forbidden), "Excluded ETF was traded")
    rows = summarize(plan, results, dates)
    require(registration_design(inputs) == design and sha(output / "registration.json") == registration_hash,
            "Inputs, code or diagnostic registration changed during evaluation")
    np.savez_compressed(output / "paths.npz", **{name + "__" + field: values
        for name, payload in results.items() for field, values in payload.items()})
    dump(output / "path_metadata.json", dict(ids=[c["id"] for c in plan["candidates"]], dates=dates,
         sha256=sha(output / "paths.npz"), cases=plan["cases"]))
    dump(output / "evaluation.json", dict(registration_sha256=registration_hash, completed_at=stamp(),
         path_sha256=sha(output / "paths.npz"), rows=rows, full_pool_report_fidelity=full_fidelity,
         independent_reference_fidelity=independent_fidelity, frozen_ids=plan["frozen_ids"],
         no_reselection=True, confirmation_failure_overturned=False, clean_oos=False,
         units="CAGR/return/DD differences are percentage points; positive DD difference means less severe drawdown; net_log_growth differences are natural-log units.",
         protocol=PROTOCOL))
    dump(output / "receipt.json", dict(completed_at=stamp(),
         artifacts_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "receipt.json"},
         inputs=design["inputs"], sources=design["sources"], no_reselection=True))
    print(json.dumps(dict(cases=len(plan["cases"]), roles=list(ROLES), paths=design["path_count"],
         full_pool_reference_cases=len(full_fidelity), independent_reference_cases=len(independent_fidelity),
         output=str(output), no_reselection=True)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.command == "register":
        value = register()
        print(json.dumps(dict(registered=True, paths=value["path_count"], no_reselection=True)), flush=True)
    else:
        run(workers=args.workers)
