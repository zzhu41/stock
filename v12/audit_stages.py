"""Independent audit of frozen V12 stage evidence; never runs a search.

All row metrics and gates are rebuilt from the saved continuous daily returns.
A small, explicitly requested set of descriptive leaders is replayed using the
Python reference, not the native simulator. Outputs live in a new audit folder;
neither source receipts nor old evidence are rewritten.
"""
import argparse
from bisect import bisect_left, bisect_right
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .data import BASE, START, END, load_inputs, sha, dump, stamp
from .reference import run_reference


SCENARIOS = (("close_1bp", 0, .0001), ("close_11bp", 0, .0011),
             ("lag1_1bp", 1, .0001), ("lag1_11bp", 1, .0011))
BLOCKS = (("early", START, "2017-12-31"), ("middle", "2018-01-01", "2021-12-31"),
          ("recent", "2022-01-01", "2025-12-31"))
NEAR_BALANCED_A = "v12_6ace1b82572e526d2b75"
CONTROL_KEYS = ("v9", "v91", "v92", "simple", "h")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def independent_metrics(returns):
    """Scalar compounded wealth, pre-first-day peak, stable scalar reductions."""
    values = [float(value) for value in returns]
    require(values and all(math.isfinite(v) and v > -1 for v in values), "Invalid daily returns")
    wealth, peak, drawdown = 1., 1., 0.
    for value in values:
        wealth *= 1. + value
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1.)
    n = len(values)
    average = math.fsum(values) / n
    sd = math.sqrt(math.fsum((value - average) ** 2 for value in values) / n)
    cagr = math.expm1(math.fsum(math.log1p(value) for value in values) * 244. / n)
    return dict(sessions=n, nav=wealth, total_return=wealth - 1., cagr=cagr, max_dd=drawdown,
                volatility=sd * math.sqrt(244.), sharpe=average / sd * math.sqrt(244.) if sd else 0.,
                calmar=cagr / abs(drawdown) if drawdown else 0.)


def independent_checks(rows, controls):
    """Ten fixed gates, directly expressed without importing the selector."""
    h = rows[controls["h"]]["scenarios"]
    v92 = rows[controls["v92"]]["scenarios"]
    result = {}
    for cid, row in rows.items():
        scenarios = row["scenarios"]
        main = scenarios["close_1bp"]
        best_main_dd = max(rows[controls[key]]["scenarios"]["close_1bp"]["full"]["max_dd"] for key in CONTROL_KEYS)
        checks = dict(main_cagr=main["full"]["cagr"] >= h["close_1bp"]["full"]["cagr"] + .01,
                      main_drawdown=main["full"]["max_dd"] >= best_main_dd + .005,
                      tail_return=main["tail"]["total_return"] >= v92["close_1bp"]["tail"]["total_return"])
        for name in ("close_11bp", "lag1_11bp"):
            old = [rows[controls[key]]["scenarios"][name]["full"] for key in CONTROL_KEYS]
            checks[name + "_cagr"] = scenarios[name]["full"]["cagr"] >= max(v["cagr"] for v in old)
            checks[name + "_drawdown"] = scenarios[name]["full"]["max_dd"] >= max(v["max_dd"] for v in old)
        for label, unused_start, unused_end in BLOCKS:
            checks[label + "_cagr"] = main["blocks"][label]["cagr"] >= h["close_1bp"]["blocks"][label]["cagr"] - .05
        result[cid] = checks
    return result


def verify_receipts(stage):
    folder = BASE / "results" / stage
    registry_path = BASE / "registered_candidates.json" if stage == "main" else folder / "registered_candidates.json"
    selected = read(folder / "selection.json")
    proof = selected["provenance"]
    files = {"registration.json": "registration_sha256", "evaluation.json": "evaluation_sha256",
             "paths.npz": "paths_sha256", "path_metadata.json": "path_metadata_sha256",
             "execution_metadata.json": "execution_metadata_sha256"}
    checked = {}
    for name, key in files.items():
        current = sha(folder / name)
        require(current == proof[key], "Changed frozen artifact: " + stage + "/" + name)
        checked[str((folder / name).relative_to(BASE))] = current
    require(sha(registry_path) == proof["registry_sha256"], "Changed candidate registry")
    checked[str(registry_path.relative_to(BASE))] = sha(registry_path)
    for group in proof["fingerprints"].values():
        for name, expected in group.items():
            require(sha(BASE / name) == expected, "Changed registered dependency: " + name)
            checked[name] = expected
    registration = read(folder / "registration.json")
    require(registration["design"]["fingerprints"] == proof["fingerprints"], "Registration fingerprint disagreement")
    registry, evaluation, path_metadata = read(registry_path), read(folder / "evaluation.json"), read(folder / "path_metadata.json")
    ids = [candidate["id"] for candidate in registry["candidates"]]
    require(len(set(ids)) == len(ids) == registry["unique_count"], "Registry id/count mismatch")
    require(ids == path_metadata["ids"] == [row["id"] for row in evaluation["rows"]], "Evaluation/path order mismatch")
    require(path_metadata["dates"] == sorted(set(path_metadata["dates"])), "Invalid observation axis")
    require(path_metadata["dates"][0] == START and path_metadata["dates"][-1] == END, "Unexpected full period")
    require(evaluation["registration_sha256"] == proof["registration_sha256"], "Evaluation registration mismatch")
    require(evaluation["paths_sha256"] == path_metadata["sha256"] == proof["paths_sha256"], "Path receipt mismatch")
    require(read(folder / "execution_metadata.json")["registration_sha256"] == proof["registration_sha256"], "Execution receipt mismatch")
    checked[str((folder / "selection.json").relative_to(BASE))] = sha(folder / "selection.json")
    return folder, registry, selected, evaluation, path_metadata, checked


def audit_metrics(registry, selected, evaluation, path_metadata, archive):
    dates, reconstructed, comparisons = path_metadata["dates"], {}, 0
    largest_error = dict(absolute=0., metric=None, id=None, scenario=None, period=None)
    for i, (candidate, saved) in enumerate(zip(registry["candidates"], evaluation["rows"])):
        cid = candidate["id"]
        reconstructed[cid] = dict(id=cid, scenarios={})
        for name, unused_lag, unused_fee in SCENARIOS:
            returns = archive[name + "__returns"][i]
            holdings = archive[name + "__holdings"][i]
            summary = archive[name + "__summary"][i]
            daily_turnover = archive[name + "__turnover_by_day"][i]
            turnover = float(archive[name + "__turnover_equivalent"][i])
            require(len(returns) == len(dates) == len(holdings) == len(daily_turnover), "Daily axis mismatch")
            require(np.isfinite(daily_turnover).all() and (daily_turnover >= 0).all(), "Invalid daily turnover")
            require(math.isclose(float(daily_turnover.sum()), turnover, rel_tol=1e-12, abs_tol=1e-12), "Turnover total mismatch")
            if candidate["kind"] == "single":
                switches = np.r_[0., holdings[1:] != holdings[:-1]]
                require(np.array_equal(switches, daily_turnover), "Single turnover differs from actual holding changes")
                require(switches.sum() == summary[3], "Native switch count mismatch")
            saved_scenario = saved["scenarios"][name]
            require(turnover == saved_scenario["turnover_equivalent"], "Saved turnover scalar mismatch")
            for field, column in (("noninitial_rebalance_count", 3), ("blocked_rebalances", 5), ("crashes", 7)):
                require(saved_scenario[field] == int(summary[column]), "Saved count differs from archived summary")
            rebuilt = dict(blocks={}, yearly={}, turnover_equivalent=turnover)
            intervals = [("full", START, END), ("tail", END[:4] + "-01-01", END)]
            intervals += [("blocks/" + label, start, end) for label, start, end in BLOCKS]
            intervals += [("yearly/" + year, year + "-01-01", year + "-12-31") for year in sorted({d[:4] for d in dates})]
            for label, start, end in intervals:
                lo, hi = bisect_left(dates, start), bisect_right(dates, end)
                calculated = independent_metrics(returns[lo:hi])
                calculated.update(start=dates[lo], end=dates[hi - 1], turnover_equivalent=float(daily_turnover[lo:hi].sum()))
                if "/" in label:
                    group, key = label.split("/")
                    expected = saved_scenario[group][key]
                    rebuilt[group][key] = calculated
                else:
                    expected = saved_scenario[label]
                    rebuilt[label] = calculated
                for metric, value in calculated.items():
                    comparisons += 1
                    if isinstance(value, str) or metric == "sessions":
                        require(expected[metric] == value, "Saved interval/date mismatch")
                        continue
                    error = abs(float(expected[metric]) - value)
                    if error > largest_error["absolute"]:
                        largest_error = dict(absolute=error, metric=metric, id=cid, scenario=name, period=label)
                    require(math.isclose(float(expected[metric]), value, rel_tol=1e-11, abs_tol=5e-12),
                            "Metric differs: %s/%s/%s/%s" % (cid, name, label, metric))
            reconstructed[cid]["scenarios"][name] = rebuilt
    expected_checks = selected["checks"]
    own_checks = independent_checks(reconstructed, registry["controls"])
    require(own_checks == expected_checks, "Independent ten-gate outcome differs")
    controls = set(registry["controls"].values())
    eligible = [c["id"] for c in registry["candidates"] if c["selectable"] and c["kind"] != "benchmark" and c["id"] not in controls]
    qualified = sorted(cid for cid in eligible if all(own_checks[cid].values()))
    require(len(qualified) == selected["qualified_count"], "Qualified count differs")
    require(qualified == sorted(selected["qualified_ids"]), "Qualified identities differ")
    if not qualified:
        require(selected["primary"] is None and selected["pareto_ids"] == [], "No-qualified stage substituted a winner")
    for label, field in (("top_return", "cagr"), ("least_drawdown", "max_dd"), ("top_tail", "total_return")):
        period = "tail" if label == "top_tail" else "full"
        # The frozen selector breaks exact numerical ties by id. Report-only
        # scalar recomputation can differ at 1 ulp, so verify its advertised
        # leader is at the global maximum to metric precision, not reselect it.
        best_value = max(reconstructed[cid]["scenarios"]["close_1bp"][period][field] for cid in eligible)
        actual = reconstructed[selected[label]]["scenarios"]["close_1bp"][period][field]
        require(math.isclose(actual, best_value, rel_tol=1e-11, abs_tol=5e-12), "Exploratory leader is not maximal")
    return reconstructed, dict(rows=len(reconstructed), scenarios=len(SCENARIOS), metric_comparisons=comparisons,
                               largest_metric_absolute_difference=largest_error, metric_tolerance=dict(rtol=1e-11, atol=5e-12),
                               all_ten_gates_equal=True, qualified_count=len(qualified), qualified_ids=qualified,
                               primary=selected["primary"], no_exploratory_replacement=True), own_checks


def feature_view(stage, arrays, meta, histories):
    if stage == "main":
        return arrays, meta
    if stage == "consensus":
        from .consensus_features import build_consensus
        extended, metadata, unused = build_consensus(arrays, meta, histories)
        receipt = read(BASE / "results/consensus/score_receipt.json")
    elif stage == "exante":
        from .exante_features import build_exante
        extended, metadata, unused = build_exante(arrays, meta, histories)
        receipt = read(BASE / "results/exante/feature_receipt.json")
    else:
        raise ValueError("Unknown audit stage")
    require(metadata == receipt["metadata"], "Rebuilt stage feature metadata changed")
    for name, values in extended.items():
        current = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        require(current == receipt["array_sha256"][name], "Rebuilt stage feature values changed: " + name)
    return extended, metadata


def audit_reference(stage, registry, selected, arrays, meta, archive, directory):
    requested = [("top_return", selected["top_return"]), ("top_tail", selected["top_tail"])]
    if stage == "main":
        requested.append(("near_balanced", NEAR_BALANCED_A))
    if selected["primary"] is not None:
        requested.append(("primary", selected["primary"]))
    records = {c["id"]: c for c in registry["candidates"]}
    positions = {c["id"]: i for i, c in enumerate(registry["candidates"])}
    ids = list(dict.fromkeys(cid for unused_role, cid in requested))
    compared, output, traces = [], {}, {}
    for cid in ids:
        record = records[cid]
        require(record["kind"] == "single", "Requested reference target must be single-security")
        view = arrays
        config = record["config"]
        if stage == "consensus" and config["score"].startswith("v12_b_"):
            view = dict(arrays, features=arrays["features"].copy())
            lane = arrays["scores"][meta["score_names"].index(config["score"])]
            view["features"][:, :, meta["feature_names"].index("valid")] *= np.isfinite(lane) & (lane > -1e90)
        elif stage == "exante":
            from .exante_features import view_for_config
            view = view_for_config(arrays, meta, config)
        for name, lag, fee in (SCENARIOS[0], SCENARIOS[3]):
            result = run_reference(view, meta, dict(deepcopy(config), lag=lag), start=START, end=END, fee=fee)
            require(result["dates"][0] == START and result["dates"][-1] == END, "Reference interval differs")
            errors, exact = {}, {}
            for field in ("returns", "holdings", "summary"):
                previous = archive[name + "__" + field][positions[cid]]
                current = result[field]
                require(current.shape == previous.shape, "Reference shape mismatch")
                exact[field] = bool(np.array_equal(current, previous))
                errors[field] = float(np.max(np.abs(current - previous)))
                require(exact[field], "Reference is not EXACT: %s/%s/%s maximum %.17g" % (cid, name, field, errors[field]))
                output[cid + "__" + name + "__" + field] = current
            traces[cid + "__" + name] = result["trace"]
            compared.append(dict(id=cid, roles=[role for role, key in requested if key == cid], scenario=name,
                                 observations=len(result["dates"]), exact=exact, max_absolute_error=errors))
            print("reference exact", stage, cid, name, flush=True)
    np.savez_compressed(directory / "reference_paths.npz", **output)
    dump(directory / "reference_traces.json", traces)
    return compared


def audit(stage, arrays, meta, histories, output):
    directory = Path(output) / stage
    if directory.exists():
        raise ValueError("Audit output exists; refusing to overwrite: " + str(directory))
    folder, registry, selected, evaluation, path_metadata, receipts = verify_receipts(stage)
    print("verified receipts", stage, len(receipts), flush=True)
    # Load once: repeated npz access otherwise decompresses entire matrices.
    with np.load(folder / "paths.npz", allow_pickle=False) as source:
        archive = {name: source[name] for name in source.files}
    reconstructed, metric_audit, checks = audit_metrics(registry, selected, evaluation, path_metadata, archive)
    extended, metadata = feature_view(stage, arrays, meta, histories)
    directory.mkdir(parents=True)
    reference = audit_reference(stage, registry, selected, extended, metadata, archive, directory)
    details = {}
    focus = set(item["id"] for item in reference) | set(registry["controls"].values())
    for cid in sorted(focus):
        scenario = reconstructed[cid]["scenarios"]["close_1bp"]
        details[cid] = dict(full=scenario["full"], tail=scenario["tail"], blocks=scenario["blocks"],
                            yearly=scenario["yearly"], failed_gates=[key for key, okay in checks[cid].items() if not okay],
                            passed_gates=sum(checks[cid].values()))
    require(verify_receipts(stage)[-1] == receipts, "Frozen evidence changed during audit")
    source_paths = [Path(__file__), BASE / "reference.py", BASE.parent / "v10_deep/reference.py",
                    BASE / "consensus_features.py", BASE / "exante_features.py"]
    report = dict(stage=stage, audited_at=stamp(), metrics=metric_audit, reference_comparisons=reference,
                  focus=details, artifact_sha256=receipts,
                  audit_source_sha256={str(path.relative_to(BASE.parent)): sha(path) for path in source_paths},
                  output_sha256={name: sha(directory / name) for name in ("reference_paths.npz", "reference_traces.json")},
                  scope="All saved rows/metrics/ten gates; selected single-security paths independently replayed. Allocation units are not independently replayed here.",
                  known_history_including_2026=True, clean_oos=False, parameter_changes=False)
    dump(directory / "audit.json", report)
    lines = ["# V12 " + stage + " 冻结证据审计", "",
             "全部 %d 行 × 4 情景的全期、分块、逐年、尾段指标及十项门槛复算一致；合格 %d 项，primary=%s。" %
             (metric_audit["rows"], metric_audit["qualified_count"], selected["primary"]), "",
             "%d 条独立 Python reference 全路径与存档 returns/holdings/summary 逐值一致。" % len(reference), "",
             "|标识|年化|最大回撤|尾段收益|失败门槛|", "|---|---:|---:|---:|---|"]
    for cid in sorted(details):
        value = details[cid]
        lines.append("|%s|%.4f%%|%.4f%%|%.4f%%|%s|" % (cid, 100 * value["full"]["cagr"],
                     100 * value["full"]["max_dd"], 100 * value["tail"]["total_return"], ", ".join(value["failed_gates"])))
    lines += ["", "这是已知历史上的实现与存档审计，不能证明样本外优势或消除此前自适应搜索偏差。"]
    (directory / "AUDIT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=("main", "consensus", "exante"), default=["main", "consensus"])
    parser.add_argument("--output", type=Path, default=BASE / "results/audit_stages")
    args = parser.parse_args()
    arrays, meta, unused = load_inputs()
    from v10_deep.data import inputs
    histories, unused_calendar, unused_fear = inputs()
    for stage in args.stages:
        report = audit(stage, arrays, meta, histories, args.output)
        print(stage, report["metrics"], flush=True)


if __name__ == "__main__":
    main()
