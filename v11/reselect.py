"""Protocol-fixed historical A/B reselection with continuous actual accounts.

A training windows always use the original candidate family. B components are
regenerated from each fold's A training results, never copied from the 2021 B
selection. Real execution is explicitly enabled at the CLI and ends in 2025.
"""
import argparse
from bisect import bisect_left, bisect_right
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .data import BASE, ROOT, START, dump, sha, protect
from .features import build, risk_view
from .combinations import generate as generate_combinations
from .scan import SCENARIOS, run_candidates, summarize_rows, select, config_for_scenario, stamp
from . import walkforward
from v10_deep.reference import DatedValues
from v10_deep.scan import metrics


OUT = BASE / "results/reselection"
END = "2025-12-31"
REPORT_START = "2018-01-01"
FOLDS = tuple(walkforward.training_folds())
SOURCE_NAMES = ("v11/reselect.py", "v11/tests/test_reselect.py", "v11/RESELECTION_PROTOCOL.md",
                "v11/PROTOCOL.md", "v11/COMBINATION_PROTOCOL.md", "v11/data.py", "v11/registry.py",
                "v11/features.py", "v11/scan.py", "v11/combinations.py", "v11/walkforward.py",
                "v10_deep/native.py", "v10_deep/native.cpp", "v10_deep/schema.py",
                "v10_deep/reference.py", "v10_deep/scan.py")


def training_blocks(end):
    blocks = [("early", START, min("2017-12-31", end))]
    if end >= "2018-01-01":
        blocks.append(("late", "2018-01-01", end))
    return tuple(blocks)


def truncate(arrays, meta, end):
    """Only the dated prefix is exposed to each training/execution engine."""
    stop = bisect_right(meta["dates"], end)
    if stop == 0:
        raise ValueError("No observations before cutoff")
    result = {name: np.ascontiguousarray(values[:, :stop] if name in ("scores", "orders") else values[:stop])
              for name, values in arrays.items()}
    metadata = deepcopy(meta)
    metadata["dates"] = meta["dates"][:stop]
    if "shape" in metadata:
        metadata["shape"][0] = stop
    return result, metadata


def load_registry():
    path = BASE / "registered_candidates.json"
    definition = json.loads((BASE / "registration.json").read_text())
    if sha(path) != definition["registry_sha256"] or definition["registry_count"] != 586:
        raise ValueError("The original A 586-candidate registration changed")
    registry = json.loads(path.read_text())
    if len(registry["candidates"]) != 586 or registry["unique_count"] != 586:
        raise ValueError("Historical A training must use exactly the original 586 candidates")
    # These receipts pin algorithms only. No old B component lists, selection
    # scores, confirmation results or 2026 outcomes are loaded by this driver.
    for directory in (BASE / "results/development", BASE / "results/combinations"):
        source = json.loads((directory / "registration.json").read_text())["sources"]
        for name, expected in source.items():
            if sha(BASE / name) != expected:
                raise ValueError("Frozen training algorithm changed: " + name)
    return registry


def fingerprints():
    return dict(sources={name: sha(ROOT / name) for name in SOURCE_NAMES},
                original_A_registry_sha256=sha(BASE / "registered_candidates.json"),
                original_A_registration_sha256=sha(BASE / "registration.json"),
                A_algorithm_receipt_sha256=sha(BASE / "results/development/registration.json"),
                B_algorithm_receipt_sha256=sha(BASE / "results/combinations/registration.json"),
                corrected_manifest_sha256=sha(ROOT / "v10_h_close/corrected_manifest.json"),
                qvix_manifest_sha256=sha(ROOT / "v10_h_close/qvix_manifest.json"),
                frozen_profiles_sha256=sha(ROOT / "v10_deep/profiles.json"))


def _same_or_create(path, value):
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError("Refusing to overwrite registered reselection inputs: " + str(path))
    else:
        dump(path, value)


def register():
    registry = load_registry()
    expected = fingerprints()
    path = OUT / "registration.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["fingerprints"] != expected:
            raise ValueError("Historical reselection sources/inputs changed after registration")
        return registry, previous
    protect()
    value = dict(registered_at=stamp(), fingerprints=expected, folds=FOLDS, scenarios=SCENARIOS,
        original_A_count=586, B_components="Regenerate from that fold's complete A training rows/checks/order",
        selection_rule="Original selector across all four training scenarios; no eligible candidate -> frozen H",
        training_blocks="2014 to min(2017, cutoff), and 2018 to cutoff when present",
        execution_start=START, execution_end=END, reporting_period=[REPORT_START, END],
        common_initialization="Frozen H from 2014 through 2017; actual account persists after each replacement",
        chosen_configuration_shared_across_pressure_scenarios=True, clean_oos=False,
        limitations=["Candidate family, H and asset pool inherit knowledge of later history.",
                     "Historical reselection is a process diagnostic, not clean out-of-sample or live trading.",
                     "Same-close fills, first overall session free, fixed linear double-sided switch costs and reconstructed TR inputs remain assumptions."])
    dump(path, value)
    return registry, value


def evaluate_training(registry, arrays, meta, cutoff, out, provenance, evaluator=run_candidates):
    """Register one fold's candidate set before evaluating it; return rows/choice."""
    out = Path(out)
    _same_or_create(out / "registered_candidates.json", registry)
    source_hash = sha(out / "registered_candidates.json")
    registration_path = out / "registration.json"
    definition = dict(registry_sha256=source_hash, period=[START, cutoff], blocks=training_blocks(cutoff),
                      scenarios=SCENARIOS, candidate_count=len(registry["candidates"]), provenance=provenance)
    if registration_path.exists():
        saved = json.loads(registration_path.read_text())
        if saved["definition"] != json.loads(json.dumps(definition)):
            raise ValueError("Fold registration changed")
    else:
        dump(registration_path, dict(registered_at=stamp(), definition=definition))
    selection_path = out / "selection.json"
    if selection_path.exists():
        choice = json.loads(selection_path.read_text())
        for name, key in (("registration.json", "registration_sha256"), ("evaluation.json", "evaluation_sha256"),
                          ("paths.npz", "paths_sha256")):
            if sha(out / name) != choice["provenance"][key]:
                raise ValueError("Frozen fold artifact changed: " + name)
        if source_hash != choice["provenance"]["registry_sha256"]:
            raise ValueError("Frozen fold candidate set changed")
        rows = json.loads((out / "evaluation.json").read_text())["rows"]
        return dict(registry=registry, rows=rows, selection=choice, directory=out)

    prefix, prefix_meta = truncate(arrays, meta, cutoff)
    results, dates = evaluator(registry["candidates"], prefix, prefix_meta, start=START, end=cutoff,
                               scenarios=SCENARIOS, workers=1)
    if not dates or dates[-1] > cutoff or dates != [d for d in prefix_meta["dates"] if START <= d <= cutoff]:
        raise AssertionError("Training evaluator crossed its cutoff or changed its observation axis")
    rows = summarize_rows(registry["candidates"], results, dates, blocks=training_blocks(cutoff))
    np.savez_compressed(out / "paths.npz", **{name + "__" + field: value for name, payload in results.items()
                                              for field, value in payload.items()})
    dump(out / "path_metadata.json", dict(dates=dates, ids=[c["id"] for c in registry["candidates"]],
                                          sha256=sha(out / "paths.npz"), actual_training_end=dates[-1]))
    dump(out / "evaluation.json", dict(period=[dates[0], dates[-1]], training_cutoff=cutoff, rows=rows))
    choice = select(rows, registry["candidates"], registry["controls"])
    # The frozen selector reports its initial 2021 development label. Its
    # formulas use supplied rows; update only receipt metadata for this fold.
    choice["selection_period"] = [START, cutoff]
    choice["status"] = "historical_training_fold_only_not_live_approval"
    choice["chosen_id"] = choice["primary"] or registry["controls"]["h"]
    choice["used_H_fallback"] = choice["primary"] is None
    choice["provenance"] = dict(registration_sha256=sha(registration_path), registry_sha256=source_hash,
        evaluation_sha256=sha(out / "evaluation.json"), paths_sha256=sha(out / "paths.npz"),
        training_cutoff=cutoff, source=provenance)
    dump(selection_path, choice)
    return dict(registry=registry, rows=rows, selection=choice, directory=out)


def train_fold(original, arrays, meta, fold, out, provenance, evaluator=run_candidates):
    """B registration depends on this fold's A outputs, never the 2021 B menu."""
    out = Path(out)
    cutoff = fold["train_end"]
    a = evaluate_training(original, arrays, meta, cutoff, out / "A",
                          dict(provenance, strategy_family="original_A", training_cutoff=cutoff), evaluator)
    b_registry = generate_combinations(original, a["rows"], a["selection"])
    if b_registry["raw_count"] > 150:
        raise AssertionError("Fold B component cross exceeded the registered maximum")
    b_provenance = dict(provenance, strategy_family="fold_specific_B_component_cross", training_cutoff=cutoff,
        A_training_selection_sha256=sha(a["directory"] / "selection.json"),
        A_training_evaluation_sha256=sha(a["directory"] / "evaluation.json"),
        A_training_registry_sha256=sha(a["directory"] / "registered_candidates.json"))
    b = evaluate_training(b_registry, arrays, meta, cutoff, out / "B", b_provenance, evaluator)
    # Grouping/order changes in the fold-specific B cross must not change H.
    a_index = next(i for i, c in enumerate(original["candidates"]) if c["id"] == original["controls"]["h"])
    b_index = next(i for i, c in enumerate(b_registry["candidates"]) if c["id"] == original["controls"]["h"])
    with np.load(a["directory"] / "paths.npz", allow_pickle=False) as a_paths, \
            np.load(b["directory"] / "paths.npz", allow_pickle=False) as b_paths:
        for scenario, unused_lag, unused_fee in SCENARIOS:
            for field in ("returns", "holdings", "summary"):
                key = scenario + "__" + field
                if not np.array_equal(a_paths[key][a_index], b_paths[key][b_index]):
                    raise AssertionError("Fold-specific B changed the H control: " + key)
    return dict(fold=deepcopy(fold), A=a, B=b, H_control_training_parity=True)


def chosen_config(training):
    cid = training["selection"]["chosen_id"]
    return deepcopy(next(c for c in training["registry"]["candidates"] if c["id"] == cid))


def schedules(original, trained):
    h = deepcopy(next(c for c in original["candidates"] if c["id"] == original["controls"]["h"]))
    initial = dict(start=START, end="2017-12-31", config=h, risk_context=h["risk_context"])
    result = {label: [deepcopy(initial)] for label in ("A", "B")}
    for item in trained:
        fold = item["fold"]
        for label in ("A", "B"):
            config = chosen_config(item[label])
            result[label].append(dict(start=fold["start"], end=fold["end"], config=config,
                                      risk_context=config["risk_context"], training_cutoff=fold["train_end"]))
    result["H"] = [dict(start=START, end=END, config=h, risk_context=h["risk_context"])]
    return result


def period_summary(result, start, end):
    dates = result["dates"]
    lo, hi = bisect_left(dates, start), bisect_right(dates, end)
    if hi <= lo:
        raise ValueError("Empty continuous-account report interval")
    value = metrics(result["returns"][lo:hi])
    navs = np.asarray([row[1] for row in result["daily"]])
    continuing_dd = navs / np.maximum.accumulate(np.r_[1., navs])[1:] - 1.
    value.update(requested_period=[start, end], date_range=[dates[lo], dates[hi - 1]],
                 prior_nav=float(navs[lo - 1]) if lo else 1., end_nav=float(navs[hi - 1]),
                 switches=sum(start <= t["date"] <= end and not t["first_session_free"] for t in result["trades"]),
                 model_changes=sum(start <= t["date"] <= end and t["model_changed"] for t in result["trace"]),
                 continuing_account_max_dd=float(np.min(continuing_dd[lo:hi])))
    return value


def execute_schedules(definitions, arrays, meta, out, factory=None):
    """Run three actual accounts independently under each common pressure case."""
    arrays, meta = truncate(arrays, meta, END)
    out = Path(out)
    if factory is None:
        def factory(values, metadata, config, context):
            return DatedValues(risk_view(values, metadata, context, score=config["score"]), metadata, config)
    periods = {"report_2018_2025": (REPORT_START, END), "full_2014_2025": (START, END)}
    periods.update({"fold_%s_%s" % (f["start"][:4], f["end"][:4]): (f["start"], f["end"]) for f in FOLDS})
    summaries, raw = {}, {}
    for label in ("A", "B", "H"):
        summaries[label], raw[label] = {}, {}
        for scenario, lag, fee in SCENARIOS:
            schedule = deepcopy(definitions[label])
            for segment in schedule:
                segment["config"] = config_for_scenario(segment["config"], lag)
            result = walkforward.run(schedule, arrays, meta, start=START, end=END, fee=fee, view_factory=factory)
            if result["dates"][-1] > END:
                raise AssertionError("Execution leaked into 2026")
            directory = out / label / scenario
            _same_or_create(directory / "schedule.json", schedule)
            walkforward.write_trace_csv(directory / "trace.csv", result)
            dump(directory / "transactions.json", dict(trades=result["trades"], boundaries=result["boundaries"],
                                                         final_account=result["final_account"], metadata=result["metadata"]))
            np.savez_compressed(directory / "paths.npz", returns=result["returns"], holdings=result["holdings"],
                                navs=np.asarray([row[1] for row in result["daily"]]), summary=result["summary"])
            dump(directory / "path_metadata.json", dict(dates=result["dates"], sha256=sha(directory / "paths.npz")))
            summaries[label][scenario] = dict(periods={name: period_summary(result, lo, hi) for name, (lo, hi) in periods.items()},
                boundaries=result["boundaries"], trades=result["trades"], final_account=result["final_account"],
                trace_path=str(directory / "trace.csv"), trace_sha256=sha(directory / "trace.csv"),
                paths_path=str(directory / "paths.npz"), paths_sha256=sha(directory / "paths.npz"),
                schedule_sha256=sha(directory / "schedule.json"), metadata=result["metadata"])
            raw[label][scenario] = result
            print("continuous", label, scenario, "complete", flush=True)
    return summaries, raw


def _fold_public(item, streams):
    fold = item["fold"]
    value = dict(train_end=fold["train_end"], execution_period=[fold["start"], fold["end"]],
                 B_components=deepcopy(item["B"]["registry"]["components"]),
                 H_control_training_parity=item["H_control_training_parity"])
    for label in ("A", "B"):
        training, selection = item[label], item[label]["selection"]
        value[label] = dict(chosen_id=selection["chosen_id"], primary=selection["primary"],
            used_H_fallback=selection["used_H_fallback"], qualified_count=selection["qualified_count"],
            candidate_count=len(training["registry"]["candidates"]),
            selection_path=str(training["directory"] / "selection.json"),
            selection_sha256=sha(training["directory"] / "selection.json"))
    key = "fold_%s_%s" % (fold["start"][:4], fold["end"][:4])
    value["execution"] = {label: {scenario: streams[label][scenario]["periods"][key] for scenario, _, _ in SCENARIOS}
                          for label in ("A", "B", "H")}
    return value


def _verify_H_prefixes(trained, continuous):
    cases = []
    for item in trained:
        training = item["A"]
        cid = training["registry"]["controls"]["h"]
        index = next(i for i, c in enumerate(training["registry"]["candidates"]) if c["id"] == cid)
        with np.load(training["directory"] / "paths.npz", allow_pickle=False) as saved:
            for scenario, unused_lag, unused_fee in SCENARIOS:
                reference = continuous["H"][scenario]
                stop = bisect_right(reference["dates"], item["fold"]["train_end"])
                for field in ("returns", "holdings"):
                    if not np.array_equal(reference[field][:stop], saved[scenario + "__" + field][index]):
                        raise AssertionError("Continuous H initialization/training prefix changed")
                cases.append(dict(train_end=item["fold"]["train_end"], scenario=scenario, exact_returns_and_holdings=True))
    return dict(passed=True, cases=cases)


def run(allow_real_post_2021=False):
    if not allow_real_post_2021:
        raise RuntimeError("Real post-2021 historical evaluation requires explicit authorization/--allow-real-post-2021")
    if (OUT / "evaluation.json").exists():
        raise ValueError("Historical reselection is already complete; do not overwrite")
    original, registration = register()
    arrays, meta = build(original["score_specs"])
    arrays, meta = truncate(arrays, meta, END)
    dump(OUT / "feature_receipt.json", dict(prepared_at=stamp(), registration_sha256=sha(OUT / "registration.json"),
        feature_fingerprints=meta["fingerprints"], cache_array_sha256=meta["cache_array_sha256"],
        execution_data_end=meta["dates"][-1]))
    trained = []
    for fold in FOLDS:
        if fingerprints() != registration["fingerprints"]:
            raise ValueError("Reselection definition changed after registration")
        item = train_fold(original, arrays, meta, fold, OUT / "folds" / fold["train_end"][:4],
                          dict(global_registration_sha256=sha(OUT / "registration.json")))
        trained.append(item)
        print("TRAIN FROZEN", fold["train_end"], "A", item["A"]["selection"]["chosen_id"],
              "B", item["B"]["selection"]["chosen_id"], flush=True)
    definitions = schedules(original, trained)
    _same_or_create(OUT / "schedules.json", definitions)
    dump(OUT / "execution_registration.json", dict(registered_at=stamp(), schedules_sha256=sha(OUT / "schedules.json"),
        global_registration_sha256=sha(OUT / "registration.json"), scenarios=SCENARIOS, period=[START, END],
        fold_selection_sha256={fold["fold"]["train_end"] + "/" + label: sha(fold[label]["directory"] / "selection.json")
                               for fold in trained for label in ("A", "B")}, chosen_once_per_fold=True))
    streams, continuous = execute_schedules(definitions, arrays, meta, OUT / "continuous")
    h_audit = _verify_H_prefixes(trained, continuous)
    if fingerprints() != registration["fingerprints"]:
        raise ValueError("Reselection definition changed during execution")
    evaluation = dict(completed_at=stamp(), period=[REPORT_START, END], initialization=[START, "2017-12-31"],
        folds=[_fold_public(item, streams) for item in trained], streams=streams,
        H_prefix_fidelity=h_audit, global_registration_sha256=sha(OUT / "registration.json"),
        execution_registration_sha256=sha(OUT / "execution_registration.json"),
        feature_receipt_sha256=sha(OUT / "feature_receipt.json"), fingerprints=registration["fingerprints"],
        protected_files_verified=protect(), clean_oos=False, no_2026_used=True,
        metric_convention="Period returns include the first boundary day's old-position P&L and actual switch fee; max_dd resets its measuring peak at the prior NAV, with full continuing-account drawdown also reported",
        no_return_stream_splicing=True, selection_shared_across_scenarios=True)
    dump(OUT / "evaluation.json", evaluation)
    lines = ["# 连续账户历史重选", "", "每折重新用当时训练数据选择A，并从当折A结果生成B组件；未使用2021组件回填过去。",
             "所有账户从2014以H开始，2018起按各训练结果切换；下表为2018–2025连续执行段，年界收益和实际换仓费用均保留。", "",
             "| 场景 | A年化/回撤 | B年化/回撤 | H年化/回撤 |", "|---|---:|---:|---:|"]
    for scenario, unused_lag, unused_fee in SCENARIOS:
        cells = ["%+.3f%% / %.3f%%" % (100 * streams[label][scenario]["periods"]["report_2018_2025"]["cagr"],
                                       100 * streams[label][scenario]["periods"]["report_2018_2025"]["max_dd"])
                 for label in ("A", "B", "H")]
        lines.append("| %s | %s |" % (scenario, " | ".join(cells)))
    lines += ["", "候选族、H和原资产池仍来自已知历史研究。此为历史过程诊断，不是干净样本外证据或真实交易记录；不更改已冻结A/B参数。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    return evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run"))
    parser.add_argument("--allow-real-post-2021", action="store_true")
    args = parser.parse_args()
    register() if args.command == "register" else run(allow_real_post_2021=args.allow_real_post_2021)
