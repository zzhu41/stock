"""Offline integration fidelity: closing targets, not close-vs-open NAV parity.

Run from the repository root with python3 -m v10_next.verify_legacy. The existing
engine is imported read-only and receives the isolated frozen price snapshot.
An independent native-indicator cache accelerates its two fixed presets.
"""
import ast
import hashlib
import json
from pathlib import Path

from .candidates import build_policy
from .data import BASE, Features, configure, load_histories, verify_protected
from .execution import run
from .frozen import strategy as frozen_strategy
from .legacy import factory


START, END = "2014-01-01", "2021-12-31"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_fidelity():
    """Comments and import location may change; every strategy statement must match."""
    def logic(path):
        module = ast.parse(path.read_text(encoding="utf-8"))
        module.body = [node for i, node in enumerate(module.body)
                       if not isinstance(node, (ast.Import, ast.ImportFrom))
                       and not (i == 0 and isinstance(node, ast.Expr)
                                and isinstance(node.value, (ast.Str, ast.Constant))
                                and isinstance(getattr(node.value, "value", getattr(node.value, "s", None)), str))]
        return ast.dump(module, include_attributes=False)
    return logic(BASE.parent / "strategy.py") == logic(BASE / "frozen/strategy.py")


def verify():
    import backtest as original_backtest
    import strategy as original_strategy
    from strategy_versions import BASE_CODES, GLOBAL_CODES, STOCK_CODES, backtest_kwargs

    protected_before = verify_protected()
    ast_equal = source_fidelity()
    if not ast_equal:
        raise AssertionError("Frozen strategy logic differs from original after excluding imports/comments")
    histories = load_histories()
    baseline_histories = {code: histories[code] for code in BASE_CODES}
    calendar = [r[0] for r in histories["510300"]]
    features = Features(histories)
    original_strategy.STOCK_POOL = list(STOCK_CODES)
    original_strategy.GLOBAL_POOL = list(GLOBAL_CODES)
    native_rank, native_decide, native_cache = original_strategy.rank, original_strategy.decide, {}
    active = {"date": None, "decisions": {}, "table_mismatches": [], "rule_mismatches": []}

    def cached_native_rank(h, on_date=None, live_prices=None):
        if h is not baseline_histories or on_date is None or live_prices is not None:
            raise ValueError("Fidelity cache requires its fixed local dated snapshot")
        active["date"] = on_date
        if on_date not in native_cache:
            native_cache[on_date] = native_rank(h, on_date=on_date)
        return [(code, dict(ind)) for code, ind in native_cache[on_date]]

    def compared_native_decide(table, holding, holding_days=99):
        answer = native_decide(table, holding, holding_days)
        date = active["date"]
        isolated_table = features.table(date, list(STOCK_CODES) + list(GLOBAL_CODES) + ["518880"])
        if table != isolated_table:
            active["table_mismatches"].append(date)
        isolated_answer = frozen_strategy.decide(isolated_table, holding, holding_days)
        if answer != isolated_answer:
            active["rule_mismatches"].append(dict(date=date, holding=holding, age=holding_days,
                                                  original=answer, isolated=isolated_answer))
        active["decisions"][date] = dict(holding=holding, age=holding_days,
                                          target=answer[0], reason=answer[1])
        return answer

    results, clean_results = {}, {}
    original_strategy.rank = cached_native_rank
    original_strategy.decide = compared_native_decide
    try:
        cases = [(candidate_id, version, start)
                 for start in (START, "2016-01-01")
                 for candidate_id, version in (("control_v9", "v9"), ("control_v91", "v9.1"))]
        for candidate_id, version, start in cases:
            print("Checking %s / %s against original %s ..." % (candidate_id, start, version), flush=True)
            configure(features.presets[version], STOCK_CODES, GLOBAL_CODES)
            frozen_strategy._state_bull = None
            frozen_strategy.BULL_HYST_PENDING = None
            active.update(date=None, decisions={}, table_mismatches=[], rule_mismatches=[])
            old = original_backtest.backtest(baseline_histories, calendar,
                                             start=start, end=END, **backtest_kwargs(version))
            policy = build_policy(candidate_id, factory(features, calendar), features)
            executed = run(histories, calendar, policy, start, END, fee=.0001, slippage=.001)
            expected = [(date, holding) for date, _, holding in old["daily"]]
            actual = [(row["date"], row["target"]) for row in policy.metadata["trace"]]
            if [d for d, _ in expected] != [d for d, _ in actual]:
                raise AssertionError("Different decision calendars in %s" % candidate_id)
            mismatches = []
            for i, (want, got) in enumerate(zip(expected, actual)):
                if want == got:
                    continue
                mismatches.append(dict(date=want[0], old_target=want[1], new_target=got[1],
                                       new_reason=policy.metadata["trace"][i]["reason"],
                                       new_actual_weights=executed["daily"][i][2],
                                       original_raw_decision=active["decisions"][want[0]],
                                       previous_old_target=expected[i - 1][1] if i else None))
            first_context = []
            if mismatches:
                first_index = next(i for i, row in enumerate(expected) if row[0] == mismatches[0]["date"])
                first_context = [dict(date=expected[i][0], old_target=expected[i][1],
                                      new_target=actual[i][1], reason=policy.metadata["trace"][i]["reason"],
                                      new_actual_weights=executed["daily"][i][2])
                                 for i in range(max(0, first_index - 3), min(len(expected), first_index + 4))]
            summary = dict(
                original_version=version, start=start, end=END, sessions=len(expected), targets_equal=not mismatches,
                mismatch_count=len(mismatches), mismatches=mismatches, first_mismatch_context=first_context,
                indicator_tables_equal=not active["table_mismatches"],
                indicator_table_mismatch_dates=active["table_mismatches"],
                same_state_raw_decisions_equal=not active["rule_mismatches"],
                same_state_raw_decision_mismatches=active["rule_mismatches"],
                original_target_sha256=_digest(expected), isolated_target_sha256=_digest(actual),
                original_crash_signals=old["crash_buys"], isolated_crash_entries=policy.metadata["crash_events"],
                isolated_deferred_rebalances=executed["diagnostics"]["deferred_rebalances"],
                isolated_trade_count=len(executed["trades"]),
            )
            (results if start == START else clean_results)[candidate_id] = summary
            print("%s: %d sessions, %d target mismatches" % (candidate_id, len(expected), len(mismatches)), flush=True)
            if mismatches:
                print(json.dumps(first_context, ensure_ascii=False), flush=True)
    finally:
        original_strategy.rank = native_rank
        original_strategy.decide = native_decide
    protected_after = verify_protected()
    gap_dates = ("2015-04-13", "2015-04-14")
    expected_difference_dates = list(gap_dates) + ["2015-04-15", "2015-04-16"]
    bars = {code: {r[0]: r for r in rows} for code, rows in histories.items()}
    gap_evidence = [dict(date=date, held_510500_bar=bars["510500"].get(date),
                         desired_510300_bar=bars["510300"].get(date))
                    for date in expected_difference_dates]
    comparable_equal = all(r["targets_equal"] for r in clean_results.values())
    explained = all([m["date"] for m in r["mismatches"]] == expected_difference_dates
                    and all(m["old_target"] == "510500" and m["new_target"] == "510300"
                            for m in r["mismatches"]) for r in results.values())
    feature_rule_equal = all(r["indicator_tables_equal"] and r["same_state_raw_decisions_equal"]
                             for r in list(results.values()) + list(clean_results.values()))
    source_paths = ("backtest.py", "strategy.py", "strategy_versions.py", "v10_next/data.py",
                    "v10_next/legacy.py", "v10_next/execution.py", "v10_next/candidates.py",
                    "v10_next/frozen/strategy.py", "v10_next/frozen/presets.json", "v10_next/verify_legacy.py")
    return dict(
        start=START, end=END, all_targets_equal=all(r["targets_equal"] for r in results.values()),
        frozen_strategy_ast_equal_except_imports_and_module_docstring=ast_equal,
        same_state_features_and_decisions_equal=feature_rule_equal,
        clean_period_all_targets_equal=comparable_equal,
        all_full_period_differences_match_documented_missing_bar_episode=explained,
        integration_checks_passed=ast_equal and feature_rule_equal and comparable_equal and explained,
        protected_files_verified_before=protected_before, protected_files_verified_after=protected_after,
        source_sha256={name: hashlib.sha256((BASE.parent / name).read_bytes()).hexdigest() for name in source_paths},
        snapshot_manifest_sha256=hashlib.sha256((BASE / "data_manifest.json").read_bytes()).hexdigest(),
        validation_scope=[
            "Compare every closing decision target between the isolated actual-fill policy and the original fixed v9/v9.1 presets.",
            "Price indicators are calculated independently by original strategy.rank and isolated Features; their caches do not share values.",
            "No return parity is expected: the original buys at same close, the isolated executor at the next open with explicit fills and costs.",
            "No source parameters or registered candidate rules are changed to obtain fidelity.",
            "The protected existing worktree is fingerprint-checked before and after the comparison.",
            "Full 2014-2021 signal-path equality is intentionally NOT claimed: the missing-bar episode changes actual execution state.",
        ], results=results, clean_period_results=clean_results,
        explained_difference=dict(
            missing_bar_asset="510500", missing_bar_dates=list(gap_dates), daily_evidence=gap_evidence,
            cause="Original backtest.daily stores same-close executed holding after the missing-bar guard, not the unfilled desired instruction.",
            original_behavior="2015-04-13/14 raw decision requests 510300 but missing 510500 bars force holding 510500; when data resumes, the strategy recomputes from that holding.",
            isolated_behavior="2015-04-13 close queues 510300; 04-14 open atomically defers; 04-15 open fills on resumption. The changed actual holding affects buffer decisions through 04-16.",
            interpretation="Execution timing and actual-state feedback deliberately differ. Do not remove deferred execution or change registered rules to force historical holding parity.",
        ),
    )


if __name__ == "__main__":
    report = verify()
    output = BASE / "results/legacy_fidelity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Written " + str(output), flush=True)
    if not report["integration_checks_passed"]:
        raise SystemExit("Unexplained integration difference; inspect evidence before candidate evaluation")
