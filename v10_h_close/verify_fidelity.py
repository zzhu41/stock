"""Fidelity only for the original-close iteration; no candidate return search."""
import bisect
from copy import deepcopy
import csv
import hashlib
import json
import math
from unittest.mock import patch

from v10_next.data import configure
from v10_next.frozen import strategy as frozen_strategy
from v10_next.frozen.metadata import STOCK_POOL, GLOBAL_POOL, CASH, GOLD
from v10_round2.v92 import QvixSeries
from v10_search.features import FeatureBank
from v10_search.verify_fidelity import check_features, independent_original_strategy, independent_wls
from .data import BASE, START, load_histories, load_fear, protect
from .engine import prepare, run
from .policy import ClosePolicy
from .registry import controls


def focused_score_checks(bank, histories, dates):
    """Verify the new WLS35 and 25/30 ensemble against independent raw prices."""
    original = independent_original_strategy()
    for key, value in bank.presets["v9.1"].items():
        if hasattr(original, key.upper()):
            setattr(original, key.upper(), deepcopy(value))
    count, maximum = 0, 0.0
    for date in dates:
        i = bisect.bisect_left(bank.calendar, date)
        scored = {windows: dict(bank.ranked("wls", windows)[i]) for windows in ((35,), (25, 30))}
        for code, data in bank.base[i].items():
            rows = [r for r in histories[code] if r[0] <= date]
            closes = [r[2] for r in rows]
            native = original.indicators(closes, [r[3] for r in rows])
            if native is None:
                raise AssertionError("Independent native warmup does not agree")
            for windows in ((35,), (25, 30)):
                expected = sum(independent_wls(closes, w, native["vol"]) for w in windows) / len(windows)
                actual = scored[windows][code]["score"]
                difference = abs(actual - expected)
                maximum = max(maximum, difference)
                if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                    raise AssertionError("Focused score mismatch: %s %s %r" % (date, code, windows))
                count += 1
    return dict(passed=True, score_checks=count, maximum_absolute_error=maximum,
                windows=[[35], [25, 30]], reference="Independent original indicators and raw-price weighted least squares")


def focused_ordinary_checks(bank, sample_dates):
    """Same-state original full decide comparisons; not performance evaluation."""
    checks = 0
    base = next(c for c in controls() if c["id"] == "c_v91")
    for windows in ((35,), (25, 30)):
        for buffer in (.01, .02, .03, .04, None):
            config = deepcopy(base)
            config["channels"] = []
            config["params"]["crash_mom5"] = 0.0
            config["score_windows"] = list(windows)
            if buffer is not None:
                config["params"]["buffer"] = buffer
                config["params"]["pool_buffer"] = {}
            policy = ClosePolicy(config, bank)
            for date in sample_dates:
                i = bisect.bisect_left(bank.calendar, date)
                table = bank.table(i, "wls", windows, policy.feature_pool, oracle=True)
                for holding in [None, CASH] + [code for code, _ in table if code in policy.trade_pool]:
                    target = policy(i, holding, 7, True)
                    configure(config["params"], config["stock_pool"], config["global_pool"])
                    expected, _ = frozen_strategy.decide(table, holding, 7)
                    if target != expected:
                        raise AssertionError("Native ordinary decision mismatch: %s %r buffer=%r holding=%r" %
                                             (date, windows, buffer, holding))
                    checks += 1
    return dict(passed=True, same_state_decision_checks=checks,
                uniform_buffers=[.01, .02, .03, .04], original_split_buffer=True,
                windows=[[35], [25, 30]])


def original_control_checks(bank, frame, fear, histories, end):
    import backtest
    import strategy
    import v10.lab as lab
    from strategy_versions import backtest_kwargs

    baseline = {code: histories[code] for code in list(STOCK_POOL) + list(GLOBAL_POOL) + [GOLD, CASH]}
    calendar = list(frame.dates)
    with (BASE / "snapshots/qvix50.csv").open(newline="") as f:
        qvix = QvixSeries([(r[0], float(r[1])) for r in csv.reader(f) if r])
    lab_fear = dict(qd=[], qz=[], qv=[], rd=[], r5=[])
    for d, value in qvix.rows:
        observation = qvix.state(d)
        if observation["available"]:
            lab_fear["qd"].append(d)
            lab_fear["qz"].append(observation["z"])
            lab_fear["qv"].append(value)
    original_rank, native_cache = strategy.rank, {}

    def native_rank(h, on_date=None, live_prices=None):
        lab.STATE["last_date"] = on_date
        if on_date not in native_cache:
            saved = lab.CFG.get("crash_volu")
            lab.CFG["crash_volu"] = 2.0
            try:
                native_cache[on_date] = original_rank(h, on_date=on_date)
            finally:
                if saved is None:
                    lab.CFG.pop("crash_volu", None)
                else:
                    lab.CFG["crash_volu"] = saved
        return [(code, dict(ind)) for code, ind in native_cache[on_date]]

    results = {}
    with patch.object(strategy, "rank", side_effect=native_rank), \
            patch.object(strategy, "STOCK_POOL", list(STOCK_POOL)), \
            patch.object(strategy, "GLOBAL_POOL", list(GLOBAL_POOL)), \
            patch.object(backtest, "FEE", .0001), patch.dict(lab.FEAR, lab_fear, clear=True):
        for config in controls():
            name = config["id"]
            cfg = (dict(fear_qz=2.5, fear_m5=-.04, crash_volu=2.0, cv_m5=-.04, cv_dep=.10)
                   if name == "c_v92" else {})
            with patch.dict(lab.CFG, cfg, clear=True):
                if name == "c_v92":
                    expected = lab.backtest_v10(baseline, calendar, start=START, end=end)
                else:
                    expected = backtest.backtest(baseline, calendar, start=START, end=end,
                                                 **backtest_kwargs("v9" if name == "c_v9" else "v9.1"))
                policy = ClosePolicy(config, bank, frame=frame, fear=fear, trace=True)
                actual = run(frame, policy, START, end, capture_daily=True, capture_trades=True)
                if actual["daily"] != expected["daily"]:
                    first = next((a, b) for a, b in zip(actual["daily"], expected["daily"]) if a != b)
                    raise AssertionError("Original close path mismatch %s: %r" % (name, first))
                if actual["trades"] != expected["trades"] or policy.metadata["crash_buys"] != expected["crash_buys"]:
                    raise AssertionError("Original close trade/crash mismatch: " + name)
                for key in ("nav", "ann", "max_dd", "switches"):
                    if actual[key] != expected[key]:
                        raise AssertionError("Original close scalar mismatch %s %s" % (name, key))
                results[name] = dict(passed=True, exact_daily_and_trades=True, sessions=actual["sessions"],
                                     maximum_absolute_nav_error=0.0, nav=actual["nav"], ann=actual["ann"],
                                     max_dd=actual["max_dd"], switches=actual["switches"],
                                     crash_buys=policy.metadata["crash_buys"],
                                     blocked_switch_days=actual["diagnostics"]["blocked_switch_days"],
                                     daily_sha256=hashlib.sha256(json.dumps(actual["daily"], separators=(",", ":")).encode()).hexdigest())
                print("Original exact close-control passed: %s (%d sessions, %.4f%% CAGR)" %
                      (name, actual["sessions"], actual["ann"] * 100), flush=True)
    return results


def verify():
    before = protect()
    files = ("engine.py", "policy.py", "data.py", "registry.py", "verify_fidelity.py")
    sources = {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in files}
    histories, end = load_histories()
    calendar = [r[0] for r in histories["510300"]]
    frame = prepare(histories, calendar)
    fear = load_fear(calendar)
    print("Building feature bank for fidelity only; common cutoff " + end, flush=True)
    bank = FeatureBank(histories, calendar)
    sample_dates = {d for i, d in enumerate(calendar) if d >= START and i % 97 == 0}
    sample_dates.update(d for d in ("2014-01-02", "2015-04-13", "2015-04-14",
                                    "2022-03-15", "2024-09-30", end) if d in calendar)
    sample_dates = sorted(sample_dates)
    original_features = check_features(bank, histories, sample_dates)
    new_scores = focused_score_checks(bank, histories, sample_dates)
    ordinary = focused_ordinary_checks(bank, sample_dates)
    original_controls = original_control_checks(bank, frame, fear, histories, end)
    after = protect()
    if any(hashlib.sha256((BASE / name).read_bytes()).hexdigest() != digest for name, digest in sources.items()):
        raise AssertionError("Fidelity source changed during the run")
    return dict(passed=True, start=START, end=end, clock="original_same_close",
                source_sha256=sources,
                snapshot_manifest_sha256=hashlib.sha256((BASE / "snapshot_manifest.json").read_bytes()).hexdigest(),
                qvix_manifest_sha256=hashlib.sha256((BASE / "qvix_manifest.json").read_bytes()).hexdigest(),
                original_features=original_features, focused_scores=new_scores,
                focused_ordinary_decisions=ordinary, controls=original_controls,
                protected_files_before=before, protected_files_after=after,
                scope="Fidelity only. No candidate return screening. Same-close full-day information and same-price execution remain idealized.")


if __name__ == "__main__":
    report = verify()
    output = BASE / "results/fidelity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Written " + str(output), flush=True)
