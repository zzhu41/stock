"""Independent fidelity gates only; never rank or screen the search candidates.

Checks frozen original price indicators, 18 predetermined decision variants,
and all three full-period controls against immutable prior actual-fill paths.
"""
import bisect
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from v10_round2.v92 import load_qvix
from .data import ASSET_ORDER, BASE, END, START, load_histories, protect
from .fast_execution import prepare, run
from .features import DYNAMIC, FeatureBank
from .policy import SearchPolicy
from .registry import BASE_STOCK, CORE_STOCK, GLOBAL_SUBSETS, _candidate, controls


SHORT_WINDOWS = (("2015-03-02", "2016-02-29"),
                 ("2022-01-04", "2022-06-30"),
                 ("2024-08-01", "2024-12-31"))


def fixed_cases(presets):
    """A fixed coverage suite, not an additional candidate-return selection."""
    base, glob = presets["v9.1"], GLOBAL_SUBSETS[-1]
    cases = controls()

    def add(name, stock=BASE_STOCK, global_=glob, **kwargs):
        item = _candidate(base, "fidelity_only", stock, global_, **kwargs)
        item["id"] = name
        cases.append(item)

    for window in (20, 30, 40):
        add("fixed_wls%d" % window, window=window)
    for mode in ("mom20_vol", "mom60_vol", "mom20"):
        add("fixed_" + mode, mode=mode)
    add("fixed_benchmark_not_tradable", stock=("159915", "510500"), global_=("513100",))
    add("fixed_sector", stock=CORE_STOCK + ("512010",), window=30)
    add("fixed_bond_global", global_=glob + ("511010",), window=40)
    add("fixed_cash_no_panic_no_overheat", overrides=dict(never_empty=False, panic_drop=0,
                                                         overheat=9.9, bear_enter_mom=0, pool_buffer={}))
    add("fixed_strict_bear_panic", overrides=dict(never_empty=True, panic_drop=.05, bear_enter_mom=.10))
    add("fixed_no_crash_uniform", stock=CORE_STOCK, global_=("513100",), crash=False,
        overrides=dict(panic_drop=.03, bear_enter_mom=.03, pool_buffer={}))
    for mode in ("open_stock", "always_bull", "all_assets"):
        add("fixed_regime_" + mode, stock=CORE_STOCK, global_=("513100",),
            regime_mode=mode, overrides=dict(never_empty=False))
    if len(cases) != 18:
        raise AssertionError("Fixed fidelity case budget changed")
    return cases


def independent_original_strategy():
    # Load the original source into a distinct module so lab monkey patches or
    # FeatureBank's frozen-module globals cannot contaminate this calculation.
    path = BASE.parent / "strategy.py"
    spec = importlib.util.spec_from_file_location("_search_independent_original_strategy", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_wls(closes, window, volatility):
    """Separate raw-price weighted regression; does not call the bank's scorer."""
    prices = closes[-window:]
    weights = list(range(1, len(prices) + 1))
    total = sum(weights)
    mx = math.fsum(w * i for i, w in enumerate(weights)) / total
    my = math.fsum(w * p for w, p in zip(weights, prices)) / total
    xy = math.fsum(w * (i - mx) * (p - my) for i, (w, p) in enumerate(zip(weights, prices)))
    xx = math.fsum(w * (i - mx) ** 2 for i, w in enumerate(weights))
    return xy / xx / my * 250 / volatility if xx > 0 and my > 0 and volatility > 0 else 0.0


def check_features(bank, histories, sample_dates=None):
    native = independent_original_strategy()
    for key, value in bank.presets["v9.1"].items():
        if hasattr(native, key.upper()):
            setattr(native, key.upper(), deepcopy(value))
    native.MIN_ROWS = max(native.MA_BULL, max(native.MOM_WINDOWS) + 1 + native.SKIP_DAYS,
                          native.MOM_MAIN + 1 + native.SKIP_DAYS) + native.VOL_DAYS
    if sample_dates is None:
        selected = {d for i, d in enumerate(bank.calendar) if d >= START and i % 61 == 0}
        selected.update(d for d in ("2014-01-02", "2015-04-13", "2015-04-14",
                                    "2022-03-15", "2024-01-31", END) if d in bank.calendar)
        for rows in histories.values():
            selected.update(r[0] for r in rows[268:271] if START <= r[0] <= END)
        sample_dates = sorted(selected)
    modes = [("wls", (w,)) for w in (20, 25, 30, 40)] + [(m, ()) for m in ("mom20_vol", "mom60_vol", "mom20")]
    checked, scores, availability = 0, 0, 0
    maximum_error = 0.0
    for date in sample_dates:
        i = bisect.bisect_left(bank.calendar, date)
        ranked = {(mode, windows): dict(bank.ranked(mode, windows)[i]) for mode, windows in modes}
        for code in ASSET_ORDER:
            if code == "511880" or code not in histories:
                continue
            rows = [r for r in histories[code] if r[0] <= date]
            native_ind = (native.indicators([r[2] for r in rows], [r[3] for r in rows])
                          if rows and rows[-1][0] == date else None)
            available = code in bank.base[i]
            availability += 1
            if available != (native_ind is not None):
                raise AssertionError("Feature availability mismatch: %s %s" % (date, code))
            if not available:
                continue
            for key in DYNAMIC:
                value, expected = bank.base[i][code][key], native_ind[key]
                if isinstance(value, bool):
                    equal = value == expected
                else:
                    equal = math.isclose(value, expected, rel_tol=1e-11, abs_tol=1e-11)
                    maximum_error = max(maximum_error, abs(value - expected))
                if not equal:
                    raise AssertionError("Original indicator mismatch %s %s %s: %r != %r" %
                                         (date, code, key, value, expected))
                checked += 1
            prior_volume = sum(r[3] for r in rows[-21:-1]) / 20
            volume_ratio = rows[-1][3] / prior_volume if prior_volume > 0 else 0.0
            if not math.isclose(bank.base[i][code]["volume_ratio20"], volume_ratio, abs_tol=1e-12):
                raise AssertionError("Volume window mismatch")
            closes = [r[2] for r in rows]
            for mode, windows in modes:
                if mode == "wls":
                    expected = independent_wls(closes, windows[0], native_ind["vol"])
                elif mode == "mom20":
                    expected = native_ind["mom20"]
                else:
                    momentum = native_ind["mom60"] if mode == "mom60_vol" else native_ind["mom20"]
                    expected = momentum / native_ind["vol"] if native_ind["vol"] else 0.0
                actual = ranked[(mode, windows)][code]["score"]
                if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                    raise AssertionError("Independent score mismatch %s %s %s %r: %r != %r" %
                                         (date, code, mode, windows, actual, expected))
                maximum_error = max(maximum_error, abs(actual - expected))
                scores += 1
    return dict(passed=True, dates=list(sample_dates), availability_checks=availability,
                independent_dynamic_fields_checked=checked, independent_scores_checked=scores,
                maximum_absolute_difference=maximum_error,
                original_source_sha256=hashlib.sha256((BASE.parent / "strategy.py").read_bytes()).hexdigest(),
                note="Original source is independently imported; every reference slices raw rows by date. Non-25 WLS uses separate math.fsum regression.")


def _signals(rows):
    return [(date, None if target is None else tuple(sorted(target.items()))) for date, target in rows]


def path_comparison(actual, expected, tolerance=1e-9):
    if len(actual["daily"]) != len(expected["daily"]):
        raise AssertionError("Daily path length differs")
    maximum = 0.0
    for a, e in zip(actual["daily"], expected["daily"]):
        if a[0] != e[0] or set(a[2]) != set(e[2]):
            raise AssertionError("Date/actual holdings mismatch: %r versus %r" % (a, e))
        error = abs(a[1] - e[1])
        maximum = max(maximum, error)
        if error > tolerance or any(abs(a[2][c] - e[2][c]) > tolerance for c in a[2]):
            raise AssertionError("Daily NAV/weights mismatch: %r versus %r" % (a, e))
    if _signals(actual["signal_target"]) != _signals(expected["signal_target"]):
        mismatch = next((a, e) for a, e in zip(_signals(actual["signal_target"]), _signals(expected["signal_target"])) if a != e)
        raise AssertionError("Closing instruction mismatch: %r" % (mismatch,))
    at = [(t["date"], t["signal_date"], tuple(sorted(t["target"].items()))) for t in actual["trades"]]
    et = [(t["date"], t["signal_date"], tuple(sorted(t["target"].items()))) for t in expected["trades"]]
    if at != et:
        raise AssertionError("Actual trade schedule differs")
    for key in ("deferred_count", "rebalance_count", "fill_count"):
        if actual["diagnostics"][key] != expected["diagnostics"][key]:
            raise AssertionError("Execution diagnostic mismatch: " + key)
    return dict(passed=True, sessions=len(actual["daily"]), maximum_absolute_nav_error=maximum,
                trade_count=len(at), deferred_count=actual["diagnostics"]["deferred_count"])


def check_oracles(bank, frame, fear, cases=None, windows=SHORT_WINDOWS):
    results = []
    for config in cases or fixed_cases(bank.presets):
        for start, end in windows:
            optimized = SearchPolicy(config, bank, frame, fear, oracle=False, trace=True)
            actual = run(frame, optimized, start, end, capture_daily=True, capture_trades=True)
            native = SearchPolicy(config, bank, frame, fear, oracle=True, trace=True)
            expected = run(frame, native, start, end, capture_daily=True, capture_trades=True)
            check = path_comparison(actual, expected)
            if optimized.metadata != native.metadata:
                raise AssertionError("Oracle policy trace/crash confirmation mismatch: " + config["id"])
            check.update(case_id=config["id"], start=start, end=end,
                         score_mode=config["score_mode"], windows=config["score_windows"],
                         regime_mode=config.get("regime_mode", "ma250"))
            results.append(check)
        print("Native decision oracle passed: " + config["id"], flush=True)
    return results


def check_controls(bank, frame, fear):
    first_path, second_path = BASE.parent / "v10_next/results/paths.json", BASE.parent / "v10_round2/results/paths.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))["results"]
    second = json.loads(second_path.read_text(encoding="utf-8"))
    reference = {"c_v9": first["control_v9"], "c_v91": first["control_v91"], "c_v92": second["c_v92"]}
    out = {}
    for config in controls():
        policy = SearchPolicy(config, bank, frame, fear, trace=True)
        actual = run(frame, policy, START, END, capture_daily=True, capture_trades=True)
        out[config["id"]] = path_comparison(actual, reference[config["id"]])
        print("Archived actual-fill control passed: " + config["id"], flush=True)
    return dict(results=out, archived_paths_sha256={
        str(p.relative_to(BASE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (first_path, second_path)})


def check_same_close_controls(bank, fear):
    """Match independent old policy paths plus already-archived ideal metrics."""
    from v10_round2.research import Simulator
    from .same_close import simulate_same_close
    reference_simulator = Simulator()
    archive_path = BASE.parent / "v10_round2/results/evaluation.json"
    archived = json.loads(archive_path.read_text(encoding="utf-8"))["ideal_close"]
    out = {}
    for config in controls():
        if config["id"] not in ("c_v91", "c_v92"):
            continue
        actual = simulate_same_close(config, bank, fear)
        expected = reference_simulator.simulate(config["id"], clock="ideal_close", slippage=0)
        summary = path_comparison(actual, expected)
        saved = archived[config["id"]]["full"]
        nav_error = abs(actual["daily"][-1][1] - saved["nav_factor"])
        ann = actual["daily"][-1][1] ** (244 / len(actual["daily"])) - 1
        if nav_error > 1e-9 or abs(ann - saved["cagr"]) > 1e-11:
            raise AssertionError("Same-close archive summary mismatch: " + config["id"])
        summary.update(final_nav=actual["daily"][-1][1], cagr=ann,
                       archived_summary_nav_error=nav_error)
        out[config["id"]] = summary
        print("Ideal-close full-path/archive control passed: " + config["id"], flush=True)
    return dict(results=out, archived_evaluation_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                note="Ideal same-close diagnostic only, including initial commission. Five close-to-close crash return intervals; never a selection clock.")


def verify():
    protected_before = protect()
    sources = ("features.py", "policy.py", "fast_execution.py", "registry.py", "data.py", "same_close.py", "verify_fidelity.py")
    hashes = {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in sources}
    histories = load_histories()
    calendar = [r[0] for r in histories["510300"]]
    frame = prepare(histories, calendar)
    print("Building shared price-indicator bank for fidelity only ...", flush=True)
    bank = FeatureBank(histories, calendar)
    qvix = load_qvix()
    fear = tuple(qvix.state(date)["active"] for date in calendar)
    feature_checks = check_features(bank, histories)
    print("Independent raw-price feature checks passed", flush=True)
    oracle_checks = check_oracles(bank, frame, fear)
    controls_checks = check_controls(bank, frame, fear)
    same_close_checks = check_same_close_controls(bank, fear)
    protected_after = protect()
    if any(hashlib.sha256((BASE / name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
        raise AssertionError("Fidelity source changed during verification; rerun on stable source")
    return dict(passed=True, start=START, end=END, nav_tolerance=1e-9,
                protected_files_before=protected_before, protected_files_after=protected_after,
                source_sha256=hashes,
                data_manifest_sha256=hashlib.sha256((BASE / "data_manifest.json").read_bytes()).hexdigest(),
                independent_feature_checks=feature_checks, fixed_oracle_checks=oracle_checks,
                archived_controls=controls_checks, same_close_controls=same_close_checks,
                scope="No candidate screening or return selection. Fixed coverage cases and immutable prior actual-fill controls only.")


if __name__ == "__main__":
    report = verify()
    destination = BASE / "results/fidelity.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Written " + str(destination), flush=True)
