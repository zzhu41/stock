"""Export frozen corrected-TR H/v9.2 curves without changing research inputs.

No live prices, accounts or future shadow NAV enter this export. Saved paths,
feature arrays and the independent reference implementation must pass their
existing frozen hashes. Reference traces supply exact filled crash dates.
Absent derived feature caches may be reproduced by the frozen builder; existing
files with a wrong hash are rejected rather than overwritten.
Run ``python3.8 -B web_v10.py`` to print the two web-compatible versions as JSON.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import sys

import numpy as np

from v10_deep.cli import load_profile
from v10_deep.data import START, END
from v10_deep.features import build as build_frozen_features
from v10_deep import reference
from v10_deep.scan import summarize


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "v10_deep"
PROFILES = BASE / "profiles.json"
RECEIPT = BASE / "results/refinements/selected_reference_fidelity.json"
SELECTED_PATHS = BASE / "results/refinements/selected_paths.json"
SELECTION = BASE / "results/refinements/selection.json"
FEATURE_META = BASE / "cache/features.json"
FEATURE_CACHE = BASE / "cache/features.npz"
TR_MANIFEST = ROOT / "v10_h_close/corrected_manifest.json"
TR_SNAPSHOTS = ROOT / "v10_h_close/corrected_snapshots"
VERSION_SPECS = (("v10-h", "growth", "V10-H 冻结研究（修正总回报）"),
                 ("v9.2-tr", "v92", "v9.2 同口径对照（修正总回报）"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Frozen web export input changed: " + str(path))
    return json.loads(raw.decode("utf-8"))


def _cache_complete(pins):
    complete = True
    for path, key in ((FEATURE_META, "feature_metadata_sha256"), (FEATURE_CACHE, "feature_cache_sha256")):
        if path.exists():
            if _sha(path) != pins[key]:
                raise ValueError("Frozen web export input changed: " + str(path))
        else:
            complete = False
    return complete


def _ensure_feature_cache(pins):
    """Only absent derived files may be rebuilt; existing wrong bytes never are."""
    if _cache_complete(pins):
        return
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with (FEATURE_CACHE.parent / ".web-v10-cache.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        # Another cold exporter may have completed while this one waited.
        if not _cache_complete(pins):
            with redirect_stdout(sys.stderr):
                build_frozen_features()
            if not _cache_complete(pins):
                raise ValueError("Frozen feature rebuild did not provide both pinned derived files")


def _load_bundle():
    # load_profile verifies frozen sources/dependencies/artifacts/manifests and
    # the actual corrected CSV/QVIX bytes, without building or changing caches.
    profiles = {variant: load_profile(variant) for _, variant, _ in VERSION_SPECS}
    pins = json.loads(PROFILES.read_text(encoding="utf-8"))
    receipt = _verified_json(RECEIPT, pins["artifact_sha256"]["results/refinements/selected_reference_fidelity.json"])
    if not receipt.get("passed") or (receipt["start"], receipt["end"]) != (START, END):
        raise ValueError("Frozen fidelity receipt does not cover the research interval")
    expected_reference = [value for name, value in receipt["source_and_cache_sha256"].items()
                          if name.endswith("/v10_deep/reference.py")]
    if len(expected_reference) != 1 or _sha(reference.__file__) != expected_reference[0]:
        raise ValueError("Frozen reference implementation changed")
    selection = _verified_json(SELECTION, pins["artifact_sha256"]["results/refinements/selection.json"])
    expected_ids = {"growth": selection["primary"], "v92": selection["controls"]["v9.2"]}
    for variant, profile in profiles.items():
        if profile["config"]["id"] != expected_ids[variant]:
            raise ValueError("Web version no longer identifies its frozen selection: " + variant)
    paths = _verified_json(SELECTED_PATHS, receipt["stage_input_sha256"]["selected_paths.json"])
    _ensure_feature_cache(pins)
    meta = _verified_json(FEATURE_META, pins["feature_metadata_sha256"])
    needed = list(dict.fromkeys(profile["config"]["score"] for profile in profiles.values()))
    indices = [meta["score_names"].index(name) for name in needed]
    # Load only two score families. No native compilation or cache rebuilding.
    with np.load(FEATURE_CACHE, allow_pickle=False) as source:
        arrays = dict(features=source["features"], scores=source["scores"][indices], fear=source["fear"])
    meta = deepcopy(meta)
    meta["score_names"] = needed
    dates = [day for day in meta["dates"] if START <= day <= END]
    if not dates or dates[0] != START or dates[-1] != END or dates != sorted(set(dates)):
        raise ValueError("Frozen feature dates do not cover the exact research interval")
    manifest = _verified_json(TR_MANIFEST, pins["input_manifests"]["corrected"])
    return dict(profiles=profiles, pins=pins, receipt=receipt, paths=paths,
                arrays=arrays, meta=meta, dates=dates, manifest=manifest)


def _web_metrics(full, switches):
    """Web percentage fields, retaining complete float precision until display."""
    return dict(nav=full["nav"], total_ret=100 * full["total_return"], ann=100 * full["cagr"],
                max_dd=100 * full["max_dd"], sharpe=full["sharpe"], calmar=full["calmar"],
                volatility=100 * full["volatility"], days=full["sessions"], switches=int(switches))


def _metadata(bundle, candidate_id=None):
    return dict(basis="corrected_tr_same_close", kind="research", frozen=True,
                period=dict(start=START, end=END), data_as_of=END,
                clock="same_close", fee_per_side=.0001, first_session_free=True,
                annualization_sessions=244, candidate_id=candidate_id,
                holding_convention="当日收盘换仓后模型持仓，当天收益主要归此前持仓",
                execution_price_provided=False, shadow_nav_included=False, clean_oos=False,
                warnings=["冻结研究历史，截至2026-09-24；不含前向影子净值。",
                          "同收盘理想成交、除息即时再投资，与14:50影子试算不同。",
                          "历史经过选型，过拟合风险仍存在。"],
                provenance=dict(selected_paths_sha256=bundle["receipt"]["stage_input_sha256"]["selected_paths.json"],
                    fidelity_receipt_sha256=bundle["pins"]["artifact_sha256"]["results/refinements/selected_reference_fidelity.json"],
                    feature_cache_sha256=bundle["pins"]["feature_cache_sha256"],
                    feature_metadata_sha256=bundle["pins"]["feature_metadata_sha256"],
                    corrected_manifest_sha256=bundle["pins"]["input_manifests"]["corrected"]),
                trace_verification="Exact daily returns/holdings versus frozen reference; crash markers from filled trace")


def _benchmarks(bundle):
    dates, manifest = bundle["dates"], bundle["manifest"]
    results = {}
    for code, label in (("510300", "沪深300ETF（修正总回报买入持有）"),
                        ("518880", "黄金ETF（修正总回报买入持有）")):
        path = TR_SNAPSHOTS / (code + ".csv")
        raw = path.read_bytes()
        expected = manifest["assets"][code]["sha256"]
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Frozen corrected benchmark changed: " + code)
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
        closes = {row[0]: float(row[2]) for row in rows if row}
        if START not in closes:
            raise ValueError("Benchmark lacks initial research close: " + code)
        base = mark = closes[START]
        previous, daily, returns = 1., [], []
        missing = []
        for date in dates:
            if date in closes:
                mark = closes[date]
            else:
                missing.append(date)  # Carry existing valuation, then catch up at the next observed close.
            if not math.isfinite(mark) or mark <= 0:
                raise ValueError("Invalid corrected benchmark close")
            nav = mark / base
            daily.append([date, nav])
            returns.append(nav / previous - 1.)
            previous = nav
        summaries = summarize(returns, dates)
        metadata = _metadata(bundle)
        metadata.update(comparison_role="buy_and_hold", snapshot_sha256=expected,
                        missing_quote_dates=missing, fees_charged=0.,
                        trace_verification="Frozen corrected-seed hash; buy-and-hold on the research calendar",
                        holding_convention="首个研究日买入并持有；缺报价日延续估值，复牌补计")
        results[code] = dict(label=label, daily=daily, daily_returns=returns,
                             metrics=_web_metrics(summaries["full"], 0), full_metrics=summaries["full"],
                             period_metrics=summaries, metadata=metadata)
    return results


def _export_one(vid, variant, label, bundle, benchmarks):
    profile = bundle["profiles"][variant]
    config, dates = profile["config"], bundle["dates"]
    saved = bundle["paths"][config["id"]]
    returns = np.asarray(saved["returns"], dtype=np.float64)
    holdings = saved["holdings"]
    if len(returns) != len(dates) or len(holdings) != len(dates) or not np.isfinite(returns).all():
        raise ValueError("Frozen path dates/returns/holdings disagree")
    if returns[0] != 0 or np.any(returns <= -1):
        raise ValueError("Frozen initial-free-day or positive-NAV convention changed")
    traced = reference.run_reference(bundle["arrays"], bundle["meta"], config, start=START, end=END, fee=.0001)
    traced_holdings = [bundle["meta"]["assets"][int(index)] if index >= 0 else None
                       for index in traced["holdings"]]
    if traced["dates"] != dates or traced_holdings != holdings or not np.array_equal(traced["returns"], returns):
        raise ValueError("Frozen saved path differs from the independently verified reference: " + vid)
    navs = np.cumprod(1. + returns)
    daily = [[date, float(nav), holding] for date, nav, holding in zip(dates, navs, holdings)]
    trades, crash_buys = [], []
    switches = 0
    for index, event in enumerate(traced["trace"]):
        if not event["filled"]:
            continue
        trades.append([dates[index], holdings[index - 1] if index else None,
                       holdings[index], float(navs[index])])
        switches += int(index > 0)
        if event["crash_requested"]:
            crash_buys.append([dates[index], holdings[index]])
    cases = [case for case in bundle["receipt"]["cases"]
             if case["id"] == config["id"] and case["fee_per_side"] == .0001]
    if len(cases) != 1 or not cases[0]["exact_holdings"] or not cases[0]["exact_native_returns"]:
        raise ValueError("No exact frozen fidelity case for " + vid)
    expected = cases[0]["summary"]
    if (switches != int(expected[3]) or len(crash_buys) != int(expected[7])
            or not math.isclose(float(navs[-1]), expected[0], rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("Exported NAV/switch/crash totals differ from frozen verification")
    summaries = summarize(returns, dates)
    metadata = _metadata(bundle, config["id"])
    metadata.update(config_sha256=config["hash"], initial_entry_in_trades=True,
                    switches_exclude_first_session=True, research_status=profile["research_status"])
    return dict(id=vid, label=label, daily=daily, daily_returns=returns.tolist(),
                trades=trades, crash_buys=crash_buys,
                metrics=_web_metrics(summaries["full"], switches), full_metrics=summaries["full"],
                period_metrics=summaries, metadata=metadata, benchmarks=deepcopy(benchmarks),
                metrics_convention=dict(initial_nav=1., returns_aligned_with_daily=True,
                    first_return_included=True, annualization_sessions=244, drawdown_initial_peak=1.,
                    percentage_fields=["ann", "max_dd", "total_ret", "volatility"],
                    interval_baseline="Previous observed NAV; use 1 only at the first research session"))


def export_versions():
    """Return only v10-h/v9.2-tr; no mutation of other web versions or live state."""
    bundle = _load_bundle()
    benchmarks = _benchmarks(bundle)
    return {vid: _export_one(vid, variant, label, bundle, benchmarks)
            for vid, variant, label in VERSION_SPECS}


if __name__ == "__main__":
    print(json.dumps(export_versions(), ensure_ascii=False, separators=(",", ":"), allow_nan=False))
