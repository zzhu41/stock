"""Bounded, deterministic registration of 5,156 ETF strategy configurations.

Generation reads the already-frozen parameter presets, not prices or returns.
Every candidate carries complete parameters and explicit trading/benchmark
pools. Controls are exposed separately and are not additional search trials.
"""
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations, product
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
PRESETS = BASE.parent / "v10_next" / "frozen" / "presets.json"
BASE_STOCK = ("159915", "588080", "510300", "510500", "563300", "512400", "512890")
CORE_STOCK = ("159915", "510300", "510500")
GLOBAL_SUBSETS = (("513100",), ("513120",), ("513100", "513120"))
DIVIDENDS = ("510880", "515080", "515100")
SECTORS = ("159928", "512010", "512070", "512880", "512800", "512200")
EXTRA_STOCK = DIVIDENDS + SECTORS
EXTRA_GLOBAL = ("513030", "513520", "159985", "511010")
WINDOWS = (20, 25, 30, 40)
GOLD, CASH, BENCHMARK = "518880", "511880", "510300"
EXPECTED_RAW, EXPECTED_UNIQUE = 5164, 5156

ROLE_DESCRIPTIONS = {
    "stock": "A-share competition pool subject to the domestic-equity bull/bear exclusion",
    "global": "Risk assets exempt from the A-share exclusion; may include overseas equity, commodities or bonds",
    "gold": "Gold role: bull fallback and bearish competition asset",
    "cash": "Money-market ETF parking asset",
    "benchmark_only": "CSI300 regime indicator only; not eligible for ordinary or crash purchases",
    "calendar_only": "Required trading calendar only; no regime signal and no ordinary/crash eligibility",
}
REGIME_DESCRIPTIONS = {
    "ma250": "Original MA250 bull/bear decision, including the domestic-equity exclusion and bearish entry gate",
    "open_stock": "Only enable bear_open_stock: retain the actual MA250 regime and the 7% bearish entry threshold",
    "always_bull": "Ignore MA250 regime; use original bull rules with stock/global competition and gold as fallback",
    "all_assets": "Ignore MA250 regime; stocks, global assets and gold compete together; retain remaining exit/crash/fallback rules",
}
ASSET_CATALOG = {
    "159915": {"name": "创业板ETF", "role": "stock", "asset_class": "domestic_equity"},
    "588080": {"name": "科创50ETF", "role": "stock", "asset_class": "domestic_equity"},
    "510300": {"name": "沪深300ETF", "role": "stock", "asset_class": "domestic_equity"},
    "510500": {"name": "中证500ETF", "role": "stock", "asset_class": "domestic_equity"},
    "563300": {"name": "中证2000ETF", "role": "stock", "asset_class": "domestic_equity"},
    "512400": {"name": "有色金属ETF", "role": "stock", "asset_class": "domestic_sector_equity"},
    "512890": {"name": "红利低波ETF", "role": "stock", "asset_class": "domestic_dividend_equity"},
    "513100": {"name": "纳指ETF", "role": "global", "asset_class": "overseas_equity"},
    "513120": {"name": "港股创新药ETF", "role": "global", "asset_class": "overseas_sector_equity"},
    "518880": {"name": "黄金ETF", "role": "gold", "asset_class": "gold"},
    "511880": {"name": "货币ETF", "role": "cash", "asset_class": "money_market"},
    "510880": {"name": "红利ETF", "role": "stock", "asset_class": "domestic_dividend_equity"},
    "515080": {"name": "中证红利ETF", "role": "stock", "asset_class": "domestic_dividend_equity"},
    "515100": {"name": "红利低波100ETF", "role": "stock", "asset_class": "domestic_dividend_equity"},
    "159928": {"name": "消费ETF", "role": "stock", "asset_class": "domestic_sector_equity"},
    "512010": {"name": "医药ETF", "role": "stock", "asset_class": "domestic_sector_equity"},
    "512070": {"name": "非银ETF", "role": "stock", "asset_class": "domestic_sector_equity"},
    "512880": {"name": "证券ETF", "role": "stock", "asset_class": "domestic_sector_equity"},
    "512800": {"name": "银行ETF", "role": "stock", "asset_class": "domestic_sector_equity"},
    "512200": {"name": "房地产ETF", "role": "stock", "asset_class": "domestic_sector_equity"},
    "513030": {"name": "德国ETF", "role": "global", "asset_class": "overseas_equity"},
    "513520": {"name": "日经ETF", "role": "global", "asset_class": "overseas_equity"},
    "159985": {"name": "豆粕ETF", "role": "global", "asset_class": "commodity"},
    "511010": {"name": "国债ETF", "role": "global", "asset_class": "government_bond"},
}
BASE_ASSET_ORDER = BASE_STOCK + GLOBAL_SUBSETS[-1] + (GOLD, CASH)
FIXED_INPUT_ASSET_ORDER = BASE_ASSET_ORDER + tuple(sorted(set(ASSET_CATALOG) - set(BASE_ASSET_ORDER)))

FAMILY_DESCRIPTIONS = {
    "pool_subsets": "127 nonempty subsets of seven A-share ETFs x three global subsets x WLS20/25/30/40 x crash on/off",
    "rule_grid": "Full A7+G2 and core A3+Nasdaq anchors x WLS25/30 x four bear thresholds x four panic thresholds x overheating on/off x never-empty on/off x uniform/split buffer x crash on/off",
    "extra_stock_single": "One of nine dividend/sector ETFs added to either A-share anchor x three global subsets x four windows x crash on/off",
    "extra_global_single": "One of four external equity/commodity/bond ETFs added to global role, with the same anchor/window/crash grid",
    "fixed_baskets": "Add all six sectors, all three dividend ETFs, or all four external assets as one predetermined basket",
    "ranking_ablation": "MOM20/VOL20, MOM60/VOL20 or plain MOM20 ranking x two A-share anchors x three global subsets x crash on/off; all entry gates remain MOM20",
    "regime_ablation": "Open-stock, always-bull and unified-all-assets regimes x two A-share anchors x three global subsets x four WLS windows x crash on/off x never-empty on/off",
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def semantic_payload(candidate):
    """Return trading semantics only; labels/provenance do not create trials."""
    return {
        "kind": candidate["kind"], "params": deepcopy(candidate["params"]),
        "stock_pool": sorted(candidate["stock_pool"]),
        "global_pool": sorted(candidate["global_pool"]),
        "gold": candidate["gold"], "cash": candidate["cash"],
        "benchmark_code": candidate["benchmark_code"],
        "score_mode": candidate["score_mode"], "score_windows": list(candidate["score_windows"]),
        "channels": sorted(candidate["channels"]), "risk_weight": candidate["risk_weight"],
        "tie_break": candidate["tie_break"], "input_asset_order": list(candidate["input_asset_order"]),
        "regime_mode": candidate["regime_mode"],
    }


def candidate_hash(candidate):
    return _hash(semantic_payload(candidate))


def required_codes(candidate):
    """Required observations include CSI300 even when it cannot be traded."""
    return tuple(sorted(set(candidate["stock_pool"]) | set(candidate["global_pool"]) |
                        {candidate["gold"], candidate["cash"], candidate["benchmark_code"]}))


def _complexity(candidate):
    params = candidate["params"]
    rules = []
    for name, enabled in (
            ("crash_entry_lock", bool(candidate["channels"])),
            ("panic_exit", params["panic_drop"] > 0),
            ("bear_entry_threshold", params["bear_enter_mom"] > 0 and candidate["regime_uses_benchmark"]),
            ("overheat_guard", params["overheat"] < 9),
            ("never_empty_fallback", params["never_empty"]),
            ("pool_specific_buffer", bool(params["pool_buffer"]))):
        if enabled:
            rules.append(name)
    return dict(risk_assets=len(candidate["stock_pool"]) + len(candidate["global_pool"]) + 1,
                score_inputs=1, active_optional_rules=rules, active_optional_rule_count=len(rules),
                regime_mode=candidate["regime_mode"], uses_regime_classifier=candidate["regime_uses_benchmark"],
                note="Descriptive structural counts, not independent statistical degrees of freedom")


def _candidate(base, family, stock, glob, window=25, crash=True, mode="wls", overrides=None,
               regime_mode="ma250"):
    stock, glob = sorted(set(stock)), sorted(set(glob))
    if not stock or not glob or set(stock) & set(glob):
        raise ValueError("Registered trading pools must be nonempty and disjoint")
    if any(ASSET_CATALOG[c]["role"] != "stock" for c in stock) or \
            any(ASSET_CATALOG[c]["role"] != "global" for c in glob):
        raise ValueError("Trading-pool roles disagree with the registered asset catalog")
    params = deepcopy(base)
    params.update(deepcopy(overrides or {}))
    if regime_mode not in REGIME_DESCRIPTIONS:
        raise ValueError("Unregistered regime mode")
    params["bear_open_stock"] = regime_mode == "open_stock"
    params["crash_mom5"] = -.08 if crash else 0.0
    params.update(score_wls=mode == "wls", score_mom=60 if mode == "mom60_vol" else 0,
                  score_plain_mom=mode == "mom20")
    if mode not in ("wls", "mom20_vol", "mom60_vol", "mom20"):
        raise ValueError("Unregistered ranking mode")
    candidate = dict(
        kind="legacy", family=family, families=[family], params=params,
        stock_pool=stock, global_pool=glob, gold=GOLD, cash=CASH, benchmark_code=BENCHMARK,
        score_mode=mode, score_windows=[window] if mode == "wls" else [],
        channels=["deep"] if crash else [], risk_weight=1.0,
        tie_break="fixed_input_asset_order", input_asset_order=list(FIXED_INPUT_ASSET_ORDER),
        regime_mode=regime_mode, regime_uses_benchmark=regime_mode in ("ma250", "open_stock"),
        benchmark_purpose="regime_and_calendar" if regime_mode in ("ma250", "open_stock") else "calendar_only",
        is_control=False, counts_as_independent_extra_control_trial=False,
    )
    pool = dict(stock=stock, global_=glob, gold=GOLD, cash=CASH)
    candidate["trade_pool_hash"] = _hash(pool)
    candidate["asset_roles"] = {code: ("stock" if code in stock else "global" if code in glob
                                      else "gold" if code == GOLD else "cash" if code == CASH
                                      else "benchmark_only" if candidate["regime_uses_benchmark"]
                                      else "calendar_only") for code in required_codes(candidate)}
    candidate["complexity"] = _complexity(candidate)
    candidate["candidate_hash"] = candidate_hash(candidate)
    candidate["id"] = "search_" + candidate["candidate_hash"][:20]
    return candidate


def _raw_candidates(base):
    for size in range(1, len(BASE_STOCK) + 1):
        for stock, glob, window, crash in product(combinations(BASE_STOCK, size), GLOBAL_SUBSETS,
                                                WINDOWS, (False, True)):
            yield _candidate(base, "pool_subsets", stock, glob, window, crash)

    anchors = ((BASE_STOCK, GLOBAL_SUBSETS[-1]), (CORE_STOCK, GLOBAL_SUBSETS[0]))
    for (stock, glob), window, bear, panic, overheat, never_empty, split_buffer, crash in product(
            anchors, (25, 30), (0.0, .03, .07, .10), (0.0, .03, .04, .05),
            (.40, 9.9), (False, True), (False, True), (False, True)):
        overrides = dict(bear_enter_mom=bear, panic_drop=panic, overheat=overheat,
                         never_empty=never_empty,
                         pool_buffer={"stock": .02, "global": .03, "gold": .03} if split_buffer else {})
        yield _candidate(base, "rule_grid", stock, glob, window, crash, overrides=overrides)

    for stock, glob, window, crash, extra in product((BASE_STOCK, CORE_STOCK), GLOBAL_SUBSETS,
                                                   WINDOWS, (False, True), EXTRA_STOCK):
        yield _candidate(base, "extra_stock_single", stock + (extra,), glob, window, crash)
    for stock, glob, window, crash, extra in product((BASE_STOCK, CORE_STOCK), GLOBAL_SUBSETS,
                                                   WINDOWS, (False, True), EXTRA_GLOBAL):
        yield _candidate(base, "extra_global_single", stock, glob + (extra,), window, crash)

    baskets = (("stock", SECTORS), ("stock", DIVIDENDS), ("global", EXTRA_GLOBAL))
    for stock, glob, window, crash, (role, extra) in product((BASE_STOCK, CORE_STOCK), GLOBAL_SUBSETS,
                                                          WINDOWS, (False, True), baskets):
        yield _candidate(base, "fixed_baskets", stock + extra if role == "stock" else stock,
                         glob + extra if role == "global" else glob, window, crash)

    for stock, glob, crash, mode in product((BASE_STOCK, CORE_STOCK), GLOBAL_SUBSETS, (False, True),
                                            ("mom20_vol", "mom60_vol", "mom20")):
        yield _candidate(base, "ranking_ablation", stock, glob, crash=crash, mode=mode)

    for regime_mode, stock, glob, window, crash, never_empty in product(
            ("open_stock", "always_bull", "all_assets"), (BASE_STOCK, CORE_STOCK),
            GLOBAL_SUBSETS, WINDOWS, (False, True), (False, True)):
        yield _candidate(base, "regime_ablation", stock, glob, window, crash,
                         overrides={"never_empty": never_empty}, regime_mode=regime_mode)


def _presets():
    return json.loads(PRESETS.read_text(encoding="utf-8"))


def build_registry():
    presets = _presets()
    by_hash, by_id, duplicates = {}, {}, []
    raw_counts, first_counts = Counter(), Counter()
    for ordinal, candidate in enumerate(_raw_candidates(presets["v9.1"]), 1):
        family, key = candidate["family"], candidate["candidate_hash"]
        raw_counts[family] += 1
        source = dict(family=family, raw_ordinal=ordinal)
        if key in by_hash:
            kept = by_hash[key]
            kept["families"] = sorted(set(kept["families"]) | {family})
            kept["registration_sources"].append(source)
            duplicates.append(dict(source, candidate_hash=key, kept_id=kept["id"], kept_family=kept["family"]))
            continue
        if candidate["id"] in by_id and by_id[candidate["id"]] != key:
            raise AssertionError("Truncated candidate identifier collision")
        candidate["registration_sources"] = [source]
        by_hash[key], by_id[candidate["id"]] = candidate, key
        first_counts[family] += 1
    candidates = list(by_hash.values())
    raw_count = sum(raw_counts.values())
    if raw_count != EXPECTED_RAW or len(candidates) != EXPECTED_UNIQUE or len(duplicates) != 8:
        raise AssertionError("The preregistered enumeration size changed")
    return dict(
        candidates=candidates, raw_count=raw_count, unique_count=len(candidates), duplicates=duplicates,
        family_counts={family: dict(raw=raw_counts[family], first_unique=first_counts[family],
                                   memberships=sum(family in c["families"] for c in candidates))
                       for family in FAMILY_DESCRIPTIONS},
        family_descriptions=deepcopy(FAMILY_DESCRIPTIONS), role_descriptions=deepcopy(ROLE_DESCRIPTIONS),
        regime_descriptions=deepcopy(REGIME_DESCRIPTIONS),
        asset_catalog=deepcopy(ASSET_CATALOG), required_snapshot_codes=sorted(ASSET_CATALOG),
        tie_break="fixed_input_asset_order", fixed_input_asset_order=list(FIXED_INPUT_ASSET_ORDER),
        frozen_params_sha256=hashlib.sha256(PRESETS.read_bytes()).hexdigest(),
        registry_sha256=_hash([c["candidate_hash"] for c in candidates]),
        caveats=[
            "All 5,156 configurations are correlated research trials, not independent statistical samples.",
            "Track distinct realized paths separately; late-listed assets can make distinct configurations dormant in an early interval.",
            "CSI300 remains a required regime feature even when excluded from the trading pool; do not permit benchmark-only crash trades.",
            "Global denotes exemption from the A-share regime gate, not geographical location or guaranteed safety.",
            "Open-stock is not removal of bull/bear classification. Always-bull and all-assets ignore MA250 for regime decisions; CSI300 is still needed for the calendar and may trade if separately selected.",
            "The overheating off setting is the inherited 990% threshold, affecting both entry and fast exit.",
            "The grid is bounded before results; the registry does not fetch data or choose an evaluation/selection interval.",
        ])


def controls():
    """Same-schema comparisons; overlapping controls are not extra trials."""
    presets = _presets()
    out = []
    for name, version in (("c_v9", "v9"), ("c_v91", "v9.1"), ("c_v92", "v9.1")):
        candidate = _candidate(presets[version], "control", BASE_STOCK, GLOBAL_SUBSETS[-1])
        if name == "c_v92":
            candidate["kind"] = "v92"
            candidate["channels"] = ["deep", "qvix", "volume"]
            candidate["candidate_hash"] = candidate_hash(candidate)
            candidate["complexity"] = _complexity(candidate)
        candidate.update(id=name, is_control=True, counts_as_independent_extra_control_trial=False,
                         control_version=version if name != "c_v92" else "v9.2")
        out.append(candidate)
    return out
