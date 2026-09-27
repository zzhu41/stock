"""Finite same-close H iteration: 5,156 prior configurations plus a fixed grid.

No prices, returns, cutoff date or live services are consulted. Strategy hashes
keep the prior semantic definition; a new hc_ identifier is not a claim that an
old rule has become a new trading strategy. Execution/data dates belong to the
separate batch registration, so differently dated runs cannot be mixed here.
"""
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

from v10_search.registry import (
    BASE_STOCK, DIVIDENDS, PRESETS, _candidate as parent_candidate,
    _complexity as parent_complexity, build_registry as parent_registry,
    candidate_hash, controls as parent_controls, required_codes, semantic_payload,
)


BASE = Path(__file__).resolve().parent
EXPECTED_WIDE = 5156
EXPECTED_FOCUSED = 1440
EXPECTED_OVERLAP = 52
EXPECTED_RAW = 6596
EXPECTED_UNIQUE = 6544
STOCK_ANCHORS = (
    ("original_a7", BASE_STOCK),
    ("a7_plus_510880", BASE_STOCK + ("510880",)),
    ("a7_plus_dividend_basket", BASE_STOCK + DIVIDENDS),
)
GLOBAL_ANCHORS = (("nasdaq", ("513100",)), ("nasdaq_and_hk_innovation", ("513100", "513120")))
SCORE_WINDOWS = ((20,), (25,), (30,), (35,), (40,), (25, 30))
BUFFERS = (("uniform_1pp", .01), ("uniform_2pp", .02), ("uniform_3pp", .03),
           ("uniform_4pp", .04), ("original_split_2_3_3", None))
CHANNEL_ORDER = ("deep", "qvix", "volume")
CHANNEL_SUBSETS = tuple(tuple(subset) for n in range(4) for subset in combinations(CHANNEL_ORDER, n))

EXECUTION_SPEC = {
    "clock": "original_same_close",
    "signal": "same-day closing information",
    "fill": "same-day closing price",
    "commission_per_side": .0001,
    "switch_nav_multiplier": 1 - 2 * .0001,
    "initial_entry_fee": 0.0,
    "slippage_per_side": 0.0,
    "annualization_trading_days": 244,
    "limitation": "Original idealized backtest convention, not evidence of an executable same-close fill.",
}
CHANNEL_DEFINITIONS = {
    "deep": dict(momentum_days=5, momentum_threshold=-.08, below_ma250=.20,
                 depth_comparison="strictly below", lock_days=5),
    "qvix": dict(previous_window=250, minimum_previous_rows=120, z_threshold=2.5,
                 z_comparison="at least", momentum_days=5, momentum_threshold=-.04,
                 below_ma250=.20, depth_comparison="strictly below", lock_days=5,
                 information="exact dated closing observation; publication timing not verified"),
    "volume": dict(previous_volume_window=20, ratio_threshold=2.0, ratio_comparison="at least",
                   momentum_days=5, momentum_threshold=-.04, below_ma250=.10,
                   depth_comparison="strictly below", lock_days=5),
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _focused(base):
    for (stock_name, stock), (global_name, glob), windows, (buffer_name, buffer), channels in product(
            STOCK_ANCHORS, GLOBAL_ANCHORS, SCORE_WINDOWS, BUFFERS, CHANNEL_SUBSETS):
        overrides = {} if buffer is None else {"buffer": buffer, "pool_buffer": {}}
        candidate = parent_candidate(base, "close_focused", stock, glob, window=windows[0],
                                     crash="deep" in channels, overrides=overrides)
        candidate["score_windows"] = list(windows)
        candidate["channels"] = list(channels)
        candidate["kind"] = "v92" if set(channels) & {"qvix", "volume"} else "legacy"
        candidate["complexity"] = parent_complexity(candidate)
        candidate["complexity"]["score_inputs"] = len(windows)
        candidate["complexity"]["external_crash_channels"] = [c for c in channels if c != "deep"]
        candidate["candidate_hash"] = candidate_hash(candidate)
        coordinates = dict(stock_anchor=stock_name, global_anchor=global_name,
                           score_windows=list(windows), buffer=buffer_name, channels=list(channels))
        yield candidate, coordinates


def build_registry():
    parent = parent_registry()
    if parent["unique_count"] != EXPECTED_WIDE:
        raise AssertionError("The inherited wide-stage configuration set changed")
    originals = parent["candidates"]
    prior_by_hash = {c["candidate_hash"]: c for c in originals}
    control_matches = {}
    for control in parent_controls():
        control_matches.setdefault(control["candidate_hash"], []).append(control["id"])
    base = json.loads(PRESETS.read_text(encoding="utf-8"))["v9.1"]
    unique, ids, duplicates = {}, {}, []
    stages = {"wide": dict(raw=0, first_unique=0), "focused": dict(raw=0, first_unique=0)}

    def register(candidate, stage, source):
        candidate = deepcopy(candidate)
        key = candidate_hash(candidate)
        if key != candidate["candidate_hash"]:
            raise AssertionError("A candidate's stored hash differs from its complete semantics")
        stages[stage]["raw"] += 1
        if key in unique:
            kept = unique[key]
            kept["stages"] = sorted(set(kept["stages"]) | {stage})
            kept["families"] = sorted(set(kept["families"]) | set(candidate["families"]))
            kept["registration_sources"].append(deepcopy(source))
            duplicates.append(dict(stage=stage, candidate_hash=key, kept_id=kept["id"], source=deepcopy(source)))
            return
        new_id = "hc_" + key[:20]
        if new_id in ids and ids[new_id] != key:
            raise AssertionError("Candidate id prefix collision")
        prior = prior_by_hash.get(key)
        prior_controls = list(control_matches.get(key, ()))
        candidate.update(
            id=new_id, stage=stage, stages=[stage],
            parent_candidate_id=prior["id"] if prior else None,
            parent_candidate_hash=prior["candidate_hash"] if prior else None,
            parent_family=prior["family"] if prior else None,
            parent_registration_sources=deepcopy(prior["registration_sources"]) if prior else [],
            prior_control_ids=prior_controls,
            previously_seen=bool(prior or prior_controls),
            previously_seen_scope="Parent v10_search registry and its three controls only; not an audit of all older research",
            new_configuration_vs_parent_registry=prior is None,
            registration_sources=[deepcopy(source)],
        )
        unique[key], ids[new_id] = candidate, key
        stages[stage]["first_unique"] += 1

    for ordinal, candidate in enumerate(originals, 1):
        register(candidate, "wide", dict(stage="wide", ordinal=ordinal,
                 parent_candidate_id=candidate["id"], parent_family=candidate["family"]))
    for ordinal, (candidate, coordinates) in enumerate(_focused(base), 1):
        register(candidate, "focused", dict(stage="focused", ordinal=ordinal, coordinates=coordinates))
    candidates = list(unique.values())
    raw_count = sum(stage["raw"] for stage in stages.values())
    if (raw_count != EXPECTED_RAW or len(candidates) != EXPECTED_UNIQUE or len(duplicates) != EXPECTED_OVERLAP
            or stages["wide"]["raw"] != EXPECTED_WIDE or stages["focused"]["raw"] != EXPECTED_FOCUSED):
        raise AssertionError("The fixed wide/focused enumeration changed size")
    for stage in stages:
        stages[stage]["memberships"] = sum(stage in c["stages"] for c in candidates)
    parent_source = BASE.parent / "v10_search" / "registry.py"
    return dict(
        candidates=candidates, raw_count=raw_count, unique_count=len(candidates), duplicates=duplicates,
        stage_counts=stages, family_counts=dict(Counter(c["family"] for c in candidates)),
        execution_spec=deepcopy(EXECUTION_SPEC), channel_definitions=deepcopy(CHANNEL_DEFINITIONS),
        focused_axes=dict(
            stock_anchors=[dict(name=name, codes=list(codes)) for name, codes in STOCK_ANCHORS],
            global_anchors=[dict(name=name, codes=list(codes)) for name, codes in GLOBAL_ANCHORS],
            score_windows=[list(windows) for windows in SCORE_WINDOWS],
            buffers=[dict(name=name, uniform_buffer=buffer) for name, buffer in BUFFERS],
            channel_subsets=[list(channels) for channels in CHANNEL_SUBSETS],
            fixed_regime_mode="ma250", fixed_panic_drop=.04, fixed_bear_enter_mom=.07,
            fixed_overheat=.40, fixed_never_empty=True),
        parent=dict(directory="v10_search", registry_sha256=parent["registry_sha256"],
                    source_sha256=hashlib.sha256(parent_source.read_bytes()).hexdigest(),
                    raw_count=parent["raw_count"], unique_count=parent["unique_count"]),
        frozen_params_sha256=parent["frozen_params_sha256"],
        asset_catalog=deepcopy(parent["asset_catalog"]), role_descriptions=deepcopy(parent["role_descriptions"]),
        regime_descriptions=deepcopy(parent["regime_descriptions"]),
        required_snapshot_codes=deepcopy(parent["required_snapshot_codes"]),
        fixed_input_asset_order=deepcopy(parent["fixed_input_asset_order"]),
        registry_sha256=_digest([c["candidate_hash"] for c in candidates]),
        candidate_id_to_hash={c["id"]: c["candidate_hash"] for c in candidates},
        caveats=[
            "A new hc_ name does not establish a new signal rule: parent ids and identical semantic hashes remain explicit.",
            "All 5,156 wide configurations were already examined under next-open execution; same-close re-ranking is a changed research objective, not unseen evidence.",
            "Focused channel thresholds and ensemble weights are fixed; only the registered channel subsets and axes vary.",
            "External channels remain enabled by channels even when deep is absent and crash_mom5 is zero.",
            "The batch runner must register one common price/QVIX cutoff per comparison; this generator neither chooses nor mixes dates.",
            "Previously_seen=False means absent from the parent registry/controls, not proof of novelty across unlogged older experiments.",
            "A best historical result after searching this family is not a clean OOS result or a future-return guarantee.",
        ])


def controls():
    """Fixed v9 references keep their version names and are not new H versions."""
    out = parent_controls()
    for candidate in out:
        candidate.update(stage="control", stages=["control"], previously_seen=True,
                         new_configuration_vs_parent_registry=candidate["id"] == "c_v92",
                         parent_candidate_id=None, prior_control_ids=[candidate["id"]],
                         reference_only=True)
    return out
