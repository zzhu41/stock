# -*- coding: utf-8 -*-
"""Ten fixed research candidates; no production imports, files or network I/O.

Factories receive isolated legacy logic and a past-only momentum reader. This
module never changes the original v9/v9.1/v9.2 files or their trading state.
The registry is a bounded hypothesis list, not a grid or a return optimizer.
"""
from copy import deepcopy
import math


CASH = "511880"
RESEARCH_START = "2014-01-01"
RESEARCH_END = "2026-09-11"  # Last common local date, including the government-bond ETF.
ROBUST_POOL = ("510300", "510500", "513100", "518880", "511010")
LEGACY_STOCK_POOL = ("159915", "588080", "510300", "510500", "563300", "512400", "512890")
LEGACY_GLOBAL_POOL = ("513100", "513120")
CORE_STOCK_POOL = ("159915", "510300", "510500")
CORE_GLOBAL_POOL = ("513100",)

PROTOCOL = {
    "candidate_budget": 10,
    "registered_before_evaluation": True,
    "research_start": RESEARCH_START,
    "research_end": RESEARCH_END,
    "signal_timing": "observed trading-day close; fill at next available trading-day open",
    "monthly_timing": "first trading session of the month, detected from the preceding calendar session",
    "momentum_unit": "244 observed trading bars per year; 61/122/244 for multi-horizon",
    "momentum_gate": "strictly positive mean cumulative return; all requested horizons required",
    "cash_asset": CASH,
    "missing_cash": "retain any selected risk weights; unassigned residual stays uninvested cash",
    "unavailable_asset": "ineligible until all lookbacks and the current-day bar are available",
    "monthly_between_rebalances": "return None; preserve units, with no daily equal-weight reset",
    "monthly_start_between_rebalances": "wait until the next month; no special entry on a mid-month run start",
    "limits": [
        "All historical data were already inspected during earlier research; no historical slice is a clean holdout.",
        "Do not expand the candidate list or tune thresholds after inspecting outcomes.",
        "High-return and robust labels describe hypotheses, not established performance.",
        "The half-exposure control tests exposure reduction, not reduced overfitting.",
        "The robust policies have no crisis dip-buy, bull/bear gate, panic exit or overheat rule.",
        "A score-window ensemble uses the fixed arithmetic mean of scores, not an average of ranks.",
    ],
}


def _legacy(candidate_id, family, base_version, hypothesis, overrides=None,
            score_windows=(25,), stock_pool=LEGACY_STOCK_POOL,
            global_pool=LEGACY_GLOBAL_POOL, risk_weight=1.0):
    return dict(
        id=candidate_id, family=family, kind="legacy", base_version=base_version,
        hypothesis=hypothesis, overrides=dict(overrides or {}),
        score_windows=tuple(score_windows), score_combination="arithmetic_mean",
        stock_pool=tuple(stock_pool), global_pool=tuple(global_pool),
        gold="518880", cash=CASH, risk_weight=risk_weight,
        rebalance="daily_signal", added_rules=(),
    )


def _monthly(candidate_id, hypothesis, lookbacks, selection, slot_weight):
    # Horizons are observed trading-bar lags, not calendar days: a 244-bar
    # return needs 245 closing-price observations including the signal day.
    return dict(
        id=candidate_id, family="robust", kind="monthly_momentum",
        hypothesis=hypothesis, pool=ROBUST_POOL, lookbacks=tuple(lookbacks),
        selection=selection, slot_weight=slot_weight,
        score_combination="arithmetic_mean_cumulative_return",
        positive_gate=0.0, tie_break="ascending_asset_code", cash=CASH,
        rebalance="first_trading_session_monthly_close",
        missing_slots="cash", intra_month_rebalance=False, added_rules=(),
    )


_CANDIDATES = (
    _legacy("control_v91", "control", "v9.1",
            "Full-exposure reference with the existing v9.1 parameters."),
    _legacy("control_v9", "control", "v9",
            "Existing v9 reference with the uniform 2% rotation buffer."),
    _legacy("control_v91_half", "control", "v9.1",
            "Same v9.1 signal with half risk exposure and half money-market ETF; an exposure control only.",
            risk_weight=.5),
    _legacy("h_no_crash", "high_return", "v9.1",
            "Remove the rare crisis-event rule and its lock to test reliance on a small event sample.",
            overrides={"crash_mom5": 0.0}),
    _legacy("h_no_bear_gate", "high_return", "v9.1",
            "Remove only the extra 7% bearish-entry momentum threshold; retain the domestic-equity bear exclusion.",
            overrides={"bear_enter_mom": 0.0}),
    _legacy("h_ensemble", "high_return", "v9.1",
            "Average fixed WLS20/25/30 scores to reduce dependence on a single selected ranking window.",
            score_windows=(20, 25, 30)),
    _legacy("h_core_pool", "high_return", "v9.1",
            "Restrict the inherited strategy to older broad equity ETFs, Nasdaq and gold; no new thematic asset.",
            stock_pool=CORE_STOCK_POOL, global_pool=CORE_GLOBAL_POOL),
    _monthly("r_m12_top2",
             "Use one annual trend horizon and monthly top-two diversification instead of daily short-term ranking.",
             (244,), "top2", .5),
    _monthly("r_multi_top2",
             "Average fixed 3/6/12-month returns before the same positive-trend gate and monthly top-two allocation.",
             (61, 122, 244), "top2", .5),
    _monthly("r_m12_broad",
             "Give each of five cross-asset sleeves a fixed fifth when its annual trend is positive; avoid winner ranking.",
             (244,), "broad", .2),
)

# A public snapshot for JSON registration. Factories always use private copies,
# so a runner annotating this snapshot cannot change subsequent policies.
CANDIDATES = tuple(deepcopy(item) for item in _CANDIDATES)


def get_candidate(candidate_id):
    for candidate in _CANDIDATES:
        if candidate["id"] == candidate_id:
            return deepcopy(candidate)
    raise KeyError("Unknown preregistered candidate: " + str(candidate_id))


def registry():
    """Return a serializable independent registration snapshot."""
    return dict(protocol=deepcopy(PROTOCOL), candidates=[deepcopy(c) for c in _CANDIDATES])


def build_policy(candidate_id, legacy_factory, momentum_reader):
    """Build one of the ten registered policies without mutating its config."""
    return build_from_config(get_candidate(candidate_id), legacy_factory, momentum_reader)


def build_from_config(candidate, legacy_factory, momentum_reader):
    """Build policy(date, observed_histories, execution_state) -> weights or None.

    Copy the supplied configuration so protocol-approved window diagnostics do
    not mutate the registry or the caller's config. Preserve its metadata id.
    legacy_factory(candidate_dict) must implement the isolated legacy strategy.
    momentum_reader(date, code, lookbacks) returns None or a dictionary with
    score, returns (integer lookback -> cumulative return), and available=True.
    It must require an exact current-day bar and max(lookbacks)+1 observations.
    momentum_reader.cash_available(date) reports availability of the cash ETF.

    The monthly policy records decisions in its in-memory .metadata. It only
    examines the date supplied by the executor and past-only reader; the full
    observed_histories prefix is accepted for the common executor signature.
    """
    candidate = deepcopy(candidate)
    candidate_id = candidate["id"]
    if candidate["kind"] == "legacy":
        policy = legacy_factory(candidate)
        if not callable(policy):
            raise TypeError("legacy_factory must return a callable policy")
        return policy
    if candidate["kind"] != "monthly_momentum":
        raise ValueError("Unknown policy kind: " + str(candidate["kind"]))
    if not callable(momentum_reader) or not callable(getattr(momentum_reader, "cash_available", None)):
        raise TypeError("momentum_reader and momentum_reader.cash_available must be callable")

    metadata = dict(candidate_id=candidate_id, candidate=deepcopy(candidate),
                    rebalance_events=[], missing_cash_dates=[])

    def policy(date, observed_histories, state):
        previous_date = state.get("previous_date")
        if previous_date is not None and previous_date >= date:
            raise ValueError("previous_date must precede the current trading date")
        if previous_date is not None and previous_date[:7] == date[:7]:
            return None

        scored, unavailable = [], []
        for code in candidate["pool"]:
            observation = momentum_reader(date, code, candidate["lookbacks"])
            if observation is None or not observation.get("available", False):
                unavailable.append(code)
                continue
            returns = observation.get("returns", {})
            if any(horizon not in returns for horizon in candidate["lookbacks"]):
                raise ValueError("Momentum reader omitted a required horizon for " + code)
            values = [returns[horizon] for horizon in candidate["lookbacks"]]
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                raise ValueError("Non-finite momentum return for " + code)
            # Compute the registered score explicitly rather than accepting an
            # accidental reader rank, volatility scaling or horizon reweighting.
            score = sum(values) / len(values)
            provided_score = observation.get("score")
            if not isinstance(provided_score, (int, float)) or not math.isfinite(provided_score):
                raise ValueError("Momentum reader score is missing or non-finite for " + code)
            if abs(provided_score - score) > 1e-12 * max(1.0, abs(score)):
                raise ValueError("Momentum reader score differs from fixed arithmetic mean for " + code)
            if score > candidate["positive_gate"]:
                scored.append((code, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        selected = scored[:2] if candidate["selection"] == "top2" else scored
        weights = {code: candidate["slot_weight"] for code, _ in selected}
        residual = max(0.0, round(1.0 - sum(weights.values()), 12))
        cash_available = None
        if residual:
            cash_available = bool(momentum_reader.cash_available(date))
            if cash_available:
                weights[candidate["cash"]] = residual
            else:
                metadata["missing_cash_dates"].append(date)
        metadata["rebalance_events"].append(dict(
            date=date, previous_date=previous_date, scores=dict(scored),
            unavailable_assets=unavailable, target=weights.copy(),
            uninvested_cash_weight=residual if residual and not cash_available else 0.0,
            cash_available=cash_available,
        ))
        return weights

    policy.metadata = metadata
    return policy


if len(_CANDIDATES) != PROTOCOL["candidate_budget"] or len({c["id"] for c in _CANDIDATES}) != len(_CANDIDATES):
    raise AssertionError("Candidate registry must contain exactly ten distinct hypotheses")
