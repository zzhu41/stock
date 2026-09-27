"""Pure two-layer allocation research; no I/O, scan, model choice or live state.

The input target tape belongs to an independently maintained base model. Its
signal/lock state NEVER depends on this allocator's positions. The allocator
owns units of corrected total-return price indices, not raw broker shares.
It calls the frozen run_weights ledger and charges its explicit legacy linear
fee approximation on actual marked security-weight turnover.

Default execution_day tapes ALREADY include the base model's lagged execution;
they must not be shifted again. lag controls the information cutoff for sigma.
time_of_signal tapes explicitly opt into shifting, and must have base_lag=0.
None means skip/preserve units. The 511880 code means a tradable money ETF.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, deque
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from numbers import Integral

from v10_deep.portfolios import CASH, run_weights


ANNUALIZATION = 244
MAIN_TARGET_VOLS = (.15, .20, .25, .30, .40, .50)
# The registered search uses only MAIN_TARGET_VOLS. The extra points are the
# protocol's +/-20% post-selection diagnostic neighborhood, never a new scan.
TARGET_VOLS = tuple(sorted({round(value * scale, 12) for value in MAIN_TARGET_VOLS for scale in (1., .8, 1.2)}))
SPEEDS = (.5, 1. / 3., 1.)
RISK_WINDOWS = {"prior20": 20, "prior60": 60}
MODES = ("full_exposure", "vol_target", "progressive")
TOLERANCE = 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value, name):
    _require(not isinstance(value, bool), name + " cannot be boolean")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(name + " must be a finite number")
    _require(math.isfinite(result), name + " must be finite")
    return result


def _normalize_prices(close_prices, days):
    _require(isinstance(close_prices, dict) and CASH in close_prices, "Prices must include cash ETF 511880")
    result = {}
    for code, values in close_prices.items():
        _require(isinstance(code, str) and code, "Asset codes must be nonempty strings")
        values = tuple(values)
        _require(len(values) == days, "Every price vector must cover the entire calendar")
        normalized = []
        for value in values:
            if value is None:
                normalized.append(None)
                continue
            _require(not isinstance(value, bool), "Price cannot be boolean")
            value = float(value)
            if math.isnan(value):
                normalized.append(None)
                continue
            _require(math.isfinite(value) and value > 0, "Prices must be positive finite values or missing")
            normalized.append(value)
        result[code] = tuple(normalized)
    return result


@lru_cache(maxsize=128)
def _prior_sigma_series(prices, window):
    """Immutable memoized output, strictly before each calendar index.

    Returns span consecutive ACTUAL quotes of the same asset. No forward fill
    creates artificial zero returns on missing sessions. A W-return estimate
    needs W+1 earlier quoted closes; the current signal-day close is excluded.
    """
    history = deque(maxlen=window)
    sigmas, previous_indices, counts = [], [], []
    previous_price = previous_index = None
    for index, price in enumerate(prices):
        counts.append(len(history))
        previous_indices.append(previous_index)
        sigma = None
        if len(history) == window:
            mean = math.fsum(history) / window
            variance = math.fsum((value - mean) ** 2 for value in history) / window
            sigma = math.sqrt(variance * ANNUALIZATION)
            _require(math.isfinite(sigma), "Historical volatility overflow")
        sigmas.append(sigma)
        if price is None:
            continue
        if previous_price is not None:
            daily = price / previous_price - 1
            _require(math.isfinite(daily), "Historical return overflow")
            history.append(daily)
        previous_price, previous_index = price, index
    return tuple(sigmas), tuple(previous_indices), tuple(counts)


def prior_annualized_volatility(close_prices, calendar, risk_context="prior20"):
    """Return dated causal sigma vectors; no target selection or strategy run."""
    dates = tuple(calendar)
    _require(dates and dates == tuple(sorted(set(dates))), "Calendar must be ascending and unique")
    _require(risk_context in RISK_WINDOWS, "Use prior20 or prior60")
    prices = _normalize_prices(close_prices, len(dates))
    return {code: _prior_sigma_series(values, RISK_WINDOWS[risk_context])[0]
            for code, values in prices.items()}


def _interval(dates, start, end):
    _require(not isinstance(start, bool) and not isinstance(end, bool), "Evaluation bounds cannot be boolean")
    lo = int(start) if isinstance(start, Integral) else bisect_left(dates, start)
    hi = int(end) if isinstance(end, Integral) else bisect_right(dates, end) - 1
    _require(0 <= lo <= hi < len(dates), "Invalid evaluation interval")
    return lo, hi


def run_allocation(close_prices, calendar, target_tape, start, end, *, mode="full_exposure",
                   target_vol=None, risk_context="prior20", speed=1., lag=0, fee=.0001,
                   target_alignment="execution_day", provenance=None):
    """Settle an explicit allocation policy against dated TR index prices.

    full_exposure: invest fully in the supplied target.
    vol_target: invest min(1,target_vol/prior_annual_sigma) in the target, the
      remainder in 511880. Incomplete/nonpositive sigma means 100% cash ETF.
    progressive: each day move speed of the distance from ACTUAL marked
      weights to the one-hot target; no panic/crash exception is invented.
      Before first investment, idle capital is represented as a cash-ETF
      anchor for this interpolation, so the remainder buys 511880 explicitly.

    A None target or a time_of_signal index <0 skips the complete rebalance.
    A missing quote on any held/requested positive leg blocks the whole trade
    (even if another leg could trade). Existing units keep carried valuations.
    A blocked proposal is discarded; next day uses that day's tape and drift.

    provenance.base_lag is mandatory: ==lag for execution_day, ==0 for
    time_of_signal. Values are checked before any evaluation. This is a new
    allocation model, not an execution-aware recalculation of the base model.
    """
    _require(mode in MODES, "Unknown allocation mode")
    _require(type(lag) is int and lag in (0, 1), "lag must be 0 or 1")
    _require(target_alignment in ("execution_day", "time_of_signal"), "Unknown target alignment")
    provenance = deepcopy(provenance or {})
    _require(type(provenance.get("base_lag")) is int and provenance["base_lag"] in (0, 1),
             "Explicit provenance.base_lag is required")
    _require(provenance["base_lag"] == (lag if target_alignment == "execution_day" else 0),
             "Target tape lag/alignment mismatch; refuse a double delay")
    speed = _number(speed, "speed")
    _require(any(abs(speed - allowed) <= 1e-15 for allowed in SPEEDS), "Use speed 1/2, 1/3 or 1")
    _require(risk_context in RISK_WINDOWS, "Use prior20 or prior60")
    if mode == "vol_target":
        target_vol = _number(target_vol, "target_vol")
        _require(any(abs(target_vol - allowed) <= 1e-15 for allowed in TARGET_VOLS), "Unregistered volatility target")
        _require(speed == 1., "Combining volatility sizing and gradual execution is not defined here")
    else:
        _require(target_vol is None, "target_vol belongs only to vol_target mode")
        _require(mode == "progressive" or speed == 1., "Full exposure cannot use partial speed")
    fee = _number(fee, "fee")
    _require(0 <= fee < .5, "Fee must lie in [0,.5)")
    dates, targets = tuple(calendar), tuple(target_tape)
    _require(dates and all(isinstance(day, str) for day in dates)
             and dates == tuple(sorted(set(dates))), "Calendar must be ascending unique dated strings")
    _require(len(targets) == len(dates), "Target tape must cover the entire calendar")
    prices = _normalize_prices(close_prices, len(dates))
    _require(all(code is None or isinstance(code, str) and code in prices for code in targets),
             "Target tape contains an unknown code")
    lo, hi = _interval(dates, start, end)
    window = RISK_WINDOWS[risk_context]
    relevant = set(code for code in targets if code is not None and code != CASH)
    risk = {code: _prior_sigma_series(prices[code], window) for code in relevant} if mode == "vol_target" else {}
    records, rejected = [], []
    reasons = Counter()

    def policy(index, state):
        signal_index = index - lag
        target_index = index if target_alignment == "execution_day" else signal_index
        code = targets[target_index] if target_index >= 0 else None
        record = dict(date=dates[index], execution_index=index, signal_index=signal_index,
            signal_date=dates[signal_index] if signal_index >= 0 else None, target_index=target_index,
            base_target=code, marked_weights=dict(state["weights"]), idle_cash_weight=state["cash_weight"],
            sigma_annual=None, sigma_last_quote_date=None, sigma_return_samples=0,
            desired_weights=None, proposed_weights=None, fallback_reason=None)
        records.append(record)
        if code is None:
            record["status"] = "skip_no_signal"
            reasons["skip_no_signal"] += 1
            return None
        fraction = 0. if code == CASH else 1.
        if mode == "vol_target" and code != CASH:
            sigma, last_quote, count = (None, None, 0) if signal_index < 0 else (
                risk[code][0][signal_index], risk[code][1][signal_index], risk[code][2][signal_index])
            record.update(sigma_annual=sigma, sigma_last_quote_date=dates[last_quote] if last_quote is not None else None,
                          sigma_return_samples=count)
            if sigma is None:
                fraction, record["fallback_reason"] = 0., "sigma_warmup_incomplete"
            elif sigma <= 0:
                fraction, record["fallback_reason"] = 0., "sigma_nonpositive"
            else:
                fraction = min(1., target_vol / sigma)
            if record["fallback_reason"]:
                reasons[record["fallback_reason"]] += 1
        desired = {CASH: 1.} if code == CASH or fraction == 0 else {code: fraction}
        if code != CASH and 0 < fraction < 1:
            desired[CASH] = 1. - fraction
        record["desired_weights"] = dict(desired)
        if mode == "progressive":
            marked = dict(state["weights"])
            # This explicitly purchases any idle-cash remainder. It does not
            # treat zero-interest cash as if it had earned the ETF's return.
            if state["cash_weight"] > 0:
                marked[CASH] = marked.get(CASH, 0.) + state["cash_weight"]
            proposed = {asset: (1. - speed) * marked.get(asset, 0.) + speed * desired.get(asset, 0.)
                        for asset in set(marked) | set(desired)}
            proposed = {asset: weight for asset, weight in proposed.items() if weight > 0}
        else:
            proposed = desired
        turnover = math.fsum(abs(proposed.get(asset, 0.) - state["weights"].get(asset, 0.))
                            for asset in set(proposed) | set(state["weights"]))
        record.update(proposed_weights=dict(proposed), requested_turnover=turnover, status="proposed")
        missing = sorted(asset for asset in set(proposed) | set(state["weights"])
                         if state["current_quotes"][asset] is None)
        if turnover > TOLERANCE and missing:
            record.update(status="blocked_missing_quote", missing_codes=missing)
            rejected.append(dict(date=dates[index], target=dict(proposed), missing_codes=missing,
                                 reason="strict_whole_rebalance_missing_leg"))
            reasons["blocked_missing_quote"] += 1
            return None
        return proposed

    result = run_weights(prices, dates, policy, lo, hi, fee=fee)
    engine_rejections = list(result["diagnostics"]["blocked_rebalances"])
    all_rejections = sorted(rejected + engine_rejections, key=lambda row: row["date"])
    traded = {trade["date"] for trade in result["trades"]}
    blocked = {row["date"] for row in all_rejections}
    for record, daily in zip(records, result["daily"]):
        if record["status"] == "proposed":
            record["status"] = "filled" if record["date"] in traded else "blocked_missing_quote" if record["date"] in blocked else "unchanged"
        record["actual_weights_after"] = dict(daily[2])
    first_target_index = lo if target_alignment == "execution_day" else lo - lag
    first_target = targets[first_target_index] if first_target_index >= 0 else None
    first_quoted = first_target is not None and prices[first_target][lo] is not None
    first_fill_free = bool(result["trades"] and result["trades"][0]["initial_session_free"])
    instant = mode == "full_exposure" or mode == "progressive" and speed == 1.
    result["diagnostics"].update(blocked_rebalance_count=len(all_rejections), blocked_rebalances=all_rejections,
        allocation_status_counts=dict(Counter(record["status"] for record in records)),
        sigma_fallback_counts={key: reasons[key] for key in ("sigma_warmup_incomplete", "sigma_nonpositive")},
        skip_no_signal_count=reasons["skip_no_signal"],
        cash_etf_code=CASH, idle_cash_interest_rate=0., risk_residual_target_is_cash_etf=True,
        execution_clock="execution_day_close_with_explicit_target_tape_alignment")
    result["allocation_decisions"] = records
    result["overlay_metadata"] = dict(schema=1, mode=mode, target_vol=target_vol,
        risk_context=risk_context if mode == "vol_target" else None,
        speed=speed if mode == "progressive" else None, lag=lag, target_alignment=target_alignment,
        provenance=provenance, annualization_sessions=ANNUALIZATION, exposure_cap=1.,
        volatility_target_scope="Six main targets plus their predeclared +/-20% diagnostic neighbors; no selection in this helper",
        target_tape_sha256=_hash(targets), prices_sha256=_hash(prices), calendar_sha256=_hash(dates),
        period=[dates[lo], dates[hi]], base_state_independent_of_allocator=True,
        sigma_cutoff="Strictly before signal_day=execution_day-lag, using prior actual asset quote returns, ddof0",
        none_target="Skip this whole rebalance and preserve actual units; no order carryover",
        cash_target="Hold tradable 511880; never substitute zero-interest idle cash",
        initial_progressive_cash="Uninvested startup cash is mapped to the cash-ETF anchor in desired interpolation; any purchase requires its quote",
        strict_missing_leg_blocks_whole_trade=True, panic_or_crash_execution_exception=False,
        first_evaluation_session_free=True, delayed_first_fill_uses_true_security_l1_fee=True,
        first_session_base_target_quoted=bool(first_quoted), first_successful_fill_was_free=first_fill_free,
        native_baseline_fidelity_check_eligible=bool(instant and first_quoted and first_fill_free),
        native_fidelity_not_asserted_here=True,
        startup_fee_limitation="If the first session cannot invest, a later first purchase from idle cash has L1=1; the old native model charges 2*fee. Do not claim native exactness or choose a different start date to hide this boundary.",
        units="Corrected TR-index units, not raw broker-share/dividend accounting",
        cost_model="Existing run_weights approximation: preNAV*(1-fee*L1(target security weights minus actual drift weights))",
        candidate_selection_performed=False, live_account=False)
    return result
