"""Pure, conditional inverse of vendor affine adjustment; no file/network I/O.

Input rows are (date, raw_or_qfq_open, raw_or_qfq_close, volume). This module
does NOT assert that a chart vendor's inferred cash is an issuer-confirmed cash
distribution. Exact split ratios and a complete event list must be supplied.
The inference path rejects affine inconsistencies and unexplained negative
offset changes. A complete caller-verified cash calendar instead reconstructs
from raw prices/actions directly; qfq then serves only as a disclosed diagnostic.

Output open equals total-return close solely as an explicit placeholder for
the existing close-only engine. It MUST NOT be used for opening-price fills.
Cash is assumed accrued on the ex-date and immediately/freely reinvested at
the observed close; real payment dates, taxes and cash-reinvestment fees are
not simulated. Original snapshots and registered strategies remain untouched.
"""
from decimal import Decimal
import math


NUMERICAL_TOLERANCE = 1e-10


def _rows(rows, adjusted=False):
    out = []
    for row in rows:
        if len(row) < 4:
            raise ValueError("Rows must contain date, open, close and volume")
        date, opening, close, volume = row[0], float(row[1]), float(row[2]), float(row[3])
        if not all(math.isfinite(x) for x in (opening, close, volume)) or volume < 0:
            raise ValueError("Nonfinite price or invalid volume: " + date)
        if not adjusted and (opening <= 0 or close <= 0):
            raise ValueError("Raw prices must be positive: " + date)
        out.append((date, opening, close, volume))
    if not out or [r[0] for r in out] != sorted({r[0] for r in out}):
        raise ValueError("Price rows must be nonempty, unique and ascending")
    return out


def adjustment_scales(dates, split_events=()):
    """a_t=1/product(shares ratios strictly after t), using exact decimal ratios.

    Each event date is the first quoted bar in the new units, not necessarily
    its legal ex-date when trading was suspended. Ratios are new/old shares.
    """
    known, supplied = {}, []
    date_set = set(dates)
    for event in split_events:
        date = event["date"]
        if date not in date_set:
            raise ValueError("Split event is not a quoted first-new-unit bar: " + date)
        if date in known:
            raise ValueError("Duplicate split date: " + date)
        ratio = Decimal(str(event.get("ratio_exact", event["ratio"])))
        if not ratio.is_finite() or ratio <= 0:
            raise ValueError("Invalid split ratio: " + date)
        if not math.isclose(float(ratio), float(event["ratio"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Exact and numeric split ratios disagree: " + date)
        known[date] = ratio
        supplied.append(dict(date=date, ratio=float(ratio), ratio_exact=str(ratio)))
    future = Decimal(1)
    scales = [0.0] * len(dates)
    for i in range(len(dates) - 1, -1, -1):
        scales[i] = float(Decimal(1) / future)
        if dates[i] in known:
            future *= known[dates[i]]
    return scales, sorted(supplied, key=lambda x: x["date"])


def affine_step(previous_raw_close, raw_close, previous_q_close, q_close,
                previous_scale=1.0, scale=1.0):
    """Exact algebra for unrounded Q=aP+b; no statistical cash-event detection."""
    if min(previous_raw_close, raw_close, previous_scale, scale) <= 0:
        raise ValueError("Raw prices and affine scales must be positive")
    old_b = previous_q_close - previous_scale * previous_raw_close
    b = q_close - scale * raw_close
    split = scale / previous_scale
    dividend = (b - old_b) / previous_scale
    gross = (split * raw_close + dividend) / previous_raw_close
    return dict(split_ratio=split, cash_per_old_share=dividend, gross=gross,
                difference_formula_gross=1 + (q_close - previous_q_close) / (previous_scale * previous_raw_close))


def _from_cash_calendar(raw, adjusted, scales, supplied_splits, cash_calendar, code, q_quantum, base):
    """Trust a caller-verified complete action table, never fit actions to qfq."""
    if isinstance(cash_calendar, dict):
        records = [dict(date=date, cash_per_old_share=value) for date, value in cash_calendar.items()]
    else:
        records = list(cash_calendar)
    dates, by_date = [r[0] for r in raw], {}
    for event in records:
        date = event["date"]
        if date not in dates[1:] or date in by_date:
            raise ValueError("Cash date must be a unique quoted event after the first row: " + date)
        cash = float(event["cash_per_old_share"])
        if not math.isfinite(cash) or cash < 0:
            raise ValueError("Cash per previous-quote share must be finite and nonnegative")
        by_date[date] = dict(event, cash_per_old_share=cash)
    cash = [by_date.get(date, {}).get("cash_per_old_share", 0.0) for date in dates]
    # Anchor the vendor intercept at the last observation, then predict earlier
    # quotes from the declared official actions. Never adjust action values to
    # make this residual vanish.
    b = [0.0] * len(raw)
    b[-1] = adjusted[-1][2] - scales[-1] * raw[-1][2]
    for i in range(len(raw) - 1, 0, -1):
        b[i - 1] = b[i] - scales[i - 1] * cash[i]
    rows, events, actions, residuals = [], [], [], []
    value, maximum = base, 0.0
    for i, (r, q, a) in enumerate(zip(raw, adjusted, scales)):
        quote_error = max(abs(q[1] - (a * r[1] + b[i])), abs(q[2] - (a * r[2] + b[i])))
        maximum = max(maximum, quote_error)
        if quote_error > q_quantum / 2 + NUMERICAL_TOLERANCE:
            residuals.append(dict(date=r[0], maximum_quote_error=quote_error))
        if i:
            split = scales[i] / scales[i - 1]
            gross = (split * r[2] + cash[i]) / raw[i - 1][2]
            if not math.isfinite(gross) or gross <= 0:
                raise ValueError("Invalid official-action total return: " + r[0])
            value *= gross
            if cash[i]:
                event = dict(by_date[r[0]], previous_quote_date=raw[i - 1][0],
                             lower=cash[i], upper=cash[i], old_unit_scale=scales[i - 1],
                             cash_yield=cash[i] / raw[i - 1][2],
                             independently_verified=True, provenance="caller_supplied_complete_official_cash_calendar")
                events.append(event)
            if cash[i] or abs(split - 1) > NUMERICAL_TOLERANCE:
                actions.append(dict(date=r[0], split_ratio=split, cash_per_old_share=cash[i], gross=gross))
        rows.append((r[0], value, value, r[3] / a))
    diagnostics = dict(
        code=code, method="raw_prices_plus_caller_verified_split_and_cash_events",
        split_events=supplied_splits, inferred_cash_event_count=0, supplied_cash_event_count=len(events),
        supplied_cash_sum=sum(cash), actions=actions, q_quantum=q_quantum,
        max_vendor_quote_residual=maximum, vendor_half_tick_residual_count=len(residuals),
        vendor_half_tick_residuals=residuals,
        vendor_model_validated_at_half_tick=not residuals,
        suppressed_rounding_noise_days=None, initial_index=base, final_index=value,
        conditional_lower_final_index=value, conditional_upper_final_index=value,
        cash_events_independently_verified=True,
        verification_responsibility="Caller supplies a complete independently checked event table; this pure function does not verify issuers or source links.",
        large_unverified_cash_events=[], open_policy="placeholder_equals_total_return_close",
        usable_for_next_open=False,
        volume_policy="raw_volume_times_product_of_future_new_to_old_share_ratios",
        reinvestment_assumption="ex-date accrual and immediate free reinvestment at each observed close",
        scope="Corporate actions determine returns. qfq disagreements remain diagnostic and are never converted into additional cash or fitted ratios.",
    )
    return dict(rows=rows, cash_events=events, diagnostics=diagnostics)


def reconstruct_total_return(raw_rows, qfq_rows, split_events=(), *, code=None,
                             q_quantum=.001, base=1.0, cash_events=None):
    """Return rows, cash_events and diagnostics; never choose strategy parameters.

    With no supplied splits, exact observed unit-slope offsets are required;
    b changes directly reproduce the source's additive cash adjustment.
    With splits, each O/C pair defines a b interval of half a quote tick. Stable
    intersecting intervals mean zero cash, preventing daily rounding noise from
    becoming dividends. A disjoint upward interval identifies a cash jump; its
    midpoint and bounds use only observations through that date. Sub-tick
    ambiguous jumps, negative jumps, and incompatible scales stop construction.

    Bounds are conditional on the supplied split list and detected event dates.
    They are NOT a guarantee that sub-resolution/omitted corporate actions do
    not exist; positive inferred cash still needs independent announcement checks.

    cash_events, if supplied, is a COMPLETE caller-verified cash calendar: a
    date->amount mapping or dicts with date/cash_per_old_share and optional
    source metadata. Amounts use the previous quoted share units. This switches
    to raw-price/action reconstruction; no cash is inferred from qfq and its
    rounding/scale residuals are reported without relaxing the inference path.
    """
    raw, adjusted = _rows(raw_rows), _rows(qfq_rows, adjusted=True)
    if [r[0] for r in raw] != [r[0] for r in adjusted]:
        raise ValueError("Raw and qfq date axes must match exactly; no silent inner join")
    if not math.isfinite(q_quantum) or q_quantum <= 0 or not math.isfinite(base) or base <= 0:
        raise ValueError("Quote quantum and initial index must be positive")
    dates = [r[0] for r in raw]
    a, supplied = adjustment_scales(dates, split_events)
    if cash_events is not None:
        return _from_cash_calendar(raw, adjusted, a, supplied, cash_events, code, q_quantum, base)
    no_split = not supplied
    eps, tol = q_quantum / 2, NUMERICAL_TOLERANCE
    offsets, intervals, max_intraday_residual = [], [], 0.0
    for r, q, scale in zip(raw, adjusted, a):
        bo, bc = q[1] - scale * r[1], q[2] - scale * r[2]
        residual = abs(bo - bc)
        max_intraday_residual = max(max_intraday_residual, residual)
        allowed = tol if no_split else q_quantum + tol
        if residual > allowed:
            raise ValueError("Declared split scale cannot explain raw/qfq open and close: %s %s residual=%g" %
                             (code or "", r[0], residual))
        low, high = max(bo, bc) - eps, min(bo, bc) + eps
        if low > high:
            low = high = (low + high) / 2  # numerical-only empty intersection, bounded above
        offsets.append(bc)
        intervals.append((low, high))

    result, cash_events, action_rows = [], [], []
    value, lower_value, upper_value = base, base, base
    feasible = intervals[0]
    noise_days = 0
    large_cash_events = []
    for i, (r, q) in enumerate(zip(raw, adjusted)):
        cash = cash_low = cash_high = 0.0
        split = a[i] / a[i - 1] if i else 1.0
        if i:
            if no_split:
                jump = offsets[i] - offsets[i - 1]
                if jump < -tol:
                    raise ValueError("Unexplained negative cash adjustment: %s %s %g" % (code or "", r[0], jump))
                if jump > tol:
                    cash = jump
                    cash_low, cash_high = max(0.0, cash - q_quantum), cash + q_quantum
            else:
                previous_low, previous_high = feasible
                current_low, current_high = intervals[i]
                low, high = max(previous_low, current_low), min(previous_high, current_high)
                if low <= high + tol:
                    feasible = (low, high) if low <= high else ((low + high) / 2,) * 2
                    if abs(offsets[i] - offsets[i - 1]) > tol:
                        noise_days += 1
                elif current_high < previous_low:
                    raise ValueError("Unexplained negative/scale adjustment after rounding bounds: %s %s" %
                                     (code or "", r[0]))
                else:
                    jump = (current_low + current_high - previous_low - previous_high) / 2
                    if jump < q_quantum - tol:
                        raise ValueError("Ambiguous sub-tick cash/scale change requires an official event: %s %s" %
                                         (code or "", r[0]))
                    cash = jump / a[i - 1]
                    cash_low = max(0.0, (current_low - previous_high) / a[i - 1])
                    cash_high = (current_high - previous_low) / a[i - 1]
                    feasible = intervals[i]
            previous = raw[i - 1][2]
            gross = (split * r[2] + cash) / previous
            if not math.isfinite(gross) or gross <= 0:
                raise ValueError("Invalid reconstructed total return: " + r[0])
            value *= gross
            lower_value *= (split * r[2] + cash_low) / previous
            upper_value *= (split * r[2] + cash_high) / previous
            if cash:
                event = dict(date=r[0], previous_quote_date=raw[i - 1][0], cash_per_old_share=cash,
                             lower=cash_low, upper=cash_high, old_unit_scale=a[i - 1],
                             cash_yield=cash / previous, independently_verified=False,
                             provenance="inferred_from_vendor_affine_adjustment")
                cash_events.append(event)
                if event["cash_yield"] >= .20:
                    large_cash_events.append(event)
            if cash or abs(split - 1) > tol:
                action_rows.append(dict(date=r[0], split_ratio=split, cash_per_old_share=cash, gross=gross))
        # Convert historical traded shares into the latest unit convention.
        # Prices here are close-index placeholders, NOT adjusted tradable opens.
        result.append((r[0], value, value, r[3] / a[i]))
    diagnostics = dict(
        code=code, method="additive_exact_observed_offset" if no_split else "official_splits_with_affine_rounding_intervals",
        split_events=supplied, inferred_cash_event_count=len(cash_events), actions=action_rows,
        max_intraday_affine_residual=max_intraday_residual, q_quantum=q_quantum,
        suppressed_rounding_noise_days=noise_days, initial_index=base, final_index=value,
        conditional_lower_final_index=lower_value, conditional_upper_final_index=upper_value,
        cash_events_independently_verified=False, large_unverified_cash_events=large_cash_events,
        open_policy="placeholder_equals_total_return_close", usable_for_next_open=False,
        volume_policy="raw_volume_times_product_of_future_new_to_old_share_ratios",
        reinvestment_assumption="ex-date accrual and immediate free reinvestment at each observed close",
        scope="Conditional vendor reconstruction; verify inferred cash and complete action coverage independently before asserting validated market total returns.",
    )
    return dict(rows=result, cash_events=cash_events, diagnostics=diagnostics)
