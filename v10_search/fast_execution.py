"""Fast causal next-open execution for full-allocation, single-asset policies.

Prepare the immutable price frame once, then reuse it for many policies:

    frame = prepare(histories, calendar)
    result = run(frame, policy, start, end)

policy(global_index, actual_holding, actual_entry_index, execution_deferred)
returns an asset code, None (keep units and any deferred order), or FLAT (sell
to uninvested, zero-interest cash). A money-market ETF is an ordinary asset.
There is no fractional allocation or leverage. Costs are charged on actual
execution notional, using the same convention as v10_next.execution.run.
"""
import bisect
import math
from numbers import Integral
from types import MappingProxyType


FLAT = ""
TRADING_DAYS = 244


class PriceFrame:
    __slots__ = ("dates", "index", "years", "codes", "opens", "closes", "coverage_end")

    def __init__(self, histories, calendar):
        dates = tuple(calendar)
        if not dates or tuple(sorted(set(dates))) != dates:
            raise ValueError("Calendar must be nonempty, unique and ascending")
        index = {d: i for i, d in enumerate(dates)}
        opens, closes, last_dates = {}, {}, []
        for code, rows in histories.items():
            if not isinstance(code, str) or not code:
                raise ValueError("Asset codes must be nonempty strings")
            if not rows:
                raise ValueError("Empty asset history: " + code)
            if any(len(row) < 3 for row in rows):
                raise ValueError("History requires dated open and close prices")
            ds = [r[0] for r in rows]
            if sorted(set(ds)) != ds:
                raise ValueError("History dates must be unique and ascending: " + code)
            op, cl = [None] * len(dates), [None] * len(dates)
            for row in rows:
                if len(row) < 3:
                    raise ValueError("History requires dated open and close prices")
                opening, closing = float(row[1]), float(row[2])
                if not math.isfinite(opening) or opening <= 0 or not math.isfinite(closing) or closing <= 0:
                    raise ValueError("Invalid price: %s %s" % (code, row[0]))
                i = index.get(row[0])
                if i is not None:
                    op[i], cl[i] = opening, closing
            opens[code], closes[code] = tuple(op), tuple(cl)
            last_dates.append(ds[-1])
        if not opens:
            raise ValueError("At least one asset history is required")
        self.dates, self.index = dates, MappingProxyType(index)
        self.years = tuple(d[:4] for d in dates)
        self.codes = frozenset(opens)
        self.opens, self.closes = MappingProxyType(opens), MappingProxyType(closes)
        self.coverage_end = min(last_dates)


def prepare(histories, calendar):
    """Copy/validate local prices once; source histories are never mutated."""
    return PriceFrame(histories, calendar)


def _bounds(frame, start, end):
    low = int(start) if isinstance(start, Integral) else bisect.bisect_left(frame.dates, start)
    high = int(end) if isinstance(end, Integral) else bisect.bisect_right(frame.dates, end) - 1
    if low < 0 or high >= len(frame.dates) or low > high:
        raise ValueError("No valid trading interval")
    if frame.dates[high] > frame.coverage_end:
        raise ValueError("Requested tail exceeds historical coverage")
    return low, high


def run(frame, policy, start, end, fee=.0001, slippage=.001,
        capture_daily=False, capture_trades=False):
    """Return streaming metrics by default; request trajectories only as needed.

    start/end accept dates or inclusive global calendar indices. Policy indices
    always address the complete prepared calendar, including pre-start history.
    capture_daily='arrays' returns only navs/holdings, avoiding daily dictionaries.
    Full True output matches the generic engine's (date, nav, weights) shape.
    Uncaptured detail fields are None, not falsely represented as zero events.
    """
    fee, slippage = float(fee), float(slippage)
    if any(not math.isfinite(x) or not 0 <= x < 1 for x in (fee, slippage)):
        raise ValueError("Invalid commission or slippage")
    if capture_daily not in (False, True, "arrays"):
        raise ValueError("capture_daily must be False, True or 'arrays'")
    full_daily = capture_daily is True
    capture_arrays = bool(capture_daily)
    low, high = _bounds(frame, start, end)
    openings, closings, dates = frame.opens, frame.closes, frame.dates
    buy_factor = (1 + slippage) * (1 + fee)
    cash, units, mark = 1.0, 0.0, None
    holding, entry_index, pending, pending_index = None, None, None, None
    filled_target = None
    trade_count = fill_count = deferred_count = missing_held_days = noops = 0
    commission_total = slippage_total = turnover_total = 0.0
    peak, max_dd, previous_nav = 1.0, 0.0, 1.0
    return_count, mean, m2 = 0, 0.0, 0.0
    daily, signals = ([], []) if full_daily else (None, None)
    navs, holdings = ([], []) if capture_arrays else (None, None)
    trades = [] if capture_trades else None
    deferred_details, missing_details = ([], []) if full_daily else (None, None)
    yearly, year, year_start_nav = {}, frame.years[low], 1.0

    for i in range(low, high + 1):
        holding_available = holding is None or openings[holding][i] is not None
        if not holding_available:
            missing_held_days += 1
            if full_daily:
                missing_details.append(dict(date=dates[i], codes=[holding]))
        deferred = False
        if pending is not None:
            target = None if pending == FLAT else pending
            target_available = target is None or openings[target][i] is not None
            if not holding_available or not target_available:
                deferred = True
                deferred_count += 1
                if full_daily:
                    missing = sorted(set(([holding] if not holding_available else [])
                                         + ([target] if not target_available else [])))
                    deferred_details.append(dict(date=dates[i], signal_date=dates[pending_index],
                                                  target={target: 1.0} if target else {}, missing_codes=missing))
            else:
                if target == holding:
                    noops += 1
                else:
                    before = cash + (units * openings[holding][i] if holding else 0.0)
                    fills = [] if capture_trades else None
                    old_holding = holding
                    reference_turnover = commission = slip_cost = 0.0
                    if holding is not None:
                        reference = units * openings[holding][i]
                        execution_price = openings[holding][i] * (1 - slippage)
                        gross = units * execution_price
                        sell_commission = gross * fee
                        proceeds = gross - sell_commission
                        cash += proceeds
                        commission += sell_commission
                        slip_cost += units * abs(execution_price - openings[holding][i])
                        reference_turnover += reference
                        fill_count += 1
                        if capture_trades:
                            fills.append(dict(code=holding, side="sell", units=units,
                                              reference_open=openings[holding][i], execution_price=execution_price,
                                              gross_notional=gross, commission=sell_commission,
                                              cash_flow=proceeds,
                                              slippage_cost=units * abs(execution_price - openings[holding][i])))
                        units, holding, entry_index = 0.0, None, None
                    if target is not None:
                        opening = openings[target][i]
                        units = cash / (opening * buy_factor)
                        execution_price = opening * (1 + slippage)
                        gross = units * execution_price
                        buy_commission = gross * fee
                        spent = gross + buy_commission
                        # The exact full-allocation solution spends all cash;
                        # eliminate floating residuals rather than borrowing.
                        if spent > cash + max(cash, 1.0) * 1e-10:
                            raise ArithmeticError("Full-allocation buy would borrow cash")
                        cash = 0.0
                        commission += buy_commission
                        slip_cost += units * abs(execution_price - opening)
                        reference_turnover += units * opening
                        fill_count += 1
                        if capture_trades:
                            fills.append(dict(code=target, side="buy", units=units,
                                              reference_open=opening, execution_price=execution_price,
                                              gross_notional=gross, commission=buy_commission,
                                              cash_flow=-spent, slippage_cost=units * abs(execution_price - opening)))
                        holding, entry_index = target, i
                    trade_count += 1
                    commission_total += commission
                    slippage_total += slip_cost
                    turnover = reference_turnover / before
                    turnover_total += turnover
                    if capture_trades:
                        after = cash + (units * openings[holding][i] if holding else 0.0)
                        trades.append(dict(date=dates[i], signal_date=dates[pending_index],
                                           target={target: 1.0} if target else {},
                                           fills=fills, nav_before=before, nav_after=after, cash_after=cash,
                                           commission=commission, slippage_cost=slip_cost, turnover=turnover,
                                           weights_after={holding: units * openings[holding][i] / after} if holding else {},
                                           from_holding=old_holding, to_holding=holding))
                filled_target = pending
                pending, pending_index = None, None

        if holding is not None:
            close = closings[holding][i]
            if close is not None:
                mark = close
            elif mark is None:
                raise ArithmeticError("Held asset has never had an observed valuation")
            nav = cash + units * mark
        else:
            mark = None
            nav = cash
        if not math.isfinite(nav) or nav <= 0:
            raise ArithmeticError("NAV is not finite and positive")
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1)
        if i > low:
            daily_return = nav / previous_nav - 1
            return_count += 1
            delta = daily_return - mean
            mean += delta / return_count
            m2 += delta * (daily_return - mean)
        if frame.years[i] != year:
            yearly[year] = previous_nav / year_start_nav - 1
            year_start_nav, year = previous_nav, frame.years[i]
        previous_nav = nav

        instruction = policy(i, holding, entry_index, deferred)
        if instruction is not None:
            if not isinstance(instruction, str) or (instruction != FLAT and instruction not in frame.codes):
                raise ValueError("Policy must return a known asset code, FLAT, or None")
            pending, pending_index = instruction, i
        if capture_arrays:
            navs.append(nav)
            holdings.append(holding)
        if full_daily:
            weights = {holding: units * mark / nav} if holding else {}
            daily.append((dates[i], nav, weights))
            signals.append((dates[i], ({instruction: 1.0} if instruction else {})
                            if instruction is not None else None))

    yearly[year] = nav / year_start_nav - 1
    sessions = high - low + 1
    std = math.sqrt(max(0.0, m2 / return_count)) if return_count else 0.0
    diagnostics = dict(deferred_count=deferred_count, deferred_rebalances=deferred_details,
                       missing_held_days=missing_held_days, missing_held_bars=missing_details,
                       total_commission=commission_total, total_slippage_cost=slippage_total,
                       total_turnover=turnover_total, completed_noop_orders=noops,
                       rebalance_count=trade_count, fill_count=fill_count, policy_calls=sessions,
                       cash_interest_rate=0.0)
    return dict(nav=nav, ann=nav ** (TRADING_DAYS / sessions) - 1, max_dd=max_dd,
                sharpe=mean / std * math.sqrt(TRADING_DAYS) if std else 0.0,
                yearly=yearly, sessions=sessions, trade_count=trade_count,
                fill_count=fill_count, deferred_count=deferred_count,
                daily=daily, navs=navs, holdings=holdings, signal_target=signals, trades=trades,
                diagnostics=diagnostics,
                final_state=dict(date=dates[high], nav=nav, cash=cash,
                                 units={holding: units} if holding else {},
                                 weights={holding: 1.0} if holding else {},
                                 holding_since={holding: dates[entry_index]} if holding else {},
                                 holding=holding, entry_index=entry_index,
                                 pending_target=({pending: 1.0} if pending else {}) if pending is not None else None,
                                 pending_signal_date=dates[pending_index] if pending_index is not None else None,
                                 filled_target=({filled_target: 1.0} if filled_target else {}) if filled_target is not None else None))
