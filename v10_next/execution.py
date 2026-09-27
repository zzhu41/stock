"""Pure, dated multi-asset execution simulator for the isolated v10 research.

At each open the engine attempts the most recent pending closing instruction.
It then marks actual units at that day's close and invokes the policy with the
actual portfolio. A policy returns target weights, {} to sell to cash, or None
to keep units (and retain any deferred instruction). A newer weight instruction
replaces an older deferred instruction.

Prices are a fixed, internally consistent adjusted-history snapshot, not live
quotes or a brokerage ledger. Cash pays zero interest. Order-book depth, limits
and market impact beyond the requested slippage are not reconstructed.
"""
import bisect
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType


class _HistoryPrefix(Sequence):
    """Read-only history prefix without copying every earlier bar each day."""
    __slots__ = ("_rows", "_stop")

    def __init__(self, rows, stop):
        self._rows, self._stop = rows, stop

    def __len__(self):
        return self._stop

    def __getitem__(self, item):
        if isinstance(item, slice):
            return tuple(self._rows[i] for i in range(*item.indices(self._stop)))
        index = item if item >= 0 else self._stop + item
        if index < 0 or index >= self._stop:
            raise IndexError("History prefix index is outside the observed date")
        return self._rows[index]


def _weights(target, known_codes):
    if not isinstance(target, Mapping):
        raise ValueError("Policy target must be a weights mapping or None")
    result = {}
    for code, weight in target.items():
        if not isinstance(code, str) or code not in known_codes:
            raise ValueError("Unknown target asset: %r" % (code,))
        if isinstance(weight, bool):
            raise ValueError("Target weights must be real numbers, not booleans")
        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("Target weights must be finite nonnegative numbers") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Target weights must be finite nonnegative numbers")
        if weight:
            result[code] = weight
    total = sum(result.values())
    if total > 1 + 1e-12:
        raise ValueError("Target weights exceed one: leverage and borrowing are forbidden")
    if total > 1:
        result = {code: weight / total for code, weight in result.items()}
    return result


def _positive(value, description):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid %s" % description) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid %s" % description)
    return value


def _rebalance(units, cash, prices, target, fee, slippage):
    """Solve target weights against NAV after all transaction costs.

    Solving the scalar post-cost NAV avoids spending the last cash unit on
    shares and then implicitly borrowing the commission. Sells execute first;
    all fills are calculated and validated before the caller changes state.
    """
    codes = sorted(set(units) | set(target))
    current = {code: units.get(code, 0.0) * prices[code] for code in codes}
    before = cash + sum(current.values())
    tolerance = max(before, 1.0) * 1e-12
    if all(abs(target.get(code, 0.0) * before - value) <= tolerance
           for code, value in current.items()):
        return dict(units), cash, [], before, before
    buy_drag = (1 + slippage) * (1 + fee) - 1
    sell_drag = 1 - (1 - slippage) * (1 - fee)

    def total_cost(after):
        return sum((target.get(code, 0.0) * after - value) * buy_drag
                   if target.get(code, 0.0) * after >= value
                   else (value - target.get(code, 0.0) * after) * sell_drag
                   for code, value in current.items())

    low, high = 0.0, before
    for _ in range(64):
        middle = (low + high) / 2
        if middle + total_cost(middle) <= before:
            low = middle
        else:
            high = middle
    intended = {code: target.get(code, 0.0) * low / prices[code] for code in codes}
    fills = []
    actual = dict(units)
    for code in codes:
        change = intended[code] - units.get(code, 0.0)
        if abs(change * prices[code]) <= tolerance:
            continue
        side = "buy" if change > 0 else "sell"
        execution = prices[code] * (1 + slippage if change > 0 else 1 - slippage)
        quantity = abs(change)
        gross = quantity * execution
        commission = gross * fee
        cash_flow = -gross - commission if change > 0 else gross - commission
        fills.append(dict(code=code, side=side, units=quantity,
                          reference_open=prices[code], execution_price=execution,
                          gross_notional=gross, commission=commission, cash_flow=cash_flow,
                          slippage_cost=quantity * abs(execution - prices[code])))
        if intended[code] > 0:
            actual[code] = intended[code]
        else:
            actual.pop(code, None)
    fills.sort(key=lambda fill: (fill["side"] != "sell", fill["code"]))
    after_cash = cash + sum(fill["cash_flow"] for fill in fills)
    if after_cash < -max(before, 1.0) * 1e-10:
        raise ArithmeticError("Rebalance would borrow cash")
    after_cash = max(after_cash, 0.0)
    after = after_cash + sum(quantity * prices[code] for code, quantity in actual.items())
    return actual, after_cash, fills, before, after


def run(histories, calendar, policy, start, end, fee=.0001, slippage=.001):
    """Run a policy against actual fills, never a preselected target path.

    policy(date, observed_histories, state) runs after the daily close.
    observed_histories is a read-only mapping of Sequence prefixes through date.
    The policy can instead use a closure containing precomputed *dated* features.
    No later bar is exposed through the sequences' public indexing interface.

    state.cash is an amount; weights includes each held ETF (including a money
    ETF if explicitly requested), while cash_weight represents uninvested cash.
    holding_since maps each held asset to its actual initial fill date, and
    trading_index/previous_date refer to the full supplied trading calendar.

    Missing any held/target bar defers the whole rebalance. Missing held bars
    keep their previous close valuation; their next observed bar automatically
    recognizes the entire price change via units. Missing bars at the end of a
    dataset are rejected as insufficient coverage, not treated as performance.
    """
    fee, slippage = float(fee), float(slippage)
    if any(not math.isfinite(x) or not 0 <= x < 1 for x in (fee, slippage)):
        raise ValueError("Fee and slippage must be finite and in [0, 1)")
    if start > end:
        raise ValueError("Start must not follow end")
    cal = tuple(calendar)
    if not cal or tuple(sorted(set(cal))) != cal:
        raise ValueError("Calendar must be nonempty, unique and ascending")
    selected = [(i, date) for i, date in enumerate(cal) if start <= date <= end]
    if not selected:
        raise ValueError("No trading sessions in requested interval")
    rows_by_code, dates_by_code, by_date = {}, {}, {}
    for code, raw_rows in histories.items():
        rows = tuple(tuple(row) for row in raw_rows)
        if not rows or any(len(row) < 3 for row in rows):
            raise ValueError("History must contain dated open/close bars: %s" % code)
        dates = tuple(row[0] for row in rows)
        if tuple(sorted(set(dates))) != dates:
            raise ValueError("History dates must be unique and ascending: %s" % code)
        if dates[-1] < selected[-1][1]:
            raise ValueError("Requested tail exceeds historical coverage: %s" % code)
        rows_by_code[code], dates_by_code[code] = rows, dates
        by_date[code] = dict(zip(dates, rows))
    known_codes = set(rows_by_code)
    if not known_codes:
        raise ValueError("At least one asset history is required")

    units, cash, marks, holding_since = {}, 1.0, {}, {}
    pending, pending_date, filled_target, last_trade_date = None, None, {}, None
    daily, trades, signals = [], [], []
    diagnostics = dict(deferred_rebalances=[], missing_held_bars=[],
                       total_commission=0.0, total_slippage_cost=0.0,
                       total_turnover=0.0, completed_noop_orders=0, cash_interest_rate=0.0)
    final_state = None

    for index, date in selected:
        today = {}
        # A bar's open may be used only for execution, its close for later mark.
        for code in known_codes:
            row = by_date[code].get(date)
            if row is not None:
                today[code] = (_positive(row[1], "%s %s open" % (code, date)),
                               _positive(row[2], "%s %s close" % (code, date)))
        missing_held = sorted(code for code in units if code not in today)
        if missing_held:
            diagnostics["missing_held_bars"].append(dict(date=date, codes=missing_held))
        current_fills, deferred = [], False
        if pending is not None:
            missing = sorted((set(units) | set(pending)) - set(today))
            if missing:
                deferred = True
                diagnostics["deferred_rebalances"].append(dict(
                    date=date, signal_date=pending_date, target=dict(pending), missing_codes=missing))
            else:
                opens = {code: pair[0] for code, pair in today.items()}
                old_units = units
                units, cash, current_fills, before, after = _rebalance(
                    old_units, cash, opens, pending, fee, slippage)
                if current_fills:
                    for code in tuple(holding_since):
                        if code not in units:
                            holding_since.pop(code)
                    for code in units:
                        if code not in old_units:
                            holding_since[code] = date
                    last_trade_date = date
                    commission = sum(fill["commission"] for fill in current_fills)
                    slip_cost = sum(fill["slippage_cost"] for fill in current_fills)
                    turnover = sum(fill["units"] * fill["reference_open"] for fill in current_fills) / before
                    trades.append(dict(date=date, signal_date=pending_date,
                                       target=dict(pending), fills=[dict(f) for f in current_fills],
                                       nav_before=before, nav_after=after, cash_after=cash,
                                       commission=commission, slippage_cost=slip_cost, turnover=turnover,
                                       weights_after={code: quantity * opens[code] / after
                                                      for code, quantity in units.items()}))
                    diagnostics["total_commission"] += commission
                    diagnostics["total_slippage_cost"] += slip_cost
                    diagnostics["total_turnover"] += turnover
                else:
                    diagnostics["completed_noop_orders"] += 1
                filled_target = dict(pending)
                pending, pending_date = None, None

        for code in units:
            if code in today:
                marks[code] = today[code][1]
            elif code not in marks:
                raise ArithmeticError("Held asset has never had an observable valuation")
        nav = cash + sum(quantity * marks[code] for code, quantity in units.items())
        if not math.isfinite(nav) or nav <= 0:
            raise ArithmeticError("Portfolio NAV is not finite and positive")
        actual_weights = {code: quantity * marks[code] / nav for code, quantity in units.items()}
        only_holding = next(iter(units)) if len(units) == 1 else None
        state = dict(date=date, nav=nav, cash=cash, cash_weight=cash / nav,
                     units=dict(units), weights=dict(actual_weights), holdings=tuple(sorted(units)),
                     holding=only_holding, holding_since=dict(holding_since),
                     trading_index=index, previous_date=cal[index - 1] if index else None,
                     last_trade_date=last_trade_date, last_fills=[dict(f) for f in current_fills],
                     filled_target=dict(filled_target),
                     pending_target=dict(pending) if pending is not None else None,
                     pending_signal_date=pending_date, execution_deferred=deferred,
                     missing_held_bars=tuple(missing_held))
        observed = MappingProxyType({
            code: _HistoryPrefix(rows, bisect.bisect_right(dates_by_code[code], date))
            for code, rows in rows_by_code.items()})
        target = policy(date, observed, state)
        if target is not None:
            pending = _weights(target, known_codes)
            pending_date = date
        signals.append((date, dict(pending) if target is not None else None))
        daily.append((date, nav, dict(actual_weights)))
        # Rebuild from owned portfolio values, not the mutable policy argument.
        final_state = dict(date=date, nav=nav, cash=cash, units=dict(units),
                           weights=dict(actual_weights), holding_since=dict(holding_since),
                           pending_target=dict(pending) if pending is not None else None,
                           pending_signal_date=pending_date)
    diagnostics.update(deferred_count=len(diagnostics["deferred_rebalances"]),
                       rebalance_count=len(trades), fill_count=sum(len(t["fills"]) for t in trades),
                       policy_calls=len(daily))
    return dict(daily=daily, nav=daily[-1][1], trades=trades, diagnostics=diagnostics,
                signal_target=signals, final_state=final_state)
