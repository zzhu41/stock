"""Ideal same-close counterfactual with actual cash/units and trading costs.

This is deliberately unattainable price/information timing, used only as a
separate diagnostic. It is never used to select next-open releases. Policies
receive a complete close before submitting a trade at that same close.
"""
import bisect
import math
from copy import deepcopy
from types import MappingProxyType

from v10_next.execution import _HistoryPrefix, _positive, _rebalance, _weights


def for_same_close(config):
    """Keep five full return intervals in a close-entry crash lock.

    The next-open policy releases at entry_index+lock-1: enter at tomorrow's
    open, receive five intraday/overnight intervals, exit the following open.
    For a close fill the first return interval is the following day, so release
    must be entry_index+lock. This adds one *decision-clock count*, not a sixth
    economic holding day. It affects only this diagnostic, not frozen rules.
    """
    c = deepcopy(config)
    if c["kind"] == "accounts":
        c["components"] = [for_same_close(x) for x in c["components"]]
    else:
        c.setdefault("overrides", {})
        c["overrides"]["crash_lock"] = c["overrides"].get("crash_lock", 5) + 1
    return c


def run_same_close(histories, calendar, policy, start, end, fee=.0001, slippage=0):
    if not 0 <= fee < 1 or not 0 <= slippage < 1:
        raise ValueError("Invalid costs")
    calendar = tuple(calendar)
    selected = [(i, d) for i, d in enumerate(calendar) if start <= d <= end]
    if not selected:
        raise ValueError("No selected trading days")
    dates = {c: [r[0] for r in rows] for c, rows in histories.items()}
    bars = {c: {r[0]: r for r in rows} for c, rows in histories.items()}
    if any(not ds or ds[-1] < selected[-1][1] for ds in dates.values()):
        raise ValueError("Insufficient end coverage")
    units, cash, marks, since = {}, 1.0, {}, {}
    pending, pending_date, filled_target = None, None, {}
    last_fills, last_trade_date, previous_deferred = [], None, False
    daily, trades, signals = [], [], []
    diag = dict(deferred_rebalances=[], missing_held_bars=[], total_commission=0.0,
                total_slippage_cost=0.0, total_turnover=0.0, completed_noop_orders=0,
                cash_interest_rate=0.0, timing="ideal_same_close")
    for i, date in selected:
        closes = {c: _positive(rows[date][2], c + " close") for c, rows in bars.items() if date in rows}
        missing_held = sorted(set(units) - set(closes))
        if missing_held:
            diag["missing_held_bars"].append(dict(date=date, codes=missing_held))
        for c in units:
            if c in closes:
                marks[c] = closes[c]
        nav = cash + sum(q * marks[c] for c, q in units.items())
        state = dict(date=date, nav=nav, cash=cash, cash_weight=cash / nav,
                     units=dict(units), weights={c: q * marks[c] / nav for c, q in units.items()},
                     holdings=tuple(sorted(units)), holding=next(iter(units)) if len(units) == 1 else None,
                     holding_since=dict(since), trading_index=i,
                     previous_date=calendar[i - 1] if i else None,
                     last_trade_date=last_trade_date, last_fills=deepcopy(last_fills),
                     filled_target=dict(filled_target),
                     pending_target=dict(pending) if pending is not None else None,
                     pending_signal_date=pending_date, execution_deferred=previous_deferred,
                     missing_held_bars=tuple(missing_held))
        observed = MappingProxyType({c: _HistoryPrefix(rows, bisect.bisect_right(dates[c], date))
                                     for c, rows in histories.items()})
        instruction = policy(date, observed, state)
        if instruction is not None:
            pending, pending_date = _weights(instruction, set(histories)), date
        signals.append((date, dict(pending) if instruction is not None else None))
        previous_deferred, last_fills = False, []
        if pending is not None:
            missing = sorted((set(units) | set(pending)) - set(closes))
            if missing:
                previous_deferred = True
                diag["deferred_rebalances"].append(dict(date=date, signal_date=pending_date,
                                                       target=dict(pending), missing_codes=missing))
            else:
                old = units
                units, cash, last_fills, before, after = _rebalance(old, cash, closes, pending, fee, slippage)
                if last_fills:
                    since = {c: since.get(c, date) for c in units}
                    last_trade_date = date
                    commission = sum(f["commission"] for f in last_fills)
                    slip = sum(f["slippage_cost"] for f in last_fills)
                    turnover = sum(f["units"] * f["reference_open"] for f in last_fills) / before
                    trades.append(dict(date=date, signal_date=pending_date, target=dict(pending),
                                       fills=deepcopy(last_fills), nav_before=before, nav_after=after,
                                       cash_after=cash, commission=commission, slippage_cost=slip,
                                       turnover=turnover, execution_reference="close"))
                    diag["total_commission"] += commission
                    diag["total_slippage_cost"] += slip
                    diag["total_turnover"] += turnover
                else:
                    diag["completed_noop_orders"] += 1
                filled_target = dict(pending)
                pending, pending_date = None, None
        for c in units:
            if c in closes:
                marks[c] = closes[c]
        nav = cash + sum(q * marks[c] for c, q in units.items())
        if not math.isfinite(nav) or nav <= 0:
            raise ValueError("Invalid portfolio NAV")
        weights = {c: q * marks[c] / nav for c, q in units.items()}
        daily.append((date, nav, weights))
    diag.update(deferred_count=len(diag["deferred_rebalances"]), rebalance_count=len(trades),
                fill_count=sum(len(t["fills"]) for t in trades), policy_calls=len(daily))
    return dict(daily=daily, nav=daily[-1][1], trades=trades, diagnostics=diag, signal_target=signals,
                final_state=dict(date=date, nav=nav, cash=cash, units=units, weights=weights,
                                 holding_since=since, pending_target=pending, pending_signal_date=pending_date))
