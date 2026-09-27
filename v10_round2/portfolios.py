"""Combine independent normalized accounts without reallocating capital.

This module reads supplied simulation results and prices only. It submits no
orders and never writes files. Gross component fills and costs are preserved,
including simultaneous opposite-side trades in the same security.
"""
from collections import defaultdict
from copy import deepcopy
import math


WEALTH_PERIODS = (
    ("2014-2017", "2014-01-01", "2017-12-31"),
    ("2018-2021", "2018-01-01", "2021-12-31"),
    ("2022-2025", "2022-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-12-31"),
)
AMOUNT_FIELDS = ("units", "gross_notional", "commission", "cash_flow", "slippage_cost")


def _close(left, right, description):
    if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10):
        raise AssertionError("Independent-account accounting mismatch: " + description)


def _specification(config):
    if config.get("kind") != "accounts":
        raise ValueError("run_accounts requires an accounts configuration")
    components = deepcopy(config.get("components", ()))
    if len(components) < 2 or len({c["id"] for c in components}) != len(components):
        raise ValueError("At least two distinctly named independent components are required")
    if any(c.get("kind") == "accounts" for c in components):
        raise ValueError("Nested account portfolios are not registered")
    weights = tuple(config.get("initial_weights", ()))
    if len(weights) != len(components) or any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w)
            or not math.isclose(w, 1.0 / len(components), rel_tol=0, abs_tol=1e-12) for w in weights):
        raise ValueError("This protocol permits equal initial capital only")
    if config.get("rebalance") != "never" or config.get("netting") is not False \
            or config.get("capital_transfers") is not False:
        raise ValueError("No rebalancing, trade netting or transfers between accounts are permitted")
    return components, weights


def run_accounts(config, histories, calendar, simulate_component):
    """Run each initial-NAV=1 component once and add its scaled economic ledger.

    simulate_component(component_config) returns the execution.run schema. All
    components must use the same dates, clock and cost assumptions. For the
    separate close-fill diagnostic, the caller supplies aggregation histories
    whose open column equals close; fills must mark execution_reference='close'.
    This is account aggregation, not a single portfolio rebalanced to average
    daily weights. A combined result has no single executable pending target:
    account_states and pending_account_instructions retain that information.
    """
    components, allocations = _specification(config)
    results = [simulate_component(deepcopy(component)) for component in components]
    names = [c["id"] for c in components]
    dates = [row[0] for row in results[0]["daily"]]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Component dates must be nonempty, unique and ascending")
    selected_calendar = [day for day in calendar if dates[0] <= day <= dates[-1]]
    if dates != selected_calendar or any([r[0] for r in result["daily"]] != dates for result in results):
        raise ValueError("All component accounts must share the identical trading-date path")
    clocks = {result.get("execution_clock", "ideal_close" if any(
        trade.get("execution_reference") == "close" for trade in result["trades"]) else "next_open")
        for result in results}
    if len(clocks) != 1 or not clocks <= {"next_open", "ideal_close"}:
        raise ValueError("Independent accounts cannot mix execution clocks")
    execution_clock = next(iter(clocks))
    execution_reference = "close" if execution_clock == "ideal_close" else "open"
    if any(trade.get("execution_reference", "open") != execution_reference
           for result in results for trade in result["trades"]):
        raise ValueError("Component fills disagree with the declared execution clock")
    for name, result in zip(names, results):
        if any(not math.isfinite(row[1]) or row[1] <= 0 for row in result["daily"]):
            raise ValueError("Component NAV must remain finite and positive: " + name)
        _close(result["nav"], result["daily"][-1][1], "component final NAV " + name)

    bars = {code: {r[0]: r for r in rows} for code, rows in histories.items()}
    trades_by_date = defaultdict(list)
    signals_by_date = defaultdict(dict)
    missing_by_date = defaultdict(lambda: dict(codes=set(), accounts=set()))
    deferred, metadata = [], {}
    for name, allocation, result in zip(names, allocations, results):
        metadata[name] = deepcopy(result.get("policy_metadata", {}))
        for trade in result["trades"]:
            if trade["date"] not in dates:
                raise ValueError("A component fill is outside its daily path")
            trades_by_date[trade["date"]].append((name, allocation, trade))
        for day, target in result.get("signal_target", ()):
            if target is not None:
                signals_by_date[day][name] = deepcopy(target)
        for row in result["diagnostics"].get("deferred_rebalances", ()):
            deferred.append(dict(deepcopy(row), account_id=name))
        for row in result["diagnostics"].get("missing_held_bars", ()):
            missing_by_date[row["date"]]["codes"].update(row["codes"])
            missing_by_date[row["date"]]["accounts"].add(name)

    daily, trades, account_wealth = [], [], []
    units, marks, cash = {}, {}, 1.0
    for index, day in enumerate(dates):
        child_rows = [result["daily"][index] for result in results]
        book_capital = [allocation * row[1] for allocation, row in zip(allocations, child_rows)]
        expected_nav = sum(book_capital)
        amounts = defaultdict(float)
        for capital, (_, _, weights) in zip(book_capital, child_rows):
            if any(not math.isfinite(w) or w < 0 for w in weights.values()) or sum(weights.values()) > 1 + 1e-9:
                raise ValueError("Invalid component closing weights")
            for code, weight in weights.items():
                amounts[code] += capital * weight
        combined_weights = {code: value / expected_nav for code, value in amounts.items() if value > 0}

        def open_price(code):
            row = bars.get(code, {}).get(day)
            if row is not None:
                return row[1]
            if code in marks:
                return marks[code]
            raise ValueError("No observable carried valuation for " + code)

        before = cash + sum(quantity * open_price(code) for code, quantity in units.items())
        fills, account_targets, signal_dates = [], {}, {}
        for name, allocation, trade in trades_by_date.get(day, ()):
            account_targets[name] = deepcopy(trade["target"])
            signal_dates[name] = trade.get("signal_date")
            for original in trade["fills"]:
                fill = deepcopy(original)
                fill.update(account_id=name, initial_capital_weight=allocation)
                for field in AMOUNT_FIELDS:
                    fill[field] *= allocation
                if fill["side"] not in ("buy", "sell"):
                    raise ValueError("Unknown component fill side")
                row = bars.get(fill["code"], {}).get(day)
                if row is None:
                    raise ValueError("Executed fill has no same-day bar")
                _close(fill["reference_open"], row[1], "fill reference must match aggregation price column")
                fills.append(fill)
        # Preserve every leg; ordering here is accounting only, not permission
        # to use another book's proceeds or waive an offsetting transaction fee.
        fills.sort(key=lambda f: (f["side"] != "sell", f["account_id"], f["code"]))
        for fill in fills:
            code = fill["code"]
            quantity = units.get(code, 0.0) + (fill["units"] if fill["side"] == "buy" else -fill["units"])
            if quantity < -1e-10:
                raise AssertionError("Aggregated component fills imply a short position")
            if abs(quantity) <= 1e-12:
                units.pop(code, None)
            else:
                units[code] = quantity
            cash += fill["cash_flow"]
        if cash < -1e-9:
            raise AssertionError("Aggregated independent accounts borrowed cash")
        if fills:
            after = cash + sum(quantity * open_price(code) for code, quantity in units.items())
            commission = sum(fill["commission"] for fill in fills)
            slippage_cost = sum(fill["slippage_cost"] for fill in fills)
            _close(after, before - commission - slippage_cost, "aggregate opening costs")
            weights_after = {code: quantity * open_price(code) / after for code, quantity in units.items()}
            unique_signal_dates = set(signal_dates.values())
            trades.append(dict(
                date=day, execution_reference=execution_reference,
                signal_date=next(iter(unique_signal_dates)) if len(unique_signal_dates) == 1 else None,
                component_signal_dates=signal_dates, component_targets=account_targets,
                target=weights_after.copy(), target_semantics="aggregate post-fill weights; no combined order",
                fills=fills, nav_before=before, nav_after=after, cash_after=cash,
                commission=commission, slippage_cost=slippage_cost,
                turnover=sum(fill["units"] * fill["reference_open"] for fill in fills) / before,
                weights_after=weights_after))
        for code in units:
            row = bars.get(code, {}).get(day)
            if row is not None:
                marks[code] = row[2]
        rebuilt = cash + sum(quantity * marks[code] for code, quantity in units.items())
        _close(rebuilt, expected_nav, "closing NAV on " + day)
        for code in set(units) | set(combined_weights):
            value = units.get(code, 0.0) * marks.get(code, 0.0)
            _close(value / expected_nav, combined_weights.get(code, 0.0), "closing asset weight " + code)
        daily.append((day, expected_nav, combined_weights))
        account_wealth.append((day, {name: amount / expected_nav for name, amount in zip(names, book_capital)}))

    initial_shares = dict(zip(names, allocations))
    periods = {}
    for label, start, end in WEALTH_PERIODS:
        indices = [i for i, day in enumerate(dates) if start <= day <= end]
        if indices:
            first, last = indices[0], indices[-1]
            periods[label] = dict(
                start=dates[first], end=dates[last],
                start_wealth_shares=deepcopy(account_wealth[first - 1][1] if first else initial_shares),
                end_wealth_shares=deepcopy(account_wealth[last][1]))
    account_states, pending_instructions, holding_since = {}, {}, {}
    for name, allocation, result in zip(names, allocations, results):
        state = result["final_state"]
        _close(state["nav"], result["nav"], "component final state " + name)
        account_states[name] = dict(
            initial_weight=allocation, final_wealth_share=account_wealth[-1][1][name],
            portfolio_wealth=allocation * result["nav"], portfolio_cash=allocation * state["cash"],
            portfolio_units={c: allocation * q for c, q in state["units"].items()},
            normalized_component_state=deepcopy(state))
        if state.get("pending_target") is not None:
            pending_instructions[name] = dict(signal_date=state.get("pending_signal_date"),
                                               target_weights=deepcopy(state["pending_target"]))
        for code, since in state.get("holding_since", {}).items():
            if state["units"].get(code, 0) > 0:
                holding_since[code] = min(since, holding_since.get(code, since))
    _close(cash, sum(item["portfolio_cash"] for item in account_states.values()), "final cash by account")
    diagnostics = dict(
        deferred_rebalances=sorted(deferred, key=lambda r: (r["date"], r["account_id"])),
        missing_held_bars=[dict(date=day, codes=sorted(row["codes"]), accounts=sorted(row["accounts"]))
                           for day, row in sorted(missing_by_date.items())],
        total_commission=sum(t["commission"] for t in trades),
        total_slippage_cost=sum(t["slippage_cost"] for t in trades),
        total_turnover=sum(t["turnover"] for t in trades),
        completed_noop_orders=sum(r["diagnostics"].get("completed_noop_orders", 0) for r in results),
        cash_interest_rate=0.0, deferred_count=len(deferred),
        deferred_days=len({r["date"] for r in deferred}),
        rebalance_count=len(trades), component_rebalance_count=sum(len(r["trades"]) for r in results),
        fill_count=sum(len(t["fills"]) for t in trades),
        policy_calls=sum(r["diagnostics"].get("policy_calls", len(dates)) for r in results),
        portfolio_observations=len(dates), netted_fills=0, capital_transfers=0,
        attribution_clock=execution_clock,
        turnover_denominator="entire portfolio pre-trade %s NAV" % execution_reference)
    crash_events = [dict(deepcopy(event), account_id=name)
                    for name, item in metadata.items() for event in item.get("crash_events", ())]
    return dict(
        daily=daily, nav=daily[-1][1], trades=trades, diagnostics=diagnostics,
        execution_clock=execution_clock,
        signal_target=[(day, deepcopy(signals_by_date.get(day)) or None) for day in dates],
        signal_target_scope="independent account instructions, not a unified target",
        final_state=dict(date=dates[-1], nav=daily[-1][1], cash=cash, units=units.copy(),
                         weights=daily[-1][2].copy(), holding_since=holding_since,
                         pending_target=None, pending_signal_date=None, independent_accounts=True,
                         account_states=account_states, pending_account_instructions=pending_instructions),
        policy_metadata=dict(accounts=metadata, crash_events=crash_events,
                             crash_event_count_unit="account entries; overlapping crisis dates are not independent samples"),
        account_summaries=[dict(id=name, config=deepcopy(component), initial_capital_weight=allocation,
                                final_component_nav=result["nav"], final_portfolio_wealth=allocation * result["nav"],
                                final_wealth_share=account_wealth[-1][1][name])
                           for name, component, allocation, result in zip(names, components, allocations, results)],
        account_nav_weights=account_wealth,
        account_wealth_shares=dict(initial=initial_shares, final=deepcopy(account_wealth[-1][1]), periods=periods),
        account_aggregation_caveat="Fixed initial budgets only; book weights drift. Each book's gross fills and costs remain payable.")
