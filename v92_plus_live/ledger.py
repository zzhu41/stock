"""Independent V9.2+ virtual units ledger; no broker orders or old shadow state.

Signal-time marks use raw prices. Cash distributions belong only to units held
before the ex-date, and are reinvested at the first observed mark on that date
(completed close for skipped days). This is an explicit virtual DRIP convention,
not a claim about a broker's dividend payment date.
"""
from copy import deepcopy
import math

VERSION = 1
CANDIDATE_ID = "vd_d2a02ab14be481f562fd"
CANDIDATE_HASH = "d2a02ab14be481f562fd674887950bb1a7876b52ed75964b43ea4edc5ae518f3"
SWITCH_COST = 0.0002


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("V9.2+ invalid " + name)
    return value


def validate_state(state):
    if state is None:
        return
    if (state.get("version") != VERSION or state.get("candidate_hash") != CANDIDATE_HASH
            or state.get("candidate_id") != CANDIDATE_ID):
        raise ValueError("V9.2+ state version/profile mismatch; refusing automatic reset")
    for field in ("units", "nav", "mark_raw_price", "peak_nav"):
        positive(state[field], field)
    if not state.get("holding") or not state.get("last_date") or not state.get("start_date"):
        raise ValueError("V9.2+ incomplete state")
    if not math.isclose(state["units"] * state["mark_raw_price"], state["nav"], rel_tol=1e-10):
        raise ValueError("V9.2+ units/mark/NAV mismatch")
    if not isinstance(state.get("events"), list):
        raise ValueError("V9.2+ missing virtual event history")


def advance(state, decision, view, quotes, signal_date):
    """Pure atomic-transaction payload. Same-date calls never mark or trade twice."""
    validate_state(state)
    if state and signal_date <= state["last_date"]:
        if signal_date == state["last_date"]:
            return deepcopy(state)
        raise ValueError("V9.2+ refuses backdated state updates")
    if decision.get("candidate_id") != CANDIDATE_ID or not decision.get("executable", False):
        raise ValueError("V9.2+ decision is not an executable frozen-profile shadow intent")
    target = decision["target"]
    target_price = positive(quotes[target]["price"], "target price")
    if quotes[target]["date"] != signal_date or signal_date not in view["calendar"]:
        raise ValueError("V9.2+ current quote/calendar mismatch")
    result = deepcopy(state) if state else {
        "version": VERSION, "candidate_id": CANDIDATE_ID, "candidate_hash": CANDIDATE_HASH,
        "start_date": signal_date, "nav": 1.0, "peak_nav": 1.0,
        "max_drawdown": 0.0, "events": [], "switches": 0,
    }
    before_nav = result["nav"]
    applied = []
    previous_holding = state["holding"] if state else None
    nav = 1.0
    if state:
        settled = state.get("last_settlement_action")
        if settled:
            confirmed = view["actions"].get(settled["code"], {}).get(settled["date"])
            if confirmed is None or any(not math.isclose(
                    float(confirmed[key]), float(settled[key]), rel_tol=1e-10, abs_tol=1e-12)
                    for key in ("split_ratio", "cash_per_old_share")):
                raise ValueError("V9.2+ previously booked provisional corporate action changed; account review required")
        if state["last_date"] not in view["calendar"]:
            raise ValueError("V9.2+ missing previous signal date in current calendar")
        held = state["holding"]
        raw_rows = {r[0]: r for r in view["raw_histories"][held]}
        actions = view["actions"][held]
        units = state["units"]
        # Require an explicit checked action entry even for an ordinary no-action
        # session. Missing data must never be interpreted as a zero dividend.
        for date in view["calendar"]:
            if not state["last_date"] < date <= signal_date:
                continue
            if date not in actions:
                raise ValueError("V9.2+ unverified corporate action: " + held + " " + date)
            action = actions[date]
            split = positive(action["split_ratio"], "split ratio")
            cash = float(action["cash_per_old_share"])
            if not math.isfinite(cash) or cash < 0:
                raise ValueError("V9.2+ invalid dividend")
            if action.get("not_observed"):
                # Only a bracketed, explicitly verified zero-action gap can
                # carry the units. Never invent a close for a suspended asset
                # or reinvest cash on an unobserved date.
                if (date == signal_date or date in raw_rows or split != 1 or cash != 0
                        or action.get("verification") != "bracketed_no_action_interval"
                        or not action.get("previous_quote_date", "") < date < action.get("next_quote_date", "")):
                    raise ValueError("V9.2+ unsafe unobserved-session action: " + date)
                continue
            entitlement = units * cash
            units *= split
            if date == signal_date:
                if quotes[held]["date"] != date:
                    raise ValueError("V9.2+ stale holding quote")
                reinvest_price = positive(quotes[held]["price"], "holding price")
            else:
                if date not in raw_rows:
                    raise ValueError("V9.2+ missing raw close on " + date)
                reinvest_price = positive(raw_rows[date][2], "completed raw close")
            units += entitlement / reinvest_price
            if split != 1 or cash:
                applied.append({"date": date, "code": held, "split_ratio": split,
                                "cash_per_old_share": cash, "cash_received": entitlement,
                                "reinvest_price": reinvest_price})
        nav = units * positive(quotes[held]["price"], "holding price")
        if held != target:
            nav *= 1 - SWITCH_COST
            result["switches"] += 1
    nav = positive(nav, "resulting NAV")
    result.update(holding=target, units=nav / target_price, nav=nav,
                  mark_raw_price=target_price, last_date=signal_date,
                  entry_date=state["entry_date"] if state and target == previous_holding else signal_date,
                  crash_trigger_date=decision.get("crash_trigger_date"),
                  crash_code=decision.get("crash_code"),
                  last_return=nav / before_nav - 1,
                  quote_timestamp=quotes[target].get("timestamp"),
                  last_decision=deepcopy(decision))
    result["last_settlement_action"] = (
        dict(code=previous_holding, date=signal_date,
             split_ratio=view["actions"][previous_holding][signal_date]["split_ratio"],
             cash_per_old_share=view["actions"][previous_holding][signal_date]["cash_per_old_share"])
        if state else None)
    result["peak_nav"] = max(result["peak_nav"], nav)
    result["max_drawdown"] = min(result["max_drawdown"], nav / result["peak_nav"] - 1)
    result["events"].append({"date": signal_date, "from": previous_holding, "to": target,
                             "nav": nav, "mark_raw_price": target_price,
                             "fee_fraction": SWITCH_COST if state and previous_holding != target else 0,
                             "actions": applied})
    validate_state(result)
    return result
