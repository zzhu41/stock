"""Continuous-account historical model reselection, never a return-stream splice.

The caller supplies an already chosen dated schedule. No parameter selection or
performance search occurs here. Identical consecutive configurations/context
reuse the same Policy, including its confirmations. A changed model creates a
fresh Policy, retaining the real Portfolio and existing per-asset cooldowns;
rotation/regime confirmations restart and this boundary is explicitly traced.
"""
from bisect import bisect_left, bisect_right
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from v10_deep.reference import DatedValues, Policy, Portfolio
from v10_deep.schema import semantic


def training_folds():
    return [dict(train_start="2014-01-02", train_end="%d-12-31" % year,
                 start="%d-01-01" % (year + 1), end="%d-12-31" % (year + 2),
                 known_history=True)
            for year in (2017, 2019, 2021, 2023)]


def _key(config, context):
    value = dict(config=semantic(config), risk_context=context or {})
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _view(arrays, meta, config, context, factory):
    if factory is None:
        if context:
            raise ValueError("Nonempty risk_context requires an explicit view_factory")
        return DatedValues(arrays, meta, config)
    return factory(arrays, meta, deepcopy(config), deepcopy(context))


def _schedule(schedule, dates, start, end):
    if not schedule:
        raise ValueError("A nonempty preselected schedule is required")
    rows = deepcopy(schedule)
    for i, row in enumerate(rows):
        if row["start"] > row["end"] or (i and row["start"] <= rows[i - 1]["end"]):
            raise ValueError("Schedule must be chronological and nonoverlapping")
        if row["config"]["lag"] not in (0, 1):
            raise ValueError("Only signal lag 0 or 1 is supported")
        row["risk_context"] = row.get("risk_context") or {}
        row["model_key"] = _key(row["config"], row["risk_context"])
    lo = bisect_left(dates, rows[0]["start"] if start is None else start)
    hi = bisect_right(dates, rows[-1]["end"] if end is None else end) - 1
    if lo > hi or lo >= len(dates):
        raise ValueError("Empty walk-forward interval")
    assignments, cursor = [], 0
    for day in range(lo, hi + 1):
        while cursor < len(rows) and dates[day] > rows[cursor]["end"]:
            cursor += 1
        if cursor >= len(rows) or dates[day] < rows[cursor]["start"]:
            raise ValueError("Every evaluated observation must belong to exactly one selected model")
        assignments.append(cursor)
    return rows, lo, hi, assignments


def run(schedule, arrays, meta, start=None, end=None, fee=.0001, view_factory=None):
    """Mark old holding, decide with q=t-lag, execute at t close across all folds.

    view_factory(arrays, meta, config, risk_context) must return a DatedValues-like
    interface for Policy. Price execution always uses the original close field
    from arrays, independent of any risk-scale substitution in that view.
    """
    if not math.isfinite(fee) or not 0 <= fee < .5:
        raise ValueError("Invalid single-sided fee")
    dates = list(meta["dates"])
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Metadata calendar must be ascending and unique")
    segments, lo, hi, assignments = _schedule(schedule, dates, start, end)
    assets = tuple(meta["assets"])
    asset_index = {code: i for i, code in enumerate(assets)}
    close = arrays["features"][:, :, meta["feature_names"].index("close")]

    def price(day, code):
        if code is None:
            return float("nan")
        if code not in asset_index:
            raise ValueError("Policy chose an asset outside the execution universe")
        value = float(close[day, asset_index[code]])
        if math.isfinite(value) and value <= 0:
            raise ValueError("Execution prices must be positive")
        return value

    account = Portfolio()
    policy, current_key, previous_segment = None, None, None
    returns, holdings, trace, daily, trades, boundaries = [], [], [], [], [], []
    for day in range(lo, hi + 1):
        segment_index = assignments[day - lo]
        segment = segments[segment_index]
        config = segment["config"]
        model_changed = current_key is not None and segment["model_key"] != current_key
        created = policy is None or model_changed
        if created:
            old_policy = policy
            policy = Policy(_view(arrays, meta, config, segment["risk_context"], view_factory), config)
            if old_policy is not None:
                for code in policy.cooldown_until:
                    policy.cooldown_until[code] = old_policy.cooldown_until.get(code, -1)
            current_key = segment["model_key"]
        if segment_index != previous_segment:
            boundaries.append(dict(date=dates[day], segment=segment_index, model_key=current_key,
                model_id=config.get("id", current_key), policy_reused=not created,
                pending_reset=model_changed, holding=account.holding, nav=account.nav,
                age=account.age, lock_until=account.lock_until, asset_peak=account.asset_peak,
                previous_mark=account.mark))
            previous_segment = segment_index

        previous_nav, previous_holding = account.nav, account.holding
        current_price = price(day, account.holding)
        can_sell = account.holding is None or math.isfinite(current_price)
        if not can_sell:
            account.missing += 1
        if day > lo:
            if account.holding is not None and can_sell:
                account.nav *= current_price / account.mark
                account.mark = current_price
            account.age += 1
        marked_nav, age_before = account.nav, account.age
        target, panic, crash = policy.intent(day, account, can_sell)
        filled, paid = False, 0.
        if target is not None and target != account.holding:
            purchase_price = price(day, target)
            if not can_sell or not math.isfinite(purchase_price):
                account.blocked += 1
            else:
                if panic and account.holding is not None and config["panic_cooldown"] > 0:
                    policy.cooldown_until[account.holding] = day + config["panic_cooldown"] + 1
                if day > lo:
                    paid = account.nav * 2 * fee
                    account.nav *= 1 - 2 * fee
                    account.switches += 1
                account.entries += 1
                account.holding, account.mark = target, purchase_price
                account.age, account.asset_peak = 0, purchase_price
                policy.rotation.clear()
                if crash:
                    account.lock_until = day + config["crash_lock"]
                    account.crashes += 1
                filled = True
                trades.append(dict(date=dates[day], **{"from": previous_holding}, to=target,
                    nav_before_cost=marked_nav, nav=account.nav, cost=paid,
                    first_session_free=day == lo, at_model_change=model_changed,
                    segment=segment_index, model_key=current_key, crash=bool(crash)))
        if not math.isfinite(account.nav) or account.nav <= 0:
            raise ArithmeticError("Invalid continuous-account NAV")
        account.nav_peak = max(account.nav_peak, account.nav)
        account.max_dd = min(account.max_dd, account.nav / account.nav_peak - 1)
        returns.append(account.nav / previous_nav - 1)
        holdings.append(asset_index[account.holding] if account.holding is not None else -1)
        daily.append([dates[day], account.nav, account.holding])
        q = day - config["lag"]
        trace.append(dict(date=dates[day], signal_date=dates[q] if q >= 0 else None,
            segment=segment_index, model_key=current_key, model_id=config.get("id", current_key),
            model_changed=model_changed, policy_created=created, pending_reset=model_changed,
            previous_holding=previous_holding, intended=target, holding=account.holding,
            panic=bool(panic), crash_requested=bool(crash), filled=filled,
            previous_nav=previous_nav, marked_nav=marked_nav, nav=account.nav, cost=paid,
            age_before_policy=age_before, age=account.age, lock_until=account.lock_until,
            asset_peak=account.asset_peak, confirmation_target=policy.rotation.candidate,
            confirmation_count=policy.rotation.count))
    count = hi - lo + 1
    summary = [account.nav, account.nav ** (244. / count) - 1, account.max_dd, account.switches,
               account.entries, account.blocked, account.missing, account.crashes,
               sum(returns), sum(r * r for r in returns)]
    return dict(returns=np.asarray(returns), holdings=np.asarray(holdings, dtype=np.int32),
        summary=np.asarray(summary), dates=dates[lo:hi + 1], daily=daily, trades=trades,
        trace=trace, boundaries=boundaries, final_account=deepcopy(vars(account)),
        metadata=dict(execution_clock="same close; signal q=t-lag, actual old holding marked at t",
            first_session_free=True, fee_per_side=fee, annualization_sessions=244,
            model_change_retains=["holding", "nav", "mark", "age", "lock_until", "asset_peak", "cooldown_until"],
            changed_model_resets=["rotation confirmation", "regime confirmation/state"],
            identical_model_reuses_policy=True, no_return_stream_splicing=True, known_history=True, clean_oos=False))


def write_trace_csv(path, result):
    """Explicit artifact writer; the run itself never writes files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(result["trace"][0]) if result["trace"] else ["date"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["trace"])
