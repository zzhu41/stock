"""Small close-only portfolio research, separate from the single-ETF scanner.

These are units of the supplied TR indices, not a live raw-share/dividend
ledger. Fees follow the explicitly requested legacy approximation:
postNAV = preNAV * (1 - fee * sum(abs(target_weight - marked_weight))).
The first evaluation day is free. A full ETF A -> ETF B switch has turnover 2
and reproduces the legacy 1-2*fee multiplier. Uninvested cash pays no interest.
No function reads open prices, changes a registered strategy, or runs a scan.
"""
import bisect
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral


CASH, GOLD, BENCHMARK = "511880", "518880", "510300"
ORIGINAL_STOCK = ("159915", "588080", "510300", "510500", "563300", "512400", "512890")
ORIGINAL_GLOBAL = ("513100", "513120")
EXTRA_STOCK = ("510880", "515080", "515100", "159928", "512010", "512070", "512880", "512800", "512200")
EXTRA_GLOBAL = ("513030", "513520", "159985", "511010")
POOLS = {
    "original": (ORIGINAL_STOCK, ORIGINAL_GLOBAL),
    "broad": (ORIGINAL_STOCK + EXTRA_STOCK, ORIGINAL_GLOBAL + EXTRA_GLOBAL),
    "macro": ((BENCHMARK,), ("513100", "513520", "159985", "511010")),
}


def prices_from_features(arrays, meta):
    """Extract ONLY the supplied close field; synthetic open is never accepted."""
    names, dates, assets = meta["feature_names"], meta["dates"], meta["assets"]
    column = names.index("close")
    cube = arrays["features"]
    if len(cube) != len(dates) or (len(cube) and len(cube[0]) != len(assets)):
        raise ValueError("Feature shape and metadata disagree")
    return {code: tuple(float(cube[i][j][column]) for i in range(len(dates)))
            for j, code in enumerate(assets)}


def _quote(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Close prices must be positive finite values, or NaN/None for missing bars")
    return value


def _weights(target, codes):
    if not isinstance(target, dict):
        raise ValueError("Policy must return a weight dict or None")
    out = {}
    for code, weight in target.items():
        if code not in codes or isinstance(weight, bool):
            raise ValueError("Unknown asset or invalid boolean weight")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Weights must be finite and nonnegative")
        if weight:
            out[code] = weight
    total = sum(out.values())
    if total > 1 + 1e-12:
        raise ValueError("Leverage and borrowing are forbidden")
    if total > 1:
        out = {code: weight / total for code, weight in out.items()}
    return out


def run_weights(close_prices, calendar, policy, start, end, fee=.0001):
    """Mark persisted model units, call policy, then attempt one same-close rebalance.

    close_prices maps code -> a vector aligned with the entire calendar.
    policy(global_index, state) returns a weight dict, {} for uninvested cash,
    or None to preserve units. A rejected rebalance is discarded, not queued.
    state.weights is marked BEFORE this close's rebalance; state.cash is an
    amount, while cash_weight is its portfolio fraction.
    """
    fee = float(fee)
    if not math.isfinite(fee) or not 0 <= fee < .5:
        raise ValueError("Fee must be finite and in [0,.5)")
    dates = tuple(calendar)
    if not dates or tuple(sorted(set(dates))) != dates:
        raise ValueError("Calendar must be nonempty, ascending and unique")
    lo = int(start) if isinstance(start, Integral) else bisect.bisect_left(dates, start)
    hi = int(end) if isinstance(end, Integral) else bisect.bisect_right(dates, end) - 1
    if lo < 0 or hi >= len(dates) or lo > hi:
        raise ValueError("Invalid evaluation interval")
    prices = {code: tuple(values) for code, values in close_prices.items()}
    if not prices or any(len(values) != len(dates) for values in prices.values()):
        raise ValueError("Every asset must have one price/missing marker per calendar row")
    codes = set(prices)
    units, marks, since, cash = {}, {}, {}, 1.0
    daily, returns, trades, signals = [], [], [], []
    blocked, missing_rows = [], []
    previous_nav, peak, max_dd = 1.0, 1.0, 0.0
    last_trade_date, previous_blocked = None, False
    total_fee = total_turnover = 0.0
    charged_changes = noops = 0

    for i in range(lo, hi + 1):
        quotes = {code: _quote(values[i]) for code, values in prices.items()}
        missing_held = sorted(code for code in units if quotes[code] is None)
        if missing_held:
            missing_rows.append(dict(date=dates[i], codes=missing_held))
        for code in units:
            if quotes[code] is not None:
                marks[code] = quotes[code]
            elif code not in marks:
                raise ArithmeticError("Held asset has no prior valuation")
        pre_nav = cash + sum(quantity * marks[code] for code, quantity in units.items())
        if not math.isfinite(pre_nav) or pre_nav <= 0:
            raise ArithmeticError("Invalid marked NAV")
        current = {code: quantity * marks[code] / pre_nav for code, quantity in units.items()}
        state = dict(date=dates[i], trading_index=i, previous_date=dates[i - 1] if i else None,
                     nav=pre_nav, cash=cash, cash_weight=cash / pre_nav,
                     units=dict(units), weights=dict(current), holdings=tuple(sorted(units)),
                     holding_since=dict(since), current_quotes=dict(quotes),
                     missing_held_bars=tuple(missing_held), last_trade_date=last_trade_date,
                     previous_rebalance_blocked=previous_blocked)
        instruction = policy(i, state)
        previous_blocked = False
        target = None if instruction is None else _weights(instruction, codes)
        signals.append((dates[i], dict(target) if target is not None else None))
        nav = pre_nav
        if target is not None:
            turnover = sum(abs(target.get(code, 0.0) - current.get(code, 0.0))
                           for code in set(target) | set(current))
            if turnover <= 1e-12:
                noops += 1
            else:
                multiplier = 1.0 if i == lo else 1 - fee * turnover
                post_nav = pre_nav * multiplier
                if post_nav <= 0:
                    raise ArithmeticError("Rebalance costs exhausted the portfolio")
                desired, missing = {}, set()
                tolerance = max(pre_nav, 1.0) * 1e-12
                for code in set(units) | set(target):
                    weight = target.get(code, 0.0)
                    price = quotes[code] if quotes[code] is not None else marks.get(code)
                    if weight and price is None:
                        missing.add(code)
                        continue
                    quantity = weight * post_nav / price if weight else 0.0
                    if quotes[code] is None and abs(quantity - units.get(code, 0.0)) * (price or 1.0) > tolerance:
                        missing.add(code)
                    if weight:
                        # An unquoted leg can remain only if its units need not
                        # change. Never fabricate a fill at its carried mark.
                        desired[code] = quantity if quotes[code] is not None else units[code]
                if missing:
                    previous_blocked = True
                    blocked.append(dict(date=dates[i], target=dict(target), missing_codes=sorted(missing)))
                else:
                    old_units = units
                    units = desired
                    cash = post_nav * max(0.0, 1 - sum(target.values()))
                    since = {code: since.get(code, dates[i]) for code in units}
                    for code in units:
                        if quotes[code] is not None:
                            marks[code] = quotes[code]
                    changes = [dict(code=code, previous_units=old_units.get(code, 0.0),
                                    new_units=units.get(code, 0.0), reference_close=quotes[code])
                               for code in sorted(set(old_units) | set(units))
                               if abs(units.get(code, 0.0) - old_units.get(code, 0.0)) * marks[code] > tolerance]
                    model_fee = pre_nav - post_nav
                    trades.append(dict(date=dates[i], target=dict(target), marked_weights=dict(current),
                                       nav_before=pre_nav, nav_after=post_nav, turnover=turnover,
                                       model_fee=model_fee, cash_after=cash, allocation_changes=changes,
                                       initial_session_free=i == lo))
                    last_trade_date, nav = dates[i], post_nav
                    total_fee += model_fee
                    total_turnover += turnover
                    if i != lo:
                        charged_changes += 1
        actual_weights = {code: quantity * marks[code] / nav for code, quantity in units.items()}
        peak, max_dd = max(peak, nav), min(max_dd, nav / max(peak, nav) - 1)
        returns.append(nav / previous_nav - 1)
        previous_nav = nav
        daily.append((dates[i], nav, actual_weights))
    return dict(daily=daily, returns=returns, nav=nav, ann=nav ** (244 / len(daily)) - 1,
                max_dd=max_dd, trades=trades, signal_target=signals,
                switches=charged_changes,
                diagnostics=dict(rebalance_count=len(trades), charged_rebalance_count=charged_changes,
                                 total_model_fee=total_fee, total_turnover=total_turnover,
                                 blocked_rebalance_count=len(blocked), blocked_rebalances=blocked,
                                 missing_held_bars=missing_rows, completed_noop_orders=noops,
                                 cash_interest_rate=0.0, order_carryover=False,
                                 execution_clock="original_same_close_portfolio_approximation",
                                 fee_convention="preNAV*fee*L1(pre-fee target minus marked security weights); first evaluation day free",
                                 live_cash_ledger=False),
                final_state=dict(date=dates[hi], nav=nav, cash=cash, units=dict(units),
                                 weights=dict(actual_weights), holding_since=dict(since)))


def targets_as_weights(code_targets, start_index=0):
    """Replay a supplied code path, not independently reconstruct its decisions.

    None in the sequence means uninvested cash ({}), while the 511880 code is a
    held security. A native fully-invested baseline must preserve its money-ETF
    code rather than replacing it by None, which has different one-sided fees.
    """
    targets = tuple(code_targets)

    def policy(i, state):
        offset = i - start_index
        if offset < 0 or offset >= len(targets):
            raise IndexError("Code target sequence does not cover this evaluation date")
        code = targets[offset]
        return {} if code is None else {code: 1.0}
    return policy


def registry():
    """Thirty-six configurations, fixed without reading any performance results."""
    candidates = []
    for topk in (1, 2, 3):
        for score in ("wls25_v20", "blend_20_40_60"):
            for pool_name in ("original", "broad", "macro"):
                stock, glob = POOLS[pool_name]
                for regime in ("legacy_gate", "all_assets"):
                    config = dict(id="top%d_%s_%s_%s" % (topk, score, pool_name, regime),
                                  topk=topk, score=score, pool=pool_name, regime=regime,
                                  stock_pool=list(stock), global_pool=list(glob), gold=GOLD, cash=CASH,
                                  benchmark=BENCHMARK, crash_channels=[], bull_entry=0.0, bear_entry=.07,
                                  overheat=.40, panic=.04, target_weight="equal among selected members",
                                  rebalance="membership changes only; preserve units otherwise",
                                  ordinary_momentum_gap_buffer=False, leverage=False)
                    semantics = {key: value for key, value in config.items() if key not in ("id", "pool")}
                    semantics["stock_pool"] = sorted(set(stock))
                    semantics["global_pool"] = sorted(set(glob))
                    config["hash"] = hashlib.sha256(json.dumps(semantics, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    candidates.append(config)
    return dict(schema=1, count=len(candidates), candidates=candidates,
                scope="Independent no-crash diversification/risk comparison, not a clone of v9.2",
                fee_model="Original close approximation on L1 security-weight turnover; no free daily weight reset",
                caveats=["Existing historical data are already known; registration does not create a clean holdout.",
                         "Top1 here is the control for this buffer-free top-k family, not the original v9.2 decision rule.",
                         "One eligible member receives full allocation; zero eligible members use the specified single-asset safe fallback.",
                         "Cash not invested in an explicitly requested ETF earns zero interest."])


def build_topk_policy(config, arrays, meta):
    """Pure dated rank policy; panic exclusion and a single safe fallback persist.

    This family intentionally omits the legacy single-position MOM-gap buffer.
    It has no crash entries, locks, dynamic capital sleeves or optimized weights.
    """
    config = deepcopy(config)
    if config.get("crash_channels") or config["topk"] not in (1, 2, 3):
        raise ValueError("Only registered no-crash top1/top2/top3 policies are supported")
    if config["regime"] not in ("legacy_gate", "all_assets"):
        raise ValueError("Unknown portfolio regime")
    assets, dates = tuple(meta["assets"]), tuple(meta["dates"])
    columns = {code: i for i, code in enumerate(assets)}
    fields = {name: i for i, name in enumerate(meta["feature_names"])}
    for name in ("close", "valid", "ma250", "mom20", "mom5", "ret1"):
        if name not in fields:
            raise ValueError("Missing required feature: " + name)
    score_index = meta["score_names"].index(config["score"])
    stock, glob = set(config["stock_pool"]), set(config["global_pool"])
    trade, safe = stock | glob | {GOLD}, glob | {GOLD}
    if not trade | {CASH, BENCHMARK} <= set(assets):
        raise ValueError("Portfolio pool is not covered by feature metadata")
    metadata = dict(config=config, decisions=[], score_lookahead=False, crash_entries=0)

    def policy(i, state):
        row, score_row = arrays["features"][i], arrays["scores"][score_index][i]

        def value(code, field):
            return float(row[columns[code]][fields[field]])

        def valid(code):
            return value(code, "valid") > .5 and math.isfinite(value(code, "close"))

        benchmark = value(BENCHMARK, "ma250") if valid(BENCHMARK) else float("nan")
        bull = True if config["regime"] == "all_assets" else (benchmark > 0 if math.isfinite(benchmark) else True)
        comp = trade if config["regime"] == "all_assets" else (stock | glob if bull else safe)
        floor = config["bull_entry"] if bull else max(config["bull_entry"], config["bear_entry"])
        panic = {code for code in state["holdings"] if config["panic"] > 0 and code != CASH and code in columns
                 and valid(code) and value(code, "ret1") <= -config["panic"]}
        ranked = [code for code in assets if code in trade and code not in panic and valid(code)
                  and math.isfinite(float(score_row[columns[code]])) and float(score_row[columns[code]]) > -1e99]
        ranked.sort(key=lambda code: (-float(score_row[columns[code]]), columns[code]))
        eligible = [code for code in ranked if code in comp and value(code, "mom20") > floor
                    and not (value(code, "mom20") > config["overheat"] and value(code, "mom5") <= 0)]
        selected = eligible[:config["topk"]]
        reason = "qualified_rank_members"
        if not selected and bull and GOLD in ranked and value(GOLD, "mom20") > 0:
            selected, reason = [GOLD], "positive_gold_fallback"
        if not selected:
            fallback = next((code for code in ranked if code in safe), None)
            selected, reason = ([fallback], "single_safe_fallback") if fallback else ([], "cash")
        target = ({code: 1.0 / len(selected) for code in selected} if selected else
                  {CASH: 1.0} if math.isfinite(value(CASH, "close")) else {})
        changed = set(target) != set(state["holdings"])
        metadata["decisions"].append(dict(date=dates[i], members=list(target), bull=bull,
                                          reason=reason, instruction_issued=changed))
        return target if changed else None

    policy.metadata = metadata
    return policy
