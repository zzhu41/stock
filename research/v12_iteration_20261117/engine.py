"""Isolated, read-only policy research with fixed-unit crash/cash sleeves.

Inputs must already contain the intended dated feature view (including the
frozen R2 prior-volatility substitution). This module never changes input arrays,
production files, or feature timing. Lag 1 means prior signal / current close.

Fees intentionally use the legacy approximation: pre-fee target security values
minus marked security values, absolute sum times one-way fee; then allocate net
NAV at target weights. This is not exact post-fee self-financing. A full switch
therefore has factor 1-2f, versus exact (1-f)/(1+f). Legacy also charges two sides
for a delayed initial entry after the first simulation session. First SESSION
is free, not first successful trade. Turnover is recorded even when fee-free.
"""
from bisect import bisect_left, bisect_right
from copy import deepcopy
import math

import numpy as np

from v10_deep.reference import CASH, GOLD, DatedValues, Portfolio
from v12.reference import Policy

LOWVOL = "512890"


class ResearchPolicy(Policy):
    """Only explicitly requested fallback and defensive eligibility changes."""
    def __init__(self, data, config, spec):
        super().__init__(data, config)
        self.hysteresis = float(spec.get("fallback_hysteresis", 0.))
        self.defensive = bool(spec.get("defensive_lowvol", False))
        if self.defensive:
            if LOWVOL not in self.stocks or LOWVOL not in data.asset_index:
                raise ValueError("Defensive lowvol must already belong to the stock asset pool")
            self.safe_pool = self.safe_pool | {LOWVOL}
        self.fallback_used = False

    def intent(self, execution_day, account, can_sell):
        self.fallback_used = False
        return super().intent(execution_day, account, can_sell)

    def choose_regular(self, day, execution_day, account, bull, panic, ranked):
        if not self.hysteresis and not self.defensive:
            return super().choose_regular(day, execution_day, account, bull, panic, ranked)
        p, d = self.config, self.data
        competition = self.trade_pool if p["regime"] == "all_assets" else (
            self.stocks | self.globals if bull or p["regime"] == "open_stock" else self.safe_pool)
        permitted = [code for code in ranked if execution_day >= self.cooldown_until[code]
                     and not (panic and code == account.holding)]
        floor = p["bull_entry"] if bull else max(p["bull_entry"], p["bear_entry"])
        entrants = [code for code in permitted if code in competition
                    and self.momentum(day, code, p["mom"]) > floor
                    and not (self.momentum(day, code, p["mom"]) > p["overheat"]
                             and self.momentum(day, code, p["fast_mom"]) <= 0)]
        if entrants:
            return entrants[0], competition
        if bull and GOLD in permitted and self.momentum(day, GOLD, p["mom"]) > 0:
            return GOLD, competition
        self.fallback_used = True
        alternatives = [code for code in permitted if code in self.safe_pool]
        if p["fallback_mode"] == "cash" or not alternatives:
            return CASH, competition
        if self.hysteresis:
            alternatives = [code for code in alternatives if (
                self.momentum(day, code, 20) >= 0 if code == account.holding
                else self.momentum(day, code, 20) > self.hysteresis)]
            if not alternatives:
                return CASH, competition
        if p["fallback_mode"] == "low_vol":
            alternatives = [code for code in alternatives if d.value(day, code, "vol20") < 1e100]
            if not alternatives:
                return CASH, competition
            choice = min(alternatives, key=lambda code: d.value(day, code, "vol20"))
        else:
            choice = alternatives[0]
        return (CASH if self.momentum(day, choice, p["mom"]) < p["fallback_floor"] else choice), competition

    def must_exit(self, day, holding, bull):
        if not self.defensive or holding != LOWVOL:
            return super().must_exit(day, holding, bull)
        p = self.config
        failed_trend = self.momentum(day, holding, p["exit_mom"]) <= p["exit_floor"]
        overheated = (self.momentum(day, holding, p["mom"]) > p["overheat"]
                      and self.momentum(day, holding, p["fast_mom"]) <= 0)
        return failed_trend or overheated


def run_portfolio(arrays, meta, base_config, spec=None, start=None, end=None, fee=.0001, lag=None):
    """Return reference-compatible arrays plus sleeve/turnover/cost diagnostics.

    spec: config_changes=dict, crash_weight in {1,.75,.5},
          fallback_hysteresis in {0,.02}, defensive_lowvol=bool.
    A crash fraction is set only on a filled target change and remains in fixed
    units, including after lock expiry, until the next filled target change.
    Missing held prices carry their last observed mark and resume with catch-up;
    a rebalance requiring any missing held/target price is blocked atomically.
    No missing cash price is backfilled using a future observation.
    """
    spec = deepcopy(spec or {})
    config = deepcopy(base_config)
    config.update(deepcopy(spec.get("config_changes", {})))
    if lag is not None:
        config["lag"] = lag
    crash_weight = float(spec.get("crash_weight", 1.))
    if crash_weight not in (1., .75, .5):
        raise ValueError("Unregistered crash sleeve weight")
    if float(spec.get("fallback_hysteresis", 0.)) not in (0., .02):
        raise ValueError("Unregistered fallback hysteresis")
    if spec.get("defensive_lowvol", False) not in (False, True):
        raise ValueError("defensive_lowvol must be boolean")
    if config["lag"] not in (0, 1) or not math.isfinite(fee) or not 0 <= fee < .5:
        raise ValueError("Unsupported lag or fee")
    dates = meta["dates"]
    lo = bisect_left(dates, dates[0] if start is None else start)
    hi = bisect_right(dates, dates[-1] if end is None else end) - 1
    if lo > hi or lo >= len(dates):
        raise ValueError("Empty simulation interval")
    data = DatedValues(arrays, meta, config)
    if CASH not in data.asset_index:
        raise ValueError("Cash ETF missing from declared asset set")
    policy, account = ResearchPolicy(data, config, spec), Portfolio()
    units, marks = {}, {}
    returns, holdings, trace, weight_rows, unit_rows, turnover_rows, cost_rows = [], [], [], [], [], [], []
    cash_missing = []
    for day in range(lo, hi + 1):
        previous_nav = account.nav
        current_price = data.price(day, account.holding)
        can_sell = account.holding is None or math.isfinite(current_price)
        if not can_sell:
            account.missing += 1
        missing_held = [code for code in units if not math.isfinite(data.price(day, code))]
        if CASH in missing_held:
            cash_missing.append(dict(date=dates[day], kind="held_cash_mark_carried", carried_mark=marks[CASH]))
        if day > lo:
            for code in units:
                price = data.price(day, code)
                if math.isfinite(price):
                    if price <= 0:
                        raise ValueError("Nonpositive observed price")
                    marks[code] = price
            if units:
                account.nav = sum(units[code] * marks[code] for code in units)
            if account.holding is not None and can_sell:
                account.mark = current_price
            account.age += 1
        marked = {code: units[code] * marks[code] for code in units}
        pretrade_nav = account.nav
        target, panic, crash = policy.intent(day, account, can_sell)
        filled, turnover, cost, blocked_missing = False, 0., 0., []
        desired = None
        if target is not None and target != account.holding:
            fraction = crash_weight if crash and target != CASH else 1.
            desired = {target: fraction}
            if fraction < 1:
                desired[CASH] = 1 - fraction
            needed = set(units) | set(desired)
            blocked_missing = sorted(code for code in needed if not math.isfinite(data.price(day, code)))
            if CASH in blocked_missing and CASH not in missing_held:
                cash_missing.append(dict(date=dates[day], kind="cash_required_trade_blocked", carried_mark=None))
            if not can_sell or blocked_missing:
                account.blocked += 1
            else:
                if any(data.price(day, code) <= 0 for code in needed):
                    raise ValueError("Nonpositive trade price")
                if panic and account.holding is not None and config["panic_cooldown"] > 0:
                    policy.cooldown_until[account.holding] = day + config["panic_cooldown"] + 1
                # Legacy delayed-first-entry also charges two sides. Uninvested
                # initial NAV is treated as a sold cash balance for this model.
                turnover_amount = sum(abs(desired.get(code, 0.) * pretrade_nav - marked.get(code, 0.))
                                      for code in needed)
                if not units:
                    turnover_amount += pretrade_nav
                turnover = turnover_amount / pretrade_nav
                if day > lo:
                    cost = fee * turnover_amount
                    account.nav = pretrade_nav - cost
                    account.switches += 1
                account.entries += 1
                units = {code: account.nav * weight / data.price(day, code) for code, weight in desired.items()}
                marks = {code: data.price(day, code) for code in units}
                account.holding, account.mark = target, data.price(day, target)
                account.age, account.asset_peak = 0, account.mark
                policy.rotation.clear()
                if crash:
                    account.lock_until = day + config["crash_lock"]
                    account.crashes += 1
                policy.after_fill(account)
                filled = True
        if not math.isfinite(account.nav) or account.nav <= 0:
            raise ArithmeticError("Invalid research NAV")
        account.nav_peak = max(account.nav_peak, account.nav)
        account.max_dd = min(account.max_dd, account.nav / account.nav_peak - 1)
        weights = {code: units[code] * marks[code] / account.nav for code in units}
        returns.append(account.nav / previous_nav - 1)
        holdings.append(data.asset_index[account.holding] if account.holding is not None else -1)
        weight_rows.append([weights.get(code, 0.) for code in data.assets])
        unit_rows.append([units.get(code, 0.) for code in data.assets])
        turnover_rows.append(turnover)
        cost_rows.append(cost)
        trace.append(dict(date=dates[day], signal_date=dates[day-config["lag"]] if day >= config["lag"] else None,
                          intended=target, holding=account.holding, panic=panic, crash_requested=crash,
                          filled=filled, lock_until=account.lock_until, age=account.age,
                          locked_panic_exit_requested=policy.locked_exit_requested,
                          confirmation_target=policy.rotation.candidate, confirmation_count=policy.rotation.count,
                          fallback_used=policy.fallback_used, weights=weights, units=dict(units),
                          desired_weights=desired, marked_values=marked, pretrade_nav=pretrade_nav,
                          nav=account.nav, turnover=turnover, cost=cost,
                          missing_held=missing_held, blocked_missing=blocked_missing,
                          initial_session_free=day == lo))
    count = hi - lo + 1
    summary = [account.nav, account.nav ** (244. / count) - 1, account.max_dd,
               account.switches, account.entries, account.blocked, account.missing, account.crashes,
               sum(returns), sum(r*r for r in returns)]
    return dict(returns=np.asarray(returns), holdings=np.asarray(holdings, dtype=np.int32),
                summary=np.asarray(summary), dates=dates[lo:hi+1], trace=trace,
                weights=np.asarray(weight_rows), units=np.asarray(unit_rows),
                turnover=np.asarray(turnover_rows), costs=np.asarray(cost_rows), cash_missing=cash_missing,
                metadata=dict(config=config, spec=spec, assets=list(data.assets),
                              allocation="fixed_units_until_target_change",
                              fee_model="legacy_pre_fee_target_turnover_approximation",
                              full_switch_exact_self_financing_factor=(1-fee)/(1+fee),
                              full_switch_legacy_factor=1-2*fee,
                              full_switch_factor_difference=2*fee*fee/(1+fee)))
