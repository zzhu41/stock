"""Independent V12 oracle: frozen Policy plus one explicitly settled lock exit.

No native parameter encoding or decision function is used here. All normal
policy rules are inherited unchanged; the executor retains actual holding/age
state and clears a crash lock only after its requested cash exit was filled.
"""
from bisect import bisect_left, bisect_right
import math
import numpy as np

from v10_deep.reference import (Policy as FrozenPolicy, Portfolio, DatedValues,
                               CASH, GOLD, BENCHMARK)
from .schema import validate_locked_panic_exit


class Policy(FrozenPolicy):
    def __init__(self, data, config):
        self.locked_panic_exit = validate_locked_panic_exit(config)
        self.locked_exit_requested = False
        super().__init__(data, config)

    def intent(self, execution_day, account, can_sell):
        self.locked_exit_requested = False
        target, panic, crash = super().intent(execution_day, account, can_sell)
        if (self.locked_panic_exit and execution_day < account.lock_until
                and account.holding not in (None, CASH) and panic
                and account.age > self.config["lag"]):
            self.rotation.clear()
            self.locked_exit_requested = True
            return CASH, panic, False
        return target, panic, crash

    def after_fill(self, account):
        """Caller must invoke only after the proposed target actually settled."""
        if self.locked_exit_requested:
            if account.holding != CASH:
                raise ValueError("Locked panic exit can clear only after a CASH fill")
            account.lock_until = -1


def run_reference(arrays, meta, config, start=None, end=None, fee=.0001):
    """Return native-compatible day arrays and summary for one configuration."""
    if config["lag"] not in (0, 1) or not math.isfinite(fee) or not 0 <= fee < .5:
        raise ValueError("Unsupported lag or fee")
    validate_locked_panic_exit(config)
    dates = meta["dates"]
    lo = bisect_left(dates, start if start is not None else dates[0])
    hi = bisect_right(dates, end if end is not None else dates[-1]) - 1
    if lo > hi or lo >= len(dates):
        raise ValueError("Empty simulation interval")
    data = DatedValues(arrays, meta, config)
    policy, account = Policy(data, config), Portfolio()
    returns, holdings, trace = [], [], []
    for day in range(lo, hi + 1):
        previous_nav = account.nav
        current_price = data.price(day, account.holding)
        can_sell = account.holding is None or math.isfinite(current_price)
        if not can_sell:
            account.missing += 1
        if day > lo:
            if account.holding is not None and can_sell:
                account.nav *= current_price / account.mark
                account.mark = current_price
            account.age += 1
        target, panic, crash = policy.intent(day, account, can_sell)
        filled = False
        if target is not None and target != account.holding:
            purchase_price = data.price(day, target)
            if not can_sell or not math.isfinite(purchase_price):
                account.blocked += 1
            else:
                if panic and account.holding is not None and config["panic_cooldown"] > 0:
                    policy.cooldown_until[account.holding] = day + config["panic_cooldown"] + 1
                if day > lo:
                    account.nav *= 1 - 2 * fee
                    account.switches += 1
                account.entries += 1
                account.holding, account.mark = target, purchase_price
                account.age, account.asset_peak = 0, purchase_price
                policy.rotation.clear()
                if crash:
                    account.lock_until = day + config["crash_lock"]
                    account.crashes += 1
                policy.after_fill(account)
                filled = True
        if not math.isfinite(account.nav) or account.nav <= 0:
            raise ArithmeticError("Invalid reference NAV")
        account.nav_peak = max(account.nav_peak, account.nav)
        account.max_dd = min(account.max_dd, account.nav / account.nav_peak - 1)
        returns.append(account.nav / previous_nav - 1)
        holdings.append(data.asset_index[account.holding] if account.holding is not None else -1)
        trace.append(dict(date=dates[day], signal_date=dates[day - config["lag"]] if day >= config["lag"] else None,
                          intended=target, holding=account.holding, panic=panic, crash_requested=crash,
                          filled=filled, lock_until=account.lock_until,
                          locked_panic_exit_requested=policy.locked_exit_requested,
                          confirmation_target=policy.rotation.candidate, confirmation_count=policy.rotation.count))
    count = hi - lo + 1
    summary = [account.nav, account.nav ** (244.0 / count) - 1, account.max_dd,
               account.switches, account.entries, account.blocked, account.missing, account.crashes,
               sum(returns), sum(r * r for r in returns)]
    return dict(returns=np.asarray(returns), holdings=np.asarray(holdings, dtype=np.int32),
                summary=np.asarray(summary), dates=dates[lo:hi + 1], trace=trace)
