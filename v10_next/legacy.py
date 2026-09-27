"""v9 decision rules adapted to a causal next-open execution clock, in isolation.

The original strategy file is untouched. Crash locks begin on actual entry;
cash allocations and signal timing are explicit, with no fixed-target replay.
"""
from copy import deepcopy

from .data import configure
from .frozen import strategy
from .frozen.metadata import CASH, GOLD


class LegacyPolicy:
    def __init__(self, candidate, features, calendar):
        self.candidate = deepcopy(candidate)
        self.features = features
        self.index = {d: i for i, d in enumerate(calendar)}
        self.params = deepcopy(features.presets[candidate.get("base_version", "v9.1")])
        self.params.update(candidate.get("overrides", {}))
        self.stock_pool = list(candidate["stock_pool"])
        self.global_pool = list(candidate["global_pool"])
        self.pool = self.stock_pool + self.global_pool + [GOLD]
        self.windows = tuple(candidate.get("score_windows", (25,)))
        self.risk_weight = candidate.get("risk_weight", 1.0)
        self.pending_crash = None
        self.lock_code = None
        self.lock_entry_index = None
        self.metadata = dict(crash_events=[], trace=[])
        configure(self.params, self.stock_pool, self.global_pool)
        strategy._state_bull = None
        strategy.BULL_HYST_PENDING = None

    def __call__(self, date, observed_histories, state):
        configure(self.params, self.stock_pool, self.global_pool)
        weights = state["weights"]
        risky = [c for c, w in weights.items() if c != CASH and w > 1e-9]
        if len(risky) > 1:
            raise ValueError("Legacy policy expects at most one risk asset")
        holding = risky[0] if risky else CASH if weights.get(CASH, 0) > 1e-9 else None
        i = self.index[date]
        if self.pending_crash and holding == self.pending_crash["code"]:
            entry = state["holding_since"].get(holding)
            if entry is None or entry < self.pending_crash["signal_date"]:
                raise AssertionError("Crash entry was not confirmed by an actual fill")
            self.lock_code = holding
            self.lock_entry_index = self.index[entry]
            self.metadata["crash_events"].append(dict(self.pending_crash, fill_date=entry))
            self.pending_crash = None
        elif self.pending_crash and state.get("execution_deferred"):
            self.metadata["trace"].append(dict(date=date, target=self.pending_crash["code"], reason="crash_order_deferred"))
            return None
        elif self.pending_crash:
            raise AssertionError("Crash order neither filled nor explicitly deferred")

        table = self.features.table(date, self.pool, self.windows)
        entry = state["holding_since"].get(holding)
        age = i - self.index[entry] + 1 if entry else 0
        target, reason = strategy.decide(table, holding, age)
        if self.lock_code and holding == self.lock_code and i - self.lock_entry_index < self.params["crash_lock"] - 1:
            target, reason = holding, "confirmed_crash_lock"
        else:
            self.lock_code = self.lock_entry_index = None
            if self.params["crash_mom5"] < 0:
                crash = next((c for c, ind in table
                              if c != holding and ind["mom5"] <= self.params["crash_mom5"]
                              and ind["dist_ma250"] < -self.params["crash_below_ma"]), None)
                if crash:
                    target, reason = crash, "deep_drop_entry"
                    self.pending_crash = dict(signal_date=date, code=crash)

        self.metadata["trace"].append(dict(date=date, target=target, reason=reason))
        if target == holding and not state.get("pending_target"):
            return None  # retain units; the half-exposure control can drift
        if target == CASH:
            return {CASH: 1.0} if self.features.cash_available(date) else {}
        target_weights = {target: self.risk_weight}
        if self.risk_weight < 1 and self.features.cash_available(date):
            target_weights[CASH] = 1 - self.risk_weight
        return target_weights


def factory(features, calendar):
    return lambda candidate: LegacyPolicy(candidate, features, calendar)
