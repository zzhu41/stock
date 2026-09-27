"""Original close-clock policy: i+5 locks and no persistent pending orders."""
from v10_search.policy import SearchPolicy


class ClosePolicy(SearchPolicy):
    """Reuse only the verified ordinary decision logic and dated feature bank.

    The inherited next-open pending/fill logic is deliberately not called.
    Crash requests can be created only when the current holding has today's
    sellable close, exactly matching the original close backtest.
    """
    def __init__(self, config, bank, frame=None, fear=None, trace=False, oracle=False):
        super().__init__(config, bank, frame=frame, fear=fear, trace=trace, oracle=oracle)
        self.lock_until = -1
        self.metadata = dict(trace=[], crash_buys=[])

    def __call__(self, i, holding, holding_days, can_sell):
        table = [(code, ind) for code, ind in self.ranked[i] if code in self.feature_pool]
        info = dict(table)
        target = self._decide(table, info, holding)
        if i < self.lock_until:
            target = holding
        elif self.channels and can_sell:
            fear = bool(self.fear[i]) if self.fear is not None and "qvix" in self.channels else False
            for code, ind in table:
                if code not in self.trade_pool or code == holding:
                    continue
                deep = ("deep" in self.channels and self.params["crash_mom5"] < 0
                        and ind["mom5"] <= self.params["crash_mom5"]
                        and ind["dist_ma250"] < -self.params["crash_below_ma"])
                qvix = fear and ind["mom5"] <= -.04 and ind["dist_ma250"] < -.20
                volume = ("volume" in self.channels and ind["volume_ratio20"] >= 2
                          and ind["mom5"] <= -.04 and ind["dist_ma250"] < -.10)
                if deep or qvix or volume:
                    target = code
                    self.lock_until = i + self.params["crash_lock"]
                    if self.trace_enabled:
                        self.metadata["crash_buys"].append((self.bank.calendar[i], code))
                    break
        if target is not None and target not in self.trade_pool and target != self.config["cash"]:
            raise AssertionError("Excluded or benchmark-only asset was bought")
        if self.trace_enabled:
            self.metadata["trace"].append((self.bank.calendar[i], target))
        return target
