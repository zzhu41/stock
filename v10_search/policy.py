"""Fast decision policy for this registry's explicitly bounded active switches.

All unsupported behavior arguments must remain at frozen v9.1 values; it is
not a general replacement for production strategy.py. A slow original-function
oracle is available for verification on the same actual portfolio state.
"""
from copy import deepcopy

from v10_next.data import configure
from v10_next.frozen import strategy
from .data import CASH, GOLD, BENCHMARK, UNIVERSE


SUPPORTED = {"pool_buffer", "buffer", "bear_enter_mom", "panic_drop", "overheat", "never_empty",
             "crash_mom5", "score_wls", "score_mom", "score_plain_mom", "bear_open_stock"}


class SearchPolicy:
    def __init__(self, config, bank, frame=None, fear=None, oracle=False, trace=False, same_close=False):
        self.config, self.bank, self.frame = deepcopy(config), bank, frame
        self.params = deepcopy(config["params"])
        for key, value in self.params.items():
            if key not in SUPPORTED and value != bank.presets["v9.1"][key]:
                raise ValueError("Unregistered indicator/engine change: " + key)
        self.stock = frozenset(config["stock_pool"])
        self.global_ = frozenset(config["global_pool"])
        self.trade_pool = self.stock | self.global_ | {GOLD}
        self.feature_pool = self.trade_pool | {BENCHMARK}
        self.ranked = bank.ranked(config["score_mode"], config["score_windows"])
        self.channels = tuple(config.get("channels", ()))
        self.regime_mode = config.get("regime_mode", "ma250")
        if self.regime_mode not in ("ma250", "open_stock", "always_bull", "all_assets"):
            raise ValueError("Unsupported regime mode")
        self.fear = fear
        self.oracle = oracle
        self.lock_duration = self.params["crash_lock"] - 1 + int(same_close)
        self.trace_enabled = trace
        self.metadata = dict(trace=[], crash_events=[])
        self.pending = None
        self.lock_code = None
        self.lock_entry = None
        self.comp_bull = self.stock | self.global_
        self.comp_bear = self.global_ | {GOLD}
        self.safe = self.comp_bear
        strategy.UNIVERSE.update(UNIVERSE)  # isolated process memory only, never a source file
        configure(self.params, config["stock_pool"], config["global_pool"])
        strategy._state_bull = None
        strategy.BULL_HYST_PENDING = None

    def _decide(self, table, info, holding):
        p = self.params
        bull = (True if self.regime_mode in ("always_bull", "all_assets")
                else info.get(BENCHMARK, {}).get("above_ma", True))
        h = info.get(holding)
        panic = bool(h and holding != CASH and p["panic_drop"] > 0 and h["ret1"] <= -p["panic_drop"])
        comp = (self.trade_pool if self.regime_mode == "all_assets" else
                self.comp_bull if (bull or p["bear_open_stock"]) else self.comp_bear)
        floor = 0.0 if bull else max(0.0, p["bear_enter_mom"])
        best = None
        for code, ind in table:
            if code not in comp or (panic and code == holding):
                continue
            if ind["mom20"] <= floor or (ind["mom20"] > p["overheat"] and ind["mom5"] <= 0):
                continue
            if ind["max_ret"] > p["max_cap"]:
                continue
            best = code
            break
        if best:
            target = best
        elif bull and GOLD in info and info[GOLD]["mom20"] > 0 and not (panic and holding == GOLD):
            target = GOLD
        elif p["never_empty"]:
            target = next((code for code, ind in table if code in self.safe and not (panic and code == holding)), CASH)
        else:
            target = CASH
        if panic:
            return target
        if holding and holding != target and h is not None:
            if not bull and not p["bear_open_stock"] and holding in self.stock:
                return target
            if (h["mom20"] <= 0 or (p["panic_drop"] > 0 and h["ret1"] <= -p["panic_drop"])
                    or (h["mom20_max"] > p["overheat"] and h["mom5"] <= 0)):
                return target
            buffer = p["buffer"]
            if p["pool_buffer"] and target in info:
                role = "gold" if target == GOLD else UNIVERSE[target][2]
                buffer = p["pool_buffer"].get(role, buffer)
            if info.get(target, {}).get("mom20", 0.0) - h["mom20"] < buffer:
                return holding
        return target

    def __call__(self, i, holding, entry_index, execution_deferred):
        if self.pending is not None:
            signal_i, code = self.pending
            if holding == code:
                if entry_index is None or entry_index < signal_i:
                    raise AssertionError("Crash fill was not confirmed")
                self.lock_code, self.lock_entry = code, entry_index
                if self.trace_enabled:
                    self.metadata["crash_events"].append(dict(signal_date=self.bank.calendar[signal_i],
                                                             code=code, fill_date=self.bank.calendar[entry_index]))
                self.pending = None
            elif execution_deferred:
                return None
            else:
                raise AssertionError("Crash entry neither filled nor deferred")
        table = [(code, ind) for code, ind in self.ranked[i] if code in self.feature_pool]
        info = dict(table)
        if self.oracle:
            # Slow, independent original-function path for checking the optimized
            # decision code against identical holdings, pool and information.
            configure(self.params, self.config["stock_pool"], self.config["global_pool"])
            oracle_table = self.bank.table(i, self.config["score_mode"], self.config["score_windows"], self.feature_pool, oracle=True)
            if self.regime_mode in ("always_bull", "all_assets"):
                oracle_table = [(code, dict(ind, above_ma=True, ma_rising=True) if code == BENCHMARK else ind)
                                for code, ind in oracle_table]
            original_global = strategy.GLOBAL_POOL
            try:
                if self.regime_mode == "all_assets":
                    strategy.GLOBAL_POOL = list(self.config["global_pool"]) + [GOLD]
                target, _ = strategy.decide(oracle_table, holding, i - entry_index + 1 if entry_index is not None else 0)
            finally:
                strategy.GLOBAL_POOL = original_global
        else:
            target = self._decide(table, info, holding)
        if self.lock_code and holding == self.lock_code and i - self.lock_entry < self.lock_duration:
            target = holding
        else:
            self.lock_code = self.lock_entry = None
            if self.channels:
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
                        self.pending = (i, code)
                        break
        if target not in self.trade_pool and target != CASH:
            raise AssertionError("A benchmark/excluded asset was accidentally traded")
        if self.trace_enabled:
            self.metadata["trace"].append((self.bank.calendar[i], target))
        if target == holding and not execution_deferred:
            return None
        if target == CASH:
            ds = self.bank.dates[CASH]
            # A money ETF need not meet the 270-bar ranking warmup, but must
            # have an actual current-day price before a new order is proposed.
            import bisect
            j = bisect.bisect_left(ds, self.bank.calendar[i])
            if j >= len(ds) or ds[j] != self.bank.calendar[i]:
                return ""
        return target
