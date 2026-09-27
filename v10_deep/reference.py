"""Readable independent Python specification for the bounded single-asset model.

It shares dated feature/score values with the native engine, but never uses its
encoded parameters, precomputed order array, decision function or portfolio
state. This is a verification oracle, not a production or search entry point.
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import math

import numpy as np


CASH, GOLD, BENCHMARK = "511880", "518880", "510300"


@dataclass
class Confirmation:
    candidate: object = None
    count: int = 0

    def clear(self):
        self.candidate, self.count = None, 0

    def observe(self, candidate, required):
        self.count = self.count + 1 if self.candidate == candidate else 1
        self.candidate = candidate
        return self.count >= required


@dataclass
class Portfolio:
    holding: object = None
    nav: float = 1.0
    mark: float = 0.0
    age: int = 0
    asset_peak: float = 0.0
    nav_peak: float = 1.0
    max_dd: float = 0.0
    switches: int = 0
    entries: int = 0
    blocked: int = 0
    missing: int = 0
    crashes: int = 0
    lock_until: int = -1


class DatedValues:
    def __init__(self, arrays, meta, config):
        self.features = arrays["features"]
        self.scores = arrays["scores"][meta["score_names"].index(config["score"])]
        self.fear = arrays["fear"]
        self.assets = tuple(meta["assets"])
        self.asset_index = {code: index for index, code in enumerate(self.assets)}
        self.fields = {name: index for index, name in enumerate(meta["feature_names"])}

    def value(self, day, code, field):
        if code is None:
            return float("nan")
        return float(self.features[day, self.asset_index[code], self.fields[field]])

    def price(self, day, code):
        return self.value(day, code, "close")

    def informed(self, day, code):
        return code not in (None, CASH) and self.value(day, code, "valid") > .5 and math.isfinite(self.price(day, code))

    def score(self, day, code):
        return float(self.scores[day, self.asset_index[code]]) if self.informed(day, code) else 0.0

    def ranked(self, day):
        # Python's stable sort preserves the declared input asset order on ties.
        usable = [code for code in self.assets if self.informed(day, code)]
        return sorted(usable, key=lambda code: -self.score(day, code))


class Regime:
    def __init__(self):
        self.current = None
        self.change = Confirmation()

    def resolve(self, data, day, config, stocks):
        mode = config["regime"]
        if mode in ("always_bull", "all_assets"):
            return True
        distance = data.value(day, BENCHMARK, config["ma"]) if data.informed(day, BENCHMARK) else 1.0
        if not math.isfinite(distance):
            distance = 1.0  # Legacy initialization behavior, not evidence of an observed bullish trend.
        if mode == "breadth":
            distances = [data.value(day, code, config["ma"]) for code in stocks if data.informed(day, code)]
            distances = [value for value in distances if math.isfinite(value)]
            raw = sum(value > 0 for value in distances) / len(distances) >= config["breadth_threshold"] if distances else distance > 0
        else:
            threshold = 0.0 if self.current is None else -config["regime_hyst"] if self.current else config["regime_hyst"]
            raw = distance > threshold
            if mode == "dual":
                raw = raw and data.value(day, BENCHMARK, "mom20") > 0
        if self.current is None:
            self.current = raw
            self.change.clear()
        elif raw == self.current:
            self.change.clear()
        elif config["regime_confirm"] <= 1 or self.change.observe(raw, config["regime_confirm"]):
            self.current = raw
            self.change.clear()
        return self.current


class Policy:
    def __init__(self, data, config):
        self.data, self.config = data, config
        self.stocks, self.globals = set(config["stock_pool"]), set(config["global_pool"])
        self.trade_pool = self.stocks | self.globals | {GOLD}
        self.safe_pool = self.globals | {GOLD}
        self.regime = Regime()
        self.rotation = Confirmation()
        self.cooldown_until = {code: -1 for code in data.assets}

    def momentum(self, day, code, window):
        return self.data.value(day, code, "mom%d" % window)

    def emergency(self, day, account):
        p, d, code = self.config, self.data, account.holding
        if not d.informed(day, code):
            return False
        account.asset_peak = max(account.asset_peak, d.price(day, code))
        threshold = p["panic"]
        if p["panic_mode"] == "volatility":
            threshold = min(.10, max(.02, threshold * d.value(day, code, "vol20")))
        daily_shock = p["panic"] > 0 and d.value(day, code, "ret1") <= -threshold
        trailing = p["trail_stop"] > 0 and account.asset_peak > 0 and d.price(day, code) / account.asset_peak - 1 <= -p["trail_stop"]
        return daily_shock or trailing

    def choose_regular(self, day, execution_day, account, bull, panic, ranked):
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
        alternatives = [code for code in permitted if code in self.safe_pool]
        if p["fallback_mode"] == "cash" or not alternatives:
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
        p = self.config
        excluded_stock = not bull and p["regime"] != "open_stock" and holding in self.stocks
        failed_trend = self.momentum(day, holding, p["exit_mom"]) <= p["exit_floor"]
        overheated = self.momentum(day, holding, p["mom"]) > p["overheat"] and self.momentum(day, holding, p["fast_mom"]) <= 0
        return excluded_stock or failed_trend or overheated

    def should_keep_healthy(self, day, target, account, competition, ranked):
        p, d, holding = self.config, self.data, account.holding
        if account.age < p["min_hold"]:
            return True
        buffer = p["buffer"]
        if d.informed(day, target):
            if target == GOLD and p["gold_buffer"] >= 0:
                buffer = p["gold_buffer"]
            elif target in self.globals and p["global_buffer"] >= 0:
                buffer = p["global_buffer"]
        if p["trend_buffer"] > 0 and self.momentum(day, holding, p["mom"]) > .10:
            buffer = max(buffer, p["trend_buffer"])
        mode = p["buffer_mode"]
        if mode == "rank":
            competitors = [code for code in ranked if code in competition]
            return holding in competitors and competitors.index(holding) + 1 <= p["rank_keep"]
        if mode == "momentum":
            target_value = self.momentum(day, target, p["mom"]) if d.informed(day, target) else 0.0
            return target_value - self.momentum(day, holding, p["mom"]) < buffer
        if mode == "score_relative":
            buffer *= max(abs(d.score(day, holding)), 1e-9)
        return d.score(day, target) - d.score(day, holding) < buffer

    def crash_candidate(self, day, execution_day, account, can_sell, ranked):
        p, d = self.config, self.data
        if not p["crash_mask"] or not can_sell:
            return None
        holding = account.holding
        risk_held = holding not in (None, CASH)
        guard = p["crash_guard"]
        if risk_held:
            if guard == "cash_only" or (guard != "always" and not d.informed(day, holding)):
                return None
            if guard == "held_negative" and not (self.momentum(day, holding, 20) <= 0):
                return None
            if guard == "held_shock" and not (d.value(day, holding, "ret1") <= -.04 or self.momentum(day, holding, 5) <= -.08):
                return None
        eligible = []
        for code in ranked:
            if code not in self.trade_pool or code == holding or execution_day < self.cooldown_until[code]:
                continue
            if p["crash_stock_only"] > .5 and code not in self.stocks:
                continue
            if guard == "candidate_better" and risk_held and d.score(day, code) <= d.score(day, holding):
                continue
            mom5, distance = self.momentum(day, code, 5), d.value(day, code, "ma250")
            deep = bool(p["crash_mask"] & 1) and mom5 <= p["deep_mom"] and distance < -p["deep_below"]
            fear = bool(p["crash_mask"] & 2) and bool(d.fear[day]) and mom5 <= p["relaxed_mom"] and distance < -.20
            volume = bool(p["crash_mask"] & 4) and d.value(day, code, "volume_ratio") >= p["volume_ratio"] \
                and mom5 <= p["relaxed_mom"] and distance < -p["volume_below"]
            if deep or fear or volume:
                eligible.append(code)
        if not eligible:
            return None
        if p["crash_pick"] == "score":
            return eligible[0]
        criterion = (lambda code: self.momentum(day, code, 5)) if p["crash_pick"] == "deepest" else (
            lambda code: d.value(day, code, "vol20"))
        eligible = [code for code in eligible if criterion(code) < 1e100]
        return min(eligible, key=criterion) if eligible else None

    def intent(self, execution_day, account, can_sell):
        p, d = self.config, self.data
        signal_day = execution_day - p["lag"]
        if signal_day < 0:
            self.rotation.clear()
            return account.holding, False, False
        bull = self.regime.resolve(d, signal_day, p, self.stocks)
        panic = self.emergency(signal_day, account)
        if execution_day < account.lock_until:
            self.rotation.clear()
            return account.holding, panic, False
        ranked = d.ranked(signal_day)
        target, competition = self.choose_regular(signal_day, execution_day, account, bull, panic, ranked)
        confirmation_used = False
        healthy_rotation = (target != account.holding and d.informed(signal_day, account.holding)
                            and not panic and not self.must_exit(signal_day, account.holding, bull))
        if healthy_rotation:
            if self.should_keep_healthy(signal_day, target, account, competition, ranked):
                target = account.holding
            elif p["switch_confirm"] > 1:
                confirmation_used = True
                if not self.rotation.observe(target, p["switch_confirm"]):
                    target = account.holding
        if not confirmation_used:
            self.rotation.clear()
        crash = self.crash_candidate(signal_day, execution_day, account, can_sell, ranked)
        if crash is not None:
            self.rotation.clear()
            # A missing current quote blocks this whole intended order. It
            # does not authorize an ordinary replacement or a fictitious lock.
            return crash, panic, True
        return target, panic, False


def run_reference(arrays, meta, config, start=None, end=None, fee=.0001):
    """Return native-compatible day arrays and summary for one configuration."""
    if config["lag"] not in (0, 1) or not math.isfinite(fee) or not 0 <= fee < .5:
        raise ValueError("Unsupported lag or fee")
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
                          confirmation_target=policy.rotation.candidate, confirmation_count=policy.rotation.count))
    count = hi - lo + 1
    summary = [account.nav, account.nav ** (244.0 / count) - 1, account.max_dd,
               account.switches, account.entries, account.blocked, account.missing, account.crashes,
               sum(returns), sum(r * r for r in returns)]
    return dict(returns=np.asarray(returns), holdings=np.asarray(holdings, dtype=np.int32),
                summary=np.asarray(summary), dates=dates[lo:hi + 1], trace=trace)
