"""Frozen simple-profile adaptation; no network, account reads, or writes.

WLS25 deliberately uses the original list/sum arithmetic, including its
20-return volatility denominator. It never substitutes the H WLS20 score.
"""
import math

from v10_deep.cli import load_profile
from v10_deep.reference import Policy, Portfolio
from v10_live.policy import (ASSETS, CASH, BENCHMARK, Snapshot, _date,
    _history_prefix, _public_indicators, _channels, _verify_reference, qvix_state)

CANDIDATE_ID = "vd_d2a02ab14be481f562fd"
CANDIDATE_HASH = "d2a02ab14be481f562fd674887950bb1a7876b52ed75964b43ea4edc5ae518f3"


def latest_features(rows, signal_date):
    out = dict(close=float("nan"), valid=0., score=0., bars=len(rows),
               ret1=float("nan"), mom5=float("nan"), mom20=float("nan"),
               mom60=float("nan"), vol20=float("nan"), ma180=float("nan"),
               ma250=float("nan"), volume_ratio=float("nan"))
    if not rows or rows[-1][0] != signal_date:
        return out
    closes = [r[1] for r in rows]
    out["close"] = closes[-1]
    if len(rows) < 270:
        return out
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - 20, len(closes))]
    mean_return = sum(rets) / len(rets)
    vol = (sum((r - mean_return) ** 2 for r in rets) / len(rets)) ** 0.5
    # Exact SCORE_WLS branch of frozen strategy.indicators with the v9.1
    # preset used by the frozen FeatureBank. No mutable strategy globals.
    seg = closes[-25:]
    xs = list(range(len(seg)))
    wts = [i + 1 for i in xs]
    wsum = sum(wts)
    wmx = sum(wts[i] * xs[i] for i in xs) / wsum
    wmy = sum(wts[i] * seg[i] for i in xs) / wsum
    wsxy = sum(wts[i] * (xs[i] - wmx) * (seg[i] - wmy) for i in xs)
    wsxx = sum(wts[i] * (xs[i] - wmx) ** 2 for i in xs)
    slope = (wsxy / wsxx) / wmy if wsxx > 0 and wmy > 0 else 0.0
    score = slope * 250 / vol if vol > 0 else 0.0
    avg_volume = sum(r[2] for r in rows[-21:-1]) / 20
    out.update(valid=1., score=score, ret1=closes[-1] / closes[-2] - 1.0,
               mom5=closes[-1] / closes[-6] - 1.0,
               mom20=closes[-1] / closes[-21] - 1.0,
               mom60=closes[-1] / closes[-61] - 1.0,
               vol20=vol, ma180=closes[-1] / (sum(closes[-180:]) / 180) - 1.,
               ma250=closes[-1] / (sum(closes[-250:]) / 250) - 1.,
               volume_ratio=rows[-1][2] / avg_volume if avg_volume > 0 else 0.)
    return out


def snapshot(histories, calendar, signal_date, fear=False):
    _date(signal_date)
    dates = [_date(d) for d in calendar]
    if dates != sorted(set(dates)) or signal_date not in dates:
        raise ValueError("V9.2+ requires an observed unique benchmark date")
    dates = [d for d in dates if d <= signal_date]
    if set(ASSETS) - set(histories):
        raise ValueError("V9.2+ missing original-universe TR histories")
    indicators = {code: latest_features(_history_prefix(histories[code], signal_date, code), signal_date)
                  for code in ASSETS}
    indicators[CASH].update(valid=0., score=0.)
    if not math.isfinite(indicators[BENCHMARK]["close"]):
        raise ValueError("V9.2+ benchmark has no same-day quote")
    return Snapshot(indicators, len(dates) - 1, fear), dates


def verified_profile():
    profile = load_profile("simple")
    _verify_reference()
    config = profile["config"]
    if config["id"] != CANDIDATE_ID or config["hash"] != CANDIDATE_HASH:
        raise ValueError("V9.2+ frozen simple profile mismatch")
    return profile


def decide(histories, calendar, signal_date, state=None, qvix_path=None):
    config = verified_profile()["config"]
    state = dict(state or {})
    qvix = qvix_state(signal_date, qvix_path)
    data, dates = snapshot(histories, calendar, signal_date, qvix["active"])
    day = len(dates) - 1
    holding = state.get("holding") or None
    if holding is not None and holding not in ASSETS:
        raise ValueError("V9.2+ unknown independent holding")
    if state.get("last_date") and state["last_date"] > signal_date:
        raise ValueError("V9.2+ signal predates saved state")
    entry = state.get("entry_date")
    if holding and not entry:
        raise ValueError("V9.2+ held position requires its actual entry date")
    if entry and (_date(entry) > signal_date or entry not in dates):
        raise ValueError("V9.2+ cannot locate actual entry date")
    trigger, lock_code = state.get("crash_trigger_date"), state.get("crash_code")
    if bool(trigger) != bool(lock_code):
        raise ValueError("V9.2+ crash date/code must be stored together")
    lock_until = -1
    if trigger:
        if lock_code != holding or _date(trigger) not in dates or trigger != entry:
            raise ValueError("V9.2+ crash lock must match actual entry and holding")
        lock_until = dates.index(trigger) + config["crash_lock"]
    active = day < lock_until
    age = day - dates.index(entry) if entry else 0
    account = Portfolio(holding=holding, age=age, asset_peak=0., lock_until=lock_until)
    model = Policy(data, config)
    can_sell = holding is None or math.isfinite(data.price(day, holding))
    intended, panic, requested_crash = model.intent(day, account, can_sell)
    can_buy = intended is not None and math.isfinite(data.price(day, intended))
    executable = bool(can_sell and can_buy)
    target = intended if executable else holding
    crash = bool(requested_crash and executable and target != holding)
    next_trigger, next_code = (trigger, lock_code) if active else (None, None)
    if crash:
        next_trigger, next_code = signal_date, target
    channels = _channels(data, day, intended, config) if requested_crash else []
    if not executable:
        reason = "持仓或目标缺当日报价，本次无有效成交建议"
    elif active:
        reason = "抄底锁仓：入场%s，第5个后续交易日恢复决策" % trigger
    elif crash:
        reason = "抄底(%s)，实际虚拟入场后锁仓5个交易日" % "∪".join(channels)
    else:
        action = "继续持有" if target == holding else "首次建仓" if holding is None else "调仓"
        reason = ("固定4%急跌退出；" if panic else "") + action + "；WLS25 / MA250 / 健康轮动最短2日"
    return dict(candidate_id=CANDIDATE_ID, signal_date=signal_date, target=target,
                intended_target=intended, reason=reason, panic=bool(panic), crash=crash,
                crash_requested=bool(requested_crash), crash_trigger_date=next_trigger,
                crash_code=next_code, executable=executable, lock_active=active or crash,
                lock_remaining_sessions=5 if crash else max(0, lock_until - day),
                research_observation_only=True, order_submission=False,
                diagnostics=dict(qvix=qvix, bull=bool(model.regime.current), ranking=data.ranked(day),
                    indicators=_public_indicators(data), panic_threshold=.04,
                    crash_channels=channels, actual_holding_age=age,
                    healthy_min_hold=2, cold_start=holding is None,
                    previous_holding=holding,
                    feature_clock="Current supplied TR quote; not the frozen final-close backtest"))
