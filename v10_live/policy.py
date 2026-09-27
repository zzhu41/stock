"""One-date frozen V10-H research observation; no account, network or file writes.

Inputs are corrected total-return index rows, NOT additive QFQ or raw prices:
(date, open_placeholder, close_index, high_placeholder, low_placeholder,
 split-normalized_volume). Only close and volume enter the model. This adapter
never imports the mutable root strategy or replays a historical portfolio.
"""
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from v10_deep.cli import load_profile
from v10_deep.features import _regression, _rolling_mean
from v10_deep import reference as reference_module
from v10_deep.reference import Policy, Portfolio

ROOT = Path(__file__).resolve().parent.parent
QVIX_PATH = ROOT / "data" / "qvix50.csv"
REFERENCE_RECEIPT = ROOT / "v10_deep/results/refinements/selected_reference_fidelity.json"
CANDIDATE_ID = "vd_7d52c290adc6c4503f7d"
ASSETS = ("159915", "588080", "510300", "510500", "563300", "512400", "512890",
          "513100", "513120", "518880", "511880")
CASH, GOLD, BENCHMARK = "511880", "518880", "510300"


def _verify_reference():
    """load_profile verifies this receipt's hash; also verify the live oracle.

    The native research CLI does not depend on reference.py and therefore does
    not pin it in its own source list. This live adapter does depend on it.
    """
    receipt = json.loads(REFERENCE_RECEIPT.read_text())
    expected = [value for name, value in receipt["source_and_cache_sha256"].items()
                if name.endswith("/v10_deep/reference.py")]
    actual = hashlib.sha256(Path(reference_module.__file__).read_bytes()).hexdigest()
    if len(expected) != 1 or actual != expected[0]:
        raise ValueError("Frozen reference.py changed since independent H fidelity verification")


def _date(value):
    if not isinstance(value, str) or datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") != value:
        raise ValueError("Invalid ISO signal/history date")
    return value


def qvix_state(signal_date, path=None):
    """Read one dated local snapshot; stale/missing/invalid QVIX disables ONLY QVIX.

    As in the frozen experiment, the reference window is up to 250 strictly
    previous observations with at least 120. No network refresh, date fill, or
    silent removal of non-market-calendar rows is performed here.
    """
    result = dict(requested_date=signal_date, date=None, value=None, z=None,
                  active=False, available=False, previous_samples=0,
                  note="当日QVIX缺失，QVIX通道已停用；深跌/量能通道仍可用",
                  publication_time_verified=False)
    path = Path(path) if path is not None else QVIX_PATH
    try:
        raw = path.read_bytes()
    except OSError:
        result["note"] = "本地QVIX文件不可用，QVIX通道已停用；深跌/量能通道仍可用"
        return result
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        rows = []
        for row in csv.reader(io.StringIO(raw.decode("utf-8-sig"))):
            if not row:
                continue
            date = _date(row[0])
            if date > signal_date:
                continue
            if len(row) < 2 or (rows and date <= rows[-1][0]):
                raise ValueError("QVIX dates must be unique and ascending")
            value = float(row[1])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid QVIX observation")
            rows.append((date, value))
        if not rows:
            return result
        result.update(date=rows[-1][0], value=rows[-1][1])
        if rows[-1][0] != signal_date:
            result["note"] = "当日%s的QVIX缺失（最新%s），QVIX通道已停用；深跌/量能通道仍可用" % (signal_date, rows[-1][0])
            return result
        previous = [value for _, value in rows[-251:-1]]
        result["previous_samples"] = len(previous)
        if len(previous) < 120:
            result["note"] = "QVIX先前观测不足120行，QVIX通道已停用；深跌/量能通道仍可用"
            return result
        mean = sum(previous) / len(previous)
        sigma = (sum((value - mean) ** 2 for value in previous) / len(previous)) ** .5
        z = (rows[-1][1] - mean) / sigma if sigma > 0 else 0.0
        result.update(z=z, active=z >= 2.5, available=True, note="")
        return result
    except (ValueError, TypeError, IndexError, UnicodeError):
        result.update(active=False, available=False,
                      note="本地QVIX记录无效，QVIX通道已停用；深跌/量能通道仍可用")
        return result


def _history_prefix(rows, signal_date, code):
    """Validate date order; inspect price/volume values only at or before asof."""
    prefix, previous = [], None
    for row in rows:
        if len(row) != 6:
            raise ValueError("%s requires six-column TR rows" % code)
        date = _date(row[0])
        if previous is not None and date <= previous:
            raise ValueError("%s history dates must be unique and ascending" % code)
        previous = date
        if date > signal_date:
            continue
        close, volume = float(row[2]), float(row[5])
        if not math.isfinite(close) or close <= 0 or not math.isfinite(volume) or volume < 0:
            raise ValueError("%s has invalid TR close/normalized volume" % code)
        prefix.append((date, close, volume))
    return prefix


def _latest_features(rows, signal_date):
    out = dict(close=float("nan"), valid=0., score=0., bars=len(rows),
               ret1=float("nan"), mom5=float("nan"), mom20=float("nan"),
               mom60=float("nan"), vol20=float("nan"), ma180=float("nan"),
               ma250=float("nan"), volume_ratio=float("nan"))
    if not rows or rows[-1][0] != signal_date:
        return out
    prices = np.asarray([row[1] for row in rows], dtype=np.float64)
    volumes = np.asarray([row[2] for row in rows], dtype=np.float64)
    out["close"] = float(prices[-1])
    if len(rows) < 270:
        return out
    # Score arithmetic matches frozen features.py BEFORE its exact-control
    # vol20 overwrite: convolution of return^2 minus squared rolling mean.
    returns = np.concatenate(([0.], prices[1:] / prices[:-1] - 1))
    mean_return = _rolling_mean(returns, 20)
    score_sigma = np.sqrt(np.maximum(0., _rolling_mean(returns * returns, 20) - mean_return * mean_return))
    mean, slope, unused_r2 = _regression(prices, 20)
    trend = np.full(len(prices), np.nan)
    trend[19:] = slope / mean * 250
    daily_score = np.divide(trend, score_sigma, out=np.zeros(len(prices)), where=score_sigma > 0)
    score = float((daily_score[-1] + daily_score[-2] + daily_score[-3]) / 3)
    # Decision vol20 and MA250 are explicitly overwritten by the legacy
    # calculation in frozen features.py. Preserve that arithmetic as well.
    closes = [row[1] for row in rows]
    ret20 = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - 20, len(closes))]
    mu = sum(ret20) / len(ret20)
    decision_sigma = (sum((value - mu) ** 2 for value in ret20) / len(ret20)) ** .5
    previous_volume = sum(row[2] for row in rows[-21:-1]) / 20
    out.update(valid=1., score=score if math.isfinite(score) else -1e100,
               ret1=closes[-1] / closes[-2] - 1,
               mom5=closes[-1] / closes[-6] - 1,
               mom20=closes[-1] / closes[-21] - 1,
               mom60=closes[-1] / closes[-61] - 1,
               vol20=decision_sigma,
               ma180=float(prices[-1] / _rolling_mean(prices, 180)[-1] - 1),
               ma250=closes[-1] / (sum(closes[-250:]) / 250) - 1,
               volume_ratio=float(volumes[-1] / previous_volume) if previous_volume > 0 else 0.)
    return out


class Snapshot:
    """Policy's dated-data interface with only the requested day's features."""
    assets = ASSETS

    def __init__(self, indicators, day, fear):
        self.indicators, self.day = indicators, day
        self.fear = [False] * (day + 1)
        self.fear[day] = bool(fear)

    def _date_guard(self, day):
        if day != self.day:
            raise ValueError("Live snapshot cannot expose another signal date")

    def value(self, day, code, field):
        self._date_guard(day)
        return float(self.indicators.get(code, {}).get(field, float("nan")))

    def price(self, day, code):
        return self.value(day, code, "close")

    def informed(self, day, code):
        return code not in (None, CASH) and self.value(day, code, "valid") > .5 and math.isfinite(self.price(day, code))

    def score(self, day, code):
        return self.value(day, code, "score") if self.informed(day, code) else 0.

    def ranked(self, day):
        return sorted((code for code in self.assets if self.informed(day, code)),
                      key=lambda code: -self.score(day, code))


def snapshot(histories, calendar, signal_date, fear=False):
    """Pure asof feature construction; supplied histories must already be TR."""
    _date(signal_date)
    dates = [_date(day) for day in calendar]
    if dates != sorted(set(dates)) or signal_date not in dates:
        raise ValueError("Signal must belong to a unique ascending benchmark calendar")
    dates = [day for day in dates if day <= signal_date]
    missing = set(ASSETS) - set(histories)
    if missing:
        raise ValueError("Missing original-universe histories: " + ",".join(sorted(missing)))
    indicators = {}
    for code in ASSETS:
        rows = _history_prefix(histories[code], signal_date, code)
        indicators[code] = _latest_features(rows, signal_date)
    indicators[CASH]["valid"] = 0.
    indicators[CASH]["score"] = 0.
    if not math.isfinite(indicators[BENCHMARK]["close"]):
        raise ValueError("Benchmark has no same-day quote")
    return Snapshot(indicators, len(dates) - 1, fear), dates


def _public_indicators(data):
    return {code: {key: (float(value) if math.isfinite(value) else None)
                   for key, value in values.items()} for code, values in data.indicators.items()}


def _channels(data, day, code, config):
    if not data.informed(day, code):
        return []
    row = data.indicators[code]
    channels = []
    if config["crash_mask"] & 1 and row["mom5"] <= config["deep_mom"] and row["ma250"] < -config["deep_below"]:
        channels.append("深跌")
    if config["crash_mask"] & 2 and data.fear[day] and row["mom5"] <= config["relaxed_mom"] and row["ma250"] < -.20:
        channels.append("QVIX恐慌")
    if config["crash_mask"] & 4 and row["volume_ratio"] >= config["volume_ratio"] and row["mom5"] <= config["relaxed_mom"] and row["ma250"] < -config["volume_below"]:
        channels.append("量能恐慌")
    return channels


def decide(histories, calendar, signal_date, state=None, qvix_path=None):
    """Propose one virtual decision; caller atomically records an actual shadow fill.

    state is read-only. New crash metadata is proposed only when both present
    holding and target have valid quotes; caller must persist it only after its
    own quote/NAV validation and successful virtual settlement. This function
    never reads a portfolio, advances NAV, submits orders or writes state.
    """
    profile = load_profile("growth")
    _verify_reference()
    config = profile["config"]
    if config["id"] != CANDIDATE_ID:
        raise ValueError("Live adapter only supports the explicitly frozen main H")
    state = dict(state or {})
    qvix = qvix_state(signal_date, qvix_path)
    data, dates = snapshot(histories, calendar, signal_date, qvix["active"])
    day = len(dates) - 1
    holding = state.get("holding") or None
    if holding is not None and holding not in ASSETS:
        raise ValueError("Unknown independent H shadow holding")
    if state.get("last_date") and state["last_date"] > signal_date:
        raise ValueError("Signal predates saved H state")
    entry = state.get("entry_date")
    if entry is not None and (_date(entry) > signal_date or entry not in dates):
        raise ValueError("Cannot locate H entry date in benchmark calendar")
    trigger = state.get("crash_trigger_date")
    lock_code = state.get("crash_code") or state.get("crash_trigger_code")
    if bool(trigger) != bool(lock_code):
        raise ValueError("Crash trigger date and code must be stored together")
    lock_until = -1
    if trigger:
        if lock_code != holding or _date(trigger) not in dates:
            raise ValueError("Crash lock must match actual shadow holding and observed calendar")
        lock_until = dates.index(trigger) + config["crash_lock"]
    active = day < lock_until
    account = Portfolio(holding=holding, age=day - dates.index(entry) if entry else 0,
                        asset_peak=0., lock_until=lock_until)
    model = Policy(data, config)
    can_sell = holding is None or math.isfinite(data.price(day, holding))
    intended, panic, requested_crash = model.intent(day, account, can_sell)
    can_buy = intended is None or math.isfinite(data.price(day, intended))
    missing_features = [code for code in ASSETS if code != CASH and data.indicators[code]["bars"] >= 270 and not data.informed(day, code)]
    # Missing risk rows are omitted by the frozen model; a missing held/bought
    # quote blocks the action. An unseasoned, newly listed ETF is simply ineligible.
    executable = bool(can_sell and can_buy)
    target = intended if executable else holding
    crash = bool(requested_crash and executable and target != holding)
    next_trigger, next_code = (trigger, lock_code) if active else (None, None)
    if crash:
        next_trigger, next_code = signal_date, target
    channels = _channels(data, day, intended, config) if requested_crash else []
    held_sigma = data.value(day, holding, "vol20") if holding else float("nan")
    threshold = min(.10, max(.02, config["panic"] * held_sigma)) if math.isfinite(held_sigma) else None
    if not executable:
        reason = "缺少当前持仓或目标的当日报价，暂停虚拟成交；保留原持仓"
    elif active:
        reason = "抄底锁仓期：触发%s，第5个后续交易日恢复决策" % trigger
    elif crash:
        reason = "抄底(%s)：MOM5 %.2f%%，距MA250 %.2f%%，量比 %.2f" % (
            "∪".join(channels), data.value(day, target, "mom5") * 100,
            data.value(day, target, "ma250") * 100, data.value(day, target, "volume_ratio"))
    else:
        prefix = "波动急跌退出（阈值 %.2f%%）；" % (100 * threshold) if panic and threshold is not None else ""
        action = "继续持有" if target == holding else "建仓" if holding is None else "调仓"
        reason = prefix + "%s；MA180%s，WLS20评分近3报价均值，沿用原动量门/分池缓冲" % (action, "牛市" if model.regime.current else "熊市")
    return dict(candidate_id=CANDIDATE_ID, signal_date=signal_date, target=target,
                intended_target=intended, reason=reason, panic=bool(panic), crash=crash,
                crash_requested=bool(requested_crash), crash_trigger_date=next_trigger,
                crash_code=next_code, executable=executable, lock_active=active or crash,
                lock_remaining_sessions=(config["crash_lock"] if crash else max(0, lock_until - day)),
                research_observation_only=True, order_submission=False,
                diagnostics=dict(qvix=qvix, bull=bool(model.regime.current), ranking=data.ranked(day),
                    indicators=_public_indicators(data), panic_threshold=threshold,
                    crash_channels=channels, missing_current_features=missing_features,
                    cold_start=holding is None, previous_holding=holding,
                    feature_clock="As-of supplied TR close; intraday quotes/volume differ from final-close research",
                    valuation="No NAV calculated; caller settles only from the independently recorded live entry"))
