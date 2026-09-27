"""Eight predeclared long-horizon trend controls, without crash/timing overlays.

Scores use each asset's own observed closes. Evaluation starts with one explicit
initial-allocation exception; subsequent orders occur only on the first observed
week/month session. Lag 1 delays that signal and its clock by one observed day.
All executions use the frozen close-only units/weight-turnover engine.
"""
import bisect
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from numbers import Integral

from v10_deep.portfolios import run_weights


CASH = "511880"
RISK_ASSETS = ("159915", "588080", "510300", "510500", "563300", "512400", "512890",
               "513100", "513120", "518880")
ASSETS = RISK_ASSETS + (CASH,)
HORIZONS = ((244,), (61, 122, 244))
MIN_OBSERVATIONS = 270
ANNUALIZATION = 244


def registry():
    """A fixed 2×2×2 family; fees and lag are evaluation scenarios, not new trials."""
    candidates = []
    for horizons in HORIZONS:
        for topk in (1, 2):
            for frequency in ("weekly", "monthly"):
                config = dict(horizons=list(horizons), topk=topk, frequency=frequency,
                    risk_assets=list(RISK_ASSETS), cash=CASH, minimum_observations=MIN_OBSERVATIONS,
                    score="mean_annualized_log_momentum", annualization_sessions=ANNUALIZATION,
                    positive_threshold=0., allocation="1/topk per selected risk asset; empty slots to cash ETF",
                    rebalance="initial allocation, then first observed week/month session; reset scheduled weights",
                    risk_exits="clock only", leverage=False, crash_channels=[])
                digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                config.update(id="pc_" + digest[:20], hash=digest,
                              label="%s-day log trend / top%d / %s" % ("+".join(map(str, horizons)), topk, frequency))
                candidates.append(config)
    return dict(schema=1, count=8, candidates=candidates, allowed_lags=[0, 1],
                engine="v10_deep.portfolios.run_weights", clean_oos=False,
                first_evaluation_session_free=True,
                fee_convention="preNAV * fee * L1(target security weights minus actual marked security weights)",
                initial_exception="First evaluation session allocates using q=t-lag if q exists; all later signals obey the clock",
                missing_quote_policy="Atomic rejected rebalance is discarded, with no order carryover")


def _date(value):
    if not isinstance(value, str):
        raise ValueError("Dates must be ISO strings")
    parsed = datetime.strptime(value, "%Y-%m-%d")
    if parsed.strftime("%Y-%m-%d") != value:
        raise ValueError("Dates must be ISO strings")
    return parsed


def _config(config):
    if (tuple(config["horizons"]) not in HORIZONS or config["topk"] not in (1, 2)
            or config["frequency"] not in ("weekly", "monthly")
            or tuple(config["risk_assets"]) != RISK_ASSETS or config["cash"] != CASH
            or config["minimum_observations"] != MIN_OBSERVATIONS):
        raise ValueError("Configuration is outside the eight registered controls")
    semantic = {key: value for key, value in config.items() if key not in ("id", "hash", "label")}
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not any(candidate["hash"] == digest for candidate in registry()["candidates"]):
        raise ValueError("Registered control semantics changed")
    if config.get("hash", digest) != digest or config.get("id", "pc_" + digest[:20]) != "pc_" + digest[:20]:
        raise ValueError("Control hash/identifier mismatch")
    return deepcopy(config)


def prepare(histories, calendar):
    """Precompute both scores without executing a strategy or reading performance.

    Input rows may have four or six columns; only date and close (index 2) are
    read. Missing current bars remain NaN, rather than carrying an old rank.
    """
    dates = tuple(calendar)
    if not dates or dates != tuple(sorted(set(dates))):
        raise ValueError("Calendar must be nonempty, unique and ascending")
    stamps = tuple(_date(date) for date in dates)
    index = {date: i for i, date in enumerate(dates)}
    if set(ASSETS) - set(histories):
        raise ValueError("All original ten risk assets and the cash ETF are required")
    prices = {code: [float("nan")] * len(dates) for code in ASSETS}
    scores = {horizons: {code: [float("nan")] * len(dates) for code in RISK_ASSETS} for horizons in HORIZONS}
    counts = {code: [0] * len(dates) for code in RISK_ASSETS}
    for code in ASSETS:
        rows, previous, closes = histories[code], None, []
        for row in rows:
            if len(row) < 3:
                raise ValueError("History row lacks a close")
            date = row[0]
            _date(date)
            if previous is not None and date <= previous:
                raise ValueError("Asset histories must have unique ascending dates")
            previous = date
            close = float(row[2])
            if not math.isfinite(close) or close <= 0:
                raise ValueError("Observed closes must be positive and finite")
            closes.append(close)
            if date not in index:
                continue
            i, own = index[date], len(closes) - 1
            prices[code][i] = close
            if code == CASH:
                continue
            counts[code][i] = len(closes)
            if len(closes) < MIN_OBSERVATIONS:
                continue
            for horizons in HORIZONS:
                if own < max(horizons):
                    continue
                scores[horizons][code][i] = sum(
                    ANNUALIZATION * math.log(close / closes[own - horizon]) / horizon
                    for horizon in horizons) / len(horizons)
    return dict(calendar=dates, prices={code: tuple(values) for code, values in prices.items()},
                scores={h: {code: tuple(values) for code, values in vectors.items()} for h, vectors in scores.items()},
                own_observations={code: tuple(values) for code, values in counts.items()},
                week_keys=tuple(stamp.isocalendar()[:2] for stamp in stamps),
                month_keys=tuple((stamp.year, stamp.month) for stamp in stamps))


def make_policy(config, prepared, start_index, lag=0):
    config = _config(config)
    if lag not in (0, 1):
        raise ValueError("Only lag 0 or 1 is registered")
    dates, scores = prepared["calendar"], prepared["scores"][tuple(config["horizons"])]
    keys = prepared["week_keys" if config["frequency"] == "weekly" else "month_keys"]
    trace = []

    def policy(i, state):
        q = i - lag
        initial = i == start_index
        scheduled = q >= 0 and (initial or q == 0 or keys[q] != keys[q - 1])
        record = dict(date=dates[i], signal_date=dates[q] if q >= 0 else None,
                      initial_allocation=initial, rebalance=scheduled)
        if not scheduled:
            record["target"] = None
            trace.append(record)
            return None
        ranked = [code for code in RISK_ASSETS if math.isfinite(scores[code][q]) and scores[code][q] > 0]
        ranked.sort(key=lambda code: (-scores[code][q], RISK_ASSETS.index(code)))
        selected = ranked[:config["topk"]]
        slot = 1. / config["topk"]
        target = {code: slot for code in selected}
        cash = (config["topk"] - len(selected)) * slot
        if cash:
            target[CASH] = cash  # No cash-quote substitution; the execution engine may reject the whole order.
        record.update(target=dict(target), selected=selected,
                      ranked_scores=[[code, scores[code][q]] for code in ranked])
        trace.append(record)
        return target

    policy.metadata = dict(config=config, lag=lag, trace=trace, close_only=True,
                           initial_exception="Only first evaluation session may allocate outside the regular clock",
                           signal_information="q=t-lag; asset-specific observed closes through q only",
                           fee_engine="Frozen run_weights; actual drifted weights; no leverage")
    return policy


def run_control(config, prepared, start, end, fee=.0001, lag=0):
    dates = prepared["calendar"]
    lo = int(start) if isinstance(start, Integral) else bisect.bisect_left(dates, start)
    policy = make_policy(config, prepared, lo, lag=lag)
    result = run_weights(prepared["prices"], dates, policy, start, end, fee=fee)
    result.update(config=deepcopy(config), lag=lag, policy_trace=policy.metadata["trace"],
                  control_metadata={key: value for key, value in policy.metadata.items() if key != "trace"})
    return result
