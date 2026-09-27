"""Trailing-only V11 score families on pinned V10 inputs, without strategy state.

Controls remain the first score rows, byte-for-byte. A new score is computed
only for the original ten risky ETFs; CASH and the thirteen extra research
assets never enter a rank percentile's cross section. Percentiles are average
ascending ranks divided by N, so a unique best=1 and a sole usable member=1.

Huber fits log prices with uniform time weights: OLS initialization, fixed
1.4826*MAD(OLS residuals) scale (floor 1e-12), delta=1.345 and exactly eight
IRLS updates. This is a bounded estimator specification, not a claim that
robust regression eliminates backtest overfitting.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from . import data as frozen_data
from .data import BASE, load_frozen
from v10_deep.features import FEATURE_NAMES
from v10_search.data import ASSET_ORDER

ORIGINAL_RISK = ("159915", "588080", "510300", "510500", "563300", "512400", "512890",
                 "513100", "513120", "518880")
CASH = "511880"
INVALID = -1e100
INVALID_CUTOFF = -1e90
ESTIMATORS = ("wls", "logwls", "olslog", "huberlog", "theilsenlog", "logmom")
HUBER_DELTA, HUBER_ITERATIONS, HUBER_SCALE_FLOOR = 1.345, 8, 1e-12
MIN_OBSERVATIONS = 270


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(_json_bytes(dict(shape=list(array.shape), dtype=array.dtype.str)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def canonical_specs(specs):
    required = {"name", "estimator", "windows", "aggregate", "smooth"}
    out = {}
    for original in specs:
        if not isinstance(original, dict) or set(original) != required:
            raise ValueError("Every score spec needs exactly name/estimator/windows/aggregate/smooth")
        name, estimator = original["name"], original["estimator"]
        if not isinstance(name, str) or not name or estimator not in ESTIMATORS:
            raise ValueError("Invalid score name or estimator")
        windows = original["windows"]
        if not isinstance(windows, (list, tuple)) or not windows:
            raise ValueError("At least one trailing window is required")
        minimum = 1 if estimator == "logmom" else 2
        if any(isinstance(w, bool) or not isinstance(w, int) or w < minimum for w in windows):
            raise ValueError("Invalid trailing window")
        if len(set(windows)) != len(windows):
            raise ValueError("Duplicate windows would introduce hidden unequal weights")
        smooth = original["smooth"]
        if isinstance(smooth, bool) or not isinstance(smooth, int) or smooth < 1:
            raise ValueError("Smoothing must use a positive number of own observations")
        if original["aggregate"] not in ("mean", "rank"):
            raise ValueError("Unknown score aggregation")
        value = dict(name=name, estimator=estimator, windows=sorted(windows),
                     aggregate=original["aggregate"], smooth=smooth)
        if name in out and out[name] != value:
            raise ValueError("One score name refers to different definitions")
        out[name] = value
    return [out[name] for name in sorted(out)]


def required_observations(spec):
    intrinsic = max(spec["windows"]) + int(spec["estimator"] == "logmom")
    if spec["aggregate"] == "rank":
        # Rank exists only once the asset is actually eligible in that dated
        # cross section; no pre-eligibility ranks are manufactured for smoothing.
        return max(MIN_OBSERVATIONS, intrinsic) + spec["smooth"] - 1
    return max(MIN_OBSERVATIONS, intrinsic + spec["smooth"] - 1)


def _rolling_mean(values, window):
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window - 1:] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def _windows(values, window):
    values = np.ascontiguousarray(values, dtype=np.float64)
    if len(values) < window:
        return np.empty((0, window), dtype=np.float64)
    return np.lib.stride_tricks.as_strided(values,
        shape=(len(values) - window + 1, window), strides=(values.strides[0], values.strides[0]), writeable=False)


def _linear(values, window, weighted):
    """Convolution arithmetic matches frozen WLS scores for non-25 windows."""
    if len(values) < window:
        return np.empty(0), np.empty(0)
    weights = np.arange(1, window + 1, dtype=float) if weighted else np.ones(window)
    positions = np.arange(window, dtype=float)
    center = np.dot(weights, positions) / weights.sum()
    denominator = np.dot(weights, (positions - center) ** 2)
    means = np.convolve(values, (weights / weights.sum())[::-1], mode="valid")
    slopes = np.convolve(values, (weights * (positions - center) / denominator)[::-1], mode="valid")
    return means, slopes


def _huber_slopes(log_prices, window):
    samples = _windows(log_prices, window)
    if not len(samples):
        return np.empty(0)
    x = np.arange(window, dtype=float) - (window - 1) / 2
    slopes = np.empty(len(samples))
    for begin in range(0, len(samples), 256):
        y = samples[begin:begin + 256]
        intercept = y.mean(axis=1)
        slope = np.sum(y * x, axis=1) / np.sum(x * x)
        residual = y - intercept[:, None] - slope[:, None] * x
        center = np.median(residual, axis=1)
        scale = np.maximum(1.4826 * np.median(np.abs(residual - center[:, None]), axis=1), HUBER_SCALE_FLOOR)
        cutoff = HUBER_DELTA * scale
        for unused in range(HUBER_ITERATIONS):
            absolute = np.abs(y - intercept[:, None] - slope[:, None] * x)
            weights = np.divide(cutoff[:, None], absolute, out=np.ones_like(absolute), where=absolute > 0)
            weights = np.minimum(1., weights)
            sw = weights.sum(axis=1)
            sx = np.sum(weights * x, axis=1)
            sy = np.sum(weights * y, axis=1)
            sxx = np.sum(weights * x * x, axis=1)
            sxy = np.sum(weights * x * y, axis=1)
            denominator = sxx - sx * sx / sw
            if np.any(denominator <= 0) or not np.isfinite(denominator).all():
                raise ArithmeticError("Huber weighted time design became singular")
            slope = (sxy - sx * sy / sw) / denominator
            intercept = (sy - slope * sx) / sw
        slopes[begin:begin + len(y)] = slope
    return slopes


def _theilsen_slopes(log_prices, window):
    samples = _windows(log_prices, window)
    if not len(samples):
        return np.empty(0)
    left, right = np.triu_indices(window, 1)
    spans = right - left
    result = np.empty(len(samples))
    # Bound temporary pairwise arrays even for the +/-20% 244-day neighborhood.
    for begin in range(0, len(samples), 64):
        rows = samples[begin:begin + 64]
        result[begin:begin + len(rows)] = np.median((rows[:, right] - rows[:, left]) / spans, axis=1)
    return result


def score_series(closes, estimator, window):
    """Unsmoothed annual trend / VOL20 on every own quotation, including warmup.

    The denominator follows the frozen convolution E[r^2]-E[r]^2 path. A
    finite trend with zero volatility maps to zero as in the frozen estimator;
    unavailable trailing histories remain NaN rather than a neutral score.
    """
    prices = np.asarray(closes, dtype=np.float64)
    if prices.ndim != 1 or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Scores require a one-dimensional positive finite TR price series")
    if estimator not in ESTIMATORS or isinstance(window, bool) or not isinstance(window, int) or window < (1 if estimator == "logmom" else 2):
        raise ValueError("Invalid estimator/window")
    result = np.full(len(prices), np.nan)
    if not len(prices):
        return result
    returns = np.r_[0., prices[1:] / prices[:-1] - 1]
    mean = _rolling_mean(returns, 20)
    sigma = np.sqrt(np.maximum(0., _rolling_mean(returns * returns, 20) - mean * mean))
    trend = np.full(len(prices), np.nan)
    if estimator == "logmom":
        if len(prices) > window:
            trend[window:] = np.log(prices[window:] / prices[:-window]) * (250. / window)
    elif len(prices) >= window:
        if estimator in ("wls", "logwls", "olslog"):
            source = prices if estimator == "wls" else np.log(prices)
            means, slopes = _linear(source, window, estimator != "olslog")
            trend[window - 1:] = slopes / means * 250 if estimator == "wls" else slopes * 250
        elif estimator == "huberlog":
            trend[window - 1:] = _huber_slopes(np.log(prices), window) * 250
        else:
            trend[window - 1:] = _theilsen_slopes(np.log(prices), window) * 250
    ready = np.isfinite(trend) & np.isfinite(sigma)
    result[ready] = 0.
    np.divide(trend, sigma, out=result, where=ready & (sigma > 0))
    result[~np.isfinite(result)] = np.nan
    return result


def _smooth_own(values, count):
    """Require a full consecutive window of own observed values; never nanmean."""
    if count == 1:
        return np.array(values, copy=True)
    if len(values) < count:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    # Latest-to-oldest order preserves the frozen 3-observation arithmetic.
    total = np.zeros(len(values) - count + 1)
    valid = np.ones(len(total), dtype=bool)
    for lag in range(count):
        view = values[count - 1 - lag:len(values) - lag if lag else None]
        valid &= np.isfinite(view)
        total += np.where(np.isfinite(view), view, 0.)
    result[count - 1:] = np.where(valid, total / count, np.nan)
    return result


def percentile_ranks(scores, eligible):
    """Cross-sectional average rank/N, only over explicitly eligible members."""
    values = np.asarray(scores, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if values.ndim != 2 or values.shape != eligible.shape:
        raise ValueError("Rank inputs must be equally shaped time-by-asset arrays")
    output = np.full(values.shape, np.nan)
    for day in range(len(values)):
        members = np.flatnonzero(eligible[day] & np.isfinite(values[day]) & (values[day] > INVALID_CUTOFF))
        if not len(members):
            continue
        ordered = members[np.argsort(values[day, members], kind="stable")]
        start = 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and values[day, ordered[end]] == values[day, ordered[start]]:
                end += 1
            output[day, ordered[start:end]] = ((start + 1 + end) / 2.) / len(ordered)
            start = end
    return output


def _validate_bundle(bundle):
    arrays, meta, histories = bundle["arrays"], bundle["meta"], bundle["histories"]
    if tuple(meta["assets"]) != tuple(ASSET_ORDER) or meta["feature_names"] != FEATURE_NAMES:
        raise ValueError("Frozen native asset/feature layout changed")
    dates = meta["dates"]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Frozen dates must be sorted and unique")
    t, a, f, s = len(dates), len(meta["assets"]), len(meta["feature_names"]), len(meta["score_names"])
    for key, shape, dtype in (("features", (t, a, f), np.float64), ("scores", (s, t, a), np.float64),
                              ("orders", (s, t, a), np.int32), ("fear", (t,), np.int32)):
        value = arrays[key]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError("Invalid frozen array layout: " + key)
    if len(set(meta["score_names"])) != s:
        raise ValueError("Frozen score names must be unique")
    for code in meta["assets"]:
        rows = histories[code]
        row_dates = [r[0] for r in rows]
        if row_dates != sorted(set(row_dates)) or any(len(r) != 4 for r in rows):
            raise ValueError("Invalid own observation sequence: " + code)
        if any(not math.isfinite(float(r[2])) or float(r[2]) <= 0 for r in rows):
            raise ValueError("Invalid TR close: " + code)


def _fingerprint(bundle, specs):
    return dict(source_sha256=_sha(__file__), data_source_sha256=_sha(frozen_data.__file__),
        input_fingerprints=deepcopy(bundle["fingerprints"]), specifications=specs,
        input_array_sha256={key: _array_sha(value) for key, value in sorted(bundle["arrays"].items())},
        input_metadata_sha256=hashlib.sha256(_json_bytes(bundle["meta"])).hexdigest(),
        histories_sha256=hashlib.sha256(_json_bytes(bundle["histories"])).hexdigest())


def _compute(bundle, specs, fingerprints):
    original, meta, histories = bundle["arrays"], deepcopy(bundle["meta"]), bundle["histories"]
    arrays = {key: np.array(value, copy=True, order="C") for key, value in original.items()}
    dates, assets = meta["dates"], meta["assets"]
    lookup = {date: i for i, date in enumerate(dates)}
    fields = {name: i for i, name in enumerate(meta["feature_names"])}
    t, a = len(dates), len(assets)
    observations = np.zeros((t, a), dtype=np.int32)
    own_map = {}
    for ai, code in enumerate(assets):
        pairs = [(j, lookup[row[0]]) for j, row in enumerate(histories[code]) if row[0] in lookup]
        own = np.asarray([j for j, unused in pairs], dtype=int)
        mapped = np.asarray([i for unused, i in pairs], dtype=int)
        observations[mapped, ai] = own + 1
        own_map[code] = own, mapped
    arrays["observations"] = observations
    risk_indices = [assets.index(code) for code in ORIGINAL_RISK]
    eligible = np.zeros((t, a), dtype=bool)
    eligible[:, risk_indices] = (arrays["features"][:, risk_indices, fields["valid"]] > .5)
    eligible &= np.isfinite(arrays["features"][:, :, fields["close"]])
    base_scores, dated_scores, ranked_scores = {}, {}, {}
    def intrinsic(estimator, window):
        key = estimator, window
        if key not in base_scores:
            values = {}
            dated = np.full((t, a), np.nan)
            for code in ORIGINAL_RISK:
                ai = assets.index(code)
                values[code] = score_series([r[2] for r in histories[code]], estimator, window)
                own, mapped = own_map[code]
                dated[mapped, ai] = values[code][own]
            base_scores[key], dated_scores[key] = values, dated
        return base_scores[key], dated_scores[key]
    new = []
    required = {name: MIN_OBSERVATIONS for name in meta["score_names"]}
    for spec in specs:
        output = np.full((t, a), INVALID)
        required[spec["name"]] = required_observations(spec)
        pieces = [intrinsic(spec["estimator"], window) for window in spec["windows"]]
        if spec["aggregate"] == "rank":
            ranked = []
            for window, (_, dated) in zip(spec["windows"], pieces):
                key = spec["estimator"], window
                if key not in ranked_scores:
                    ranked_scores[key] = percentile_ranks(dated, eligible)
                ranked.append(ranked_scores[key])
            combined = np.mean(ranked, axis=0)
            for code in ORIGINAL_RISK:
                ai = assets.index(code)
                own, mapped = own_map[code]
                series = _smooth_own(combined[mapped, ai], spec["smooth"])
                usable = np.isfinite(series) & eligible[mapped, ai] & (observations[mapped, ai] >= required[spec["name"]])
                output[mapped[usable], ai] = series[usable]
        else:
            for code in ORIGINAL_RISK:
                ai = assets.index(code)
                combined = np.mean([piece[0][code] for piece in pieces], axis=0)
                smoothed = _smooth_own(combined, spec["smooth"])
                own, mapped = own_map[code]
                usable = np.isfinite(smoothed[own]) & eligible[mapped, ai] & (observations[mapped, ai] >= required[spec["name"]])
                output[mapped[usable], ai] = smoothed[own[usable]]
        new.append(output)
    if new:
        arrays["scores"] = np.ascontiguousarray(np.concatenate((arrays["scores"], np.asarray(new)), axis=0))
        new_orders = np.argsort(-np.asarray(new), axis=2, kind="stable").astype(np.int32)
        arrays["orders"] = np.ascontiguousarray(np.concatenate((arrays["orders"], new_orders), axis=0))
    meta["score_names"] = list(meta["score_names"]) + [spec["name"] for spec in specs]
    meta.update(fingerprints=fingerprints, score_specs=deepcopy(specs),
                required_observations_by_score=required, original_risk_assets=list(ORIGINAL_RISK),
                control_score_count=original["scores"].shape[0],
                rank_convention="Within original ten risk assets, base.valid and finite current score; average ascending rank/N, singleton=1; no neutral imputation.",
                smoothing_convention="Full trailing own-quotation windows; rank needs mature dated cross sections before smoothing; logmom needs L+1 prices.",
                huber=dict(delta=HUBER_DELTA, iterations=HUBER_ITERATIONS, scale="Fixed 1.4826 * MAD of OLS residuals", scale_floor=HUBER_SCALE_FLOOR),
                usage="Call risk_view(..., score=config['score']) to apply that score's warmup/sentinel validity mask before native execution.")
    return arrays, meta


def _check_controls(arrays, meta, bundle):
    source = bundle["arrays"]
    count = len(bundle["meta"]["score_names"])
    if meta["score_names"][:count] != bundle["meta"]["score_names"]:
        raise ValueError("Cached control score order changed")
    if not np.array_equal(arrays["scores"][:count], source["scores"]) or not np.array_equal(arrays["orders"][:count], source["orders"]):
        raise ValueError("Cached control scores/rankings changed")
    if _array_sha(arrays["features"]) != _array_sha(source["features"]) or not np.array_equal(arrays["fear"], source["fear"]):
        raise ValueError("Cached frozen features/fear changed")


def _atomic_cache(array_path, metadata_path, arrays, meta):
    temporary_arrays = temporary_meta = None
    try:
        with tempfile.NamedTemporaryFile("w+b", dir=str(array_path.parent), prefix=".features-", suffix=".npz", delete=False) as stream:
            temporary_arrays = stream.name
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        meta["cache_array_sha256"] = _sha(temporary_arrays)
        with tempfile.NamedTemporaryFile("wb", dir=str(array_path.parent), prefix=".features-", suffix=".json", delete=False) as stream:
            temporary_meta = stream.name
            stream.write(_json_bytes(meta) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_arrays, str(array_path))
        temporary_arrays = None
        os.replace(temporary_meta, str(metadata_path))
        temporary_meta = None
    finally:
        for path in (temporary_arrays, temporary_meta):
            if path is not None and os.path.exists(path):
                os.unlink(path)


def build(specs=None, force=False):
    """Append registered scores to frozen controls; cache only inside v11/cache."""
    if specs is None:
        from .registry import score_specs
        specs = score_specs() if callable(score_specs) else score_specs
    specs = canonical_specs(specs)
    bundle = load_frozen()
    _validate_bundle(bundle)
    if set(spec["name"] for spec in specs) & set(bundle["meta"]["score_names"]):
        raise ValueError("A new spec cannot overwrite an exact frozen control score")
    fingerprints = _fingerprint(bundle, specs)
    key = hashlib.sha256(_json_bytes(fingerprints)).hexdigest()
    directory = Path(BASE) / "cache"
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / ("features-" + key[:24] + ".npz")
    metadata_path = array_path.with_suffix(".json")
    if not force and array_path.is_file() and metadata_path.is_file():
        meta = json.loads(metadata_path.read_text())
        if meta.get("fingerprints") != fingerprints or meta.get("cache_array_sha256") != _sha(array_path):
            raise ValueError("V11 feature cache integrity mismatch; rebuild explicitly with force=True")
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
        _check_controls(arrays, meta, bundle)
        return arrays, meta
    arrays, meta = _compute(bundle, specs, fingerprints)
    meta["cache_paths"] = dict(arrays=str(array_path), metadata=str(metadata_path))
    _check_controls(arrays, meta, bundle)
    _atomic_cache(array_path, metadata_path, arrays, meta)
    return arrays, meta


def risk_view(arrays, meta, context, score=None):
    """Copy decision features, replacing VOL20 only (plus necessary validity).

    The previous sigma is from the previous actual quoted row of that same
    asset, never a forward fill from a future row or the previous market date.
    Scores, orders and other arrays are shared and must be treated as read-only.
    Current20 control features remain exact when their score is available.
    """
    if context not in ("current20", "prior20", "prior60", "prior_max20_60"):
        raise ValueError("Unknown volatility timing context")
    fields = {name: i for i, name in enumerate(meta["feature_names"])}
    source = arrays["features"]
    view = dict(arrays)
    cube = np.array(source, copy=True, order="C")
    valid = source[:, :, fields["valid"]] > .5
    if context != "current20":
        column = np.full(source.shape[:2], np.nan)
        for asset in range(source.shape[1]):
            quoted = np.flatnonzero(np.isfinite(source[:, asset, fields["close"]]) & (source[:, asset, fields["close"]] > 0))
            if len(quoted) < 2:
                continue
            if context == "prior_max20_60":
                sigma = np.maximum(source[quoted[:-1], asset, fields["vol20"]], source[quoted[:-1], asset, fields["vol60"]])
            else:
                sigma = source[quoted[:-1], asset, fields["vol20" if context == "prior20" else "vol60"]]
            column[quoted[1:], asset] = sigma
        cube[:, :, fields["vol20"]] = column
        valid &= np.isfinite(column)
    if score is not None:
        if score not in meta["score_names"]:
            raise ValueError("Unknown score for validity mask")
        index = meta["score_names"].index(score)
        scores = arrays["scores"][index]
        valid &= np.isfinite(scores) & (scores > INVALID_CUTOFF)
        required = meta["required_observations_by_score"][score]
        valid &= arrays["observations"] >= required
    cube[:, :, fields["valid"]][~valid] = 0.
    view["features"] = cube
    return view
