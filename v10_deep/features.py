"""Dated feature arrays; exact legacy features plus trailing-only alternatives."""
import json
import math
import numpy as np

from v10_search.features import FeatureBank
from .data import BASE, ASSET_ORDER, CASH, inputs, sha, dump

MOM_WINDOWS = (3, 5, 10, 15, 20, 30, 40, 60, 90, 120, 180, 240)
VOL_WINDOWS = (10, 20, 40, 60, 120)
MA_WINDOWS = (80, 100, 120, 150, 180, 200, 250, 300, 360)
SCORE_WINDOWS = (10, 15, 20, 25, 30, 35, 40, 50, 60, 90, 120)
FEATURE_NAMES = (["close", "valid", "ret1", "volume_ratio", "er20"] +
                 ["mom%d" % n for n in MOM_WINDOWS] + ["vol%d" % n for n in VOL_WINDOWS] +
                 ["ma%d" % n for n in MA_WINDOWS] + ["dd%d" % n for n in (20, 60, 120)] +
                 ["ema120", "ema250"])
F = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _rolling_mean(x, window):
    result = np.full(len(x), np.nan)
    if len(x) >= window:
        result[window - 1:] = np.convolve(x, np.ones(window) / window, mode="valid")
    return result


def _regression(x, window, weighted=True):
    if len(x) < window:
        return np.empty(0), np.empty(0), np.empty(0)
    w = np.arange(1, window + 1, dtype=float) if weighted else np.ones(window)
    positions = np.arange(window, dtype=float)
    mx = np.dot(w, positions) / w.sum()
    xx = np.dot(w, (positions - mx) ** 2)
    mean = np.convolve(x, (w / w.sum())[::-1], mode="valid")
    slope = np.convolve(x, (w * (positions - mx) / xx)[::-1], mode="valid")
    yy = np.convolve(x * x, w[::-1], mode="valid") - w.sum() * mean * mean
    r2 = np.divide(slope * slope * xx, yy, out=np.zeros_like(slope), where=yy > 1e-15)
    return mean, slope, np.clip(r2, 0, 1)


def build(force=False):
    cache = BASE / "cache"
    cache.mkdir(exist_ok=True)
    path = cache / "features.npz"
    metadata_path = cache / "features.json"
    fp = {"source": sha(__file__), "data": sha(BASE.parent / "v10_h_close/corrected_manifest.json"),
          "legacy_features": sha(BASE.parent / "v10_search/features.py"),
          "legacy_strategy": sha(BASE.parent / "v10_next/frozen/strategy.py")}
    for name in ('v10_deep/data.py','v10_next/data.py','v10_next/frozen/presets.json','v10_search/data.py',
                 'v10_h_close/corrected_scan.py','v10_h_close/data.py','v10_h_close/qvix_manifest.json','v10_h_close/snapshots/qvix50.csv'):
        fp[name]=sha(BASE.parent/name)
    if not force and path.exists() and metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if meta["fingerprints"] == fp:
            with np.load(path) as loaded:
                return {k: loaded[k] for k in loaded.files}, meta
    histories, calendar, fear = inputs()
    old = FeatureBank(histories, calendar)
    T, A = len(calendar), len(ASSET_ORDER)
    cube = np.full((T, A, len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    cube[:, :, F["valid"]] = 0
    date_lookup = {d: i for i, d in enumerate(calendar)}
    scores = {}
    def save_score(name, ai, indices, values):
        if name not in scores:
            scores[name] = np.full((T, A), -1e100)
        scores[name][indices, ai] = values
    for ai, code in enumerate(ASSET_ORDER):
        rows = histories[code]
        prices = np.asarray([r[2] for r in rows]); volumes = np.asarray([r[3] for r in rows])
        own = np.asarray([i for i, r in enumerate(rows) if r[0] in date_lookup], dtype=int)
        idx = np.asarray([date_lookup[rows[i][0]] for i in own], dtype=int)
        ret = np.concatenate(([0.0], prices[1:] / prices[:-1] - 1))
        arrays = {"close": prices, "ret1": ret, "valid": (np.arange(len(rows)) >= 269).astype(float)}
        for w in MOM_WINDOWS:
            values = np.full(len(rows), np.nan); values[w:] = prices[w:] / prices[:-w] - 1
            arrays["mom%d" % w] = values
        for w in VOL_WINDOWS:
            avg = _rolling_mean(ret, w)
            arrays["vol%d" % w] = np.sqrt(np.maximum(0, _rolling_mean(ret * ret, w) - avg * avg))
        for w in MA_WINDOWS:
            arrays["ma%d" % w] = prices / _rolling_mean(prices, w) - 1
        for w in (20, 60, 120):
            values = np.full(len(rows), np.nan)
            for j in range(w - 1, len(rows)):
                values[j] = prices[j] / max(prices[j - w + 1:j + 1]) - 1
            arrays["dd%d" % w] = values
        volume_average = np.roll(_rolling_mean(volumes, 20), 1)
        arrays["volume_ratio"] = np.divide(volumes, volume_average, out=np.zeros(len(rows)), where=volume_average > 0)
        arrays["volume_ratio"][:20] = np.nan
        path_length = _rolling_mean(np.concatenate(([0.0], np.abs(np.diff(prices)))), 20) * 20
        change = np.abs(prices - np.roll(prices, 20))
        arrays["er20"] = np.divide(change, path_length, out=np.ones(len(rows)), where=path_length > 0)
        arrays["er20"][:20] = np.nan
        for w in (120, 250):
            ema = np.empty(len(rows)); ema[0] = prices[0]; alpha = 2.0 / (w + 1)
            for j in range(1, len(rows)): ema[j] = alpha * prices[j] + (1 - alpha) * ema[j - 1]
            arrays["ema%d" % w] = prices / ema - 1
        for name, values in arrays.items(): cube[idx, ai, F[name]] = values[own]
        for window in SCORE_WINDOWS:
            for family in ("wls", "logwls", "ols"):
                source = np.log(prices) if family == "logwls" else prices
                mean, slope, r2 = _regression(source, window, family != "ols")
                trend = np.full(len(rows), np.nan)
                trend[window - 1:] = slope * 250 if family == "logwls" else slope / mean * 250
                for vol_window in (20, 40, 60):
                    score = np.divide(trend, arrays["vol%d" % vol_window], out=np.zeros(len(rows)), where=arrays["vol%d" % vol_window] > 0)
                    save_score("%s%d_v%d" % (family, window, vol_window), ai, idx, score[own])
                if family == "wls":
                    save_score("wls%d_raw" % window, ai, idx, trend[own])
                    shaped = np.full(len(rows), np.nan); shaped[window - 1:] = r2
                    quality = np.divide(trend * shaped, arrays["vol20"], out=np.zeros(len(rows)), where=arrays["vol20"] > 0)
                    save_score("wls%d_r2" % window, ai, idx, quality[own])
        for window in (10, 20, 40, 60, 120, 240):
            for vol_window in (20, 40, 60):
                score = np.divide(arrays["mom%d" % window] * 250 / window, arrays["vol%d" % vol_window],
                                  out=np.zeros(len(rows)), where=arrays["vol%d" % vol_window] > 0)
                save_score("mom%d_v%d" % (window, vol_window), ai, idx, score[own])
            save_score("mom%d_raw" % window, ai, idx, arrays["mom%d" % window][own])
        # Preserve exact original arithmetic for controls and the baseline score.
        for ti in idx:
            ind = old.base[ti].get(code)
            if ind:
                for name in ("mom5", "mom20", "mom60", "ret1"):
                    cube[ti, ai, F[name]] = ind[name]
                cube[ti, ai, F["vol20"]] = ind["vol"]
                cube[ti, ai, F["ma250"]] = ind["dist_ma250"]
                cube[ti, ai, F["volume_ratio"]] = ind["volume_ratio20"]
                scores["wls25_v20"][ti, ai] = ind["score"]
    for windows in ((15,25,40), (20,40,60), (10,20,40), (25,30), (20,25,30), (20,60), (40,60,120)):
        scores["blend_" + "_".join(map(str, windows))] = np.mean([scores["wls%d_v20" % w] for w in windows], axis=0)
    for w in (20, 40, 60):
        smooth = np.full((T, A), -1e100)
        for ai, code in enumerate(ASSET_ORDER):
            observed = np.asarray([date_lookup[r[0]] for r in histories[code] if r[0] in date_lookup], dtype=int)
            series = scores["wls%d_v20" % w][observed, ai]
            smooth[observed[2:], ai] = (series[2:] + series[1:-1] + series[:-2]) / 3
        scores["wls%d_smooth3" % w] = smooth
    names = sorted(scores)
    cube[:, ASSET_ORDER.index(CASH), F['valid']] = 0
    score_cube = np.asarray([scores[n] for n in names])
    score_cube[:, ~np.isfinite(cube[:, :, F["close"]]) | (cube[:, :, F["valid"]] == 0)] = -1e100
    score_cube[:, :, ASSET_ORDER.index(CASH)] = -1e100
    score_cube[~np.isfinite(score_cube)] = -1e100
    orders = np.argsort(-score_cube, axis=2, kind="stable").astype(np.int32)
    values = dict(features=np.ascontiguousarray(cube), scores=np.ascontiguousarray(score_cube), orders=np.ascontiguousarray(orders),
                  fear=np.asarray(fear,dtype=np.int32), years=np.asarray([int(d[:4]) for d in calendar],dtype=np.int32))
    meta = dict(fingerprints=fp, dates=calendar, assets=list(ASSET_ORDER), feature_names=FEATURE_NAMES,
                score_names=names, shape=list(cube.shape), note="Only trailing observations; exact legacy features overwrite default control arithmetic.")
    np.savez_compressed(path, **values); dump(metadata_path, meta)
    print("Feature cache prepared:", cube.shape, len(names), "score families", flush=True)
    return values, meta
