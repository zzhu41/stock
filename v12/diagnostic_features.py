"""Pure, explicit feature views for preregistered V12 one-axis diagnostics.

No files, cache, simulations or candidate selection. Original score lanes are
preserved; exact control centers are reused rather than recomputed. Histories
are frozen four-column (date, open_placeholder, TR_close, normalized_volume).
"""
from copy import deepcopy
import math

import numpy as np

from v10_deep.features import _regression, _rolling_mean

CENTERS = {"wls25_v20": (25, 1), "wls20_smooth3": (20, 3)}
CASH, BENCHMARK = "511880", "510300"
INVALID = -1e100


def _integer(value, label, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (label, minimum))
    return int(value)


def _dated_prices(histories, code, end):
    if code not in histories:
        raise ValueError("Missing frozen history: " + code)
    dates, values, previous = [], [], None
    for row in histories[code]:
        if len(row) != 4 or not isinstance(row[0], str):
            raise ValueError("Four-column dated TR history required: " + code)
        date = row[0]
        if previous is not None and date <= previous:
            raise ValueError("History must have unique ascending dates: " + code)
        previous = date
        if date > end:
            continue
        price = float(row[2])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid observed TR close: " + code)
        dates.append(date)
        values.append(price)
    return dates, np.asarray(values, dtype=np.float64)


def _legacy25(prices):
    """Exact frozen WLS25/VOL20 arithmetic, before any investability mask."""
    raw = np.full(len(prices), np.nan)
    closes = [float(p) for p in prices]
    xs = list(range(25))
    weights = [i+1 for i in xs]
    wsum = sum(weights)
    mx = sum(weights[i]*xs[i] for i in xs)/wsum
    xx = sum(weights[i]*(xs[i]-mx)**2 for i in xs)
    for j in range(24, len(closes)):
        seg = closes[j-24:j+1]
        my = sum(weights[i]*seg[i] for i in xs)/wsum
        xy = sum(weights[i]*(xs[i]-mx)*(seg[i]-my) for i in xs)
        slope = (xy/xx)/my if xx > 0 and my > 0 else 0.
        rets = [closes[i]/closes[i-1]-1. for i in range(j-19,j+1)]
        mean = sum(rets)/len(rets)
        vol = (sum((r-mean)**2 for r in rets)/len(rets))**.5
        raw[j] = slope*250/vol if vol > 0 else 0.
    return raw


def _raw_scores(prices, window):
    if window == 25:
        return _legacy25(prices)
    if not len(prices):
        return np.empty(0, dtype=np.float64)
    returns = np.concatenate(([0.],prices[1:]/prices[:-1]-1.))
    mean_return = _rolling_mean(returns,20)
    sigma = np.sqrt(np.maximum(0.,_rolling_mean(returns*returns,20)-mean_return*mean_return))
    mean,slope,unused = _regression(prices,window)
    trend = np.full(len(prices),np.nan)
    if len(prices) >= window:
        trend[window-1:] = slope/mean*250
    raw = np.divide(trend,sigma,out=np.zeros(len(prices)),where=sigma > 0)
    raw[:min(window-1,len(raw))] = np.nan
    raw[~np.isfinite(sigma)] = np.nan
    return raw


def make_view(arrays, meta, histories, config, *, window=None, smooth=None, ma_window=None):
    """Return (copied_arrays, copied_metadata, diagnostic_config) for native.

    Defaults retain the supplied center's score and MA. Non-center MA values
    borrow only the benchmark's ma180 slot; ma250 stays intact for crash rules.
    Diagnostic identity must include metadata['v12_diagnostic_features'], not
    only the returned config (different MA windows can share the encoded slot).
    """
    names = list(meta["score_names"])
    if len(names) != 2 or set(names) != set(CENTERS):
        raise ValueError("Expected exactly the two original V12 score lanes")
    score = config.get("score")
    if score not in CENTERS or config.get("ma") not in ("ma180","ma250"):
        raise ValueError("Diagnostics must start from an original V12 score/MA center")
    window = _integer(CENTERS[score][0] if window is None else window,"window",2)
    smooth = _integer(CENTERS[score][1] if smooth is None else smooth,"smooth",1)
    ma_window = _integer(int(config["ma"][2:]) if ma_window is None else ma_window,"ma_window",2)
    dates, assets = list(meta["dates"]),list(meta["assets"])
    if not dates or dates != sorted(set(dates)) or len(assets) != len(set(assets)):
        raise ValueError("Unique ascending observation dates and asset identities required")
    if BENCHMARK not in assets or CASH not in assets:
        raise ValueError("Benchmark and cash roles must remain in the feature universe")
    fields = {name:i for i,name in enumerate(meta["feature_names"])}
    required_fields = {"close","valid","ma180","ma250"}
    if not required_fields <= set(fields):
        raise ValueError("Missing native feature fields")
    t,a = len(dates),len(assets)
    if (arrays["features"].shape != (t,a,len(fields)) or arrays["scores"].shape != (2,t,a)
            or arrays["orders"].shape != (2,t,a)):
        raise ValueError("Feature/score observation axes do not match metadata")
    view = {name:np.array(value,copy=True,order="C") for name,value in arrays.items()}
    metadata, candidate = deepcopy(meta),deepcopy(config)
    old_id,old_hash = candidate.get("id"),candidate.get("hash")
    required = max(270,window+smooth-1)
    metadata.setdefault("required_observations_by_score",{}).update({name:270 for name in CENTERS})
    center_lane = next((name for name,pair in CENTERS.items() if pair == (window,smooth)),None)
    provenance = dict(diagnostic_only=True,window=window,smooth=smooth,
        required_observations=required,source_score=score,source_config_ma=config["ma"],
        source_config_id=old_id,source_config_hash=old_hash,
        actual_ma_window=ma_window,score_center_reused=center_lane,
        score_arithmetic="frozen_lane" if center_lane else "legacy_list_sum_WLS25_VOL20" if window == 25 else "frozen_convolution_WLS_VOL20",
        smooth_clock="trailing own quoted observations; raw scores before investability mask",
        cash_rank_excluded=True,ma250_crash_column_preserved=True)
    lookup = {date:i for i,date in enumerate(dates)}
    if center_lane is not None:
        candidate["score"] = center_lane
    else:
        new_name = "v12_diag_wls%d_smooth%d" % (window,smooth)
        scores = np.full((t,a),INVALID,dtype=np.float64)
        for ai,code in enumerate(assets):
            if code == CASH:
                continue
            own_dates,prices = _dated_prices(histories,code,dates[-1])
            raw = _raw_scores(prices,window)
            smoothed = _rolling_mean(raw,smooth) if smooth > 1 else raw
            for j,date in enumerate(own_dates):
                if date not in lookup:
                    continue
                i = lookup[date]
                if j+1 >= required and np.isfinite(smoothed[j]):
                    scores[i,ai] = smoothed[j]
        valid_score = np.isfinite(scores) & (scores > INVALID/2)
        view["features"][:,:,fields["valid"]][~valid_score] = 0.
        view["scores"] = np.ascontiguousarray(np.concatenate((view["scores"],scores[None,:,:]),axis=0))
        # Stable ties preserve declared asset order, as in the frozen bank.
        order = np.argsort(-scores,axis=1,kind="stable").astype(np.int32)
        view["orders"] = np.ascontiguousarray(np.concatenate((view["orders"],order[None,:,:]),axis=0))
        metadata["score_names"] = names+[new_name]
        metadata["required_observations_by_score"][new_name] = required
        candidate["score"] = new_name
    view["features"][:,assets.index(CASH),fields["valid"]] = 0.
    # Never overwrite ma250: it is also the crash-distance input, even for the
    # benchmark when 510300 is an eligible trade candidate.
    if ma_window in (180,250):
        candidate["ma"] = "ma%d" % ma_window
        provenance.update(ma_center_reused=True,ma_slot=candidate["ma"],ma_overridden_asset=None)
    else:
        own_dates,prices = _dated_prices(histories,BENCHMARK,dates[-1])
        distance = prices/_rolling_mean(prices,ma_window)-1.
        benchmark = assets.index(BENCHMARK)
        view["features"][:,benchmark,fields["ma180"]] = np.nan
        for j,date in enumerate(own_dates):
            if date in lookup:
                view["features"][lookup[date],benchmark,fields["ma180"]] = distance[j]
        candidate["ma"] = "ma180"
        provenance.update(ma_center_reused=False,ma_slot="ma180",ma_overridden_asset=BENCHMARK,
            missing_ma_behavior="NaN before enough benchmark observations; existing native initialization fallback unchanged")
    provenance["active_score"] = candidate["score"]
    metadata["v12_diagnostic_features"] = provenance
    if ((window,smooth) != CENTERS[score] or ma_window != int(config["ma"][2:])):
        # Do not mislabel a counterfactual as the frozen parent candidate.
        candidate.pop("id",None)
        candidate.pop("hash",None)
    return view,metadata,candidate
