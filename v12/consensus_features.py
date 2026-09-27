"""Pure fixed B-stage score construction; no I/O, simulation or selection.

New scores use the original ten risk ETFs only. All original feature fields,
score lanes and order lanes are preserved. Because native.valid does not read
score sentinels, callers must additionally mask a copied validity column by
the active new lane's finite/non-sentinel values before simulating that lane.
"""
from copy import deepcopy

import numpy as np

from v10_next.frozen.metadata import UNIVERSE,CASH
from .diagnostic_features import make_view,_dated_prices,INVALID

RISK_ASSETS = tuple(code for code in UNIVERSE if code != CASH)
SCORE_NAMES = dict(equal_score="v12_b_equal_score",
                   equal_rank="v12_b_equal_rank",
                   wls25_smooth3="v12_b_wls25_smooth3")
INPUT_NAMES = ("wls25_v20","wls20_smooth3")
VALID_CUTOFF = -1e90


def _ranks(values, eligible):
    """Ascending average rank / current eligible N; no ordinal tie bias."""
    result = np.full(values.shape,INVALID,dtype=np.float64)
    for day in range(len(values)):
        members = np.flatnonzero(eligible[day])
        if not len(members):
            continue
        order = members[np.argsort(values[day,members],kind="stable")]
        start = 0
        while start < len(order):
            end = start+1
            while end < len(order) and values[day,order[end]] == values[day,order[start]]:
                end += 1
            result[day,order[start:end]] = ((start+1+end)/2.)/len(order)
            start = end
    return result


def build_consensus(arrays, meta, histories):
    """Append three fixed lanes and return (arrays, metadata, name_mapping).

    Rank denominators use the same contemporaneous, jointly valid original
    risk universe for both component scores. Eligibility also requires an
    actual same-day history row and 270 own quoted observations. No score is
    made investable by forward-filling or smoothing a sentinel.
    """
    names,assets,dates = list(meta["score_names"]),list(meta["assets"]),list(meta["dates"])
    if len(names) != 2 or set(names) != set(INPUT_NAMES):
        raise ValueError("B consensus expects only the two original V12 score lanes")
    if len(assets) != len(set(assets)) or not set(RISK_ASSETS+(CASH,)) <= set(assets):
        raise ValueError("B consensus requires all original eleven assets")
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Unique ascending observation dates required")
    fields = {name:i for i,name in enumerate(meta["feature_names"])}
    if not {"close","valid"} <= set(fields):
        raise ValueError("Missing close/valid fields")
    t,a = len(dates),len(assets)
    if (arrays["features"].shape != (t,a,len(fields))
            or arrays["scores"].shape != (2,t,a) or arrays["orders"].shape != (2,t,a)):
        raise ValueError("Consensus feature axes do not match metadata")
    view = {key:np.array(value,copy=True,order="C") for key,value in arrays.items()}
    metadata = deepcopy(meta)
    cube = arrays["features"]
    history_eligible = np.zeros((t,a),dtype=bool)
    lookup = {date:i for i,date in enumerate(dates)}
    for code in RISK_ASSETS:
        ai = assets.index(code)
        own_dates,unused_prices = _dated_prices(histories,code,dates[-1])
        for j,date in enumerate(own_dates):
            if j >= 269 and date in lookup:
                history_eligible[lookup[date],ai] = True
    eligible = (history_eligible & (cube[:,:,fields["valid"]] > .5)
                & np.isfinite(cube[:,:,fields["close"]]) & (cube[:,:,fields["close"]] > 0))
    h = arrays["scores"][names.index("wls20_smooth3")]
    wls25 = arrays["scores"][names.index("wls25_v20")]
    common = eligible & np.isfinite(h) & np.isfinite(wls25) & (h > VALID_CUTOFF) & (wls25 > VALID_CUTOFF)
    equal_score = np.full((t,a),INVALID,dtype=np.float64)
    equal_score[common] = .5*h[common]+.5*wls25[common]
    equal_rank = np.full((t,a),INVALID,dtype=np.float64)
    h_rank,wls25_rank = _ranks(h,common),_ranks(wls25,common)
    equal_rank[common] = .5*h_rank[common]+.5*wls25_rank[common]

    # The helper computes raw legacy WLS25 before the 270-observation mask,
    # then smooths three own quotes. Its altered features are NOT adopted.
    smooth_view,smooth_meta,smooth_config = make_view(arrays,meta,histories,
        dict(score="wls25_v20",ma="ma250"),window=25,smooth=3)
    raw_smooth = smooth_view["scores"][smooth_meta["score_names"].index(smooth_config["score"])]
    wls25_smooth3 = np.full((t,a),INVALID,dtype=np.float64)
    smooth_valid = eligible & np.isfinite(raw_smooth) & (raw_smooth > VALID_CUTOFF)
    wls25_smooth3[smooth_valid] = raw_smooth[smooth_valid]
    extra = np.stack((equal_score,equal_rank,wls25_smooth3))
    view["scores"] = np.ascontiguousarray(np.concatenate((view["scores"],extra),axis=0))
    extra_orders = np.argsort(-extra,axis=2,kind="stable").astype(np.int32)
    view["orders"] = np.ascontiguousarray(np.concatenate((view["orders"],extra_orders),axis=0))
    metadata["score_names"] = names+list(SCORE_NAMES.values())
    metadata.setdefault("required_observations_by_score",{}).update({name:270 for name in metadata["score_names"]})
    metadata["v12_consensus_features"] = dict(
        mapping=dict(SCORE_NAMES),risk_assets=list(RISK_ASSETS),minimum_own_observations=270,
        equal_score_weights=[.5,.5],components=["wls20_smooth3","wls25_v20"],
        rank="ascending average rank/N on jointly valid original risk assets at this observation",
        rank_weights=[.5,.5],rank_extra_smoothing=0,
        final_tie_break="stable declared asset order",
        wls25_smoothing="three trailing own raw legacy WLS25 scores before investability mask",
        original_arrays_preserved=True,non_original_assets_and_cash="sentinel in every new lane",
        native_active_score_validity_mask_required=True,
        active_score_validity_rule="copied feature.valid &= isfinite(active_score) and active_score > -1e90",
        missing_or_unseasoned="sentinel; no forward fill",future_observations_used=False)
    return view,metadata,dict(SCORE_NAMES)
