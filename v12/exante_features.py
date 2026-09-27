"""Pure C-stage scores and causal decision-volatility views.

No I/O, cache rebuild, simulation or selection. Rankings always use the fixed
current-observation score arrays; an ex-ante view replaces only decision VOL20
and necessary validity. The caller owns dated prefix truncation and config IDs.
"""
import numpy as np

from v11.features import risk_view
from .consensus_features import build_consensus,RISK_ASSETS,VALID_CUTOFF
from .diagnostic_features import _dated_prices,_raw_scores,INVALID

MULTI_NAME = "v12_c_wls_10_20_40_mean_s1"


def build_exante(arrays, meta, histories):
    """Return (six-score arrays, metadata, B mapping plus multi_10_20_40)."""
    view,metadata,mapping = build_consensus(arrays,meta,histories)
    dates,assets = list(meta["dates"]),list(meta["assets"])
    lookup = {date:i for i,date in enumerate(dates)}
    fields = {name:i for i,name in enumerate(meta["feature_names"])}
    cube = arrays["features"]
    output = np.full((len(dates),len(assets)),INVALID,dtype=np.float64)
    for code in RISK_ASSETS:
        ai = assets.index(code)
        own_dates,prices = _dated_prices(histories,code,dates[-1])
        pieces = [_raw_scores(prices,window) for window in (10,20,40)]
        combined = np.mean(pieces,axis=0)
        for j,date in enumerate(own_dates):
            if j < 269 or date not in lookup or not np.isfinite(combined[j]):
                continue
            i = lookup[date]
            if cube[i,ai,fields["valid"]] > .5 and np.isfinite(cube[i,ai,fields["close"]]):
                output[i,ai] = combined[j]
    view["scores"] = np.ascontiguousarray(np.concatenate((view["scores"],output[None,:,:]),axis=0))
    order = np.argsort(-output,axis=1,kind="stable").astype(np.int32)
    view["orders"] = np.ascontiguousarray(np.concatenate((view["orders"],order[None,:,:]),axis=0))
    metadata["score_names"] = list(metadata["score_names"])+[MULTI_NAME]
    metadata["required_observations_by_score"][MULTI_NAME] = 270
    mapping = dict(mapping,multi_10_20_40=MULTI_NAME)
    metadata["v12_exante_features"] = dict(
        multi_score=MULTI_NAME,known_v11_lane="v11_wls_10_20_40_mean_s1",
        windows=[10,20,40],aggregate="equal mean of each generic WLS/VOL20 score",
        smooth=1,minimum_own_observations=270,risk_assets=list(RISK_ASSETS),
        source_arithmetic="frozen convolution WLS and convolution E[r^2]-E[r]^2 VOL20",
        current_score_arrays_unchanged_by_risk_context=True,
        decision_contexts=["current20","prior20","prior60","prior_max20_60"],
        prior_clock="previous actual quoted row of that same asset, never current/future or an invented missing bar",
        current_original_scores_preserve_frozen_validity=True,
        caller_must_preserve_noncurrent_context_in_config_hash=True,
        caller_truncates_requested_end_before_execution=True)
    return view,metadata,mapping


def view_for_config(arrays, meta, config):
    """Copy decision features; keep all ranking scores/orders unchanged.

    C's registry restricts the volatility-dependent choice rules so that this
    substitution affects panic thresholds, not a low-volatility asset picker.
    As with v11.risk_view, unchanged arrays in the returned dict are shared and
    must be treated as read-only. No config fields are removed or normalized.
    """
    context = config.get("risk_context","current20")
    score = config.get("score")
    if score not in meta["score_names"]:
        raise ValueError("Unknown active score for C decision view")
    view = risk_view(arrays,meta,context,score=None)
    # Preserve the original controls bit-for-bit, including their two initial
    # pre-study 510880 H-score sentinel rows in May 2012. B also leaves original
    # controls unmasked. This exception never applies to a new score/prior view.
    if context == "current20" and score in ("wls25_v20","wls20_smooth3"):
        return view
    lane = arrays["scores"][meta["score_names"].index(score)]
    valid = meta["feature_names"].index("valid")
    view["features"][:,:,valid] *= np.isfinite(lane) & (lane > VALID_CUTOFF)
    return view
