"""Nine fixed round-two hypotheses, including two comparison controls.

This registry imports only the first round's pure configuration factory. It
does not mutate that registry, read performance results, or run a simulation.
"""
from copy import deepcopy

from v10_next.candidates import get_candidate as first_round_candidate


START, END = "2014-01-01", "2026-09-11"
FEE, SLIPPAGE = .0001, .001
HIGH_RETURN_IDS = ("c_v91", "h_qvix_only", "h_v91_no_overheat", "h_qvix_score25_30")
STABILITY_IDS = ("s_v91_wls30", "s_v91_wls30_no_crash", "s_accounts25_30", "s_accounts20_25_30")
ROBUST_IDS = STABILITY_IDS

PROTOCOL = dict(
    candidate_budget=9, controls=2, formal_candidates=7,
    start=START, end=END, commission_per_side=FEE, main_slippage_per_side=SLIPPAGE,
    primary_execution="causal next-open, costs on actual executed notional",
    high_return_reference="c_v92", high_return_requirement="strictly higher net CAGR than the same-clock c_v92",
    stability_minimum_net_cagr=.30,
    ideal_close_gate="Defined by the round-two protocol before selection; not chosen after seeing results",
    account_policy="equal initial capital; independently executed; no capital transfers or periodic resets",
    account_trade_netting=False, leverage=False,
    known_history=True,
    caveats=(
        "All historical periods and several candidates were examined previously; this is not a fresh holdout.",
        "All S candidates inherit the original universe and short-horizon momentum architecture; they do not eliminate selection bias.",
        "WLS30's previously reported replay performance is not evidence for this revised causal execution engine.",
        "Independent accounts diversify a parameter choice, not independent sources of alpha; their wealth shares drift.",
        "Do not add candidates or tune weights after observing this bounded batch's outcomes.",
        "If no fixed candidate satisfies a target, report that target as unmet.",
    ),
)


def _single(candidate_id, family, hypothesis, windows=(25,), overrides=None,
            kind="legacy", channels=None, previously_seen=False, prior_note=""):
    item = first_round_candidate("control_v91")
    item.update(id=candidate_id, family=family, kind=kind, hypothesis=hypothesis,
                score_windows=tuple(windows), overrides=dict(overrides or {}),
                previously_seen=previously_seen, prior_note=prior_note)
    if channels is not None:
        item["channels"] = tuple(channels)
    return item


def _accounts(candidate_id, windows):
    components = tuple(_single(
        "%s:book_wls%d" % (candidate_id, window), "component",
        "Fixed v9.1 component account with WLS%d; no independent candidate selection." % window,
        windows=(window,), previously_seen=True,
        prior_note="The ranking window was included in earlier public parameter sensitivity work.")
        for window in windows)
    return dict(
        id=candidate_id, family="robust", kind="accounts",
        hypothesis="Start equal independent v9.1 accounts at windows %s; retain each account's own wealth and trades." % (windows,),
        components=components, initial_weights=tuple(1.0 / len(components) for _ in components),
        rebalance="never", netting=False, capital_transfers=False,
        previously_seen=False, prior_note="New account construction using previously examined WLS windows.",
        inherited_selection_bias=True,
    )


_CANDIDATES = (
    _single("c_v91", "control", "Same-clock v9.1 control; also eligible for H as removal of both external-event channels.",
            previously_seen=True, prior_note="Existing v9.1 and first-round growth reference."),
    _single("c_v92", "control", "Same-clock v9.2 benchmark with the original three-channel crash-entry union.",
            kind="v92", channels=("deep", "qvix", "volume"), previously_seen=True,
            prior_note="Existing v9.2 research/shadow rules, reevaluated under identical causal execution."),
    _single("h_qvix_only", "high_return", "Remove volume-based entries from v9.2; retain deep-drop and dated QVIX entries.",
            kind="v92", channels=("deep", "qvix"), previously_seen=True,
            prior_note="The QVIX-only fz25_m4 mechanism was examined in earlier research."),
    _single("h_v91_no_overheat", "high_return", "Raise the existing overheating threshold to 990%; changes both its entry veto and fast-exit condition.",
            overrides={"overheat": 9.9}, prior_note="Overheat ablation is a previously studied mechanism family; this exact causal v9.1 comparison is preregistered here."),
    _single("h_qvix_score25_30", "high_return", "Remove the volume channel, then average WLS25 and WLS30 scores at fixed equal weights.",
            windows=(25, 30), kind="v92", channels=("deep", "qvix"),
            prior_note="Fixed combination of known QVIX-only and ranking-window ideas; no optimized ensemble weights."),
    _single("s_v91_wls30", "robust", "Recheck the publicly observed WLS30 neighborhood under the causal execution clock.",
            windows=(30,), previously_seen=True,
            prior_note="Prior parameter_stability report used fixed-target replay; those numbers are not interchangeable with this engine."),
    _single("s_v91_wls30_no_crash", "robust", "Use WLS30 and remove the rare crisis-entry/lock rule; no replacement event condition.",
            windows=(30,), overrides={"crash_mom5": 0.0},
            prior_note="A fixed simplification of the known WLS30 candidate; inherits the remaining original strategy choices."),
    _accounts("s_accounts25_30", (25, 30)),
    _accounts("s_accounts20_25_30", (20, 25, 30)),
)
CANDIDATES = tuple(deepcopy(candidate) for candidate in _CANDIDATES)


def get_candidate(candidate_id):
    for candidate in _CANDIDATES:
        if candidate["id"] == candidate_id:
            return deepcopy(candidate)
    raise KeyError("Unknown round-two candidate: " + str(candidate_id))


def registry():
    return dict(protocol=deepcopy(PROTOCOL), candidates=[deepcopy(c) for c in _CANDIDATES],
                high_return_ids=list(HIGH_RETURN_IDS), stability_ids=list(STABILITY_IDS))


if len(_CANDIDATES) != 9 or len({c["id"] for c in _CANDIDATES}) != 9:
    raise AssertionError("Round two must remain a fixed nine-item registry")
