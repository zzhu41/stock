"""Post-search disclosure for the user's explicit regime-removal question.

No new candidate or selection is generated. The reported open-stock variant is
already in the frozen 5,156-item matrix and selected here with hindsight. Three
additional execution diagnostics are labelled as such, not silently added to
the original pre-search plan or used to replace its frozen champion.
"""
from copy import deepcopy
import json
import statistics
import numpy as np

from . import scan
from .data import BASE, END, protect, sha
from .registry import candidate_hash, controls
from .same_close import simulate_same_close
from v10_next.metrics import summarize


OUT = BASE / "results"


def collect_pairs(report, configs):
    rows = {r["id"]: r for r in report["results"]}
    by_hash = {c["candidate_hash"]: c["id"] for c in configs}
    base = next(c for c in controls() if c["id"] == "c_v91")
    anchor = {}
    for mode in ("ma250", "open_stock", "always_bull", "all_assets"):
        c = deepcopy(base)
        c["regime_mode"] = mode
        c["params"]["bear_open_stock"] = mode == "open_stock"
        name = by_hash.get(candidate_hash(c))
        anchor[mode] = rows[name] if name else None
    paired = {}
    for mode in ("open_stock", "always_bull", "all_assets"):
        cases, unmatched = [], []
        for c in configs:
            if c["regime_mode"] != mode:
                continue
            b = deepcopy(c)
            b["regime_mode"] = "ma250"
            b["params"]["bear_open_stock"] = False
            other = by_hash.get(candidate_hash(b))
            if other is None:
                unmatched.append(c["id"])
                continue
            a, ref = rows[c["id"]], rows[other]
            cases.append(dict(candidate=c["id"], baseline=other,
                              full_cagr_difference=a["full"]["cagr"] - ref["full"]["cagr"],
                              validation_cagr_difference=a["validation"]["cagr"] - ref["validation"]["cagr"]))
        paired[mode] = dict(matched=len(cases), unmatched_ids=unmatched,
                            higher_full_cagr=sum(x["full_cagr_difference"] > 1e-12 for x in cases),
                            median_full_difference=statistics.median(x["full_cagr_difference"] for x in cases),
                            comparisons=cases)
    return anchor, paired


def main():
    protect()
    r = json.loads((OUT / "evaluation.json").read_text())
    configs = json.loads((OUT / "registry.json").read_text())["candidates"]
    by_id = {c["id"]: c for c in configs}
    candidate = deepcopy(by_id[r["regime_best"]["open_stock"]["id"]])
    selection_before = sha(OUT / "selection.json")
    anchor, paired = collect_pairs(r, configs)
    scan.setup([candidate])
    dates = [d for d in scan._FRAME.dates if scan.START <= d <= END]
    exact = scan.simulate(candidate, END, details=True)
    primary_metrics, reproduced_returns = scan.summaries(exact, dates)
    index = r["regime_best"]["open_stock"]["index"]
    original_returns = np.load(OUT / "full_returns.npy", mmap_mode="r")[:, index]
    if not np.allclose(reproduced_returns, original_returns, rtol=1e-11, atol=1e-12):
        raise AssertionError("Post-search replay differs from the registered matrix")
    costs = {str(bp): scan.summaries(scan.simulate(candidate, END, slippage=bp / 10000), dates)[0]
             for bp in (20, 50)}
    ideal = summarize(simulate_same_close(candidate, scan._BANK, scan._FEAR))
    registered_neighbors = []
    by_hash = {c["candidate_hash"]: c["id"] for c in configs}
    rows = {x["id"]: x for x in r["results"]}
    for window in (20, 25, 30, 40):
        c = deepcopy(candidate)
        c["score_windows"] = [window]
        name = by_hash.get(candidate_hash(c))
        if name:
            registered_neighbors.append(dict(window=window, existing_candidate=name, metrics=rows[name]))
    report = dict(label="Post-search, hindsight-selected regime candidate; not the development champion",
                  rationale="User explicitly asked to examine removal of the bull/bear restriction",
                  computed_at=scan.stamp(), registration_sha256=sha(OUT / "registration.json"),
                  candidate=candidate, selection_sha256=selection_before,
                  evaluation_sha256=sha(OUT / "evaluation.json"),
                  original_v91_anchor_ablation=anchor, matched_regime_pairs=paired,
                  primary=rows[candidate["id"]], costs=costs, ideal_close=ideal,
                  registered_window_neighbors=registered_neighbors,
                  added_candidate_count=0, added_diagnostic_scenarios=3,
                  main_scenario_reproduction=True,
                  caution="Pool and window also differ from original v9.1; the total improvement is not attributable to removing one gate.")
    scan.dump(OUT / "regime_followup.json", report)
    scan.dump(OUT / "regime_candidate_path.json", exact)
    assert sha(OUT / "selection.json") == selection_before
    protect()
    print("Regime post-search diagnostics saved:", candidate["id"], costs["20"]["full"]["cagr"], ideal["cagr"])


if __name__ == "__main__":
    main()
