"""Read-only path comparison for every C configuration previously in A/B."""
import json
import numpy as np
from .data import BASE, sha, dump


def run():
    destination=BASE/"results/audit_stages/exante/overlap.json"
    if destination.exists(): raise ValueError("Overlap audit already frozen")
    current=BASE/"results/exante"
    current_meta=json.loads((current/"path_metadata.json").read_text())
    positions={cid:i for i,cid in enumerate(current_meta["ids"])}
    reports=[]
    with np.load(current/"paths.npz",allow_pickle=False) as source:
        # Decompress each matrix only once.
        values={key:source[key] for key in source.files if key.rsplit("__",1)[-1] in ("returns","holdings","summary")}
    for stage in ("main","consensus"):
        folder=BASE/"results"/stage
        meta=json.loads((folder/"path_metadata.json").read_text())
        if meta["dates"]!=current_meta["dates"]: raise AssertionError("Date axes differ")
        common=[(j,positions[cid],cid) for j,cid in enumerate(meta["ids"]) if cid in positions]
        with np.load(folder/"paths.npz",allow_pickle=False) as old:
            for key,new in values.items():
                prior=old[key]
                for j,i,cid in common:
                    if not np.array_equal(prior[j],new[i]):
                        raise AssertionError("Repeated semantic ID changed: "+stage+" "+cid+" "+key)
        reports.append(dict(stage=stage,common_count=len(common),ids=[cid for j,i,cid in common],
                            all_four_scenarios_returns_holdings_summary_exact=True,
                            prior_path_sha256=sha(folder/"paths.npz")))
    value=dict(comparisons=reports,current_path_sha256=sha(current/"paths.npz"),
               audit_source_sha256=sha(BASE/"audit_overlap.py"))
    dump(destination,value)
    print(json.dumps([dict(stage=r["stage"],common_count=r["common_count"],exact=True) for r in reports]))


if __name__=="__main__": run()
