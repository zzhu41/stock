"""Frozen A/B/C union training followed by continuous actual-account execution."""
import argparse
from bisect import bisect_left, bisect_right
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from .data import BASE, ROOT, START, load_inputs, sha, dump, stamp
from .exante_features import build_exante, view_for_config
from .scan import run_candidates, summary_rows, SCENARIOS, NUMERIC_FIELDS
from .selection import select
from .reference import DatedValues
from . import walkforward
from v10_deep.data import inputs
from v10_deep.schema import identifier
from v11.features import risk_view

END = "2025-12-31"
OUT = BASE / "results/reselection"
FOLDS = tuple(walkforward.training_folds())
REGISTRY_PATHS = (BASE/"registered_candidates.json", BASE/"results/consensus/registered_candidates.json",
                  BASE/"results/exante/registered_candidates.json")
SOURCE_PATHS = tuple(BASE/name for name in (
    "reselection.py","RESELECTION_PROTOCOL.md","tests/test_reselection.py","data.py","scan.py","selection.py",
    "exante_features.py","consensus_features.py","diagnostic_features.py","reference.py","walkforward.py",
    "native.py","native.cpp","schema.py","allocation.py","audit_stages.py")) + tuple(ROOT/name for name in (
    "v11/features.py","v10_deep/reference.py","v10_deep/schema.py","v10_deep/portfolios.py"))


def require(value, message):
    if not value: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def same_or_create(path, value):
    if path.exists():
        require(read(path)==json.loads(json.dumps(value)), "Frozen definition changed: "+str(path))
    else: dump(path,value)


def truncate(arrays, meta, end):
    stop=bisect_right(meta["dates"],end)
    require(stop>0,"Empty training prefix")
    view={name:np.array(v[:,:stop] if name in ("scores","orders") else v[:stop],copy=True,order="C")
          for name,v in arrays.items()}
    metadata=deepcopy(meta);metadata["dates"]=meta["dates"][:stop]
    if "shape" in metadata: metadata["shape"][0]=stop
    return view,metadata


def prefix_features(arrays, meta, histories, end):
    clipped,metadata=truncate(arrays,meta,end)
    own={code:[row for row in rows if row[0]<=end] for code,rows in histories.items()}
    extended,metadata,unused=build_exante(clipped,metadata,own)
    return extended,metadata


def union_registry(registries):
    require(len(registries)==3,"All three registries are required")
    controls=registries[0]["controls"];result={}
    for stage,registry in zip(("A","B","C"),registries):
        require(registry["controls"]==controls,"Stage controls differ")
        for candidate in registry["candidates"]:
            cid=candidate["id"];key={"single":"config","allocation":"allocation","benchmark":"benchmark"}[candidate["kind"]]
            if candidate["kind"]=="single":
                require(cid=="v12_"+identifier(candidate["config"])[:20],"ID lost semantic configuration")
            if cid in result:
                previous=result[cid]
                require(previous["kind"]==candidate["kind"] and previous[key]==candidate[key],"ID semantics differ")
                require(previous["complexity"]==candidate["complexity"],"Duplicate complexity changed")
                previous["selectable"]=previous["selectable"] and candidate["selectable"]
                previous["families"]=sorted(set(previous["families"])|set(candidate["families"]))
                previous["source_stages"].append(stage)
            else: result[cid]=dict(deepcopy(candidate),source_stages=[stage])
    return dict(candidates=[result[cid] for cid in sorted(result)],controls=deepcopy(controls),
                unique_count=len(result),adaptive_fixed_union=True,clean_oos=False)


def load_union():
    # Pins only; no full-period performance table or winner is used.
    for stage in ("main","consensus","exante"):
        registered=read(BASE/"results"/stage/"registration.json")
        for group in registered["design"]["fingerprints"].values():
            for name,expected in group.items():
                require(sha(BASE/name)==expected,"Registered dependency changed: "+name)
    registry=union_registry([read(path) for path in REGISTRY_PATHS])
    require(registry["unique_count"]==1807,"Fixed union must have 1807 records")
    return registry


def fingerprints():
    paths=list(REGISTRY_PATHS)+[BASE/name for name in (
        "inputs/manifest.json","inputs/features.json","inputs/features.npz.gz","PROTOCOL.md","EXANTE_PROTOCOL.md",
        "results/main/registration.json","results/consensus/registration.json","results/exante/registration.json")]
    return dict(sources={str(p.relative_to(ROOT)):sha(p) for p in SOURCE_PATHS},
                inputs={str(p.relative_to(ROOT)):sha(p) for p in paths})


def register(out=OUT):
    out=Path(out);registry=load_union()
    same_or_create(out/"registered_candidates.json",registry)
    design=dict(fingerprints=fingerprints(),registry_sha256=sha(out/"registered_candidates.json"),
                folds=FOLDS,scenarios=SCENARIOS,count=1807,execution_start=START,execution_end=END,
                fallback=registry["controls"]["h"],allocation_requires_execution_review=True,clean_oos=False)
    if (out/"registration.json").exists():
        require(read(out/"registration.json")["design"]==json.loads(json.dumps(design)),"Registration changed")
    else: dump(out/"registration.json",dict(registered_at=stamp(),design=design))
    return registry,read(out/"registration.json")


def run_union(records, arrays, meta, end, evaluator=run_candidates, cache_dir=None):
    positions={c["id"]:i for i,c in enumerate(records)};groups={}
    for record in records:
        if record["kind"]=="single":
            config=record["config"]
            groups.setdefault((config.get("risk_context","current20"),config["score"]),[]).append(record)
    merged,observed,assigned=None,None,set()
    def merge(subset,view):
        nonlocal merged,observed
        result,dates=evaluator(subset,view,meta,start=START,end=end,scenarios=SCENARIOS,workers=1,cache_dir=cache_dir)
        require(dates and dates[-1]<=end,"Training crossed cutoff")
        if merged is None:
            observed=dates
            merged={name:{field:np.empty((len(records),)+values[field].shape[1:],dtype=values[field].dtype)
                          for field in NUMERIC_FIELDS} for name,values in result.items()}
            for values in merged.values(): values["metadata"]={}
        require(observed==dates,"Group date axes differ")
        for j,record in enumerate(subset):
            cid=record["id"];i=positions[cid]
            for name,values in result.items():
                for field in NUMERIC_FIELDS:
                    if cid in assigned:
                        require(np.array_equal(merged[name][field][i],values[field][j]),"Fixed anchor changed")
                    else: merged[name][field][i]=values[field][j]
                merged[name]["metadata"][cid]=values["metadata"][cid]
            assigned.add(cid)
    for subset in groups.values(): merge(subset,view_for_config(arrays,meta,subset[0]["config"]))
    portfolios=[c for c in records if c["kind"]!="single"]
    if portfolios:
        bases={c["allocation"]["base_id"] for c in portfolios if c["kind"]=="allocation"}
        merge([c for c in records if c["id"] in bases]+portfolios,arrays)
    require(assigned==set(positions),"Incomplete training family")
    return merged,observed


def train_fold(registry, arrays, meta, histories, fold, out, evaluator=run_union):
    out=Path(out);cutoff=fold["train_end"]
    require(not (out/"selection.json").exists(),"Fold already frozen")
    extended,metadata=prefix_features(arrays,meta,histories,cutoff)
    receipt=dict(cutoff=cutoff,actual_end=metadata["dates"][-1],metadata=metadata,
        array_sha256={name:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for name,v in extended.items()})
    same_or_create(out/"feature_receipt.json",receipt)
    results,dates=evaluator(registry["candidates"],extended,metadata,cutoff)
    require(dates==[d for d in metadata["dates"] if START<=d<=cutoff],"Fold date axis changed")
    np.savez_compressed(out/"paths.npz",**{name+"__"+field:values[field] for name,values in results.items() for field in NUMERIC_FIELDS})
    dump(out/"path_metadata.json",dict(ids=[c["id"] for c in registry["candidates"]],dates=dates,sha256=sha(out/"paths.npz")))
    dump(out/"execution_metadata.json",{name:values["metadata"] for name,values in results.items()})
    rows=summary_rows(registry["candidates"],results,dates,period_end=cutoff)
    dump(out/"evaluation.json",dict(rows=rows,period=[START,cutoff],clean_oos=False))
    selection=select(rows,registry["candidates"],registry["controls"],period_end=cutoff)
    chosen=selection["primary"] or registry["controls"]["h"]
    record=next(c for c in registry["candidates"] if c["id"]==chosen)
    selection.update(chosen_id=chosen,chosen_kind=record["kind"],used_h_fallback=selection["primary"] is None,
        provenance={name:sha(out/name) for name in ("paths.npz","path_metadata.json","evaluation.json","feature_receipt.json","execution_metadata.json")})
    dump(out/"selection.json",selection)
    return dict(fold=fold,selection=selection,directory=str(out))


def selected_schedules(registry, trained):
    records={c["id"]:c for c in registry["candidates"]};h=records[registry["controls"]["h"]]
    def segment(record,start,end):
        config=dict(deepcopy(record["config"]),id=record["id"])
        return dict(start=start,end=end,config=config,risk_context=config.get("risk_context","current20"))
    schedule=[segment(h,START,"2017-12-31")]
    for item in trained:
        record=records[item["selection"]["chosen_id"]]
        require(record["kind"]=="single","Allocation selected: preserve result and review continuous execution")
        schedule.append(segment(record,item["fold"]["start"],item["fold"]["end"]))
    return dict(reselected=schedule,H=[segment(h,START,END)])


def grandfathered_factory(arrays, meta, config, context):
    require(config["buffer_mode"]=="momentum" and config["crash_guard"]=="always","Unscored inherited holding cannot enter score comparisons")
    view=view_for_config(arrays,meta,config)
    original=risk_view(arrays,meta,context or "current20",score=None)
    pool=set(config["stock_pool"])|set(config["global_pool"])|{"518880"}
    valid=meta["feature_names"].index("valid")
    lane=arrays["scores"][meta["score_names"].index(config["score"])]
    for i,code in enumerate(meta["assets"]):
        if code not in pool and code!="511880":
            view["features"][:,i,valid]=original["features"][:,i,valid]
    class GrandfatheredValues(DatedValues):
        def ranked(self,day):
            eligible=[code for code in self.assets if code in pool and self.informed(day,code)
                      and np.isfinite(lane[day,self.asset_index[code]]) and lane[day,self.asset_index[code]]>-1e90]
            return sorted(eligible,key=lambda code:-self.score(day,code))
        def score(self,day,code):
            if code is None or not self.informed(day,code): return 0.
            value=float(lane[day,self.asset_index[code]])
            return value if math.isfinite(value) and value>-1e90 else 0.
    return GrandfatheredValues(view,meta,config)


def execute(registry, trained, arrays, meta, histories, out):
    definitions=selected_schedules(registry,trained)
    extended,metadata=prefix_features(arrays,meta,histories,END)
    from .audit_stages import independent_metrics
    results,raw={},{}
    for label,definition in definitions.items():
        results[label],raw[label]={},{}
        for name,lag,fee in SCENARIOS:
            schedule=deepcopy(definition)
            for segment in schedule: segment["config"]["lag"]=lag
            result=walkforward.run(schedule,extended,metadata,start=START,end=END,fee=fee,view_factory=grandfathered_factory)
            directory=Path(out)/"execution"/label/name
            same_or_create(directory/"schedule.json",schedule)
            walkforward.write_trace_csv(directory/"trace.csv",result)
            dump(directory/"transactions.json",dict(trades=result["trades"],boundaries=result["boundaries"],final_account=result["final_account"],metadata=result["metadata"]))
            np.savez_compressed(directory/"paths.npz",returns=result["returns"],holdings=result["holdings"],summary=result["summary"],nav=np.asarray([r[1] for r in result["daily"]]))
            periods={"full_2014_2025":(START,END),"report_2018_2025":("2018-01-01",END)}
            periods.update({"fold_"+f["start"][:4]:(f["start"],f["end"]) for f in FOLDS})
            periods.update({"year_"+y:(y+"-01-01",y+"-12-31") for y in sorted({d[:4] for d in result["dates"]})})
            metrics={}
            for period,(begin,end) in periods.items():
                lo,hi=bisect_left(result["dates"],begin),bisect_right(result["dates"],end)
                value=independent_metrics(result["returns"][lo:hi])
                value.update(start=result["dates"][lo],end=result["dates"][hi-1],
                    switches=sum(begin<=t["date"]<=end and not t["first_session_free"] for t in result["trades"]))
                metrics[period]=value
            results[label][name]=dict(periods=metrics,path_sha256=sha(directory/"paths.npz"),trace_sha256=sha(directory/"trace.csv"),boundaries=result["boundaries"])
            raw[label][name]=result
    index=next(i for i,c in enumerate(registry["candidates"]) if c["id"]==registry["controls"]["h"])
    proofs=[]
    for item in trained:
        with np.load(Path(item["directory"])/"paths.npz",allow_pickle=False) as paths:
            for name,unused_lag,unused_fee in SCENARIOS:
                stop=bisect_right(raw["H"][name]["dates"],item["fold"]["train_end"])
                for field in ("returns","holdings"):
                    require(np.array_equal(raw["H"][name][field][:stop],paths[name+"__"+field][index]),"Continuous H prefix differs")
                proofs.append(dict(train_end=item["fold"]["train_end"],scenario=name,exact=True))
    return dict(streams=results,h_prefix_fidelity=proofs,known_history=True,clean_oos=False)


def run(phase="register", out=OUT):
    out=Path(out);registry,registration=register(out)
    expected=registration["design"]["fingerprints"]
    require(fingerprints()==expected,"Sources changed after registration")
    if phase=="register": return registration
    arrays,meta,unused=load_inputs();histories,unused_calendar,unused_fear=inputs()
    arrays,meta=truncate(arrays,meta,END)
    histories={code:[r for r in rows if r[0]<=END] for code,rows in histories.items()}
    trained=[]
    for fold in FOLDS:
        directory=out/"folds"/fold["train_end"][:4]
        if (directory/"selection.json").exists():
            selection=read(directory/"selection.json")
            for name,digest in selection["provenance"].items():
                require(sha(directory/name)==digest,"Frozen fold artifact changed")
            item=dict(fold=fold,selection=selection,directory=str(directory))
        else:
            require(phase=="train","Execution requires all frozen fold selections")
            item=train_fold(registry,arrays,meta,histories,fold,directory)
        trained.append(item)
        require(fingerprints()==expected,"Source changed during training")
        print("TRAIN FROZEN",fold["train_end"],item["selection"]["chosen_id"],item["selection"]["chosen_kind"],item["selection"]["qualified_count"],flush=True)
    summary=dict(folds=[dict(fold=t["fold"],chosen_id=t["selection"]["chosen_id"],chosen_kind=t["selection"]["chosen_kind"],
        qualified_count=t["selection"]["qualified_count"],used_h_fallback=t["selection"]["used_h_fallback"]) for t in trained],
        registration_sha256=sha(out/"registration.json"),source_fingerprints=expected,clean_oos=False)
    same_or_create(out/"training_summary.json",summary)
    if phase=="execute":
        require(not (out/"evaluation.json").exists(),"Continuous result already frozen")
        result=execute(registry,trained,arrays,meta,histories,out)
        require(fingerprints()==expected,"Source changed during execution")
        result.update(training_summary_sha256=sha(out/"training_summary.json"),registration_sha256=sha(out/"registration.json"))
        dump(out/"evaluation.json",result)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("register","train","execute"),default="register")
    parser.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args()
    run(args.phase,args.output)
