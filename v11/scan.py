"""Development-only batch evaluation and a deterministic preregistered selector."""
import argparse
from bisect import bisect_left,bisect_right
from collections import defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import time
import numpy as np

from .data import BASE,ROOT,START,DEV_END,CONFIRM_END,END,sha,dump,protect,load_frozen
from .registry import register
from v10_deep.native import Simulator
from v10_deep.reference import run_reference
from v10_deep.scan import metrics

SCENARIOS=(('close_1bp',0,.0001),('close_11bp',0,.0011),('lag1_1bp',1,.0001),('lag1_11bp',1,.0011))
DEV_BLOCKS=(('early','2014-01-02','2017-12-31'),('late','2018-01-01',DEV_END))


def stamp():return datetime.now(timezone.utc).isoformat()


def verify_registered_sources(sources, registry_hash):
    changed=[name for name,expected in sources.items() if sha(BASE/name)!=expected]
    if changed or sha(BASE/'registered_candidates.json')!=registry_hash:
        raise ValueError('Research definition changed during execution: '+', '.join(changed))


def config_for_scenario(config,lag):
    c=deepcopy(config);c['lag']=lag
    return c


def run_candidates(candidates,arrays,meta,start=START,end=DEV_END,scenarios=SCENARIOS,workers=2):
    from .features import risk_view
    groups={};view_keys={}
    fvalid=meta['feature_names'].index('valid')
    for index,c in enumerate(candidates):
        lookup=(c['risk_context'],c['score'])
        if lookup not in view_keys:
            view=risk_view(arrays,meta,c['risk_context'],score=c['score'])
            key=(c['risk_context'],hashlib.sha256(np.ascontiguousarray(view['features'][:,:,fvalid]).tobytes()).hexdigest())
            view_keys[lookup]=key
            if key not in groups:groups[key]=dict(view=view,indices=[])
        groups[view_keys[lookup]]['indices'].append(index)
    lo,hi=bisect_left(meta['dates'],start),bisect_right(meta['dates'],end)
    dates=meta['dates'][lo:hi];count,days=len(candidates),len(dates)
    if not days:raise ValueError('Empty dated evaluation')
    results={name:dict(returns=np.empty((count,days)),holdings=np.empty((count,days),dtype=np.int32),
                      summary=np.empty((count,10))) for name,_,_ in scenarios}
    for number,(key,group) in enumerate(groups.items(),1):
        indices=group['indices'];sim=Simulator(group['view'],meta)
        for name,lag,fee in scenarios:
            for begin in range(0,len(indices),128):
                positions=indices[begin:begin+128]
                configs=[config_for_scenario(candidates[i],lag) for i in positions]
                result=sim.run(configs,start=start,end=end,fee=fee,workers=workers)
                for field in ('returns','holdings','summary'):results[name][field][positions]=result[field]
        print('evaluated view %d/%d: %s, %d configs'%(number,len(groups),key[0],len(indices)),flush=True)
    for prefix in ('close','lag1'):
        low,high=prefix+'_1bp',prefix+'_11bp'
        if low in results and high in results and not np.array_equal(results[low]['holdings'],results[high]['holdings']):
            raise AssertionError('Fees unexpectedly changed a price-only single-position target path')
    return results,dates


def summarize_rows(candidates,results,dates,blocks=DEV_BLOCKS):
    rows=[]
    for i,c in enumerate(candidates):
        value=dict(id=c['id'],families=c['families'],scenarios={})
        for name,payload in results.items():
            daily=payload['returns'][i]
            whole=metrics(daily);sub={}
            for label,start,end in blocks:
                a,b=bisect_left(dates,start),bisect_right(dates,end)
                if b>a:sub[label]=metrics(daily[a:b])
            value['scenarios'][name]=dict(full=whole,blocks=sub,switches=int(payload['summary'][i,3]),
                                         crashes=int(payload['summary'][i,7]))
        rows.append(value)
    return rows


def complexity(config,baseline):
    ignored={'id','hash','families','parents','stage','risk_context'}
    return sum(config.get(k)!=v for k,v in baseline.items() if k not in ignored)+int(config['risk_context']!='current20')


def select(rows,candidates,control_ids):
    mapping={r['id']:r for r in rows};configs={c['id']:c for c in candidates}
    benchmark=mapping[control_ids['h']];base=configs[control_ids['h']]
    tests={};scores={}
    for row in rows:
        scenarios=row['scenarios'];bs=benchmark['scenarios']
        m,b=scenarios['close_1bp']['full'],bs['close_1bp']['full']
        checks=dict(return_floor=m['cagr']>=max(.30,.90*b['cagr']),main_drawdown=m['max_dd']>=b['max_dd']-.02)
        excess=[]
        for name in ('close_11bp','lag1_11bp'):
            current,reference=scenarios[name]['full'],bs[name]['full']
            checks[name+'_growth']=current['cagr']>=reference['cagr']-1e-12
            checks[name+'_drawdown']=current['max_dd']>=reference['max_dd']-.02
            for block in scenarios[name]['blocks']:
                excess.append(scenarios[name]['blocks'][block]['cagr']-bs[name]['blocks'][block]['cagr'])
        tests[row['id']]=checks
        scores[row['id']]=dict(worst_block_excess=min(excess),median_block_excess=float(np.median(excess)),
                               switches=scenarios['close_1bp']['switches'],complexity=complexity(configs[row['id']],base))
    def key(row):
        value=scores[row['id']]
        return (-value['worst_block_excess'],-value['median_block_excess'],value['switches'],value['complexity'],row['id'])
    # Frozen controls can qualify but are never renamed to a new V11 discovery.
    noncontrols=[r for r in rows if r['id'] not in set(control_ids.values())]
    qualified=sorted([r for r in noncontrols if all(tests[r['id']].values())],key=key)
    leaders={}
    for row in qualified:
        for family in row['families']:
            if family not in leaders:leaders[family]=row['id']
    parents=[]
    for row in qualified:
        if row['id'] in leaders.values() and row['id'] not in parents:parents.append(row['id'])
        if len(parents)==4:break
    return dict(primary=qualified[0]['id'] if qualified else None,parents=parents,qualified_count=len(qualified),
                top_development=[r['id'] for r in sorted(noncontrols,key=key)[:30]],
                top_return=max(noncontrols,key=lambda r:r['scenarios']['close_1bp']['full']['cagr'])['id'],
                family_leaders=leaders,checks=tests,scores=scores,controls=control_ids,
                frozen_at=stamp(),selection_period=[START,DEV_END],clean_oos=False,
                status='development_only_not_confirmation_or_live_approval')


def control_audit(registry,arrays,meta,results,dates):
    from .features import risk_view
    index={c['id']:i for i,c in enumerate(registry['candidates'])};cases=[]
    for key,cid in registry['controls'].items():
        i=index[cid];c=registry['candidates'][i]
        view=risk_view(arrays,meta,c['risk_context'],score=c['score'])
        for name,lag,fee in SCENARIOS:
            result=run_reference(view,meta,config_for_scenario(c,lag),start=START,end=DEV_END,fee=fee)
            same_returns=np.array_equal(result['returns'],results[name]['returns'][i])
            same_holdings=np.array_equal(result['holdings'],results[name]['holdings'][i])
            if not (same_returns and same_holdings):raise AssertionError('Control fidelity failure: '+key+' '+name)
            cases.append(dict(control=key,scenario=name,exact_returns=same_returns,exact_holdings=same_holdings,
                              metrics=metrics(result['returns'])))
    return dict(passed=True,period=[dates[0],dates[-1]],cases=cases)


def develop():
    from .features import build
    protect();registry=register();out=BASE/'results/development'
    if (out/'selection.json').exists():raise RuntimeError('Development selection already frozen; do not overwrite')
    arrays,meta=build(registry['score_specs'])
    sources={n:sha(BASE/n) for n in ('PROTOCOL.md','data.py','registry.py','features.py','scan.py')}
    registry_hash=sha(BASE/'registered_candidates.json')
    dump(out/'registration.json',dict(registered_at=stamp(),sources=sources,
        registry_sha256=registry_hash,scenarios=SCENARIOS,development_blocks=DEV_BLOCKS,
        feature_fingerprints=meta.get('fingerprints'),candidate_count=registry['unique_count'],
        only_evaluated_until=DEV_END,clean_oos=False))
    results,dates=run_candidates(registry['candidates'],arrays,meta)
    audit=control_audit(registry,arrays,meta,results,dates);dump(out/'control_fidelity.json',audit)
    # Float64 paths; packed artifacts are ignored while final archives are staged.
    packed={}
    for name,data in results.items():
        for field,value in data.items():packed[name+'__'+field]=value
    np.savez_compressed(out/'paths.npz',**packed)
    dump(out/'path_metadata.json',dict(dates=dates,ids=[c['id'] for c in registry['candidates']],sha256=sha(out/'paths.npz')))
    rows=summarize_rows(registry['candidates'],results,dates)
    dump(out/'evaluation.json',dict(period=[dates[0],dates[-1]],rows=rows))
    verify_registered_sources(sources,registry_hash)
    selection=select(rows,registry['candidates'],registry['controls'])
    selection['provenance']=dict(registration_sha256=sha(out/'registration.json'),
        evaluation_sha256=sha(out/'evaluation.json'),paths_sha256=sha(out/'paths.npz'),
        registry_sha256=registry_hash,source_sha256=sources)
    dump(out/'selection.json',selection)
    protect()
    print(json.dumps({k:selection[k] for k in ('primary','qualified_count','parents','top_return','family_leaders')},indent=2),flush=True)
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['develop']);args=parser.parse_args()
    if args.command=='develop':develop()
