"""Finite development-adaptive component cross, explicitly separate from V11-A."""
from copy import deepcopy
from itertools import product
import json
import numpy as np
from .data import BASE,START,DEV_END,dump,sha,protect
from .registry import canonical
from .scan import SCENARIOS,run_candidates,summarize_rows,select,stamp
from v10_deep.schema import identifier

RISK_KEYS=('panic_mode','panic','risk_context')
TURN_KEYS=('buffer_mode','buffer','global_buffer','gold_buffer','rank_keep','min_hold','switch_confirm','trend_buffer')


def generate(registry,rows,selection=None):
    selection=selection or select(rows,registry['candidates'],registry['controls'])
    configs={c['id']:c for c in registry['candidates']};h=configs[registry['controls']['h']]
    def key(c):
        cid=c['id'];s=selection['scores'][cid]
        return (not all(selection['checks'][cid].values()),-s['worst_block_excess'],-s['median_block_excess'],
                s['switches'],s['complexity'],cid)
    ordered=sorted(registry['candidates'],key=key)
    def packages(keys,families,initial):
        out=[];seen=set()
        def add(config,source):
            value={k:config[k] for k in keys};encoded=json.dumps(value,sort_keys=True)
            if encoded in seen:return False
            seen.add(encoded);out.append(dict(values=value,source=source));return True
        for cid in initial:add(configs[cid],cid)
        for family in families:
            count=0
            for c in ordered:
                if c['id'] in registry['controls'].values() or family not in c['families']:continue
                if add(c,c['id']):count+=1
                if count==2:break
        return out
    score=packages(('score',),('robust_regression','multi_horizon_consensus'),(registry['controls']['h'],registry['controls']['v92']))
    risk=packages(RISK_KEYS,('prior_risk_scale','fixed_risk_scale'),(registry['controls']['h'],))
    turnover=packages(TURN_KEYS,('healthy_momentum_retention','healthy_rank_retention'),(registry['controls']['h'],))
    values={};raw=0;original_ids={c['id'] for c in registry['candidates']};membership={}
    for sp,rp,tp in product(score,risk,turnover):
        c=deepcopy(h)
        for part in (sp,rp,tp):c.update(part['values'])
        c=canonical(c);digest=identifier(c);cid='v11_'+digest[:20];raw+=1
        values[cid]=dict(c,id=cid,hash=digest,families=['development_component_combination'],
                        parents=sorted({sp['source'],rp['source'],tp['source']}),stage='combinations')
        membership[cid]=dict(previously_in_stage_a=cid in original_ids)
    # Controls are evaluation references and must not become a new discovery.
    for cid in registry['controls'].values():
        if cid in values:
            values[cid]['families']=sorted(set(values[cid]['families'])|set(configs[cid]['families']))
        else:values[cid]=deepcopy(configs[cid])
    return dict(schema=1,stage='combinations',raw_count=raw,unique_count=len(values),
                combination_unique_count=len(membership),candidates=[values[cid] for cid in sorted(values)],
                controls=registry['controls'],components=dict(score=score,risk=risk,turnover=turnover),
                membership=membership,score_specs=registry['score_specs'],
                scope='Adaptive development-only extension, not initially preregistered and not clean OOS')


def develop():
    from .features import build
    origin=BASE/'results/development';out=BASE/'results/combinations'
    if (out/'selection.json').exists():raise RuntimeError('B selection is already frozen')
    a=json.loads((BASE/'registered_candidates.json').read_text())
    evaluation=json.loads((origin/'evaluation.json').read_text());selection=json.loads((origin/'selection.json').read_text())
    proof=selection['provenance']
    for filename,key in (('registration.json','registration_sha256'),('evaluation.json','evaluation_sha256'),('paths.npz','paths_sha256')):
        if sha(origin/filename)!=proof[key]:raise ValueError('Stage A receipt changed: '+filename)
    if sha(BASE/'registered_candidates.json')!=proof['registry_sha256']:raise ValueError('Stage A registry changed')
    for name,expected in proof['source_sha256'].items():
        if sha(BASE/name)!=expected:raise ValueError('Stage A source changed: '+name)
    registry=generate(a,evaluation['rows'],selection)
    dump(out/'registered_candidates.json',registry)
    arrays,meta=build(registry['score_specs'])
    sources={n:sha(BASE/n) for n in ('COMBINATION_PROTOCOL.md','combinations.py','registry.py','features.py','scan.py','data.py')}
    dump(out/'registration.json',dict(registered_at=stamp(),sources=sources,
        registry_sha256=sha(out/'registered_candidates.json'),a_selection_sha256=sha(origin/'selection.json'),
        a_evaluation_sha256=sha(origin/'evaluation.json'),scenarios=SCENARIOS,period=[START,DEV_END],
        feature_fingerprints=meta['fingerprints'],feature_array_sha256=meta['cache_array_sha256'],
        note='One adaptive extension after seeing only development diagnostics; all confirmation remains undisclosed'))
    results,dates=run_candidates(registry['candidates'],arrays,meta)
    packed={name+'__'+field:values for name,result in results.items() for field,values in result.items()}
    np.savez_compressed(out/'paths.npz',**packed)
    dump(out/'path_metadata.json',dict(dates=dates,ids=[c['id'] for c in registry['candidates']],sha256=sha(out/'paths.npz')))
    rows=summarize_rows(registry['candidates'],results,dates)
    dump(out/'evaluation.json',dict(period=[dates[0],dates[-1]],rows=rows))
    for name,h in sources.items():
        if sha(BASE/name)!=h:raise ValueError('B definition changed during execution')
    selected=select(rows,registry['candidates'],registry['controls'])
    selected['provenance']=dict(registration_sha256=sha(out/'registration.json'),
        registry_sha256=sha(out/'registered_candidates.json'),evaluation_sha256=sha(out/'evaluation.json'),
        paths_sha256=sha(out/'paths.npz'),source_sha256=sources)
    selected['scope']=registry['scope'];dump(out/'selection.json',selected);protect()
    print(json.dumps(dict(raw=registry['raw_count'],unique=registry['unique_count'],
        new_to_a=sum(not item['previously_in_stage_a'] for item in registry['membership'].values()),
        primary=selected['primary'],qualified_count=selected['qualified_count'],parents=selected['parents'],top_return=selected['top_return']),indent=2),flush=True)
    return selected

if __name__=='__main__':develop()
