"""Final explicitly adaptive C batch: ex-ante risk-scale interactions."""
from bisect import bisect_right
from copy import deepcopy
from itertools import product
import hashlib
import json
import numpy as np
from .data import BASE,START,END,load_inputs,stamp,sha,dump,protect
from .registry import canonical
from .scan import run_candidates,summary_rows,SCENARIOS,NUMERIC_FIELDS
from .selection import select
from .exante_features import build_exante,view_for_config
from v10_deep.data import inputs
from v10_deep.schema import identifier

OUT=BASE/'results/exante'


def verify_stage(folder,registry_path):
    selection=json.loads((folder/'selection.json').read_text());proof=selection['provenance']
    for name,key in (('registration.json','registration_sha256'),('evaluation.json','evaluation_sha256'),
                     ('paths.npz','paths_sha256'),('path_metadata.json','path_metadata_sha256'),
                     ('execution_metadata.json','execution_metadata_sha256')):
        if sha(folder/name)!=proof[key]:raise ValueError('Prior evidence changed: '+str(folder/name))
    if sha(registry_path)!=proof['registry_sha256']:raise ValueError('Prior registry changed')
    for group in proof['fingerprints'].values():
        for name,h in group.items():
            if sha(BASE/name)!=h:raise ValueError('Prior source/input changed: '+name)
    return json.loads(registry_path.read_text()),selection


def generate(a,b,mapping):
    prior={c['id']:c for c in a['candidates']+b['candidates']};h=prior[a['controls']['h']]['config']
    records={cid:deepcopy(prior[cid]) for cid in a['controls'].values()}
    scores=('wls20_smooth3','wls25_v20',mapping['wls25_smooth3'],mapping['multi_10_20_40'])
    for score,context,mult,ma,hold,buffer,flag in product(scores,
            ('current20','prior20','prior60','prior_max20_60'),(1.25,1.5,1.75),
            ('ma180','ma250'),(0,2,3),('original','uniform2'),(0,1)):
        c=canonical(dict(h,score=score,panic_mode='volatility',panic=mult,ma=ma,min_hold=hold,
                         crash_mask=5,locked_panic_exit=flag))
        if buffer=='uniform2':c.update(buffer=.02,global_buffer=.02,gold_buffer=.02)
        if context!='current20':c['risk_context']=context
        digest=identifier(c);cid='v12_'+digest[:20]
        records[cid]=dict(id=cid,hash=digest,kind='single',config=c,selectable=True,
            families=['exante_'+context],complexity=sum(c.get(k)!=v for k,v in h.items())+int(context!='current20')+int(score==mapping['multi_10_20_40']),
            previously_seen=cid in prior,previous_ids=[cid] if cid in prior else [],
            direction_previously_researched=True)
    if len(records)!=1157:raise AssertionError('C family count changed')
    return dict(schema=1,candidates=[records[cid] for cid in sorted(records)],unique_count=len(records),
        controls=a['controls'],score_mapping=mapping,adaptive_after_a_b=True,
        known_history_including_2026=True,prior_stage_duplicate_ids=sorted(set(records)&set(prior)))


def run_c(records,arrays,meta,start=START,end=END,scenarios=SCENARIOS,workers=1,cache_dir=None):
    stop=bisect_right(meta['dates'],end)
    clipped={name:np.ascontiguousarray(value[:,:stop] if name in ('scores','orders') else value[:stop])
             for name,value in arrays.items()}
    m=deepcopy(meta);m['dates']=meta['dates'][:stop]
    if 'shape' in m:m['shape'][0]=stop
    positions={c['id']:i for i,c in enumerate(records)};groups={}
    for c in records:groups.setdefault((c['config'].get('risk_context','current20'),c['config']['score']),[]).append(c)
    merged=None;dates=None
    for group,subset in groups.items():
        view=view_for_config(clipped,m,subset[0]['config'])
        result,observed=run_candidates(subset,view,m,start=start,end=end,scenarios=scenarios,workers=workers,cache_dir=cache_dir)
        if merged is None:
            dates=observed;merged={name:{field:np.empty((len(records),)+v[field].shape[1:],dtype=v[field].dtype)
                      for field in NUMERIC_FIELDS} for name,v in result.items()}
            for data in merged.values():data['metadata']={}
        if observed!=dates:raise AssertionError('C grouped dates differ')
        ids=[positions[c['id']] for c in subset]
        for name,data in result.items():
            for field in NUMERIC_FIELDS:merged[name][field][ids]=data[field]
            for cid,details in data['metadata'].items():
                merged[name]['metadata'][cid]=dict(details,risk_context=group[0],score=group[1])
    return merged,dates


def fingerprints():
    names=('exante.py','exante_features.py','consensus_features.py','diagnostic_features.py','data.py','registry.py',
        'native.py','native.cpp','schema.py','scan.py','selection.py','allocation.py',
        'PROTOCOL.md','CONSENSUS_PROTOCOL.md','EXANTE_PROTOCOL.md','tests/test_exante.py','tests/test_exante_features.py')
    return dict(sources={n:sha(BASE/n) for n in names},inputs={name:sha(BASE/name) for name in (
        'registered_candidates.json','registration.json','inputs/manifest.json','inputs/features.json','inputs/features.npz.gz',
        'results/main/selection.json','results/consensus/selection.json','results/exante/registered_candidates.json','results/exante/feature_receipt.json')})


def evaluate(workers=1):
    if (OUT/'selection.json').exists():raise ValueError('C already frozen')
    a,sa=verify_stage(BASE/'results/main',BASE/'registered_candidates.json')
    b,sb=verify_stage(BASE/'results/consensus',BASE/'results/consensus/registered_candidates.json')
    arrays,meta,unused=load_inputs();histories,calendar,fear=inputs()
    extended,m,mapping=build_exante(arrays,meta,histories);registry=generate(a,b,mapping)
    path=OUT/'registered_candidates.json'
    if path.exists() and json.loads(path.read_text())!=registry:raise ValueError('C definitions changed')
    if not path.exists():dump(path,registry)
    feature=dict(metadata=m,array_sha256={name:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for name,v in extended.items()},
                 protocol_sha256=sha(BASE/'EXANTE_PROTOCOL.md'),source_sha256=sha(BASE/'exante_features.py'))
    if (OUT/'feature_receipt.json').exists():
        if json.loads((OUT/'feature_receipt.json').read_text())!=feature:raise ValueError('C feature definitions changed')
    else:dump(OUT/'feature_receipt.json',feature)
    expected=fingerprints();design=dict(fingerprints=expected,candidate_count=registry['unique_count'],period=[START,END],
        scenarios=[list(s) for s in SCENARIOS],adaptive_after_a_b=True,a_primary_unchanged=sa['primary'],
        b_primary_unchanged=sb['primary'],known_history_including_2026=True,clean_oos=False)
    if (OUT/'registration.json').exists():
        if json.loads((OUT/'registration.json').read_text())['design']!=design:raise ValueError('C source registration changed')
    else:dump(OUT/'registration.json',dict(registered_at=stamp(),design=design))
    reg_hash=sha(OUT/'registration.json')
    results,dates=run_c(registry['candidates'],extended,m,workers=workers)
    if fingerprints()!=expected:raise ValueError('C source/input changed during evaluation')
    positions={c['id']:i for i,c in enumerate(registry['candidates'])};parity=[]
    for directory in ('main','consensus'):
        folder=BASE/'results'/directory;oldmeta=json.loads((folder/'path_metadata.json').read_text())
        if oldmeta['dates']!=dates:raise ValueError('Prior/C date axes differ')
        common=[(positions[cid],j,cid) for j,cid in enumerate(oldmeta['ids']) if cid in positions]
        with np.load(folder/'paths.npz',allow_pickle=False) as old:
            for name,lag,fee in SCENARIOS:
                for field in ('returns','holdings','summary'):
                    previous=old[name+'__'+field]
                    for i,j,cid in common:
                        if not np.array_equal(results[name][field][i],previous[j]):
                            raise AssertionError('C changed existing exact configuration '+directory+' '+cid+' '+name+' '+field)
        parity.append(dict(stage=directory,shared_ids=[cid for i,j,cid in common],exact=True))
    np.savez_compressed(OUT/'paths.npz',**{name+'__'+field:data[field] for name,data in results.items() for field in NUMERIC_FIELDS})
    dump(OUT/'path_metadata.json',dict(ids=[c['id'] for c in registry['candidates']],dates=dates,sha256=sha(OUT/'paths.npz')))
    dump(OUT/'execution_metadata.json',dict(registration_sha256=reg_hash,prior_parity=parity,scenarios={name:r['metadata'] for name,r in results.items()}))
    rows=summary_rows(registry['candidates'],results,dates,period_end=END)
    dump(OUT/'evaluation.json',dict(rows=rows,period=[START,END],registration_sha256=reg_hash,paths_sha256=sha(OUT/'paths.npz'),clean_oos=False))
    choice=select(rows,registry['candidates'],registry['controls'],period_end=END)
    if fingerprints()!=expected:raise ValueError('C source/input changed while selecting')
    choice.update(frozen_at=stamp(),adaptive_after_a_b=True,provenance=dict(registration_sha256=reg_hash,
        registry_sha256=sha(path),evaluation_sha256=sha(OUT/'evaluation.json'),paths_sha256=sha(OUT/'paths.npz'),
        path_metadata_sha256=sha(OUT/'path_metadata.json'),execution_metadata_sha256=sha(OUT/'execution_metadata.json'),fingerprints=expected))
    dump(OUT/'selection.json',choice);protect()
    print(json.dumps({k:choice[k] for k in ('primary','qualified_count','top_return','top_tail','least_drawdown')},indent=2))
    return choice


if __name__=='__main__':evaluate(workers=2)
