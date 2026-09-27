"""Explicitly adaptive B registry and evaluation; A evidence remains immutable."""
from copy import deepcopy
from itertools import product
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from .data import BASE,ROOT,START,END,load_inputs,stamp,sha,dump,protect
from .registry import canonical
from .native import Simulator
from .scan import run_candidates,summary_rows,SCENARIOS,NUMERIC_FIELDS
from .selection import select
from .consensus_features import build_consensus
from v10_deep.data import inputs
from v10_deep.schema import identifier

OUT=BASE/'results/consensus'


def verify_a():
    selection=json.loads((BASE/'results/main/selection.json').read_text());proof=selection['provenance']
    for name,key in (('registration.json','registration_sha256'),('evaluation.json','evaluation_sha256'),
                     ('paths.npz','paths_sha256'),('path_metadata.json','path_metadata_sha256'),
                     ('execution_metadata.json','execution_metadata_sha256')):
        if sha(BASE/'results/main'/name)!=proof[key]:raise ValueError('A evidence changed: '+name)
    if sha(BASE/'registered_candidates.json')!=proof['registry_sha256']:raise ValueError('A registry changed')
    for group in proof['fingerprints'].values():
        for name,h in group.items():
            if sha(BASE/name)!=h:raise ValueError('A pinned source/input changed: '+name)
    return json.loads((BASE/'registered_candidates.json').read_text()),selection


def generate(a,mapping):
    old={c['id']:c for c in a['candidates']};h=old[a['controls']['h']]['config'];records={}
    for cid in a['controls'].values():records[cid]=deepcopy(old[cid])
    for score,ma,risk,buffer,hold,flag in product(mapping.values(),('ma180','ma250'),
            (('fixed',.04),('volatility',1.35),('volatility',1.5),('volatility',1.65)),
            ('original','uniform2'),(0,2,3),(0,1)):
        c=dict(h,score=score,ma=ma,panic_mode=risk[0],panic=risk[1],min_hold=hold,crash_mask=5,locked_panic_exit=flag)
        if buffer=='uniform2':c.update(buffer=.02,global_buffer=.02,gold_buffer=.02)
        c=canonical(c);digest=identifier(c);cid='v12_'+digest[:20]
        records[cid]=dict(id=cid,hash=digest,kind='single',config=c,selectable=True,
            families=['consensus_'+next(k for k,v in mapping.items() if v==score)],
            complexity=sum(c[k]!=v for k,v in h.items())+int(score in (mapping['equal_score'],mapping['equal_rank'])),
            previously_seen=False,previous_ids=[])
    if len(records)!=293:raise AssertionError('B registry size changed')
    return dict(schema=1,candidates=[records[cid] for cid in sorted(records)],unique_count=len(records),
        controls=a['controls'],score_mapping=mapping,stage='adaptive_consensus',
        adaptive_after_a=True,known_history_including_2026=True,source_a_selection_sha256=sha(BASE/'results/main/selection.json'))


def run_b(records,arrays,meta,start=START,end=END,scenarios=SCENARIOS,workers=1):
    """Per-score eligibility cannot silently resurrect a sentinel-ranked ETF."""
    positions={c['id']:i for i,c in enumerate(records)}
    groups={}
    for record in records:
        key=record['config']['score'] if record['config']['score'].startswith('v12_b_') else 'original_controls'
        groups.setdefault(key,[]).append(record)
    merged=None;dates=None
    for score,subset in groups.items():
        view=arrays
        if score!='original_controls':
            lane=arrays['scores'][meta['score_names'].index(score)]
            view=dict(arrays,features=arrays['features'].copy())
            valid=meta['feature_names'].index('valid')
            view['features'][:,:,valid]*=np.isfinite(lane)&(lane>-1e90)
        result,observed=run_candidates(subset,view,meta,start=start,end=end,scenarios=scenarios,workers=workers)
        if merged is None:
            dates=observed;merged={name:{field:np.empty((len(records),)+data[field].shape[1:],dtype=data[field].dtype)
                          for field in NUMERIC_FIELDS} for name,data in result.items()}
            for data in merged.values():data['metadata']={}
        if observed!=dates:raise AssertionError('B grouped date axes differ')
        ids=[positions[c['id']] for c in subset]
        for name,data in result.items():
            for field in NUMERIC_FIELDS:merged[name][field][ids]=data[field]
            merged[name]['metadata'].update(data['metadata'])
    return merged,dates


def fingerprints():
    names=('consensus.py','consensus_features.py','diagnostic_features.py','data.py','registry.py',
           'native.py','native.cpp','schema.py','scan.py','selection.py','allocation.py',
           'PROTOCOL.md','CONSENSUS_PROTOCOL.md','tests/test_consensus.py','tests/test_consensus_features.py')
    return dict(sources={name:sha(BASE/name) for name in names},
        inputs={name:sha(BASE/name) for name in ('registered_candidates.json','registration.json','PROTOCOL.md',
        'CONSENSUS_PROTOCOL.md','inputs/manifest.json','inputs/features.json','inputs/features.npz.gz',
        'results/main/selection.json','results/consensus/registered_candidates.json','results/consensus/score_receipt.json')})


def evaluate(workers=1):
    if (OUT/'selection.json').exists():raise ValueError('B already frozen; refusing overwrite')
    a,sa=verify_a();arrays,meta,unused=load_inputs();histories,calendar,fear=inputs()
    extended,m,mapping=build_consensus(arrays,meta,histories)
    registry=generate(a,mapping);path=OUT/'registered_candidates.json'
    if path.exists() and json.loads(path.read_text())!=registry:raise ValueError('B registered definitions differ')
    if not path.exists():dump(path,registry)
    score_receipt=dict(mapping=mapping,metadata=m,source_sha256=sha(BASE/'consensus_features.py'),
        array_sha256={name:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for name,value in extended.items()},
        a_selection_sha256=sha(BASE/'results/main/selection.json'),protocol_sha256=sha(BASE/'CONSENSUS_PROTOCOL.md'))
    if (OUT/'score_receipt.json').exists():
        if json.loads((OUT/'score_receipt.json').read_text())!=score_receipt:raise ValueError('B feature receipt changed')
    else:dump(OUT/'score_receipt.json',score_receipt)
    expected=fingerprints()
    design=dict(fingerprints=expected,candidate_count=293,period=[START,END],scenarios=[list(s) for s in SCENARIOS],
        adaptive_after_a=True,a_primary_unchanged=sa['primary'],known_history_including_2026=True,clean_oos=False)
    if (OUT/'registration.json').exists():
        if json.loads((OUT/'registration.json').read_text())['design']!=design:raise ValueError('B source registration changed')
    else:dump(OUT/'registration.json',dict(registered_at=stamp(),design=design))
    reg_hash=sha(OUT/'registration.json')
    results,dates=run_b(registry['candidates'],extended,m,workers=workers)
    if fingerprints()!=expected or sha(OUT/'registration.json')!=reg_hash:raise ValueError('B inputs/source changed during evaluation')
    # The same controls must preserve all earlier A path bytes.
    oldmeta=json.loads((BASE/'results/main/path_metadata.json').read_text())
    if oldmeta['dates']!=dates:raise ValueError('A/B date axes differ')
    positions={c['id']:i for i,c in enumerate(registry['candidates'])}
    with np.load(BASE/'results/main/paths.npz',allow_pickle=False) as old:
        for cid in registry['controls'].values():
            i,j=positions[cid],oldmeta['ids'].index(cid)
            for name,lag,fee in SCENARIOS:
                for field in ('returns','holdings','summary'):
                    if not np.array_equal(results[name][field][i],old[name+'__'+field][j]):
                        raise AssertionError('B changed a frozen A control: '+cid+' '+name+' '+field)
    np.savez_compressed(OUT/'paths.npz',**{name+'__'+field:data[field] for name,data in results.items() for field in NUMERIC_FIELDS})
    dump(OUT/'path_metadata.json',dict(ids=[c['id'] for c in registry['candidates']],dates=dates,sha256=sha(OUT/'paths.npz')))
    dump(OUT/'execution_metadata.json',dict(registration_sha256=reg_hash,scenarios={name:r['metadata'] for name,r in results.items()}))
    rows=summary_rows(registry['candidates'],results,dates,period_end=END)
    dump(OUT/'evaluation.json',dict(rows=rows,period=[START,END],registration_sha256=reg_hash,paths_sha256=sha(OUT/'paths.npz'),clean_oos=False))
    choice=select(rows,registry['candidates'],registry['controls'],period_end=END)
    if fingerprints()!=expected:raise ValueError('B source/input changed while selecting')
    choice.update(frozen_at=stamp(),adaptive_after_a=True,provenance=dict(registration_sha256=reg_hash,
        registry_sha256=sha(path),evaluation_sha256=sha(OUT/'evaluation.json'),paths_sha256=sha(OUT/'paths.npz'),
        path_metadata_sha256=sha(OUT/'path_metadata.json'),execution_metadata_sha256=sha(OUT/'execution_metadata.json'),fingerprints=expected))
    dump(OUT/'selection.json',choice);protect()
    print(json.dumps({k:choice[k] for k in ('primary','qualified_count','top_return','top_tail','least_drawdown')},indent=2))
    return choice


if __name__=='__main__':evaluate(workers=2)
