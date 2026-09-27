"""Append-only independent replay of the one gate-passing diagnostic neighbor.

This does not change eligibility, selection, the frozen sensitivity evaluator,
or any profile. Only two already-registered scenarios are replayed in Python.
"""
import json
from pathlib import Path

import numpy as np

from .data import BASE,ROOT,START,END,sha,dump,stamp,load_inputs
from .reference import run_reference
from .sensitivity import prepare_view
from v10_deep.data import inputs

OUT=BASE/'results/sensitivity/independent_smooth4'
CANDIDATE='v12d_5e020674285fc2d9db44'
SCENARIOS=(('close_1bp',0,.0001),('lag1_11bp',1,.0011))


def run():
    if (OUT/'audit.json').exists():raise ValueError('Independent diagnostic audit is already frozen')
    folder=BASE/'results/sensitivity'
    evaluation=json.loads((folder/'evaluation.json').read_text())
    registry=json.loads((folder/'registered_candidates.json').read_text())
    registration=json.loads((folder/'registration.json').read_text())
    metadata=json.loads((folder/'path_metadata.json').read_text())
    if (sha(folder/'registered_candidates.json')!=registration['registry_sha256']
            or sha(folder/'registration.json')!=evaluation['registration_sha256']
            or sha(folder/'paths.npz')!=evaluation['paths_sha256']):
        raise ValueError('Frozen diagnostic evidence changed')
    for name,digest in evaluation['fingerprints']['sources'].items():
        if sha(BASE/name)!=digest:raise ValueError('Frozen sensitivity source changed: '+name)
    for name,digest in evaluation['fingerprints']['frozen_dependencies'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen dependency changed: '+name)
    for name,digest in evaluation['fingerprints']['inputs'].items():
        if sha(BASE/name)!=digest:raise ValueError('Frozen diagnostic input changed: '+name)
    record=next(r for r in registry['records'] if r['id']==CANDIDATE)
    if (record['feature_spec']!=dict(window=25,smooth=4,ma_window=180)
            or record['selectable'] or record['config'].get('risk_context')!='prior_max20_60'):
        raise ValueError('Unexpected diagnostic identity or selection status')
    definition=dict(candidate_id=CANDIDATE,feature_spec=record['feature_spec'],source_config=record['config'],
        scenarios=SCENARIOS,period=[START,END],no_reselection=True,official_primary=None,
        source_sha256={name:sha(BASE/name) for name in ('audit_sensitivity.py','reference.py','sensitivity.py',
            'diagnostic_features.py','exante_features.py')},
        frozen_evidence_sha256={name:sha(folder/name) for name in
            ('evaluation.json','registration.json','registered_candidates.json','paths.npz','path_metadata.json')})
    path=OUT/'registration.json'
    if path.exists():
        if json.loads(path.read_text())['definition']!=json.loads(json.dumps(definition)):
            raise ValueError('Independent audit definition changed')
    else:dump(path,dict(registered_at=stamp(),definition=definition))
    arrays,meta,unused=load_inputs();histories,unused,unused=inputs()
    view,m,effective=prepare_view(arrays,meta,histories,record)
    index=metadata['ids'].index(CANDIDATE)
    proofs,paths={},{}
    with np.load(folder/'paths.npz',allow_pickle=False) as saved:
        for name,lag,fee in SCENARIOS:
            config=dict(effective,lag=lag)
            result=run_reference(view,m,config,start=START,end=END,fee=fee)
            if result['dates']!=metadata['dates']:raise AssertionError('Reference dates changed')
            fields={}
            for field in ('returns','holdings','summary'):
                expected=saved[name+'__'+field][index]
                actual=result[field]
                equal=np.array_equal(actual,expected)
                fields[field]=dict(exact=bool(equal),max_abs_difference=float(np.max(np.abs(actual-expected))))
                if not equal:raise AssertionError('Independent diagnostic path differs: '+name+' '+field+' '+str(fields[field]))
                paths[name+'__'+field]=actual
            paths[name+'__nav']=np.cumprod(1+result['returns'])
            proofs[name]=dict(fields=fields,observations=len(result['dates']),effective_config=config,
                              crash_fills=sum(r['filled'] and r['crash_requested'] for r in result['trace']))
            dump(OUT/(name+'_trace.json'),result['trace'])
    np.savez_compressed(OUT/'paths.npz',**paths)
    dump(OUT/'path_metadata.json',dict(candidate_id=CANDIDATE,dates=metadata['dates'],scenarios=[s[0] for s in SCENARIOS],
                                      sha256=sha(OUT/'paths.npz')))
    receipt=dict(completed_at=stamp(),passed=True,candidate_id=CANDIDATE,feature_spec=record['feature_spec'],
        results=proofs,no_reselection=True,official_primary=None,registration_sha256=sha(path),
        paths_sha256=sha(OUT/'paths.npz'),frozen_evaluation_sha256=sha(folder/'evaluation.json'),
        trace_sha256={s[0]:sha(OUT/(s[0]+'_trace.json')) for s in SCENARIOS})
    dump(OUT/'audit.json',receipt)
    print(json.dumps(dict(passed=True,candidate_id=CANDIDATE,scenarios=[s[0] for s in SCENARIOS],
                         paths=str(OUT/'paths.npz')),ensure_ascii=False))
    return receipt


if __name__=='__main__':run()
