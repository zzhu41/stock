"""Frozen-winner ablations and local perturbations; never reselect a candidate."""
import argparse
from copy import deepcopy
import json
import numpy as np

from .data import BASE, ASSET_ORDER, CASH, inputs, dump, sha
from .features import build, F, _regression, _rolling_mean
from .native import Simulator
from .scan import summarize, stamp
from .schema import baseline, identifier, semantic
from .registry import canonical


def extra_score(histories,calendar,arrays,window,smoothing):
    """Same trailing-only WLS construction as the registered feature builder."""
    out=np.full((len(calendar),len(ASSET_ORDER)),-1e100);lookup={d:i for i,d in enumerate(calendar)}
    for ai,code in enumerate(ASSET_ORDER):
        if code==CASH:continue
        rows=histories[code];prices=np.asarray([r[2] for r in rows])
        ret=np.r_[0.,prices[1:]/prices[:-1]-1];avg=_rolling_mean(ret,20)
        vol=np.sqrt(np.maximum(0.,_rolling_mean(ret*ret,20)-avg*avg))
        mean,slope,_=_regression(prices,window)
        trend=np.full(len(prices),np.nan);trend[window-1:]=slope/mean*250
        score=np.divide(trend,vol,out=np.zeros(len(prices)),where=vol>0)
        own=np.asarray([j for j,row in enumerate(rows) if row[0] in lookup],dtype=int)
        values=score[own]
        if smoothing>1:values=_rolling_mean(values,smoothing)
        for k,j in enumerate(own):
            if j>=269 and np.isfinite(values[k]):out[lookup[rows[j][0]],ai]=values[k]
    return out


def run(stage='refinements'):
    path=BASE/'results'/stage
    chosen=json.loads((path/'selection.json').read_text());registry=json.loads((path/'registered_candidates.json').read_text())
    main=next(c for c in registry['candidates'] if c['id']==chosen['primary']);base=canonical(baseline())
    scenarios=[]
    def add(label,changes):
        c=deepcopy(semantic(main));c.update(changes);c=canonical(c)
        c['hash']=identifier(c);c['id']='diagnostic_'+c['hash'][:20]
        scenarios.append(dict(label=label,config=c))
    add('frozen_center',{})
    add('reference_v92',base)
    add('restore_WLS25',dict(score='wls25_v20'))
    add('restore_MA250',dict(ma='ma250'))
    add('restore_fixed_panic4',dict(panic_mode='fixed',panic=.04))
    add('drop_QVIX',dict(crash_mask=main['crash_mask']&~2))
    add('deep_only',dict(crash_mask=1))
    add('no_crash_channels',dict(crash_mask=0))
    for ma in ('ma150','ma200'):add('neighbor_'+ma,dict(ma=ma))
    if main['panic_mode']=='volatility':
        for factor in (.9,1.1):add('panic_multiple_x'+str(factor),dict(panic=main['panic']*factor))
    # Diagnostic windows are fixed +/-10% around the final WLS20, not a new search grid.
    generated=[]
    if main['score']=='wls20_smooth3':
        for window,smoothing in ((20,1),(20,2),(20,4),(18,3),(22,3)):
            name='diagnostic_wls%d_smooth%d'%(window,smoothing)
            generated.append((name,window,smoothing));add(name,dict(score=name))
    registration=dict(registered_at=stamp(),selected_id=main['id'],selection_sha256=sha(path/'selection.json'),
                      script_sha256=sha(__file__),scenarios=scenarios,generated_scores=generated,
                      reselect=False,clean_oos=False,scope='Postselection ablations/local sensitivity only')
    out=path/'neighborhood';out.mkdir(exist_ok=True)
    if (out/'registration.json').exists():raise RuntimeError('Neighborhood diagnostics already registered')
    dump(out/'registration.json',registration)
    arrays,meta=build();histories,calendar,_=inputs()
    # Verify independently rebuilt center first, including first eligibility and suspensions.
    if main['score']=='wls20_smooth3':
        reproduced=extra_score(histories,calendar,arrays,20,3)
        original=arrays['scores'][meta['score_names'].index('wls20_smooth3')]
        valid=(arrays['features'][:,:,F['valid']]>0)&np.isfinite(arrays['features'][:,:,F['close']])
        if not np.allclose(reproduced[valid],original[valid],atol=1e-9,rtol=1e-10):
            raise AssertionError('Dynamic-score builder does not reproduce the frozen center')
    meta=deepcopy(meta)
    scores=[arrays['scores']]
    for name,w,smoothing in generated:
        values=extra_score(histories,calendar,arrays,w,smoothing);scores.append(values[None,:,:]);meta['score_names'].append(name)
    arrays=dict(arrays);arrays['scores']=np.ascontiguousarray(np.concatenate(scores,axis=0))
    arrays['orders']=np.ascontiguousarray(np.argsort(-arrays['scores'],axis=2,kind='stable').astype(np.int32))
    sim=Simulator(arrays,meta);result=sim.run([x['config'] for x in scenarios],workers=1)
    rows=[]
    for i,x in enumerate(scenarios):
        rows.append(dict(label=x['label'],config=x['config'],metrics=summarize(result['returns'][i],result['dates']),
                         switches=int(result['summary'][i,3])))
    center=next(r for r in rows if r['label']=='frozen_center')
    frozen=json.loads((path/'evaluation.json').read_text())
    expected=next(r for r in frozen['rows'] if r['id']==main['id'])
    if abs(center['metrics']['full']['cagr']-expected['metrics']['full']['cagr'])>1e-12:raise AssertionError('Center changed')
    dump(out/'evaluation.json',dict(registration_sha256=sha(out/'registration.json'),rows=rows,reselect=False))
    for r in rows:print(r['label'],r['metrics']['full']['cagr'],r['metrics']['full']['max_dd'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',default='refinements');a=p.parse_args();run(a.stage)
