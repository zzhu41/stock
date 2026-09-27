"""Preregistered simple long-trend controls, never candidates for posthoc rescue."""
import json
import time
import numpy as np
from .data import BASE,START,DEV_END,load_frozen,sha,dump,protect
from .portfolio_controls import registry,prepare,run_control
from .scan import SCENARIOS,DEV_BLOCKS,stamp
from v10_deep.scan import metrics
from bisect import bisect_left,bisect_right


def main():
    out=BASE/'results/long_trend_controls'
    if (out/'evaluation.json').exists():raise RuntimeError('Controls already evaluated; do not overwrite')
    controls=registry();bundle=load_frozen();prepared=prepare(bundle['histories'],bundle['meta']['dates'])
    sources={n:sha(BASE/n) for n in ('portfolio_controls.py','long_trend_check.py','PROTOCOL.md','data.py')}
    dump(out/'registration.json',dict(registered_at=stamp(),controls=controls,source_sha256=sources,
        scenarios=SCENARIOS,period=[START,DEV_END],scope='Supplemental controls, not eligible to replace development-selected primary',clean_oos=False))
    rows=[];packed={};dates=None
    for config in controls['candidates']:
        row=dict(id=config['id'],config=config,scenarios={})
        for name,lag,fee in SCENARIOS:
            result=run_control(config,prepared,start=START,end=DEV_END,fee=fee,lag=lag)
            returns=np.asarray(result['returns']);ds=[r[0] for r in result['daily']]
            if dates is None:dates=ds
            if dates!=ds:raise ValueError('Control calendar differs')
            blocks={}
            for label,a,b in DEV_BLOCKS:
                lo,hi=bisect_left(dates,a),bisect_right(dates,b)
                blocks[label]=metrics(returns[lo:hi])
            row['scenarios'][name]=dict(full=metrics(returns),blocks=blocks,switches=result['switches'],
                total_turnover=result['diagnostics']['total_turnover'],total_model_fee=result['diagnostics']['total_model_fee'])
            packed[config['id']+'__'+name]=returns
        rows.append(row)
        print(config['id'],'dev CAGR %.3f%%'%(row['scenarios']['close_1bp']['full']['cagr']*100),flush=True)
    np.savez_compressed(out/'paths.npz',**packed)
    for name,expected in sources.items():
        if sha(BASE/name)!=expected:raise ValueError('Control definition changed during run')
    dump(out/'evaluation.json',dict(period=[dates[0],dates[-1]],rows=rows,
        registration_sha256=sha(out/'registration.json'),paths_sha256=sha(out/'paths.npz'),source_sha256=sources))
    dump(out/'path_metadata.json',dict(dates=dates,keys=list(packed)))
    protect()

if __name__=='__main__':main()
