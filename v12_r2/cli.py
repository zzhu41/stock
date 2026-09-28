"""Read-only frozen R2 path inspection; no live quotes, accounts or orders."""
import argparse
from bisect import bisect_right
import json
import numpy as np
from .data import BASE,END,verify_inputs
from .research import load_results
from v10_deep.scan import metrics


def evaluate(role='primary',end=END,lag=0,fee_bp=1):
    verify_inputs();registry,evaluation,selection=load_results()
    if role=='primary':cid=selection['primary']
    elif role=='balanced':cid=selection['balanced_reference']
    elif role in registry['controls']:cid=registry['controls'][role]
    else:raise ValueError('Unknown frozen role')
    if cid is None:raise ValueError('No candidate qualified for this frozen tier; no automatic substitute')
    if lag not in (0,1) or fee_bp not in (1,11):raise ValueError('Only four registered execution scenarios')
    meta=json.loads((BASE/'results/path_metadata.json').read_text())
    if end not in meta['dates']:raise ValueError('End must be a frozen observed date')
    stop=bisect_right(meta['dates'],end);scenario=('lag1' if lag else 'close')+'_%dbp'%fee_bp
    index=meta['ids'].index(cid)
    with np.load(BASE/'results/paths.npz',allow_pickle=False) as z:
        returns=z[scenario+'__returns'][index,:stop];holdings=z[scenario+'__holdings'][index,:stop]
    assets=json.loads((BASE/'results/feature_metadata.json').read_text())['assets']
    config=next(r for r in registry['records'] if r['id']==cid)
    out=dict(role=role,candidate_id=cid,feature_spec=config['feature_spec'],config=config['config'],
        period=[meta['dates'][0],end],scenario=scenario,full=metrics(returns),
        yearly={year:metrics(returns[np.asarray([d[:4]==year for d in meta['dates'][:stop]])]) for year in sorted({d[:4] for d in meta['dates'][:stop]})},
        historical_target=assets[int(holdings[-1])] if holdings[-1]>=0 else None,
        status='known-history research; strict and balanced tiers are distinct',clean_oos=False,deployed=False,order_submission=False,
        execution='TR ideal close, first session free, 244 observations/year; lag1 is next close, not next open or 14:50')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('backtest','signal'))
    p.add_argument('role',choices=('primary','balanced','c4','c','h','simple'),nargs='?',default='primary')
    p.add_argument('--end',default=END);p.add_argument('--lag',type=int,choices=(0,1),default=0)
    p.add_argument('--fee-bp',type=int,choices=(1,11),default=1);a=p.parse_args()
    result=evaluate(a.role,a.end,a.lag,a.fee_bp)
    if a.command=='signal':result={k:v for k,v in result.items() if k not in ('full','yearly')}
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
