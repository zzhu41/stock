"""Read-only backtest/historical signal for frozen, unqualified V12 references."""
import argparse
from bisect import bisect_right
from copy import deepcopy
import json
import numpy as np

from .data import BASE, ROOT, START, END, sha, load_inputs
from .exante_features import build_exante, view_for_config
from .reference import run_reference
from v10_deep.data import inputs
from v10_deep.schema import identifier
from v10_deep.scan import metrics


def load_profile(name):
    receipt = json.loads((BASE/'runtime_manifest.json').read_text())
    for group in ('source_sha256','artifact_sha256'):
        for relative,expected in receipt[group].items():
            path = ROOT/relative
            if not path.is_file() or sha(path) != expected:
                raise ValueError('Frozen V12 runtime dependency changed: '+relative)
    document = json.loads((BASE/'profiles.json').read_text())
    if document['qualified_primary'] is not None or name not in document['variants']:
        raise ValueError('Unknown exploratory variant or unexpected qualification status')
    profile = deepcopy(document['variants'][name])
    if identifier(profile['config']) != profile['hash'] or 'v12_'+profile['hash'][:20] != profile['id']:
        raise ValueError('Frozen V12 profile semantic hash changed')
    return profile


def evaluate(name='balanced_c',end=END,lag=0,fee=.0001):
    profile = load_profile(name)
    arrays,meta,unused = load_inputs()
    if end < START or end > END or end not in meta['dates']:
        raise ValueError('End must be an observed date within the frozen research period')
    if lag not in (0,1) or fee not in (.0001,.0005,.0011,.0021):
        raise ValueError('Only registered lag 0/1 and per-leg fee 1/5/11/21bp are supported')
    stop = bisect_right(meta['dates'],end)
    arrays = {key:np.ascontiguousarray(value[:,:stop] if key in ('scores','orders') else value[:stop])
              for key,value in arrays.items()}
    meta = deepcopy(meta);meta['dates']=meta['dates'][:stop];meta['shape'][0]=stop
    histories,unused_calendar,unused_fear = inputs()
    histories = {code:[row for row in rows if row[0] <= end] for code,rows in histories.items()}
    expanded,m,unused_mapping = build_exante(arrays,meta,histories)
    config = dict(profile['config'],lag=lag)
    view = view_for_config(expanded,m,config)
    # Python reference avoids even compiling/writing a native cache.
    result = run_reference(view,m,config,START,end,fee)
    return profile,result,m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('backtest','signal'))
    parser.add_argument('variant',choices=('balanced_a','balanced_b','balanced_c','h','v9','v91','v92','simple'),
                        nargs='?',default='balanced_c')
    parser.add_argument('--end',default=END)
    parser.add_argument('--lag',type=int,choices=(0,1),default=0)
    parser.add_argument('--fee-bp',type=int,choices=(1,5,11,21),default=1)
    args = parser.parse_args()
    profile,result,meta = evaluate(args.variant,args.end,args.lag,args.fee_bp/10000.)
    out = dict(variant=args.variant,candidate_id=profile['id'],status=profile['status'],
        qualified_primary=None,data_date=result['dates'][-1],deployed=False,order_submission=False,clean_oos=False,
        execution=dict(lag_closes=args.lag,fee_per_leg_bp=args.fee_bp,first_session_free=True,
                       annualization_sessions=244,dividend_reinvestment='Immediate free reinvestment at ex-date close'),
        limitations='Known-history research including 2026; no observed 14:50/next-open prices, lots or payment delays; no future performance guarantee')
    if args.command == 'backtest':
        out['full'] = metrics(result['returns'])
        dates = np.asarray(result['dates'])
        out['yearly'] = {year:metrics(result['returns'][np.asarray([d[:4] == year for d in dates])])
                         for year in sorted({d[:4] for d in dates})}
        out['yearly_return_label'] = 'total_return is each calendar period cumulative return; last year may be incomplete'
        out['switches'] = int(result['summary'][3])
    else:
        held = int(result['holdings'][-1])
        out['historical_close_target'] = meta['assets'][held] if held >= 0 else None
        out['warning'] = 'Frozen historical model target, not a current intraday order instruction'
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
