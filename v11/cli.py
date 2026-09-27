"""Read-only reproducible entry for frozen V11 research candidates."""
import argparse
from copy import deepcopy
import json

from .data import BASE, ROOT, START, DEV_END, CONFIRM_END, END, sha
from .features import build, risk_view
from .confirmation import _rows
from v10_deep.native import Simulator
from v10_deep.schema import identifier, semantic


def load_profile(variant='b'):
    profiles=json.loads((BASE/'profiles.json').read_text())
    if variant not in profiles['variants']:raise ValueError('Unknown frozen variant')
    for section,root in (('source_sha256',BASE),('dependency_sha256',ROOT),('artifact_sha256',BASE)):
        for name,expected in profiles[section].items():
            if sha(root/name)!=expected:raise ValueError('Frozen V11 evidence changed: '+name)
    result=deepcopy(profiles['variants'][variant])
    digest=identifier(result['config'])
    if digest!=result['config']['hash'] or result['config']['id']!='v11_'+digest[:20]:
        raise ValueError('Frozen candidate definition changed')
    # A self-consistent hash alone does not identify the frozen selection: a
    # different valid registered config must not silently be relabeled A/B/H.
    stage='combinations' if variant=='b' else 'development'
    selection_path='results/'+stage+'/selection.json'
    registry_path='results/combinations/registered_candidates.json' if variant=='b' else 'registered_candidates.json'
    if selection_path not in profiles['artifact_sha256'] or registry_path not in profiles['artifact_sha256']:
        raise ValueError('Frozen role selection/registry receipts are missing')
    selection=json.loads((BASE/selection_path).read_text())
    registry=json.loads((BASE/registry_path).read_text())
    expected=registry['controls']['h'] if variant=='h' else selection['primary']
    original=next((c for c in registry['candidates'] if c['id']==expected),None)
    if original is None or result['config']['id']!=expected or semantic(original)!=semantic(result['config']):
        raise ValueError('Frozen variant differs from its selected role')
    report_path='results/report_2026/evaluation.json'
    if report_path not in profiles['artifact_sha256']:
        raise ValueError('Frozen final-report receipt is missing')
    report=json.loads((BASE/report_path).read_text());role={'a':'v11_a','b':'v11_b','h':'v10_h'}[variant]
    verdict=report['confirmation_verdicts'].get(role)
    status='existing_control' if variant=='h' else ('research_only_confirmation_passed_not_clean_oos'
             if verdict['passed'] else 'research_only_failed_confirmation')
    if report['roles'][role]!=expected or result.get('confirmation_verdict')!=verdict or result['status']!=status:
        raise ValueError('Frozen research status differs from final evidence')
    return result,profiles


def evaluate(variant='b',end=END,lag=0,fee_bp=1):
    if lag not in (0,1) or fee_bp not in (1,5,11,21):
        raise ValueError('Use a registered timing/cost scenario')
    profile,profiles=load_profile(variant)
    arrays,meta=build(profiles['score_specs'])
    if meta['fingerprints']!=profiles['feature_fingerprints']:
        raise ValueError('Frozen feature inputs differ')
    if meta.get('cache_array_sha256')!=profiles['feature_array_sha256']:
        raise ValueError('Frozen feature array bytes differ; do not accept a new self-consistent cache')
    if end<START or end>END or end not in meta['dates']:
        raise ValueError('End must be an observed date within the frozen historical interval')
    config=dict(profile['config'],lag=lag)
    view=risk_view(arrays,meta,config['risk_context'],score=config['score'])
    result=Simulator(view,meta).run([config],start=START,end=end,fee=fee_bp/10000.,workers=1)
    return profile,result,meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('backtest','signal'))
    parser.add_argument('variant',nargs='?',choices=('a','b','h'),default='b')
    parser.add_argument('--end',default=END)
    parser.add_argument('--lag',type=int,choices=(0,1),default=0)
    parser.add_argument('--fee-bp',type=int,choices=(1,5,11,21),default=1)
    args=parser.parse_args()
    profile,result,meta=evaluate(args.variant,args.end,args.lag,args.fee_bp)
    out=dict(version=profile['name'],candidate_id=profile['config']['id'],
             data_date=result['dates'][-1],status=profile['status'],
             confirmation_verdict=profile.get('confirmation_verdict'),
             deployed=False,order_submission=False,clean_oos=False,
             signal_lag_days=args.lag,single_side_fee_bp=args.fee_bp,
             assumptions='Corrected total-return, ideal closing-price execution; immediate dividend reinvestment; first session free; 244 sessions/year. No actual 14:50 fills, lots or dividend payment delays.')
    if args.command=='backtest':
        periods=[('development',START,DEV_END),('confirmation','2022-01-01',CONFIRM_END),
                 ('report_only_2026','2026-01-01',END),('full',START,args.end)]
        out['metrics']=_rows([profile['config']],{'execution':result},result['dates'],periods)[0]['scenarios']['execution']
        out['switches']=int(result['summary'][0,3])
    else:
        held=int(result['holdings'][0,-1])
        out['historical_close_target']=meta['assets'][held] if held>=0 else None
        out['warning']='Frozen historical target only; not a live quote, order, or recommendation.'
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
