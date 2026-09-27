"""Read-only entry for the frozen v10 research candidate and comparisons."""
import argparse
from copy import deepcopy
import json
from .data import BASE, ROOT, START, END, sha, inputs
from .schema import identifier
from .features import build
from .native import Simulator
from .scan import summarize


def load_profile(variant):
    profiles=json.loads((BASE/'profiles.json').read_text())
    if variant not in profiles['variants']:raise ValueError('Unknown frozen variant: '+variant)
    for name,expected in profiles['source_sha256'].items():
        if sha(BASE/name)!=expected:raise ValueError('Frozen research source changed: '+name)
    for name,expected in profiles['dependency_sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen dependency changed: '+name)
    for name,expected in profiles['artifact_sha256'].items():
        if sha(BASE/name)!=expected:raise ValueError('Frozen selection/artifact changed: '+name)
    for name,path in (('corrected',ROOT/'v10_h_close/corrected_manifest.json'),('qvix',ROOT/'v10_h_close/qvix_manifest.json')):
        if sha(path)!=profiles['input_manifests'][name]:raise ValueError('Frozen input manifest changed: '+name)
    profile=deepcopy(profiles['variants'][variant])
    if identifier(profile['config'])!=profile['config']['hash']:raise ValueError('Frozen configuration changed')
    # Verify the actual dated CSV bytes even when using an existing feature cache.
    inputs()
    return profile


def evaluate(variant='growth',end=END):
    profile=load_profile(variant)
    profiles=json.loads((BASE/'profiles.json').read_text())
    expected=(('cache/features.npz','feature_cache_sha256'),('cache/features.json','feature_metadata_sha256'))
    for name,key in expected:
        if (BASE/name).exists() and sha(BASE/name)!=profiles[key]:raise ValueError('Frozen feature cache or metadata differs: '+name)
    arrays,meta=build()
    for name,key in expected:
        if sha(BASE/name)!=profiles[key]:raise ValueError('Rebuilt feature cache or metadata differs: '+name)
    if end<START or end>END or end not in meta['dates']:raise ValueError('End must be an observed date within the frozen research period')
    result=Simulator(arrays,meta).run([profile['config']],end=end,workers=1)
    return profile,result,meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('backtest','signal'));p.add_argument('variant',choices=('growth','simple','v92'),nargs='?',default='growth')
    p.add_argument('--end',default=END);args=p.parse_args()
    profile,result,meta=evaluate(args.variant,args.end)
    out=dict(version=profile['name'],candidate_id=profile['config']['id'],selected_by=profile['selected_by'],
             data_date=result['dates'][-1],not_deployed=True,order_submission=False,clean_oos=False,
             clock='Ideal same-close, corrected total-return indices, first session free; switch NAV times 0.9998',
             assumptions='Immediate free dividend reinvestment on ex-date; payment delays, lots and real 14:50 prices not modeled',
             research_status=profile.get('research_status','Research only; generalization remains unproven'))
    if args.command=='backtest':
        out['metrics']=summarize(result['returns'][0],result['dates'])
        out['switches']=int(result['summary'][0,3])
    else:
        held=int(result['holdings'][0,-1]);out['historical_close_target']=meta['assets'][held] if held>=0 else None
        out['warning']='Historical model target, not a current intraday order instruction'
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
