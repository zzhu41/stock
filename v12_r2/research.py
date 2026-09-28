"""Finite preregistered R2 grid, deterministic gates, full paths and audits."""
import argparse
from bisect import bisect_left,bisect_right
from copy import deepcopy
from datetime import datetime,timezone
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import BASE,ROOT,START,END,SCENARIOS,BLOCKS,C4_ID,read,dump,sha,verify_inputs,load_base,load_npz,controls
from .features import build,view,effective_config,lane_name
from v12.native import Simulator
from v12.reference import run_reference
from v10_deep.registry import canonical as legacy_canonical
from v10_deep.schema import semantic
from v10_deep.scan import metrics

GRID=dict(window=(20,25,30),smooth=(3,4,5),panic=(1.1,1.25,1.4,1.5),min_hold=(0,2,3),
          buffer=('original','uniform2'),risk_context=('current20','prior20','prior60','prior_max20_60'))
SOURCE_NAMES=('data.py','features.py','diagnose.py','research.py','cli.py','PROTOCOL.md','tests/test_research.py')


def stamp():return datetime.now(timezone.utc).isoformat()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def normalized(config,spec):
    c=legacy_canonical(semantic(config));c.update(score=lane_name(spec['window'],spec['smooth']),ma='ma%d'%spec['ma_window'],lag=0)
    if c.get('risk_context','current20')=='current20':c.pop('risk_context',None)
    if c['min_hold']==1:c['min_hold']=0
    return c


def identity(config,spec):return digest(dict(config=normalized(config,spec),feature_spec=spec))


def prior_ids():
    result={};known={'wls25_v20':(25,1),'wls20_smooth3':(20,3),'v12_b_wls25_smooth3':(25,3)}
    for path in ('v12/registered_candidates.json','v12/results/consensus/registered_candidates.json',
                 'v12/results/exante/registered_candidates.json','v12/results/sensitivity/registered_candidates.json'):
        registry=read(path)
        for row in registry.get('candidates',registry.get('records',[])):
            if row['kind']!='single':continue
            c=row['config']
            if 'feature_spec' in row:spec=row['feature_spec']
            elif c['score'] in known:
                w,s=known[c['score']];spec=dict(window=w,smooth=s,ma_window=int(c['ma'][2:]))
            else:continue
            result.setdefault(identity(c,spec),set()).add(row['id'])
    return result


def generate():
    anchor=controls();previous=prior_ids();records={};control_ids={}
    def add(config,spec,role=None):
        c=normalized(config,spec);h=identity(c,spec);cid='v12r2_'+h[:20]
        if cid not in records:
            records[cid]=dict(id=cid,hash=h,config=c,feature_spec=deepcopy(spec),kind='single',selectable=True,
                previous_v12_ids=sorted(previous.get(h,())),roles=[])
        if role:
            records[cid]['selectable']=False;records[cid]['roles'].append(role);control_ids[role]=cid
        return cid
    for role,record in anchor.items():add(record['config'],record['feature_spec'],role)
    for w,s,panic,hold,buffer,context in product(*(GRID[key] for key in ('window','smooth','panic','min_hold','buffer','risk_context'))):
        config=deepcopy(anchor['c4']['config'])
        config.update(ma='ma180',panic_mode='volatility',panic=panic,min_hold=hold,crash_mask=5,locked_panic_exit=0,risk_context=context)
        if buffer=='uniform2':config.update(buffer=.02,global_buffer=.02,gold_buffer=.02)
        add(config,dict(window=w,smooth=s,ma_window=180))
    center=records[control_ids['c4']]
    for record in records.values():
        record['complexity']=sum(record['config'].get(k)!=v for k,v in center['config'].items() if k not in ('score','ma'))
        record['complexity']+=sum(record['feature_spec'][k]!=v for k,v in center['feature_spec'].items())
    if len(records)!=866 or sum(r['selectable'] for r in records.values())!=862:raise AssertionError('Registered grid count changed')
    return dict(grid=GRID,raw_grid_count=864,unique_count=866,selectable_count=862,controls=control_ids,
                records=sorted(records.values(),key=lambda r:r['id']),period=[START,END],known_history=True,clean_oos=False)


def old_baselines():
    registry=read('v12/results/sensitivity/registered_candidates.json');old=read('v12/results/sensitivity/evaluation.json')
    by_source={source:r['id'] for r in registry['records'] for source in r['source_ids']}
    rows={r['id']:r for r in old['rows']}
    return {role:deepcopy(rows[record['source_id'] if role=='c4' else by_source[record['source_id']]])
            for role,record in controls().items()}


def source_hashes():return {name:sha(BASE/name) for name in SOURCE_NAMES}


def register():
    verify_inputs();registry=generate();baseline=old_baselines()
    definition=dict(source_sha256=source_hashes(),frozen_input_receipt_sha256=sha(BASE/'frozen_inputs.json'),
        diagnosis_sha256=sha(BASE/'diagnosis/2021.json'),grid_count=866,scenarios=SCENARIOS,
        statistical_parameters=dict(draws=2000,block_lengths=[20,60],seed=20260928),
        no_new_year_specific_rules=True,no_post_failure_expansion=True,clean_oos=False)
    path=BASE/'registration.json'
    if path.exists():
        receipt=json.loads(path.read_text())
        if (receipt['definition']!=json.loads(json.dumps(definition)) or sha(BASE/'registered_candidates.json')!=receipt['registry_sha256']
                or sha(BASE/'baseline_metrics.json')!=receipt['baseline_sha256']):
            raise ValueError('Registered source/inputs changed')
        return registry,baseline,receipt
    dump(BASE/'registered_candidates.json',registry);dump(BASE/'baseline_metrics.json',baseline)
    receipt=dict(registered_at=stamp(),definition=definition,registry_sha256=sha(BASE/'registered_candidates.json'),
                 baseline_sha256=sha(BASE/'baseline_metrics.json'),protocol_sha256=sha(BASE/'PROTOCOL.md'))
    dump(path,receipt);return registry,baseline,receipt


def summarize(results,dates,records):
    rows=[]
    for i,record in enumerate(records):
        row=dict(id=record['id'],scenarios={})
        for scenario,unused,unused in SCENARIOS:
            r=results[scenario]['returns'][i]
            def part(begin,end):
                lo,hi=bisect_left(dates,begin),bisect_right(dates,end)
                return metrics(r[lo:hi])
            row['scenarios'][scenario]=dict(full=metrics(r),yearly={y:part(y+'-01-01',y+'-12-31') for y in sorted({d[:4] for d in dates})},
                blocks={name:part(start,end) for name,start,end in BLOCKS},
                switches=int(results[scenario]['summary'][i,3]),crashes=int(results[scenario]['summary'][i,7]))
        rows.append(row)
    return rows


def select(rows,registry,baselines):
    records={r['id']:r for r in registry['records']};b=baselines['c4']['scenarios']
    target2021=dict(strict=baselines['h']['scenarios']['close_1bp']['yearly']['2021']['total_return'],
                    balanced=baselines['c']['scenarios']['close_1bp']['yearly']['2021']['total_return'])
    checks={};ranking={}
    for row in rows:
        cid=row['id'];s=row['scenarios'];checks[cid]={}
        for tier in ('strict','balanced'):
            tight=tier=='strict';main=s['close_1bp'];base=b['close_1bp']
            rules=dict(full_cagr=main['full']['cagr']>=base['full']['cagr'],drawdown=main['full']['max_dd']>=-.21,
                recovery_2021=main['yearly']['2021']['total_return']>=target2021[tier],
                return_2026=main['yearly']['2026']['total_return']>=base['yearly']['2026']['total_return']-(0 if tight else .10))
            for name,unused,unused in SCENARIOS:
                rules[name+'_2021']=s[name]['yearly']['2021']['total_return']>=b[name]['yearly']['2021']['total_return']
            for name in ('close_11bp','lag1_11bp','lag1_1bp'):
                allowed=(0 if tight else .01) if name!='lag1_1bp' else (.01 if tight else .02)
                rules[name+'_cagr']=s[name]['full']['cagr']>=b[name]['full']['cagr']-allowed
                rules[name+'_dd']=s[name]['full']['max_dd']>=b[name]['full']['max_dd']-(.01 if tight else .02)
            for name in ('close_1bp','close_11bp','lag1_11bp'):
                for block,unused,unused in BLOCKS:
                    rules[name+'_'+block]=s[name]['blocks'][block]['cagr']>=b[name]['blocks'][block]['cagr']-.05
            checks[cid][tier]=rules
        stress=[s[name]['blocks'][block]['cagr']-b[name]['blocks'][block]['cagr']
                for name in ('close_11bp','lag1_11bp') for block,unused,unused in BLOCKS]
        ranking[cid]=dict(worst_stress_block_excess=min(stress),median_stress_block_excess=float(np.median(stress)),
            cagr=s['close_1bp']['full']['cagr'],switches=s['close_1bp']['switches'],complexity=records[cid]['complexity'])
    def key(cid):
        r=ranking[cid];return (-r['worst_stress_block_excess'],-r['median_stress_block_excess'],-r['cagr'],r['switches'],r['complexity'],cid)
    qualified={tier:sorted([r['id'] for r in registry['records'] if r['selectable'] and all(checks[r['id']][tier].values())],key=key)
               for tier in ('strict','balanced')}
    eligible=[r['id'] for r in registry['records'] if r['selectable']]
    lookup={r['id']:r for r in rows}
    exploratory={label:min(eligible,key=lambda cid:(-getter(lookup[cid]),cid)) for label,getter in (
        ('highest_2021',lambda r:r['scenarios']['close_1bp']['yearly']['2021']['total_return']),
        ('highest_cagr',lambda r:r['scenarios']['close_1bp']['full']['cagr']))}
    return dict(primary=qualified['strict'][0] if qualified['strict'] else None,
        balanced_reference=qualified['balanced'][0] if qualified['balanced'] else None,
        qualified_ids=qualified,qualified_counts={k:len(v) for k,v in qualified.items()},checks=checks,scores=ranking,
        exploratory=exploratory,clean_oos=False,deployed=False,strict_target_2021=target2021['strict'],balanced_target_2021=target2021['balanced'])


def run():
    if (BASE/'results/selection.json').exists():raise ValueError('R2 selection already frozen')
    registry,baseline,receipt=register();arrays,meta,histories=load_base()
    expanded,m=build(arrays,meta,histories,[r['feature_spec'] for r in registry['records']])
    dump(BASE/'results/feature_metadata.json',m)
    np.savez_compressed(BASE/'results/features.npz',**expanded)
    dump(BASE/'results/feature_receipt.json',dict(array_sha256=sha(BASE/'results/features.npz'),metadata_sha256=sha(BASE/'results/feature_metadata.json'),
        source_sha256=sha(BASE/'features.py'),registration_sha256=sha(BASE/'registration.json')))
    records=registry['records'];positions={r['id']:i for i,r in enumerate(records)};groups={}
    for r in records:groups.setdefault((lane_name(r['feature_spec']['window'],r['feature_spec']['smooth']),r['config'].get('risk_context','current20')),[]).append(r)
    results,dates=None,None
    for gi,(group,subset) in enumerate(sorted(groups.items()),1):
        v=view(expanded,m,subset[0]);engine=Simulator(v,m,cache_dir=BASE/'cache')
        for name,lag,fee in SCENARIOS:
            batch=engine.run([effective_config(r,lag) for r in subset],START,END,fee=fee,workers=1)
            if results is None:
                dates=batch['dates'];results={s:{field:np.empty((len(records),)+batch[field].shape[1:],dtype=batch[field].dtype)
                       for field in ('returns','holdings','summary')} for s,unused,unused in SCENARIOS}
            if dates!=batch['dates']:raise AssertionError('Date axis changed')
            idx=[positions[r['id']] for r in subset]
            for field in ('returns','holdings','summary'):results[name][field][idx]=batch[field]
        print('R2 group %d/%d: %s %s, %d records'%(gi,len(groups),group[0],group[1],len(subset)),flush=True)
    previous=load_npz('results/sensitivity/paths.npz');pm=read('v12/results/sensitivity/path_metadata.json')
    prior_reg=read('v12/results/sensitivity/registered_candidates.json')
    by_source={source:r['id'] for r in prior_reg['records'] for source in r['source_ids']};proof=[]
    for role,record in controls().items():
        original=record['source_id'] if role=='c4' else by_source[record['source_id']]
        i,j=positions[registry['controls'][role]],pm['ids'].index(original)
        for name,unused,unused in SCENARIOS:
            for field in ('returns','holdings','summary'):
                if not np.array_equal(results[name][field][i],previous[name+'__'+field][j]):raise AssertionError('Old control changed: '+role+'/'+name+'/'+field)
            proof.append(dict(role=role,scenario=name,exact=True))
    rows=summarize(results,dates,records)
    if source_hashes()!=receipt['definition']['source_sha256']:raise ValueError('Registered source changed')
    verify_inputs()
    np.savez_compressed(BASE/'results/paths.npz',**{name+'__'+field:data[field] for name,data in results.items() for field in data})
    dump(BASE/'results/path_metadata.json',dict(ids=[r['id'] for r in records],dates=dates,sha256=sha(BASE/'results/paths.npz')))
    dump(BASE/'results/evaluation.json',dict(rows=rows,control_fidelity=proof,registration_sha256=sha(BASE/'registration.json'),
        paths_sha256=sha(BASE/'results/paths.npz'),clean_oos=False))
    choice=select(rows,registry,baseline);choice.update(frozen_at=stamp(),registry_sha256=sha(BASE/'registered_candidates.json'),
        evaluation_sha256=sha(BASE/'results/evaluation.json'),paths_sha256=sha(BASE/'results/paths.npz'),
        registration_sha256=sha(BASE/'registration.json'))
    dump(BASE/'results/selection.json',choice)
    print(json.dumps({k:choice[k] for k in ('primary','balanced_reference','qualified_counts')},ensure_ascii=False))
    return choice


def load_results():
    registry=json.loads((BASE/'registered_candidates.json').read_text());evaluation=json.loads((BASE/'results/evaluation.json').read_text());choice=json.loads((BASE/'results/selection.json').read_text())
    receipt=json.loads((BASE/'registration.json').read_text())
    if source_hashes()!=receipt['definition']['source_sha256']:raise ValueError('R2 registered source changed')
    for name,key in (('registered_candidates.json','registry_sha256'),('results/evaluation.json','evaluation_sha256'),('results/paths.npz','paths_sha256'),('registration.json','registration_sha256')):
        if sha(BASE/name)!=choice[key]:raise ValueError('R2 evidence changed: '+name)
    return registry,evaluation,choice


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('register','run'));args=p.parse_args()
    register() if args.command=='register' else run()
