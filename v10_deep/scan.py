"""Registered stage runner: train, freeze choice, then display the known 2026 tail."""
import argparse
import bisect
from copy import deepcopy
from datetime import datetime, timezone
import json
import time
from functools import lru_cache
import numpy as np

from .data import BASE, ROOT, START, END, TRAIN_END, dump, sha, protect, freeze_protected
from .features import build
from .native import Simulator
from .registry import stage_one, canonical
from .schema import baseline, identifier

SOURCE_FILES=('data.py','features.py','schema.py','registry.py','native.cpp','native.py','scan.py','PROTOCOL.md')
PERIODS={'selection':(START,TRAIN_END),'early_2014_2017':(START,'2017-12-31'),
         'middle_2018_2021':('2018-01-01','2021-12-31'), 'recent_2022_2025':('2022-01-01',TRAIN_END),
         'report_only_2026':('2026-01-01',END),'full':(START,END)}


def stamp():return datetime.now(timezone.utc).isoformat()


def metrics(returns):
    r=np.asarray(returns,dtype=np.float64)
    nav=np.cumprod(1+r);factor=float(nav[-1]);peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
    std=float(r.std());ann=float(np.expm1(np.log1p(r).sum()*244/len(r)));dd=float(np.min(nav/peak-1))
    return dict(sessions=len(r),nav=factor,total_return=factor-1,cagr=ann,max_dd=dd,
                volatility=std*244**.5,sharpe=float(r.mean()/std*244**.5) if std else 0.,
                calmar=ann/abs(dd) if dd else 0.)


@lru_cache(maxsize=4)
def slices(dates):
    return ({name:(bisect.bisect_left(dates,lo),bisect.bisect_right(dates,hi)) for name,(lo,hi) in PERIODS.items()},
            {y:(bisect.bisect_left(dates,y+'-01-01'),bisect.bisect_right(dates,y+'-12-31')) for y in sorted({d[:4] for d in dates})})


def summarize(returns,dates):
    out={}
    periods,years=slices(tuple(dates));returns=np.asarray(returns)
    for name,(lo,hi) in periods.items():
        if hi>lo:out[name]=metrics(returns[lo:hi])
    out['yearly']={y:float(np.expm1(np.log1p(returns[lo:hi]).sum())) for y,(lo,hi) in years.items()}
    return out


def stress_returns(returns,holdings,fee=.0005):
    charged=np.r_[False,np.asarray(holdings)[1:]!=np.asarray(holdings)[:-1]]
    adjusted=(1+np.asarray(returns))*np.where(charged,(1-2*fee)/(1-2*.0001),1.)-1
    return adjusted,int(charged.sum())


def fingerprint(meta):
    return dict(sources={n:sha(BASE/n) for n in SOURCE_FILES},features=meta['fingerprints'],
                corrected_data_sha256=sha(ROOT/'v10_h_close/corrected_manifest.json'),
                qvix_manifest_sha256=sha(ROOT/'v10_h_close/qvix_manifest.json'),
                feature_cache_sha256=sha(BASE/'cache/features.npz'),
                protected_manifest_sha256=sha(BASE/'protected_manifest.json'))


def stage_configs(stage):
    if stage=='mechanisms':return stage_one()
    path=BASE/'results'/stage/'registered_candidates.json'
    return json.loads(path.read_text())


def control_ids():
    versions={}
    for name,changes in [('v9',dict(crash_mask=1,global_buffer=.02,gold_buffer=.02)),('v9.1',dict(crash_mask=1)),('v9.2',{})]:
        c=baseline();c.update(changes);versions[name]='vd_'+identifier(canonical(c))[:20]
    return versions


def choose(rows,configs):
    ids=control_ids();base=next(r for r in rows if r['id']==ids['v9.2'])
    configs={c['id']:c for c in configs}
    def key(r):
        c=configs[r['id']];b=canonical(baseline())
        changes=sum(c[k]!=v for k,v in b.items())
        external=int(bool(c['crash_mask']&2))+int(bool(c['crash_mask']&4))
        score_inputs=len(c['score'].split('_'))-1 if c['score'].startswith('blend_') else 1
        return (-r['metrics']['selection']['cagr'],external,score_inputs,changes,r['selection_switches'],r['id'])
    ordered=sorted(rows,key=key);qualified=[];checks={}
    for row in ordered:
        m,b=row['metrics'],base['metrics'];f,bf=row['fee5'],base['fee5']
        tests=dict(material_return=m['selection']['cagr']>=b['selection']['cagr']+.02,
                   drawdown=abs(m['selection']['max_dd'])<=abs(b['selection']['max_dd'])+.05,
                   recent=m['recent_2022_2025']['cagr']>=b['recent_2022_2025']['cagr']-.02,
                   fee_relative=f['selection']['cagr']>=bf['selection']['cagr'])
        checks[row['id']]=tests
        if all(tests.values()):qualified.append(row)
    return dict(primary=ordered[0]['id'],guarded=qualified[0]['id'] if qualified else None,
                qualified_count=len(qualified),top_ids=[r['id'] for r in ordered[:50]],checks=checks,
                controls=ids,selected_at=stamp(),selection_period='2014-2025 already researched history',clean_oos=False)


def run_stage(stage,workers=2):
    freeze_protected();protect()
    out=BASE/'results'/stage;out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():raise RuntimeError('Selection already frozen; use a new registered stage')
    registry=stage_configs(stage);configs=registry['candidates'];dump(out/'registered_candidates.json',registry)
    arrays,meta=build();sim=Simulator(arrays,meta);fp=fingerprint(meta)
    dump(out/'registration.json',dict(registered_at=stamp(),fingerprints=fp,registry_sha256=sha(out/'registered_candidates.json'),
              candidate_count=len(configs),start=START,train_end=TRAIN_END,end=END,fee=.0001,stress_fee=.0005,clean_oos=False))
    dates=[d for d in meta['dates'] if START<=d<=TRAIN_END];N,D=len(configs),len(dates)
    returns=np.lib.format.open_memmap(str(out/'selection_returns.npy'),mode='w+',dtype=np.float64,shape=(N,D))
    fees=np.lib.format.open_memmap(str(out/'selection_fee5_returns.npy'),mode='w+',dtype=np.float64,shape=(N,D))
    rows=[];started=time.monotonic()
    for begin in range(0,N,128):
        chunk=configs[begin:begin+128];result=sim.run(chunk,end=TRAIN_END,workers=workers)
        stress_direct=sim.run(chunk,end=TRAIN_END,fee=.0005,workers=workers)
        if not np.array_equal(result['holdings'],stress_direct['holdings']):raise AssertionError('Fees changed a price-only policy path')
        for j,c in enumerate(chunk):
            r=result['returns'][j];f,sw=stress_returns(r,result['holdings'][j])
            if sw!=int(result['summary'][j,3]):raise AssertionError('Fee switch count differs')
            if not np.allclose(f,stress_direct['returns'][j],rtol=1e-10,atol=1e-12):raise AssertionError('Algebraic fee pressure disagrees with fresh simulation')
            returns[begin+j]=r;fees[begin+j]=f
            rows.append(dict(id=c['id'],hash=c['hash'],families=c['families'],index=begin+j,
                             metrics=summarize(r,dates),fee5=summarize(f,dates),switches=sw,selection_switches=sw,
                             crashes=int(result['summary'][j,7]),blocked=int(result['summary'][j,5]),
                             missing_held=int(result['summary'][j,6])))
        print(stage,'selection',min(begin+128,N),'/',N,'%.2fs'%(time.monotonic()-started),flush=True)
    returns.flush();fees.flush()
    dump(out/'development.json',dict(rows=rows,dates=dates,all_fee5_paths_verified_by_fresh_simulation=True));chosen=choose(rows,configs)
    chosen.update(registration_sha256=sha(out/'registration.json'),development_sha256=sha(out/'development.json'),
                  selection_returns_sha256=sha(out/'selection_returns.npy'),selection_fee5_sha256=sha(out/'selection_fee5_returns.npy'))
    assert fingerprint(meta)==fp;dump(out/'selection.json',chosen)
    print('FROZEN',chosen['primary'],chosen['guarded'],'qualified',chosen['qualified_count'],flush=True)
    # Full-period diagnostics after the selection has been written, never used to amend it.
    dates=[d for d in meta['dates'] if START<=d<=END];Dfull=len(dates)
    full=np.lib.format.open_memmap(str(out/'full_returns.npy'),mode='w+',dtype=np.float64,shape=(N,Dfull))
    fullfee=np.lib.format.open_memmap(str(out/'full_fee5_returns.npy'),mode='w+',dtype=np.float64,shape=(N,Dfull))
    fullrows=[];paths={};keep=set(chosen['top_ids'][:10])|set(chosen['controls'].values())|{chosen['primary'],chosen['guarded']}
    for begin in range(0,N,128):
        chunk=configs[begin:begin+128];result=sim.run(chunk,workers=workers)
        for j,c in enumerate(chunk):
            r=result['returns'][j];f,sw=stress_returns(r,result['holdings'][j])
            if not np.array_equal(r[:len(returns[begin+j])],returns[begin+j]):raise AssertionError('Full run altered training prefix')
            if not np.array_equal(f[:len(fees[begin+j])],fees[begin+j]):raise AssertionError('Full run altered fee5 training prefix')
            full[begin+j]=r;fullfee[begin+j]=f
            fullrows.append(dict(id=c['id'],hash=c['hash'],families=c['families'],index=begin+j,
                                 metrics=summarize(r,dates),fee5=summarize(f,dates),switches=sw,
                                 selection_switches=int(np.sum(result['holdings'][j,1:len(returns[begin+j])]!=result['holdings'][j,:len(returns[begin+j])-1])),
                                 crashes=int(result['summary'][j,7])))
            if c['id'] in keep:
                paths[c['id']]=dict(returns=r.tolist(),holdings=[meta['assets'][a] if a>=0 else None for a in result['holdings'][j]])
        print(stage,'full',min(begin+128,N),'/',N,flush=True)
    full.flush();fullfee.flush();assert fingerprint(meta)==fp
    fullbest=max(fullrows,key=lambda x:x['metrics']['full']['cagr'])
    dump(out/'evaluation.json',dict(rows=fullrows,dates=dates,selection=chosen,hindsight_best=fullbest['id'],
              fingerprints=fp,completed_at=stamp(),full_returns_sha256=sha(out/'full_returns.npy'),full_fee5_sha256=sha(out/'full_fee5_returns.npy'),
              protected_files_verified=protect(),clean_oos=False))
    dump(out/'selected_paths.json',paths)
    named={r['id']:r for r in fullrows}
    for n in dict.fromkeys([chosen['primary'],chosen['guarded'],chosen['controls']['v9.2'],fullbest['id']]):
        if n:print('RESULT',n,named[n]['metrics']['full'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage');p.add_argument('--workers',type=int,choices=(1,2),default=2)
    a=p.parse_args();run_stage(a.stage,a.workers)
