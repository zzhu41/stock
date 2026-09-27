"""Registered post-search sensitivity, ablation and paired pool deletion.

All diagnostic records are non-selectable. Feature overrides are part of the
diagnostic identity; a borrowed native MA slot alone is not an identity.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from decimal import Decimal
import io
import json
from pathlib import Path

import numpy as np

from .data import BASE,ROOT,START,END,load_inputs,stamp,sha,dump,protect
from .exante import verify_stage
from .exante_features import view_for_config
from .diagnostic_features import make_view,INVALID
from .consensus_features import RISK_ASSETS
from .scan import SCENARIOS,NUMERIC_FIELDS,run_candidates,summary_rows,digest
from .selection import select as gate_diagnostic
from v10_deep.data import inputs
from v10_deep.registry import canonical as old_canonical
from v10_deep.schema import semantic

OUT=BASE/'results/sensitivity'
PARENTS=dict(balanced_b='v12_bb36c1de9e534b1cf332',balanced_c='v12_7f06d74ee08531d6cdbe',
             balanced_a='v12_6ace1b82572e526d2b75')
SPECS={'wls25_v20':(25,1),'wls20_smooth3':(20,3),'v12_b_wls25_smooth3':(25,3)}
SOURCES=('sensitivity.py','tests/test_sensitivity.py','DIAGNOSTIC_PROTOCOL.md','PROTOCOL.md',
         'diagnostic_features.py','exante_features.py','consensus_features.py','data.py',
         'native.py','native.cpp','schema.py','scan.py','selection.py','exante.py')


def normalize(config):
    c=old_canonical(semantic(config))
    if c['min_hold']==1:c['min_hold']=0
    if c.get('risk_context','current20')=='current20':c.pop('risk_context',None)
    c['locked_panic_exit']=int(c.get('locked_panic_exit',0))
    return c


def feature_spec(config):
    w,s=SPECS[config['score']]
    return dict(window=w,smooth=s,ma_window=int(config['ma'][2:]))


def identity(config,spec):
    c=normalize(config);c.pop('score');c.pop('ma')
    return digest(dict(policy=c,features=spec))


def generate(records,controls,parents=None):
    """Pure, fixed-parent registry. Never reads performance or ranks a neighbor."""
    records={r['id']:r for r in records} if not isinstance(records,dict) else records
    parents=dict(PARENTS if parents is None else parents);parents['h']=controls['h']
    h=normalize(records[controls['h']]['config'])
    configs,links,control_ids={}, {}, {}

    def add(c,spec,role,source_id=None):
        c=normalize(c);spec=deepcopy(spec);hash_=identity(c,spec);cid='v12d_'+hash_[:20]
        if cid not in configs:
            configs[cid]=dict(id=cid,hash=hash_,kind='single',config=c,feature_spec=spec,
                selectable=False,complexity=0,families=[],source_ids=[])
        configs[cid]['families']=sorted(set(configs[cid]['families'])|{role})
        if source_id:configs[cid]['source_ids']=sorted(set(configs[cid]['source_ids'])|{source_id})
        return cid

    for role,source_id in controls.items():
        c=records[source_id]['config']
        control_ids[role]=add(c,feature_spec(c),'old_control',source_id)
    for role,source_id in parents.items():
        c=normalize(records[source_id]['config']);spec=feature_spec(c)
        center=add(c,spec,'center',source_id)
        links[role]=dict(center=center,source_id=source_id,neighbors=[],ablations=[],proposals=[])
        link=links[role]

        def propose(kind,label,changes=None,features=None,invalid_reason=None):
            if invalid_reason:
                link['proposals'].append(dict(kind=kind,label=label,invalid_reason=invalid_reason,no_op=True))
                return
            value=dict(deepcopy(c),**(changes or {}));fs=dict(spec,**(features or {}))
            cid=add(value,fs,kind);bucket=link['neighbors' if kind=='neighbor' else 'ablations']
            duplicate=cid in bucket
            if cid!=center and not duplicate:bucket.append(cid)
            link['proposals'].append(dict(kind=kind,label=label,id=cid,no_op=cid==center,
                deduplicated=duplicate,changes=changes or {},feature_changes=features or {}))

        for factor in (.8,1.2):
            propose('neighbor','window_x'+str(factor),features=dict(window=max(2,round(spec['window']*factor))))
            propose('neighbor','MA_x'+str(factor),features=dict(ma_window=max(2,round(spec['ma_window']*factor))))
        for step in (-1,1):
            smooth=spec['smooth']+step
            propose('neighbor','smooth_%+d'%step,features=dict(smooth=smooth),
                    invalid_reason='smoothing below one' if smooth<1 else None)
            hold=c['min_hold']+step
            propose('neighbor','min_hold_%+d'%step,changes=dict(min_hold=hold),
                    invalid_reason='negative holding duration' if hold<0 else None)
        for factor in (.9,1.1):
            propose('neighbor','panic_x'+str(factor),changes=dict(panic=float(Decimal(str(c['panic']))*Decimal(str(factor)))))
        for shift in (-.005,.005):
            propose('neighbor','buffers_%+.3f'%shift,changes={k:float(Decimal(str(c[k]))+Decimal(str(shift)))
                    for k in ('buffer','global_buffer','gold_buffer')})
        for key in sorted(set(c)|set(h)):
            old=c.get(key,'current20' if key=='risk_context' else None)
            new=h.get(key,'current20' if key=='risk_context' else None)
            if old==new:continue
            if key=='score':propose('ablation','restore_H_score',changes={key:new},
                                   features={k:v for k,v in feature_spec(h).items() if k!='ma_window'})
            elif key=='ma':propose('ablation','restore_H_ma',changes={key:new},features=dict(ma_window=int(new[2:])))
            else:propose('ablation','restore_H_'+key,changes={key:new})
        propose('ablation','disable_locked_panic_exit',changes=dict(locked_panic_exit=0))
        if role=='balanced_c':propose('ablation','restore_current20',changes=dict(risk_context='current20'))

    codes=sorted(set(h['stock_pool'])|set(h['global_pool']))
    if len(codes)!=9 or {'511880','518880'}&set(codes):raise ValueError('Expected original nine deletable equities')
    deletions=[]
    for code in codes:
        hc=deepcopy(h)
        for key in ('stock_pool','global_pool'):hc[key]=[c for c in hc[key] if c!=code]
        paired=add(hc,feature_spec(h),'paired_H_delete')
        for role,source_id in parents.items():
            c=normalize(records[source_id]['config'])
            for key in ('stock_pool','global_pool'):c[key]=[s for s in c[key] if s!=code]
            cid=add(c,feature_spec(c),'leave_one_out')
            deletions.append(dict(parent=role,excluded=code,id=cid,paired_h=paired))
    return dict(schema=1,parents=links,controls=control_ids,source_controls=controls,
        records=sorted(configs.values(),key=lambda r:r['id']),candidate_count=len(configs),
        leave_one_out=deletions,period=[START,END],scenarios=SCENARIOS,
        official_primary=None,no_reselection=True,clean_oos=False,
        conventions=dict(identity='normalized policy plus actual score/MA specification',
            minimum_hold='1 is identical to 0 in once-per-observation decisions',
            score='raw legacy WLS25; generic convolution WLS otherwise; own quote smoothing',
            moving_average='benchmark ma180 scratch slot only; original ma250 remains untouched',
            risk_context='applied once after feature transformation from the original base cube',
            deletions='trading eligibility only; benchmark and historical price information preserved'))


def load_registered_parents():
    registries,records={},{}
    for stage,path in (('main',BASE/'registered_candidates.json'),
            ('consensus',BASE/'results/consensus/registered_candidates.json'),
            ('exante',BASE/'results/exante/registered_candidates.json')):
        r,s=verify_stage(BASE/'results'/stage,path)
        if s['primary'] is not None:raise ValueError('Diagnostic protocol expects all three searches to have no primary')
        registries[stage]=r;records.update({c['id']:c for c in r['candidates']})
    return records,registries['main']['controls']


def fingerprints():
    artifacts=['registered_candidates.json','registration.json','inputs/manifest.json','inputs/features.json',
               'inputs/features.npz.gz','protected_manifest.json']
    for stage in ('main','consensus','exante'):
        artifacts += ['results/'+stage+'/'+name for name in
            ('registration.json','selection.json','evaluation.json','paths.npz','path_metadata.json')]
        if stage!='main':artifacts.append('results/'+stage+'/registered_candidates.json')
    return dict(sources={name:sha(BASE/name) for name in SOURCES},
        frozen_dependencies={name:sha(ROOT/name) for name in ('v11/features.py','v10_deep/features.py',
            'v10_deep/registry.py','v10_deep/reference.py','v10_deep/schema.py','v10_deep/scan.py')},
        inputs={name:sha(BASE/name) for name in artifacts})


def register():
    records,controls=load_registered_parents();registry=generate(records,controls)
    definition=dict(fingerprints=fingerprints(),period=[START,END],scenarios=SCENARIOS,
                    candidate_count=registry['candidate_count'],no_reselection=True)
    path=OUT/'registration.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['definition']!=json.loads(json.dumps(definition)) or sha(OUT/'registered_candidates.json')!=old['registry_sha256']:
            raise ValueError('Sensitivity registration changed; do not overwrite')
        if json.loads((OUT/'registered_candidates.json').read_text())!=json.loads(json.dumps(registry)):
            raise ValueError('Sensitivity registry differs from its fixed generation rule')
        return registry,old
    protect();dump(OUT/'registered_candidates.json',registry)
    receipt=dict(registered_at=stamp(),definition=definition,registry_sha256=sha(OUT/'registered_candidates.json'))
    dump(path,receipt)
    return registry,receipt


def prepare_view(arrays,meta,histories,record):
    spec=record['feature_spec']
    base=dict(record['config'],score='wls25_v20',ma='ma250' if spec['ma_window']==250 else 'ma180')
    feature,m,c=make_view(arrays,meta,histories,base,**spec)
    if c['score'] not in ('wls25_v20','wls20_smooth3'):
        lane=m['score_names'].index(c['score'])
        outside=[i for i,code in enumerate(m['assets']) if code not in RISK_ASSETS]
        feature['scores'][lane,:,outside]=INVALID
        feature['orders'][lane]=np.argsort(-feature['scores'][lane],axis=1,kind='stable').astype(np.int32)
    return view_for_config(feature,m,c),m,c


def run_registered(registry,arrays,meta,histories,evaluator=run_candidates,output=None):
    records=registry['records'];groups={};positions={r['id']:i for i,r in enumerate(records)}
    for r in records:
        key=digest(dict(feature=r['feature_spec'],context=r['config'].get('risk_context','current20')))
        groups.setdefault(key,[]).append(r)
    merged,dates=None,None
    for number,(key,group) in enumerate(sorted(groups.items()),1):
        view,m,example=prepare_view(arrays,meta,histories,group[0])
        effective=[]
        for record in group:
            item=deepcopy(record)
            item['config'].update(score=example['score'],ma=example['ma'])
            effective.append(item)
        if output is not None:
            import hashlib
            dump(Path(output)/'feature_receipts'/(key+'.json'),dict(group=key,
                ids=[r['id'] for r in group],feature_metadata=m['v12_diagnostic_features'],
                risk_context=example.get('risk_context','current20'),
                effective_configs={r['id']:r['config'] for r in effective},
                array_sha256={name:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for name,v in view.items()}))
        with redirect_stdout(io.StringIO()):
            result,observed=evaluator(effective,view,m,start=START,end=END,scenarios=SCENARIOS,workers=1)
        if merged is None:
            dates=observed;merged={name:{field:np.empty((len(records),)+v[field].shape[1:],dtype=v[field].dtype)
                            for field in NUMERIC_FIELDS} for name,v in result.items()}
            for value in merged.values():value['metadata']={}
        if observed!=dates:raise AssertionError('Diagnostic date axes differ')
        ids=[positions[r['id']] for r in group]
        for name,value in result.items():
            for field in NUMERIC_FIELDS:merged[name][field][ids]=value[field]
            merged[name]['metadata'].update(value['metadata'])
        print('sensitivity group %d/%d complete (%d records)'%(number,len(groups),len(group)),flush=True)
    return merged,dates


def verify_centers(registry,results,dates):
    positions={r['id']:i for i,r in enumerate(registry['records'])};cases=[]
    requested={source:cid for cid in positions for source in registry['records'][positions[cid]]['source_ids']}
    for stage in ('main','consensus','exante'):
        folder=BASE/'results'/stage;m=json.loads((folder/'path_metadata.json').read_text())
        if m['dates']!=dates:raise AssertionError('Center reference dates differ')
        matching=[(source,positions[cid],m['ids'].index(source)) for source,cid in requested.items() if source in m['ids']]
        with np.load(folder/'paths.npz',allow_pickle=False) as old:
            for name,unused,unused in SCENARIOS:
                for field in ('returns','holdings','summary'):
                    saved=old[name+'__'+field]
                    for source,i,j in matching:
                        if not np.array_equal(results[name][field][i],saved[j]):
                            raise AssertionError('Center changed: '+stage+' '+source+' '+name+' '+field)
                cases.extend(dict(stage=stage,source_id=source,scenario=name,exact=True) for source,i,j in matching)
    return dict(passed=True,cases=cases)


def analyze(registry,rows):
    lookup={r['id']:r for r in rows}
    gates=gate_diagnostic(rows,registry['records'],registry['controls'],END)
    if gates['primary'] is not None or gates['selectable_count']!=0:raise AssertionError('Diagnostics must never select a strategy')
    parents={}
    def changes(left,right):
        return dict(cagr_pp=100*(left['full']['cagr']-right['full']['cagr']),
                    max_dd_pp=100*(left['full']['max_dd']-right['full']['max_dd']),
                    tail_return_pp=100*(left['tail']['total_return']-right['tail']['total_return']))
    for label,link in registry['parents'].items():
        center=lookup[link['center']];details={};worst=[]
        for cid in link['neighbors']:
            row=lookup[cid];deltas={s:changes(row['scenarios'][s],center['scenarios'][s]) for s,_,_ in SCENARIOS}
            pressure=[100*(row['scenarios'][s]['blocks'][b]['cagr']-center['scenarios'][s]['blocks'][b]['cagr'])
                      for s in ('close_11bp','lag1_11bp') for b in ('early','middle','recent')]
            worst.append(min(pressure))
            details[cid]=dict(deltas=deltas,worst_paired_pressure_block_delta_pp=min(pressure),
                             passes_original_gates=all(gates['checks'][cid].values()))
        values=[d['deltas']['close_1bp']['cagr_pp'] for d in details.values()]
        parents[label]=dict(center=link['center'],source_id=link['source_id'],center_metrics=center['scenarios'],
            neighbor_count=len(details),neighbors=details,original_gate_pass_count=sum(d['passes_original_gates'] for d in details.values()),
            own_center_cagr_delta_pp=dict(median=float(np.median(values)),q25=float(np.percentile(values,25))),
            worst_paired_pressure_delta_pp=dict(median=float(np.median(worst)),q25=float(np.percentile(worst,25))),
            ablations={cid:{s:changes(lookup[cid]['scenarios'][s],center['scenarios'][s]) for s,_,_ in SCENARIOS} for cid in link['ablations']})
    deletion=[]
    for entry in registry['leave_one_out']:
        row,paired=lookup[entry['id']],lookup[entry['paired_h']]
        center=lookup[registry['parents'][entry['parent']]['center']]
        deletion.append(dict(entry,scenarios={s:dict(metrics=row['scenarios'][s],
            versus_paired_h=changes(row['scenarios'][s],paired['scenarios'][s]),
            versus_own_full_pool=changes(row['scenarios'][s],center['scenarios'][s])) for s,_,_ in SCENARIOS}))
    return dict(parents=parents,leave_one_out=deletion,original_gate_checks=gates['checks'],no_reselection=True)


def evaluate():
    if (OUT/'evaluation.json').exists():raise ValueError('Sensitivity is already frozen')
    registry,receipt=register();arrays,meta,unused=load_inputs();histories,unused,unused=inputs()
    results,dates=run_registered(registry,arrays,meta,histories,output=OUT)
    fidelity=verify_centers(registry,results,dates)
    rows=summary_rows(registry['records'],results,dates,period_end=END)
    if fingerprints()!=receipt['definition']['fingerprints']:raise ValueError('Diagnostic source/input changed during execution')
    np.savez_compressed(OUT/'paths.npz',**{name+'__'+field:value[field] for name,value in results.items() for field in NUMERIC_FIELDS})
    dump(OUT/'path_metadata.json',dict(ids=[r['id'] for r in registry['records']],dates=dates,sha256=sha(OUT/'paths.npz')))
    dump(OUT/'execution_metadata.json',{name:value['metadata'] for name,value in results.items()})
    value=dict(completed_at=stamp(),rows=rows,analysis=analyze(registry,rows),center_fidelity=fidelity,
        registration_sha256=sha(OUT/'registration.json'),paths_sha256=sha(OUT/'paths.npz'),
        fingerprints=receipt['definition']['fingerprints'],protected_files_verified=protect(),
        official_primary=None,no_reselection=True,clean_oos=False)
    dump(OUT/'evaluation.json',value)
    lines=['# 冻结对照敏感性（不重选）','','四父、四既定情景；所有曲线均为连续账户，原A/B/C主选择继续为null。','',
        '| 对照 | 邻居数 | 过原门槛 | 自身年化差中位数/Q25(pp) | 最差压力块差中位数/Q25(pp) |',
        '|---|---:|---:|---:|---:|']
    for label,p in value['analysis']['parents'].items():
        a,b=p['own_center_cagr_delta_pp'],p['worst_paired_pressure_delta_pp']
        lines.append('| %s | %d | %d | %.3f / %.3f | %.3f / %.3f |'%(label,p['neighbor_count'],p['original_gate_pass_count'],a['median'],a['q25'],b['median'],b['q25']))
    lines+=['','差值为邻居减自身中心。删票保留原价格/指标，候选与H同删；正回撤差表示回撤减小。',
            '完整邻域、同义/无效提案、逐项消融、36组删票、逐年/分块及日收益见JSON/NPZ。',
            '这些是看过全期后固定的探索对照，不构成干净样本外或合格主策略。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('register','run'));args=p.parse_args()
    register() if args.command=='register' else evaluate()
