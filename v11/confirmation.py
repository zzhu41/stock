"""Frozen finalist confirmation, then last 2026 reporting; never reselect."""
import argparse
from bisect import bisect_left,bisect_right
from copy import deepcopy
import json
import numpy as np

from .data import BASE,START,DEV_END,CONFIRM_END,END,dump,sha,protect
from .features import build,risk_view
from .scan import run_candidates,SCENARIOS,stamp
from .registry import canonical
from v10_deep.schema import identifier
from v10_deep.scan import metrics
from v10_deep.reference import run_reference

PRESSURES=tuple((('lag1' if lag else 'close')+'_%dbp'%bp,lag,bp/10000.) for lag in (0,1) for bp in (1,5,11,21))


def validate_path_metadata(folder,ids,dates,scenarios,expected_sha):
    metadata=json.loads((folder/'path_metadata.json').read_text())
    if metadata['ids']!=ids or metadata['dates']!=dates or metadata['sha256']!=expected_sha:
        raise ValueError('Path metadata does not cover the exact registered IDs/dates: '+str(folder))
    if sha(folder/'paths.npz')!=expected_sha:raise ValueError('Path matrix hash differs')
    with np.load(folder/'paths.npz',allow_pickle=False) as matrices:
        for name,_,_ in scenarios:
            for field,width in (('returns',len(dates)),('holdings',len(dates)),('summary',10)):
                if matrices[name+'__'+field].shape!=(len(ids),width):
                    raise ValueError('Registered path shape differs')
    return metadata


def validate_confirmation_before_2026(finalists,meta):
    folder=BASE/'results/confirmation'
    evaluation=json.loads((folder/'evaluation.json').read_text())
    registration=json.loads((folder/'registration.json').read_text())
    if sha(folder/'registration.json')!=evaluation['registration_sha256']:
        raise ValueError('Confirmation registration hash differs')
    if registration['finalists']!=finalists or registration['evaluation_end']!=CONFIRM_END:
        raise ValueError('Finalists changed after confirmation')
    if registration['scenarios']!=[list(s) for s in PRESSURES]:
        raise ValueError('Confirmation scenario registration differs')
    if evaluation['roles']!=finalists['roles'] or not evaluation['no_reselection']:
        raise ValueError('Confirmation role/selection receipt differs')
    for name,h in evaluation['sources'].items():
        if sha(BASE/name)!=h:raise ValueError('Confirmation source changed: '+name)
    dates=[date for date in meta['dates'] if START<=date<=CONFIRM_END]
    return validate_path_metadata(folder,[c['id'] for c in finalists['candidates']],dates,PRESSURES,evaluation['paths_sha256'])


def validate_neighborhood_evidence(finalists,meta):
    """An existing file is not proof that a registered diagnostic completed."""
    evidence={};roles=finalists['roles']
    dates=[d for d in meta['dates'] if START<=d<=DEV_END]
    for directory,role in (('neighborhood','v11_a'),('b_neighborhood','v11_b'),('h_neighborhood','v10_h')):
        folder=BASE/'results'/directory
        evaluation=json.loads((folder/'evaluation.json').read_text())
        if evaluation['period']!=[START,DEV_END] or evaluation['new_gate_added'] is not False:
            raise ValueError('Neighborhood period/gate changed: '+directory)
        for name,key in (('registration.json','registration_sha256'),
                         ('registered_candidates.json','registry_sha256'),
                         ('feature_receipt.json','feature_receipt_sha256'),('paths.npz','paths_sha256')):
            if sha(folder/name)!=evaluation[key]:raise ValueError('Neighborhood receipt hash differs: '+directory+' '+name)
        record=json.loads((folder/'registered_candidates.json').read_text())
        registered=json.loads((folder/'registration.json').read_text())
        features=json.loads((folder/'feature_receipt.json').read_text())
        if (registered['registry_sha256']!=evaluation['registry_sha256'] or
            features['registration_sha256']!=evaluation['registration_sha256'] or
            features['execution_last_date']>DEV_END):
            raise ValueError('Neighborhood registration/feature receipt differs')
        parent=roles[role]
        if parent not in record['parents'] or record['parent_map'][parent]['center']!=parent:
            raise ValueError('Neighborhood parent differs')
        if directory=='neighborhood':
            if (evaluation['primary_unchanged']!=parent or evaluation['reselected'] is not False or
                evaluation['parent_selection_sha256']!=finalists['a_selection_sha256']):
                raise ValueError('A neighborhood selection changed')
        else:
            if (evaluation['diagnostic_parent']!=parent or registered['parent']!=parent or
                evaluation['A_primary_unchanged']!=roles['v11_a'] or
                evaluation['B_primary_unchanged']!=roles['v11_b'] or
                evaluation['selected_here'] is not False or evaluation['heldout_performance_accessed'] is not False):
                raise ValueError('B/H neighborhood parent or role changed')
            if sha(folder/'matched_comparison_registration.json')!=registered['matched_registration_sha256']:
                raise ValueError('Matched neighborhood registration changed')
            if registered['parent_override_sha256'] and sha(folder/'parent_override.json')!=registered['parent_override_sha256']:
                raise ValueError('Fixed control parent override changed')
        ids=[c['id'] for c in record['candidates']]
        if not ids or [r['id'] for r in evaluation['rows']]!=ids:
            raise ValueError('Neighborhood evaluation does not cover registered candidates')
        validate_path_metadata(folder,ids,dates,SCENARIOS,evaluation['paths_sha256'])
        evidence[directory]=sha(folder/'evaluation.json')
    return evidence


def _verified_stage(folder,registry_path):
    selection=json.loads((folder/'selection.json').read_text());proof=selection['provenance']
    for name,key in (('registration.json','registration_sha256'),('evaluation.json','evaluation_sha256'),('paths.npz','paths_sha256')):
        if sha(folder/name)!=proof[key]:raise ValueError('Stage artifact changed: '+str(folder/name))
    if sha(registry_path)!=proof['registry_sha256']:raise ValueError('Stage registry changed')
    for name,value in proof['source_sha256'].items():
        if sha(BASE/name)!=value:raise ValueError('Stage source changed: '+name)
    return json.loads(registry_path.read_text()),selection


def frozen_finalists():
    a,sa=_verified_stage(BASE/'results/development',BASE/'registered_candidates.json')
    b,sb=_verified_stage(BASE/'results/combinations',BASE/'results/combinations/registered_candidates.json')
    combined={c['id']:c for c in a['candidates']+b['candidates']}
    roles={'v10_h':a['controls']['h'],'v92':a['controls']['v92'],'simple_reference':a['controls']['simple']}
    if sa['primary']:roles['v11_a']=sa['primary']
    if sb['primary']:roles['v11_b']=sb['primary']
    roles['a_highest_development_return']=sa['top_return']
    for i,cid in enumerate(sa['parents']):roles['a_parent_%d'%i]=cid
    # Previously verified versions are comparison controls, never new selections.
    for name,changes in [('v9',dict(crash_mask=1,global_buffer=.02,gold_buffer=.02)),('v91',dict(crash_mask=1))]:
        c=deepcopy(combined[a['controls']['v92']]);c.update(changes);c=canonical(c);digest=identifier(c);cid='v11_'+digest[:20]
        combined[cid]=dict(c,id=cid,hash=digest,families=['existing_version_comparison'],parents=[],stage='confirmation_control')
        roles[name]=cid
    candidates=[deepcopy(combined[cid]) for cid in sorted(set(roles.values()))]
    return dict(candidates=candidates,roles=roles,score_specs=a['score_specs'],
                a_selection_sha256=sha(BASE/'results/development/selection.json'),
                b_selection_sha256=sha(BASE/'results/combinations/selection.json'))


def _rows(candidates,results,dates,periods):
    rows=[]
    for i,c in enumerate(candidates):
        row=dict(id=c['id'],scenarios={})
        for scenario,values in results.items():
            daily=values['returns'][i];holdings=values['holdings'][i]
            changes=np.r_[False,holdings[1:]!=holdings[:-1]]
            summary={}
            for name,start,end in periods:
                lo,hi=bisect_left(dates,start),bisect_right(dates,end)
                if hi>lo:
                    summary[name]=dict(metrics(daily[lo:hi]),switches=int(changes[lo:hi].sum()))
            summary['yearly']={y:metrics(daily[bisect_left(dates,y+'-01-01'):bisect_right(dates,y+'-12-31')])
                               for y in sorted({d[:4] for d in dates})}
            row['scenarios'][scenario]=summary
        rows.append(row)
    return rows


def gates(row,benchmark,period='confirmation'):
    c,b=row['scenarios'],benchmark['scenarios']
    primary,reference=c['close_1bp'][period],b['close_1bp'][period]
    checks=dict(return_floor=primary['cagr']>=max(.30,.9*reference['cagr']),
                main_drawdown=primary['max_dd']>=reference['max_dd']-.02)
    for scenario in ('close_11bp','lag1_11bp'):
        current,base=c[scenario][period],b[scenario][period]
        checks[scenario+'_growth']=current['cagr']>=base['cagr']-1e-12
        checks[scenario+'_drawdown']=current['max_dd']>=base['max_dd']-.02
    return dict(passed=all(checks.values()),checks=checks)


def _fidelity(finalists,arrays,meta,results,dates,end):
    cases=[];lookup={c['id']:i for i,c in enumerate(finalists['candidates'])}
    # Independently verify both frozen primaries across the two decision clocks.
    for role in ('v11_a','v11_b','v10_h'):
        if role not in finalists['roles']:continue
        cid=finalists['roles'][role];index=lookup[cid];config=finalists['candidates'][index]
        view=risk_view(arrays,meta,config['risk_context'],score=config['score'])
        for name,lag,fee in (('close_1bp',0,.0001),('lag1_11bp',1,.0011)):
            current=dict(config,lag=lag)
            oracle=run_reference(view,meta,current,start=START,end=end,fee=fee)
            exact=all(np.array_equal(oracle[field],results[name][field][index]) for field in ('returns','holdings','summary'))
            if not exact:raise AssertionError('Finalist reference mismatch: '+role+' '+name)
            cases.append(dict(role=role,id=cid,scenario=name,exact=True,days=len(dates)))
    return cases


def run(report_2026=False):
    out=BASE/'results'/('report_2026' if report_2026 else 'confirmation')
    if (out/'evaluation.json').exists():raise RuntimeError('Frozen evaluation exists; do not overwrite')
    finalists=frozen_finalists()
    # Complete preregistered development neighborhood checks before unsealing.
    arrays,meta=build(finalists['score_specs'])
    evidence=validate_neighborhood_evidence(finalists,meta)
    development_dates=[date for date in meta['dates'] if START<=date<=DEV_END]
    for stage,rpath in (('development',BASE/'registered_candidates.json'),
                        ('combinations',BASE/'results/combinations/registered_candidates.json')):
        registered,selected=_verified_stage(BASE/'results'/stage,rpath)
        validate_path_metadata(BASE/'results'/stage,[c['id'] for c in registered['candidates']],
                               development_dates,SCENARIOS,selected['provenance']['paths_sha256'])
    if report_2026:
        validate_confirmation_before_2026(finalists,meta)
        evidence['confirmation']=sha(BASE/'results/confirmation/evaluation.json')
    end=END if report_2026 else CONFIRM_END
    sources={name:sha(BASE/name) for name in ('confirmation.py','scan.py','features.py','registry.py','data.py','PROTOCOL.md','COMBINATION_PROTOCOL.md')}
    dump(out/'registration.json',dict(registered_at=stamp(),finalists=finalists,
        evaluation_end=end,scenarios=PRESSURES,sources=sources,evidence=evidence,
        feature_fingerprints=meta['fingerprints'],no_reselection=True,clean_oos=False))
    results,dates=run_candidates(finalists['candidates'],arrays,meta,start=START,end=end,scenarios=PRESSURES)
    # Identical definitions must leave every earlier development return intact.
    index={c['id']:i for i,c in enumerate(finalists['candidates'])}
    n=bisect_right(dates,DEV_END)
    for stage in ('development','combinations'):
        source=BASE/'results'/stage;metadata=json.loads((source/'path_metadata.json').read_text())
        with np.load(source/'paths.npz',allow_pickle=False) as old:
            for j,cid in enumerate(metadata['ids']):
                if cid not in index:continue
                for name,_,_ in SCENARIOS:
                    for field in ('returns','holdings'):
                        if not np.array_equal(old[name+'__'+field][j],results[name][field][index[cid],:n]):
                            raise AssertionError('Future evaluation changed a frozen development prefix')
    if report_2026:
        prior_dir=BASE/'results/confirmation';prior_meta=json.loads((prior_dir/'path_metadata.json').read_text())
        with np.load(prior_dir/'paths.npz',allow_pickle=False) as prior:
            for name,_,_ in PRESSURES:
                for field in ('returns','holdings'):
                    if not np.array_equal(prior[name+'__'+field],results[name][field][:,:len(prior_meta['dates'])]):
                        raise AssertionError('2026 evaluation changed confirmation prefix')
    cases=_fidelity(finalists,arrays,meta,results,dates,end)
    periods=[('development',START,DEV_END),('confirmation','2022-01-01',CONFIRM_END),('full',START,end)]
    if report_2026:periods.append(('report_only_2026','2026-01-01',END))
    rows=_rows(finalists['candidates'],results,dates,periods)
    row_map={r['id']:r for r in rows};benchmark=row_map[finalists['roles']['v10_h']]
    verdicts={role:gates(row_map[finalists['roles'][role]],benchmark) for role in ('v11_a','v11_b') if role in finalists['roles']}
    for name,h in sources.items():
        if sha(BASE/name)!=h:raise ValueError('Finalist evaluation source changed')
    packed={name+'__'+field:value for name,data in results.items() for field,value in data.items()}
    np.savez_compressed(out/'paths.npz',**packed)
    dump(out/'path_metadata.json',dict(ids=[c['id'] for c in finalists['candidates']],dates=dates,sha256=sha(out/'paths.npz')))
    dump(out/'evaluation.json',dict(rows=rows,roles=finalists['roles'],confirmation_verdicts=verdicts,
        no_reselection=True,clean_oos=False,reference_fidelity=cases,
        registration_sha256=sha(out/'registration.json'),paths_sha256=sha(out/'paths.npz'),sources=sources))
    protect()
    print(json.dumps(dict(period_end=end,confirmation_verdicts=verdicts),indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report-2026',action='store_true');args=parser.parse_args()
    run(report_2026=args.report_2026)
