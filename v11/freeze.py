"""Freeze failed-confirmation research candidates and reproducibility receipts."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import BASE, ROOT, START, END, CONFIRM_END, dump, sha, protect
from .confirmation import frozen_finalists, validate_confirmation_before_2026, validate_path_metadata, PRESSURES, gates
from .scan import SCENARIOS
from .features import build


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def verify_hashes(root,values):
    require(bool(values),'Empty evidence hash map')
    for name,expected in values.items():
        require(sha(root/name)==expected,'Frozen evidence hash differs: '+str(name))


def verify_final_report(finalists,meta):
    folder=BASE/'results/report_2026'
    report,registration=read(folder/'evaluation.json'),read(folder/'registration.json')
    require(sha(folder/'registration.json')==report['registration_sha256'],'Final report registration hash differs')
    require(registration['finalists']==finalists and report['roles']==finalists['roles'],'Final report roles/configurations changed')
    require(registration['evaluation_end']==END and registration['no_reselection'] is True
            and report['no_reselection'] is True,'Final report is incomplete or reselected')
    require(registration['scenarios']==[list(s) for s in PRESSURES],'Final report must contain all eight registered scenarios')
    require(report['sources']==registration['sources'],'Final report source receipts disagree')
    verify_hashes(BASE,registration['sources'])
    require(registration['feature_fingerprints']==meta['fingerprints'],'Final report feature inputs differ')
    ids=[c['id'] for c in finalists['candidates']]
    dates=[d for d in meta['dates'] if START<=d<=END]
    validate_path_metadata(folder,ids,dates,PRESSURES,report['paths_sha256'])
    require([r['id'] for r in report['rows']]==ids,'Final report evaluation rows differ')
    rows={r['id']:r for r in report['rows']};h=rows[finalists['roles']['v10_h']]
    expected={role:gates(rows[finalists['roles'][role]],h) for role in ('v11_a','v11_b')}
    require(report['confirmation_verdicts']==expected,'Final report confirmation verdict does not match its metrics')
    return report


def verify_report_receipt():
    receipt=read(BASE/'results/report_receipt.json')
    require(receipt['source_sha256']==sha(BASE/'report.py') and receipt['candidate_selection_performed'] is False,
            'Report rendering source or no-reselection receipt differs')
    needed_inputs={'results/report_2026/evaluation.json','results/report_2026/paths.npz',
        'results/final_audit/audit.json','results/leave_one_out/evaluation.json','results/reselection/evaluation.json',
        'results/combined_development_diagnostics/diagnostics.json','results/b_neighborhood/comparison.json'}
    needed_outputs={'REPORT.md','results/equity_drawdown.png','results/full_comparison.csv',
                    'results/annual_returns.csv','results/execution_stress.csv'}
    require(needed_inputs.issubset(receipt['inputs']) and needed_outputs.issubset(receipt['outputs']),
            'Report receipt omits required inputs or outputs')
    verify_hashes(BASE,receipt['inputs']);verify_hashes(BASE,receipt['outputs'])


def verify_final_audit(report,meta):
    folder=BASE/'results/final_audit';audit=read(folder/'audit.json')
    require(audit['passed'] is True and audit['no_reselection'] is True and audit['period']==[START,END],
            'Independent audit did not pass the complete frozen interval')
    verify_hashes(ROOT,audit['input_sha256']);verify_hashes(folder,audit['artifacts'])
    checks=audit['checks'];days=sum(START<=d<=END for d in meta['dates'])
    require(checks['confirmation_prefix_exact'] is True and checks['exact_reference_returns_holdings_summary'] is True,
            'Independent audit parity checks did not pass')
    require(checks['reference_days']==days and set(checks['reference_roles'])=={'v10_h','v11_a','v11_b','simple_reference'},
            'Independent reference audit coverage is incomplete')
    require(checks['quote_repriced_paths']==len(report['rows'])*len(PRESSURES)
            and checks['recomputed_metric_values']>0
            and np.isfinite(checks['maximum_daily_return_error']) and 0<=checks['maximum_daily_return_error']<=1e-12
            and np.isfinite(checks['maximum_metric_absolute_error']) and 0<=checks['maximum_metric_absolute_error']<=1e-10,
            'Independent P&L/metric audit did not meet its numerical tolerance')
    # Failed candidate confirmation verdicts are expected evidence, not failed
    # accounting checks; never recursively require every `passed` field true.
    require(audit['confirmation_verdicts']==report['confirmation_verdicts'],'Audit confirmation verdicts differ')


def verify_leave_one_out(finalists):
    folder=BASE/'results/leave_one_out';evaluation=read(folder/'evaluation.json');receipt=read(folder/'receipt.json')
    registration=read(folder/'registration.json');design=registration['design']
    require(evaluation['registration_sha256']==sha(folder/'registration.json'),'Pool diagnostic registration differs')
    require(evaluation['path_sha256']==sha(folder/'paths.npz')==read(folder/'path_metadata.json')['sha256'],
            'Pool diagnostic paths differ')
    require(evaluation['no_reselection'] is True and receipt['no_reselection'] is True
            and evaluation['confirmation_failure_overturned'] is False,'Pool diagnostic changed the frozen verdict')
    frozen={role:finalists['roles'][name] for role,name in (('a','v11_a'),('b','v11_b'),('h','v10_h'))}
    require(evaluation['frozen_ids']==design['plan']['frozen_ids']==frozen,'Pool diagnostic changed primary IDs')
    require(receipt['inputs']==design['inputs'] and receipt['sources']==design['sources'],'Pool diagnostic receipts disagree')
    verify_hashes(ROOT,receipt['sources']);verify_hashes(folder,receipt['artifacts_sha256'])
    for item in receipt['inputs'].values():
        require(sha(Path(item['path']))==item['sha256'],'Pool diagnostic source input changed')
    for key,count in (('full_pool_report_fidelity',12),('independent_reference_fidelity',6)):
        require(len(evaluation[key])==count and all(c['exact'] is True for c in evaluation[key]),
                'Pool diagnostic fidelity is incomplete: '+key)


def verify_reselection(meta):
    folder=BASE/'results/reselection';evaluation=read(folder/'evaluation.json');registration=read(folder/'registration.json')
    execution=read(folder/'execution_registration.json');feature=read(folder/'feature_receipt.json')
    for name,key in (('registration.json','global_registration_sha256'),('execution_registration.json','execution_registration_sha256'),
                     ('feature_receipt.json','feature_receipt_sha256')):
        require(sha(folder/name)==evaluation[key],'Reselection receipt differs: '+name)
    require(evaluation['period']==['2018-01-01',CONFIRM_END] and evaluation['no_2026_used'] is True
            and evaluation['no_return_stream_splicing'] is True and evaluation['selection_shared_across_scenarios'] is True,
            'Reselection execution scope differs')
    require(registration['scenarios']==execution['scenarios']==[list(s) for s in SCENARIOS]
            and execution['period']==[START,CONFIRM_END] and execution['chosen_once_per_fold'] is True,
            'Reselection scenario/clock registration differs')
    require(execution['global_registration_sha256']==sha(folder/'registration.json')
            and execution['schedules_sha256']==sha(folder/'schedules.json'),'Reselection execution schedule changed')
    fp=registration['fingerprints'];require(evaluation['fingerprints']==fp,'Reselection input receipts disagree')
    verify_hashes(ROOT,fp['sources'])
    paths={'original_A_registry_sha256':BASE/'registered_candidates.json',
        'original_A_registration_sha256':BASE/'registration.json',
        'A_algorithm_receipt_sha256':BASE/'results/development/registration.json',
        'B_algorithm_receipt_sha256':BASE/'results/combinations/registration.json',
        'corrected_manifest_sha256':ROOT/'v10_h_close/corrected_manifest.json',
        'qvix_manifest_sha256':ROOT/'v10_h_close/qvix_manifest.json',
        'frozen_profiles_sha256':ROOT/'v10_deep/profiles.json'}
    for key,path in paths.items():require(sha(path)==fp[key],'Reselection input changed: '+key)
    require(feature['registration_sha256']==sha(folder/'registration.json')
            and feature['feature_fingerprints']==meta['fingerprints']
            and feature['cache_array_sha256']==meta['cache_array_sha256'],'Reselection feature receipt differs')
    dates=[d for d in meta['dates'] if START<=d<=CONFIRM_END]
    require(feature['execution_data_end']==dates[-1],'Reselection exposed the wrong execution end')
    parity=evaluation['H_prefix_fidelity']
    require(parity['passed'] is True and len(parity['cases'])==16
            and all(c['exact_returns_and_holdings'] is True for c in parity['cases']),'Reselection H-prefix fidelity did not pass')
    folds=evaluation['folds']
    require([f['train_end'] for f in folds]==['2017-12-31','2019-12-31','2021-12-31','2023-12-31'],
            'Reselection training folds differ')
    for fold in folds:
        require(fold['H_control_training_parity'] is True,'Fold B changed H training path')
        for label in ('A','B'):
            training=folder/'folds'/fold['train_end'][:4]/label;selection=read(training/'selection.json')
            proof=selection['provenance'];registry=read(training/'registered_candidates.json')
            require(sha(training/'selection.json')==fold[label]['selection_sha256']==execution['fold_selection_sha256'][fold['train_end']+'/'+label],
                    'Fold selection changed')
            for name,key in (('registration.json','registration_sha256'),('registered_candidates.json','registry_sha256'),
                             ('evaluation.json','evaluation_sha256'),('paths.npz','paths_sha256')):
                require(sha(training/name)==proof[key],'Fold artifact changed: '+str(training/name))
            require(selection['selection_period']==[START,fold['train_end']]
                    and selection['chosen_id']==fold[label]['chosen_id'],'Fold selection clock/ID changed')
            validate_path_metadata(training,[c['id'] for c in registry['candidates']],
                [d for d in dates if d<=fold['train_end']],SCENARIOS,proof['paths_sha256'])
    require(set(evaluation['streams'])=={'A','B','H'},'Reselection stream coverage differs')
    for label,streams in evaluation['streams'].items():
        require(set(streams)=={s[0] for s in SCENARIOS},'Reselection pressure scenario missing')
        for scenario,value in streams.items():
            directory=folder/'continuous'/label/scenario
            for name,key in (('paths.npz','paths_sha256'),('trace.csv','trace_sha256'),('schedule.json','schedule_sha256')):
                require(sha(directory/name)==value[key],'Continuous-account artifact changed')
            metadata=read(directory/'path_metadata.json')
            require(metadata['dates']==dates and metadata['sha256']==value['paths_sha256'],'Continuous-account date/hash mismatch')
            with np.load(directory/'paths.npz',allow_pickle=False) as saved:
                require(all(saved[k].shape==(len(dates),) for k in ('returns','holdings','navs'))
                        and saved['summary'].shape==(10,),'Continuous-account path shape mismatch')


def verify_archives():
    manifest=read(BASE/'path_archives.json');rows=manifest['archives']
    expected={str(p.relative_to(BASE)) for p in (BASE/'results').rglob('*.npz')}
    require(rows and len({r['path'] for r in rows})==len(rows)
            and {r['path'] for r in rows}==expected,'Archives must cover every raw results NPZ exactly once')
    for row in rows:
        for name in (row['path'],row['archive']):
            path=Path(name)
            require(not path.is_absolute() and '..' not in path.parts and BASE.resolve() in (BASE/path).resolve().parents,
                    'Archive path escapes V11')
        raw,archive=BASE/row['path'],BASE/row['archive']
        require(sha(raw)==row['sha256'] and raw.stat().st_size==row['bytes']>0,'Raw archive input differs')
        require(sha(archive)==row['archive_sha256'],'Compressed archive hash differs')
        digest=hashlib.sha256();size=0
        with gzip.open(archive,'rb') as stream:
            for chunk in iter(lambda:stream.read(1048576),b''):
                digest.update(chunk);size+=len(chunk)
        require(size==row['bytes'] and digest.hexdigest()==row['sha256'],'Archive does not reconstruct its exact raw bytes')
    return len(rows)


def main():
    target=BASE/'profiles.json'
    if target.exists():raise ValueError('V11 profile already frozen; refusing overwrite')
    finalists=frozen_finalists();arrays,meta=build(finalists['score_specs'])
    require(meta['cache_array_sha256']==read(BASE/'results/combinations/registration.json')['feature_array_sha256'],
            'Feature bytes differ from the already registered B development cache')
    validate_confirmation_before_2026(finalists,meta)
    report=verify_final_report(finalists,meta)
    for name in ('results/reselection/evaluation.json','results/leave_one_out/evaluation.json',
                 'results/final_audit/audit.json','results/report_receipt.json','path_archives.json'):
        if not (BASE/name).is_file():raise ValueError('Research evidence incomplete: '+name)
    verify_report_receipt();verify_final_audit(report,meta);verify_leave_one_out(finalists);verify_reselection(meta)
    archived_paths_verified=verify_archives()
    configs={c['id']:c for c in finalists['candidates']};variants={}
    for key,role,name in (('a','v11_a','V11-A frozen research'),('b','v11_b','V11-B frozen research'),('h','v10_h','V10-H reference')):
        verdict=report['confirmation_verdicts'].get(role)
        variants[key]=dict(name=name,config=configs[finalists['roles'][role]],
            status='existing_control' if key=='h' else ('research_only_confirmation_passed_not_clean_oos' if verdict['passed'] else 'research_only_failed_confirmation'),
            confirmation_verdict=verdict,
            selected_by='fixed_existing_control' if key=='h' else 'frozen_development_multiscenario_selector',
            deployable_by_this_research=False)
    source_files=sorted(p for p in BASE.glob('*.py'))
    source_files+=sorted(p for p in BASE.glob('*PROTOCOL.md'))
    artifact_files=[BASE/'registered_candidates.json',BASE/'registration.json',BASE/'protected_manifest.json',
                    BASE/'path_archives.json',BASE/'REPORT.md',BASE/'README.md',BASE/'LITERATURE.md']
    artifact_files+=sorted(p for p in (BASE/'results').rglob('*') if p.is_file() and
                           p.suffix in ('.json','.csv','.md','.py','.png','.gz') and '__pycache__' not in p.parts)
    protected=json.loads((BASE/'protected_manifest.json').read_text())
    dump(target,dict(schema=1,period=[START,END],clean_oos=False,production_integration=False,
        variants=variants,score_specs=finalists['score_specs'],feature_fingerprints=meta['fingerprints'],
        feature_array_sha256=meta['cache_array_sha256'],archived_paths_verified=archived_paths_verified,
        source_sha256={str(p.relative_to(BASE)):sha(p) for p in source_files},
        dependency_sha256=protected['sha256'],artifact_sha256={str(p.relative_to(BASE)):sha(p) for p in artifact_files},
        protected_files_verified=protect(),
        limitations=['Previously examined history; confirmation is a rejection test, not new unseen evidence.',
                     'A and B remain development-selected candidates; no reselection using 2022–2026.',
                     'Research outputs are not a live account or a guarantee of lower future overfitting.']))
    print(json.dumps({key:value['status'] for key,value in variants.items()},indent=2))


if __name__=='__main__':main()
