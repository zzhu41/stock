"""Post-freeze execution and cash-amount diagnostics; no candidate selection."""
from copy import deepcopy
import hashlib
import json
import numpy as np

from .data import BASE, ROOT, START, END, SCORES, load_inputs, sha, dump, stamp, protect
from .finalists import records_and_sources
from .exante_features import build_exante, view_for_config
from .exante import run_c
from .scan import NUMERIC_FIELDS, SCENARIOS, summary_rows
from .reference import run_reference
from v10_deep.data import inputs
from v10_deep.precision import SCENARIOS as CASH_CASES, build_scenario
from v10_deep.stress import precision_features
from v11.diagnostics import event_cluster_diagnostics

OUTPUT = BASE / 'results/pressure'
FEES = (.0001, .0005, .0011, .0021)
EXECUTION = tuple(('%s_%dbp' % ('close' if lag == 0 else 'lag1', round(fee*10000)), lag, fee)
                  for lag in (0, 1) for fee in FEES)


def fingerprints():
    names = ('pressure.py', 'finalists.py', 'DIAGNOSTIC_PROTOCOL.md', 'data.py', 'exante.py',
             'exante_features.py', 'consensus_features.py', 'diagnostic_features.py', 'native.py',
             'native.cpp', 'schema.py', 'scan.py', 'reference.py')
    old = ('v10_deep/precision.py', 'v10_deep/stress.py', 'v10_deep/features.py',
           'v10_deep/reference.py', 'v10_deep/scan.py', 'v11/diagnostics.py', 'v11/features.py')
    return dict(sources={str((BASE/n).relative_to(ROOT)): sha(BASE/n) for n in names},
                dependencies={n: sha(ROOT/n) for n in old},
                inputs={str(p.relative_to(ROOT)): sha(p) for p in
                        [BASE/'inputs/manifest.json', BASE/'inputs/features.npz.gz', BASE/'inputs/features.json']+
                        [BASE/'results'/s/'selection.json' for s in ('main','consensus','exante')]})


def save_case(name, records, results, dates, registration_sha):
    directory = OUTPUT / name
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory/'paths.npz', **{scenario+'__'+key: payload[key]
                       for scenario,payload in results.items() for key in NUMERIC_FIELDS})
    dump(directory/'path_metadata.json', dict(ids=[r['id'] for r in records], dates=dates,
        sha256=sha(directory/'paths.npz'), registration_sha256=registration_sha))
    dump(directory/'evaluation.json', dict(rows=summary_rows(records,results,dates,END),
        data_case=name, period=[START,END], registration_sha256=registration_sha,
        paths_sha256=sha(directory/'paths.npz'), qualified_primary=None, diagnostic_only=True))


def check_original(records, results, dates, sources):
    checked = []
    for stage in ('main','consensus','exante'):
        directory = BASE/'results'/stage
        metadata = json.loads((directory/'path_metadata.json').read_text())
        if metadata['dates'] != dates:
            raise AssertionError('Pressure observation axis changed')
        with np.load(directory/'paths.npz', allow_pickle=False) as archive:
            # Read each matrix once, not once per row of a compressed ZIP.
            for scenario,unused_lag,unused_fee in SCENARIOS:
                for key in ('returns','holdings','summary'):
                    prior = archive[scenario+'__'+key]
                    for i,record in enumerate(records):
                        cid = record['id']
                        if sources[cid][0] != stage:
                            continue
                        j = metadata['ids'].index(cid)
                        if not np.array_equal(results[scenario][key][i],prior[j]):
                            raise AssertionError('Pressure original center changed: '+cid+'/'+scenario+'/'+key)
                        checked.append(dict(id=cid,scenario=scenario,field=key,exact=True))
    return checked


def independent_references(records,roles,arrays,meta,results,dates):
    ids = {role:roles[role] for role in ('h','balanced_a','balanced_b','balanced_c')}
    traces, comparisons, clusters = {}, [], {}
    positions = {record['id']:i for i,record in enumerate(records)}
    by_id = {record['id']:record for record in records}
    reference_paths = {}
    for role,cid in ids.items():
        config = by_id[cid]['config']
        view = view_for_config(arrays,meta,config)
        for scenario,lag,fee in SCENARIOS:
            result = run_reference(view,meta,dict(deepcopy(config),lag=lag),START,END,fee)
            if result['dates'] != dates:
                raise AssertionError('Reference observation axis changed')
            for field in ('returns','holdings','summary'):
                current = result[field]
                if not np.array_equal(current,results[scenario][field][positions[cid]]):
                    raise AssertionError('Independent reference changed: '+cid+'/'+scenario+'/'+field)
                reference_paths[role+'__'+scenario+'__'+field] = current
            traces[role+'__'+scenario] = result['trace']
            comparisons.append(dict(role=role,id=cid,scenario=scenario,observations=len(dates),exact=True))
            print('reference exact',role,scenario,flush=True)
    for scenario,unused_lag,unused_fee in SCENARIOS:
        h = results[scenario]['returns'][positions[roles['h']]]
        for role,cid in ids.items():
            # Count only an actual crash entry which filled and started its lock.
            # A same-target intent does not refresh the frozen engine's lock.
            event_dates = [row['date'] for row in traces[role+'__'+scenario]
                           if row['crash_requested'] and row['filled']]
            own = results[scenario]['returns'][positions[cid]]
            clusters[role+'__'+scenario] = event_cluster_diagnostics(own,h,dates,event_dates,
                pre=5,post=20,merge_gap=20)
    np.savez_compressed(OUTPUT/'reference_paths.npz', **reference_paths)
    dump(OUTPUT/'reference_traces.json',traces)
    dump(OUTPUT/'event_clusters.json',dict(definition='Reference crash requested and filled: actual crash entry which starts a lock; same-target intents excluded',
        pre=5,post=20,merge_gap=20,results=clusters))
    return comparisons


def evaluate():
    if (OUTPUT/'completion.json').exists():
        raise ValueError('Pressure results already frozen')
    protect()
    records,roles,sources = records_and_sources()
    arrays,meta,unused_profiles = load_inputs()
    histories,calendar,fear = inputs()
    expanded,expanded_meta,unused_map = build_exante(arrays,meta,histories)
    expected = fingerprints()
    design = dict(records=records,roles=roles,sources=sources,period=[START,END],
        execution=[list(v) for v in EXECUTION],cash_cases=['base']+list(CASH_CASES),
        cash_execution=[list(v) for v in SCENARIOS],fingerprints=expected,
        cash_meaning='Frozen point baseline, cash lower/upper intervals, official 515100 event replacement; not strategy return bounds',
        no_reselection=True,qualified_primary=None,clean_oos=False)
    registration = OUTPUT/'registration.json'
    if registration.exists():
        if json.loads(registration.read_text())['design'] != design:
            raise ValueError('Pressure registration changed')
    else:
        dump(registration,dict(registered_at=stamp(),design=design))
    reg_hash = sha(registration)
    results,dates = run_c(records,expanded,expanded_meta,scenarios=EXECUTION,workers=1)
    parity = check_original(records,results,dates,sources)
    save_case('base',records,results,dates,reg_hash)
    exact = independent_references(records,roles,expanded,expanded_meta,results,dates)
    for case in CASH_CASES:
        alternative,scenario_meta = build_scenario(case)
        dump(OUTPUT/case/'input_receipt.json',scenario_meta)
        full,full_meta = precision_features(alternative,scenario_meta,fear,calendar)
        lanes = [full_meta['score_names'].index(name) for name in SCORES]
        base = {key:np.ascontiguousarray(value[lanes] if key in ('scores','orders') else value)
                for key,value in full.items()}
        narrowed = deepcopy(full_meta);narrowed['score_names']=list(SCORES)
        alt,alt_meta,unused_mapping = build_exante(base,narrowed,alternative)
        output,observed = run_c(records,alt,alt_meta,scenarios=SCENARIOS,workers=1)
        save_case(case,records,output,observed,reg_hash)
        print('cash diagnostic complete',case,flush=True)
    if fingerprints() != expected:
        raise ValueError('Registered pressure source/input changed')
    protect()
    artifacts = {str(path.relative_to(OUTPUT)):sha(path) for path in OUTPUT.rglob('*')
                 if path.is_file() and path.name != 'completion.json'}
    dump(OUTPUT/'completion.json',dict(completed_at=stamp(),registration_sha256=reg_hash,
        original_parity=parity,independent_reference=exact,output_sha256=artifacts,
        qualified_primary=None,clean_oos=False))


if __name__ == '__main__':
    evaluate()
