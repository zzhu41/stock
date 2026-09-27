"""Input/native portability checks against already frozen control paths only."""
import gzip
import io
import json
import numpy as np
from .data import BASE,ROOT,START,END,load_inputs,sha,dump,protect
from .registry import generate
from .native import Simulator

SCENARIOS=(('close_1bp',0,.0001),('close_11bp',0,.0011),('lag1_1bp',1,.0001),('lag1_11bp',1,.0011))


def controls():
    out=BASE/'results/control_fidelity.json'
    if out.exists():raise ValueError('Control fidelity already recorded')
    arrays,meta,unused=load_inputs();record=generate();mapping={c['id']:c for c in record['candidates']}
    selected=[mapping[record['controls'][name]] for name in ('v9','v91','v92','simple','h')]
    source=ROOT/'v11/results/report_2026';report=json.loads((source/'evaluation.json').read_text())
    oldmeta=json.loads((source/'path_metadata.json').read_text())
    archive=json.loads((ROOT/'v11/path_archives.json').read_text())
    item=next(row for row in archive['archives'] if row['path']=='results/report_2026/paths.npz')
    packed=ROOT/'v11'/item['archive']
    if sha(packed)!=item['archive_sha256']:raise ValueError('Frozen control archive differs')
    raw=gzip.decompress(packed.read_bytes())
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=item['sha256'] or item['sha256']!=report['paths_sha256']:
        raise ValueError('Frozen control raw bytes differ')
    cases=[];sim=Simulator(arrays,meta)
    with np.load(io.BytesIO(raw),allow_pickle=False) as previous:
        for name,lag,fee in SCENARIOS:
            result=sim.run([dict(c['config'],lag=lag) for c in selected],start=START,end=END,fee=fee,workers=1)
            if result['dates']!=oldmeta['dates']:raise AssertionError('Frozen control dates changed')
            for i,role in enumerate(('v9','v91','v92','simple_reference','v10_h')):
                j=oldmeta['ids'].index(report['roles'][role])
                for field in ('returns','holdings','summary'):
                    if not np.array_equal(result[field][i],previous[name+'__'+field][j]):
                        raise AssertionError('V12 control mismatch '+role+' '+name+' '+field)
                cases.append(dict(role=role,scenario=name,exact=True,observations=len(result['dates'])))
    dump(out,dict(passed=True,cases=cases,configurations={c['id']:c['config'] for c in selected},
        only_existing_controls=True,new_candidate_performance_computed=False,
        source_sha256={n:sha(BASE/n) for n in ('fidelity.py','data.py','native.py','native.cpp','schema.py','registry.py')},
        input_manifest_sha256=sha(BASE/'inputs/manifest.json'),frozen_control_archive_sha256=sha(packed),
        protected_files_verified=protect()))
    print('Existing-control paths match exactly:',len(cases))


if __name__=='__main__':controls()
