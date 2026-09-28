"""Explicit frozen-model/data pins; production files are outside this study."""
from copy import deepcopy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(__file__).resolve().parent
START,END='2014-01-02','2026-09-24'
C4_ID='v12d_5e020674285fc2d9db44'
SCENARIOS=(('close_1bp',0,.0001),('close_11bp',0,.0011),('lag1_1bp',1,.0001),('lag1_11bp',1,.0011))
BLOCKS=(('early',START,'2017-12-31'),('middle','2018-01-01','2021-12-31'),('recent','2022-01-01','2025-12-31'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def scalar(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError('Unsupported JSON value: '+type(value).__name__)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False,default=scalar)+'\n')


def read(relative):return json.loads((ROOT/relative).read_text())


def frozen_paths():
    paths=['v12/'+name for name in ('native.py','native.cpp','schema.py','reference.py','diagnostic_features.py',
        'exante_features.py','consensus_features.py','inputs/manifest.json','inputs/features.npz.gz','inputs/features.json',
        'profiles.json','path_archives.json','release_receipt.json','results/sensitivity/registered_candidates.json',
        'results/sensitivity/evaluation.json','results/sensitivity/path_metadata.json',
        'results/sensitivity/paths.npz.gz','results/sensitivity/independent_smooth4/audit.json',
        'results/sensitivity/independent_smooth4/path_metadata.json','results/sensitivity/independent_smooth4/paths.npz.gz',
        'results/pressure/completion.json','results/pressure/reference_paths.npz.gz','results/pressure/reference_traces.json.gz',
        'results/pressure/base/path_metadata.json','results/pressure/base/evaluation.json','results/pressure/base/paths.npz.gz')]
    paths += ['v10_deep/'+name for name in ('data.py','features.py','reference.py','schema.py','registry.py','scan.py')]
    paths += ['v11/features.py','v11/diagnostics.py','v10_search/features.py','v10_search/data.py',
              'v10_next/data.py','v10_next/frozen/strategy.py','v10_next/frozen/metadata.py','v10_next/frozen/presets.json',
              'v10_h_close/corrected_manifest.json','v10_h_close/qvix_manifest.json','v10_h_close/snapshots/qvix50.csv']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'v10_h_close/corrected_snapshots').glob('*.csv'))]
    return sorted(paths)


def pin_inputs():
    path=BASE/'frozen_inputs.json'
    if path.exists():return verify_inputs()
    # The release seal excludes mutable runtime state. Verify just the frozen
    # artifacts used here, not its historical whole-repository protection map.
    release=read('v12/release_receipt.json')
    for name in frozen_paths():
        if name.startswith('v12/') and name[4:] in release['file_sha256']:
            if sha(ROOT/name)!=release['file_sha256'][name[4:]]:raise ValueError('Frozen release changed: '+name)
    value=dict(period=[START,END],sha256={p:sha(ROOT/p) for p in frozen_paths()},
        scope='Explicit immutable model/data/evidence only; live push maintenance is permitted',
        clean_oos=False,known_optimization_years=[2021,2026])
    dump(path,value);return value


def verify_inputs():
    value=json.loads((BASE/'frozen_inputs.json').read_text())
    for name,expected in value['sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen input/source changed: '+name)
    receipt=read('v12/inputs/manifest.json')
    for name,expected in receipt['manifests'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Original input manifest chain changed: '+name)
    manifest=read('v10_h_close/corrected_manifest.json')
    for code,item in manifest['assets'].items():
        if sha(ROOT/'v10_h_close/corrected_snapshots'/(code+'.csv'))!=item['sha256']:
            raise ValueError('Corrected history differs from its frozen manifest: '+code)
    return value


def load_npz(relative):
    """Read an archived V12 array in memory without restoring old files."""
    manifest=read('v12/path_archives.json')
    record=next(r for r in manifest['archives'] if r['path']==relative)
    path=ROOT/'v12'/relative
    raw=path.read_bytes() if path.exists() else gzip.decompress((ROOT/'v12'/record['archive']).read_bytes())
    expected=record.get('sha256',record.get('source_sha256',record.get('raw_sha256')))
    if expected is None:raise ValueError('Unknown frozen archive hash schema')
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Archived path bytes changed: '+relative)
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:return {k:z[k] for k in z.files}


def load_base():
    verify_inputs()
    receipt=read('v12/inputs/manifest.json')
    raw=gzip.decompress((ROOT/'v12/inputs/features.npz.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest()!=receipt['array_sha256']:raise ValueError('Base array bytes changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:arrays={k:np.ascontiguousarray(z[k]) for k in z.files}
    meta=read('v12/inputs/features.json');histories={}
    for code in meta['assets']:
        with (ROOT/'v10_h_close/corrected_snapshots'/(code+'.csv')).open() as f:
            histories[code]=[(r[0],float(r[1]),float(r[2]),float(r[3])) for r in csv.reader(f) if r]
    return arrays,meta,histories


def controls():
    profiles=read('v12/profiles.json')['variants']
    r=read('v12/results/sensitivity/registered_candidates.json')
    c4=next(c for c in r['records'] if c['id']==C4_ID)
    out=dict(c4=dict(config=deepcopy(c4['config']),feature_spec=deepcopy(c4['feature_spec']),source_id=C4_ID))
    for role,name,feature in (('c','balanced_c',(25,3,180)),('h','h',(20,3,180)),('simple','simple',(25,1,250))):
        out[role]=dict(config=deepcopy(profiles[name]['config']),feature_spec=dict(zip(('window','smooth','ma_window'),feature)),source_id=profiles[name]['id'])
    return out
