"""Pinned V12 inputs; feature preparation never writes old research caches."""
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import subprocess
import numpy as np
from v10_deep.data import sha,dump,inputs as frozen_histories
from v10_deep.cli import load_profile

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
START,END='2014-01-02','2026-09-24'
SCORES=('wls25_v20','wls20_smooth3')


def stamp():return datetime.now(timezone.utc).isoformat()


def protect():
    document=json.loads((BASE/'protected_manifest.json').read_text())
    changed=[n for n,h in document['sha256'].items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    if changed:raise ValueError('Files outside V12 changed: '+', '.join(changed))
    return len(document['sha256'])


def freeze_protected():
    path=BASE/'protected_manifest.json'
    if path.exists():return protect()
    names=subprocess.check_output(['git','ls-files','-z'],cwd=str(ROOT)).decode().split('\0')
    # Runtime observations are allowed to advance under the existing scheduler.
    paths={n:sha(ROOT/n) for n in names if n and not n.startswith(('v12/','signals/','data/')) and (ROOT/n).is_file()}
    dump(path,dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),sha256=paths))
    return len(paths)


def profiles():
    values={name:load_profile(variant)['config'] for name,variant in [('h','growth'),('v92','v92'),('simple','simple')]}
    values['v9']=dict(deepcopy(values['v92']),crash_mask=1,global_buffer=.02,gold_buffer=.02)
    values['v91']=dict(deepcopy(values['v92']),crash_mask=1)
    return values


def prepare_inputs():
    """Copy exact registered features into a small portable V12 input archive."""
    path=BASE/'inputs/manifest.json'
    if path.exists():return load_inputs()
    frozen=json.loads((ROOT/'v10_deep/profiles.json').read_text())
    ap,mp=ROOT/'v10_deep/cache/features.npz',ROOT/'v10_deep/cache/features.json'
    for p,key in ((ap,'feature_cache_sha256'),(mp,'feature_metadata_sha256')):
        if not p.is_file() or sha(p)!=frozen[key]:
            raise ValueError('Exact frozen feature cache is required for initial V12 registration: '+str(p))
    original=json.loads(mp.read_text())
    selected=[original['score_names'].index(n) for n in SCORES]
    with np.load(ap,allow_pickle=False) as source:
        arrays={key:np.ascontiguousarray(source[key][selected] if key in ('scores','orders') else source[key]) for key in source.files}
    meta=deepcopy(original);meta['score_names']=list(SCORES)
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);raw=buffer.getvalue()
    target=BASE/'inputs/features.npz.gz';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as output:
        with gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0) as stream:stream.write(raw)
    dump(BASE/'inputs/features.json',meta)
    import hashlib
    dump(path,dict(schema=1,prepared_at=stamp(),base_array_sha256=sha(ap),base_metadata_sha256=sha(mp),
        array_sha256=hashlib.sha256(raw).hexdigest(),archive_sha256=sha(target),metadata_sha256=sha(BASE/'inputs/features.json'),
        source_sha256=sha(BASE/'data.py'),fixed_score_names=list(SCORES),
        manifests={name:sha(ROOT/name) for name in ('v10_deep/profiles.json','v10_h_close/corrected_manifest.json','v10_h_close/qvix_manifest.json')}))
    return load_inputs()


def load_inputs():
    import hashlib
    receipt=json.loads((BASE/'inputs/manifest.json').read_text())
    for name,value in receipt['manifests'].items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen data manifest changed: '+name)
    if sha(BASE/'inputs/features.npz.gz')!=receipt['archive_sha256'] or sha(BASE/'inputs/features.json')!=receipt['metadata_sha256']:
        raise ValueError('V12 input archive changed')
    with gzip.open(BASE/'inputs/features.npz.gz','rb') as f:raw=f.read()
    if hashlib.sha256(raw).hexdigest()!=receipt['array_sha256']:raise ValueError('V12 feature bytes changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
    meta=json.loads((BASE/'inputs/features.json').read_text())
    # Recheck underlying CSV bytes via immutable loaders as well as the archive.
    histories,calendar,fear=frozen_histories()
    if list(calendar)!=meta['dates'] or not np.array_equal(arrays['fear'],fear):raise ValueError('Frozen observation axis differs')
    return arrays,meta,profiles()
