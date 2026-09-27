"""Pinned existing inputs, independent V11 artifacts, no live data access."""
import hashlib
import json
from pathlib import Path
import subprocess
from copy import deepcopy
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'v11'
START,DEV_END,CONFIRM_END,END='2014-01-02','2021-12-31','2025-12-31','2026-09-24'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def freeze_protected():
    path=BASE/'protected_manifest.json'
    if path.exists():return protect()
    # Runtime files legitimately change under the scheduler. Protect immutable
    # strategy/research code and frozen datasets, not daily accounts/cache data.
    names=subprocess.check_output(['git','ls-files','-z'],cwd=str(ROOT)).decode().split('\0')
    excluded=('signals/','data/','v11/')
    files={n:sha(ROOT/n) for n in names if n and not n.startswith(excluded) and (ROOT/n).is_file()}
    dump(path,dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT)).decode().strip(),sha256=files,note='Production schedules may update runtime data; immutable tracked strategy/research inputs stay pinned.'))
    return len(files)

def protect():
    manifest=json.loads((BASE/'protected_manifest.json').read_text())
    changed=[n for n,h in manifest['sha256'].items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    if changed:raise AssertionError('Protected files changed: '+', '.join(changed))
    return len(manifest['sha256'])

def load_frozen():
    from web_v10 import _load_bundle
    from v10_deep.cli import load_profile
    from v10_deep.data import inputs
    bundle=_load_bundle()
    histories,calendar,fear=inputs()
    profiles={k:load_profile(v)['config'] for k,v in [('h','growth'),('v92','v92'),('simple','simple')]}
    arrays={k:np.array(v,copy=True) for k,v in bundle['arrays'].items()}
    meta=deepcopy(bundle['meta'])
    arrays['orders']=np.argsort(-arrays['scores'],axis=2,kind='stable').astype(np.int32)
    return dict(arrays=arrays,meta=meta,histories=histories,profiles=profiles,
                fingerprints=dict(corrected=sha(ROOT/'v10_h_close/corrected_manifest.json'),
                                  qvix=sha(ROOT/'v10_h_close/qvix_manifest.json'),
                                  frozen_profiles=sha(ROOT/'v10_deep/profiles.json')))
