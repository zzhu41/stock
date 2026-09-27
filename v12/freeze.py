"""Freeze read-only exploratory profiles; never promote a failed qualification."""
import json

from .data import BASE, ROOT, START, END, sha, dump, stamp, protect
from .finalists import records_and_sources, PARENTS
from v10_deep.schema import identifier


def freeze_runtime():
    target = BASE/'profiles.json'
    if target.exists() or (BASE/'runtime_manifest.json').exists():
        raise ValueError('V12 runtime is already frozen; do not overwrite it')
    protect()
    records,roles,sources = records_and_sources()
    by_id = {r['id']:r for r in records}
    variants = {}
    for role,cid in roles.items():
        record = by_id[cid]
        if identifier(record['config']) != record['hash']:
            raise ValueError('Configuration identity differs')
        variants[role] = dict(id=cid,hash=record['hash'],config=record['config'],
            status='exploratory_failed_qualification' if role in PARENTS else 'existing_frozen_control',
            source_stages=sources[cid])
    dump(target,dict(schema=1,frozen_at=stamp(),period=[START,END],variants=variants,
        qualified_primary=None,clean_oos=False,deployed=False,
        selection_policy='DIAGNOSTIC_PROTOCOL.md freezes three descriptive references after all A/B/C qualification failures',
        goal_status='Partial historical improvements only; lower overfitting and future superiority are unproven'))
    # This closure pins model/input loaders, not unrelated production UI/bot files.
    local = ('cli.py','data.py','schema.py','reference.py','exante_features.py',
             'consensus_features.py','diagnostic_features.py')
    names = {str((BASE/name).relative_to(ROOT)) for name in local}
    prior = json.loads((ROOT/'v10_deep/profiles.json').read_text())
    names.update('v10_deep/'+name for name in prior['source_sha256'])
    names.update(prior['dependency_sha256'])
    names.update(('v10_deep/cli.py','v10_deep/reference.py','v10_deep/schema.py','v10_deep/scan.py',
                  'v10_deep/features.py','v10_deep/data.py','v11/features.py','v11/data.py'))
    artifacts = ('v12/profiles.json','v12/inputs/manifest.json','v12/inputs/features.json',
                 'v12/inputs/features.npz.gz','v10_deep/profiles.json')
    dump(BASE/'runtime_manifest.json',dict(schema=1,frozen_at=stamp(),
        source_sha256={name:sha(ROOT/name) for name in sorted(names)},
        artifact_sha256={name:sha(ROOT/name) for name in artifacts},
        scope='Actual V12/V10/V11 model and immutable input dependencies; no production UI or live-account receipts',
        qualified_primary=None))
    print('Frozen',len(variants),'read-only references; qualified primary remains null')


if __name__ == '__main__':
    freeze_runtime()
