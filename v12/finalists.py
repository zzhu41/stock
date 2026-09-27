"""Frozen exploratory references, explicitly not qualified production profiles."""
from copy import deepcopy
import json

from .data import BASE
from .exante import verify_stage

PARENTS = {
    'balanced_b': 'v12_bb36c1de9e534b1cf332',
    'balanced_c': 'v12_7f06d74ee08531d6cdbe',
    'balanced_a': 'v12_6ace1b82572e526d2b75',
}
STAGES = ('main', 'consensus', 'exante')


def records_and_sources(verify=True):
    records, sources, controls = {}, {}, None
    for stage in STAGES:
        folder = BASE / 'results' / stage
        path = BASE / 'registered_candidates.json' if stage == 'main' else folder / 'registered_candidates.json'
        if verify:
            registry, selection = verify_stage(folder, path)
        else:
            registry = json.loads(path.read_text())
            selection = json.loads((folder / 'selection.json').read_text())
        if selection['primary'] is not None or selection['qualified_count'] != 0:
            raise ValueError('This exploratory freeze expects all stages to have failed qualification')
        controls = controls or registry['controls']
        if registry['controls'] != controls:
            raise ValueError('Stage controls differ')
        for record in registry['candidates']:
            cid = record['id']
            records.setdefault(cid, deepcopy(record))
            sources.setdefault(cid, []).append(stage)
    roles = dict(controls, **PARENTS)
    chosen = [records[cid] for cid in roles.values()]
    if len(chosen) != len({r['id'] for r in chosen}) or any(r['kind'] != 'single' for r in chosen):
        raise ValueError('Exploratory references must be distinct single-security configurations')
    return chosen, roles, {cid: sources[cid] for cid in roles.values()}
