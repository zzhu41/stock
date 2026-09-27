"""Adaptive second stage, using only recorded pre-2026 first-stage diagnostics.

At most two mechanism groups are combined. This is a declared in-sample
extension, not a fresh validation set or an unseen family-level test.
"""
from copy import deepcopy
from itertools import combinations
import json
import numpy as np
from .data import BASE, dump, sha
from .registry import canonical
from .schema import baseline, identifier, semantic


def group(family):
    if family.startswith('economic_pool_'):return 'pool'
    if family.startswith('rank_'):return 'ranking'
    if family.startswith('regime_'):return 'regime'
    if family.startswith('crash_'):return 'crisis'
    if family.startswith(('rotation_','healthy_')):return 'rotation'
    if family.startswith(('panic_','trailing_','overheat_','exit_','absolute_','defensive_','momentum_')):return 'entry_exit'
    return None


def register():
    source=BASE/'results/mechanisms';target=BASE/'results/combinations'
    if (target/'registered_candidates.json').exists():raise RuntimeError('Combination design already registered')
    first=json.loads((source/'registered_candidates.json').read_text());dev=json.loads((source/'development.json').read_text())
    configs={c['id']:c for c in first['candidates']};rows={r['id']:r for r in dev['rows']}
    values=np.load(source/'selection_returns.npy',mmap_mode='r')
    base=canonical(baseline())
    families=sorted({f for c in configs.values() for f in c['families'] if group(f)})
    chosen=[]
    for family in families:
        members=[r for r in dev['rows'] if family in configs[r['id']]['families']]
        orders=[sorted(members,key=lambda r:(-r['metrics']['selection']['cagr'],r['selection_switches'],r['id'])),
                sorted(members,key=lambda r:(-r['metrics']['selection']['calmar'],-r['metrics']['selection']['cagr'],r['id']))]
        for label,ordered in zip(('return','calmar'),orders):
            seen=set();count=0
            for r in ordered:
                c=configs[r['id']]
                if all(c[k]==v for k,v in base.items()):continue
                key=np.round(values[r['index']],12).tobytes()
                if key in seen:continue
                seen.add(key);chosen.append(dict(id=c['id'],family=family,group=group(family),criterion=label));count+=1
                if count==3:break
    unique={c['hash']:deepcopy(c) for c in configs.values()};attempted=0;added=0
    for a,b in combinations(chosen,2):
        if a['group']==b['group'] or a['id']==b['id']:continue
        ca,cb=configs[a['id']],configs[b['id']]
        da={k:v for k,v in semantic(ca).items() if v!=base[k]};db={k:v for k,v in semantic(cb).items() if v!=base[k]}
        if any(da[k]!=db[k] for k in set(da)&set(db)):continue
        changes=dict(da,**db)
        if len(changes)>6:continue
        c=deepcopy(base);c.update(changes);c=canonical(c);h=identifier(c);attempted+=1
        if h in unique:continue
        unique[h]=dict(c,id='vd_'+h[:20],hash=h,families=['combination_'+a['group']+'_'+b['group']],
                       parents=[a['id'],b['id']],stage='combinations');added+=1
    result=dict(stage='combinations',candidates=list(unique.values()),unique_count=len(unique),new_unique_count=added,
                raw_pair_attempts=attempted,parent_choices=chosen,parent_stage='mechanisms',
                parent_registry_sha256=sha(source/'registered_candidates.json'),parent_development_sha256=sha(source/'development.json'),
                parent_selection_matrix_sha256=sha(source/'selection_returns.npy'),
                generator_sha256=sha(__file__),
                design='Per family top 3 distinct training paths by CAGR and top 3 by Calmar; combine two different groups with compatible fields, at most 6 changed fields; retain entire first stage',
                clean_oos=False,adaptive=True,read_full_or_2026_results=False)
    dump(target/'registered_candidates.json',result)
    print('Combination registry:',added,'new,',len(unique),'total; parent choices',len(chosen),flush=True)


if __name__=='__main__':register()
