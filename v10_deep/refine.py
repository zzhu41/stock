"""One recorded three-mechanism extension; no new numerical thresholds."""
from copy import deepcopy
from itertools import permutations
import json
import numpy as np
from .data import BASE,dump,sha
from .combine import group
from .registry import canonical
from .schema import baseline,identifier,semantic

GROUPS=('pool','ranking','regime','crisis','rotation','entry_exit')


def groups(config):
    out=set()
    for family in config['families']:
        if family.startswith('combination_'):
            for a,b in permutations(GROUPS,2):
                if family=='combination_'+a+'_'+b:out.update((a,b))
        elif group(family):out.add(group(family))
    return out


def register():
    first=BASE/'results/mechanisms';second=BASE/'results/combinations';out=BASE/'results/refinements'
    if (out/'registered_candidates.json').exists():raise RuntimeError('Third-stage registration already exists')
    registry=json.loads((second/'registered_candidates.json').read_text());dev=json.loads((second/'development.json').read_text())
    configs={c['id']:c for c in registry['candidates']};base=canonical(baseline())
    paths=np.load(second/'selection_returns.npy',mmap_mode='r')
    parents=[]
    for metric,limit in (('cagr',60),('calmar',40)):
        ordered=sorted(dev['rows'],key=lambda r:(-r['metrics']['selection'][metric],-r['metrics']['selection']['cagr'],r['id']))
        seen=set();count=0
        for r in ordered:
            c=configs[r['id']]
            if all(c[k]==v for k,v in base.items()):continue
            key=np.round(paths[r['index']],12).tobytes()
            if key in seen:continue
            seen.add(key);parents.append(dict(id=r['id'],criterion=metric));count+=1
            if count==limit:break
    donor_records=registry['parent_choices']
    unique={c['hash']:deepcopy(c) for c in configs.values()};attempted=0
    def add(c,origins,family):
        nonlocal attempted
        c=canonical(c);h=identifier(c);attempted+=1
        if h not in unique:
            unique[h]=dict(c,id='vd_'+h[:20],hash=h,families=[family],parents=origins,stage='refinements')
    for p in parents:
        parent=configs[p['id']];pg=groups(parent)
        da={k:v for k,v in semantic(parent).items() if v!=base[k]}
        for d in donor_records:
            if d['group'] in pg or parent['id']==d['id']:continue
            donor=configs[d['id']];db={k:v for k,v in semantic(donor).items() if v!=base[k]}
            if any(da[k]!=db[k] for k in set(da)&set(db)):continue
            changes=dict(da,**db)
            if len(changes)>6:continue
            c=deepcopy(base);c.update(changes)
            add(c,[parent['id'],donor['id']],'third_mechanism_'+d['group'])
        # Explicit simplifications, not additional threshold fitting.
        for mask in (0,1,5,7):
            c=semantic(parent);c['crash_mask']=mask
            add(c,[parent['id']],'crisis_channel_simplification')
        for score in ('wls20_v20','wls25_v20','wls30_v20','blend_20_25_30'):
            c=semantic(parent);c['score']=score
            add(c,[parent['id']],'score_simplification')
    candidates=list(unique.values())
    record=dict(stage='refinements',candidates=candidates,unique_count=len(candidates),new_unique_count=len(candidates)-len(configs),
                raw_extension_attempts=attempted,parent_choices=parents,donors=donor_records,
                parent_registry_sha256=sha(second/'registered_candidates.json'),parent_development_sha256=sha(second/'development.json'),
                parent_selection_matrix_sha256=sha(second/'selection_returns.npy'),generator_sha256=sha(__file__),
                design='Top60 distinct training paths by CAGR plus Top40 by Calmar; add at most one absent group from recorded stage1 donors, max6 changed fields; explicit fixed score/channel simplifications; retain all prior candidates',
                adaptive=True,clean_oos=False,read_full_or_2026_results=False)
    dump(out/'registered_candidates.json',record)
    print('Third-stage registry:',record['new_unique_count'],'new;',len(candidates),'total',flush=True)


if __name__=='__main__':register()
