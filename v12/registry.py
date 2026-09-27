"""One bounded registry built without reading new performance or rankings."""
from copy import deepcopy
from itertools import product
import hashlib
import json
from .data import BASE,ROOT,dump,sha,stamp,profiles
from v10_deep.registry import canonical as old_canonical
from v10_deep.schema import semantic,identifier
from v10_deep.portfolios import ORIGINAL_STOCK,ORIGINAL_GLOBAL,EXTRA_STOCK,EXTRA_GLOBAL

POOLS={
 'original':(ORIGINAL_STOCK,ORIGINAL_GLOBAL),
 'broad_macro':(('159915','510300','510500'),('513100','513030','513520','159985','511010')),
 'domestic_macro':(('159915','510300','510500','510880'),('159985','511010')),
 'all24':(ORIGINAL_STOCK+EXTRA_STOCK,ORIGINAL_GLOBAL+EXTRA_GLOBAL),
}


def canonical(c):
    c=old_canonical(semantic(c));c['locked_panic_exit']=int(c.get('locked_panic_exit',0))
    c.pop('risk_context',None)
    return c


def key(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def prior_signatures():
    paths=[ROOT/'v10_deep/results/refinements/registered_candidates.json',ROOT/'v11/registered_candidates.json']
    paths+=sorted((ROOT/'v11/results').rglob('registered_candidates.json'))
    hashes={};sources={}
    for path in paths:
        r=json.loads(path.read_text());records=r.get('candidates',[]) if isinstance(r,dict) else r
        sources[str(path.relative_to(ROOT))]=sha(path)
        for c in records:
            if c.get('risk_context','current20')!='current20':continue
            if not all(name in c for name in ('score','stock_pool','global_pool')):continue
            config=canonical(c);config.pop('locked_panic_exit')
            hashes.setdefault(identifier(config),[]).append(c['id'])
    return hashes,sources


def generate():
    anchors=profiles();prior,prior_sources=prior_signatures();records={};controls={};raw=0
    def add_single(source,family,selectable=True):
        nonlocal raw
        raw+=1;c=canonical(source);h=identifier(c);cid='v12_'+h[:20]
        old=deepcopy(c);old.pop('locked_panic_exit')
        seen=prior.get(identifier(old),[]) if not c['locked_panic_exit'] else []
        base=canonical(anchors['h'])
        complexity=sum(c.get(k)!=v for k,v in base.items())
        if cid in records:
            records[cid]['families']=sorted(set(records[cid]['families'])|{family})
        else:
            records[cid]=dict(id=cid,hash=h,kind='single',config=c,families=[family],selectable=selectable,
                complexity=complexity,previously_seen=bool(seen),previous_ids=sorted(set(seen)))
        if not selectable:records[cid]['selectable']=False
        return cid
    for name,c in anchors.items():controls[name]=add_single(c,'existing_'+name,False)
    risks=(('fixed',.04),('volatility',1.35),('volatility',1.5),('volatility',1.65))
    for score,ma,risk,buffer,hold,mask,flag in product(('wls25_v20','wls20_smooth3'),('ma180','ma250'),
            risks,('original','uniform2'),(0,2,3),(7,5),(0,1)):
        c=dict(anchors['h'],score=score,ma=ma,panic_mode=risk[0],panic=risk[1],
               min_hold=hold,crash_mask=mask,locked_panic_exit=flag)
        if buffer=='uniform2':c.update(buffer=.02,global_buffer=.02,gold_buffer=.02)
        add_single(c,'locked_risk' if flag else 'bounded_existing_interactions')
    pool_controls={}
    for pool_name,(stocks,glob) in POOLS.items():
        if pool_name=='original':continue
        for anchor,flag in product(('h','simple'),(0,1)):
            c=dict(anchors[anchor],crash_mask=5,locked_panic_exit=flag,stock_pool=list(stocks),global_pool=list(glob))
            add_single(c,'economic_pool_'+pool_name)
        c=dict(anchors['h'],stock_pool=list(stocks),global_pool=list(glob))
        pool_controls[pool_name]=add_single(c,'matched_pool_control',False)
    allocation_controls={}
    def add_alloc(anchor,mode,selectable=True,**changes):
        nonlocal raw
        raw+=1;spec=dict(base_id=controls[anchor],mode=mode,**changes);h=key(spec);cid='v12_'+h[:20]
        records[cid]=dict(id=cid,hash=h,kind='allocation',allocation=spec,selectable=selectable,
            families=['allocation_'+mode],complexity=records[controls[anchor]]['complexity']+1,
            previously_seen=False,previous_ids=[])
        return cid
    for anchor in ('h','simple'):
        for target,context in product((.15,.20,.25,.30,.40,.50),('prior20','prior60')):
            add_alloc(anchor,'vol_target',target_vol=target,risk_context=context)
        for speed in (.5,1./3.):add_alloc(anchor,'progressive',speed=speed)
        allocation_controls[anchor]=add_alloc(anchor,'full_exposure',False)
        for weight in (.5,.75):add_alloc(anchor,'fixed_exposure',False,weight=weight)
    for code in ('510300','513100','518880'):
        spec=dict(mode='buy_hold',assets=[code]);h=key(spec);cid='v12_'+h[:20]
        records[cid]=dict(id=cid,hash=h,kind='benchmark',benchmark=spec,selectable=False,
                         families=['fixed_benchmark'],complexity=0,previously_seen=False,known_benchmark=True,previous_ids=[])
    spec=dict(mode='monthly_equal',assets=['510300','513100','518880','159985','511010']);h=key(spec)
    cid='v12_'+h[:20];records[cid]=dict(id=cid,hash=h,kind='benchmark',benchmark=spec,selectable=False,
            families=['fixed_benchmark'],complexity=0,previously_seen=False,known_benchmark=True,previous_ids=[])
    return dict(schema=1,raw_proposals=raw+4,candidates=[records[cid] for cid in sorted(records)],
        unique_count=len(records),controls=controls,pool_controls=pool_controls,allocation_controls=allocation_controls,
        pools={name:[list(s),list(g)] for name,(s,g) in POOLS.items()},prior_registry_sha256=prior_sources,
        known_history_including_2026=True,new_performance_read=False)


def register():
    path=BASE/'registered_candidates.json'
    if path.exists():
        receipt=json.loads((BASE/'registration.json').read_text())
        if receipt['registry_sha256']!=sha(path) or receipt['protocol_sha256']!=sha(BASE/'PROTOCOL.md'):
            raise ValueError('V12 registry/protocol changed after registration')
        return json.loads(path.read_text())
    record=generate();dump(path,record)
    dump(BASE/'registration.json',dict(registered_at=stamp(),protocol_sha256=sha(BASE/'PROTOCOL.md'),
        registry_sha256=sha(path),source_sha256=sha(BASE/'registry.py'),
        count=record['unique_count'],raw_proposals=record['raw_proposals'],known_history_including_2026=True))
    return record
