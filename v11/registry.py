"""Finite mechanism families, declared without loading candidate returns."""
from collections import Counter
from copy import deepcopy
from itertools import product
import json

from .data import BASE, ROOT, dump, sha
from v10_deep.registry import canonical as legacy_canonical
from v10_deep.schema import identifier, semantic


def score_name(estimator, windows, aggregate='mean', smooth=3):
    return 'v11_%s_%s_%s_s%d' % (estimator, '_'.join(map(str, windows)), aggregate, smooth)


WINDOW_GROUPS=((10,20,40),(20,40,60),(20,61,122),(20,61,244),(61,122,244),(20,60,120))


def score_specs():
    values={}
    def add(estimator, windows, aggregate, smooth):
        name=score_name(estimator,windows,aggregate,smooth)
        values[name]=dict(name=name,estimator=estimator,windows=list(windows),aggregate=aggregate,smooth=smooth)
    for estimator,w,smooth in product(('olslog','huberlog','theilsenlog'),(15,20,30,40,60),(1,3,5)):
        add(estimator,(w,),'mean',smooth)
    for w,smooth in product((15,20,30,40,60,90,120),(1,3,5)):
        add('logwls',(w,),'mean',smooth)
    for estimator,windows,aggregate,smooth in product(('wls','logmom'),WINDOW_GROUPS,('mean','rank'),(1,3)):
        add(estimator,windows,aggregate,smooth)
    return [values[name] for name in sorted(values)]


def controls():
    payload=json.loads((ROOT/'v10_deep/profiles.json').read_text())
    return {key:semantic(payload['variants'][variant]['config']) for key,variant in
            [('h','growth'),('v92','v92'),('simple','simple')]}


def canonical(config):
    c=legacy_canonical(semantic(config))
    c.setdefault('risk_context','current20')
    if c['panic_mode']=='fixed' or c['panic']<=0:c['risk_context']='current20'
    return c


def existing_signatures():
    registry=json.loads((ROOT/'v10_deep/results/refinements/registered_candidates.json').read_text())
    return {row['hash']:row['id'] for row in registry['candidates']}


def _prior_alias(config, specs):
    c=deepcopy(config)
    if c.pop('risk_context','current20')!='current20':return None
    spec=specs.get(c['score'])
    if spec:
        if spec['aggregate']!='mean' or spec['smooth']!=1:return None
        windows=tuple(spec['windows'])
        if spec['estimator']=='logwls' and len(windows)==1:
            c['score']='logwls%d_v20'%windows[0]
        elif spec['estimator']=='wls' and windows in ((10,20,40),(20,40,60)):
            c['score']='blend_'+'_'.join(map(str,windows))
        else:return None
    return identifier(legacy_canonical(c))


def registry():
    anchors=controls();h=anchors['h'];specs={s['name']:s for s in score_specs()}
    previous=existing_signatures();values={};raw=0
    def add(family, changes=None, parent=None):
        nonlocal raw
        c=deepcopy(parent or h);c.update(changes or {});c=canonical(c);raw+=1
        digest=identifier(c)
        if digest in values:
            values[digest]['families']=sorted(set(values[digest]['families'])|{family})
        else:
            prior=previous.get(_prior_alias(c,specs))
            values[digest]=dict(c,id='v11_'+digest[:20],hash=digest,families=[family],parents=[],stage='mechanisms')
            # Keep audit metadata separate; it must not accidentally change the
            # strategy hash or be encoded as an engine parameter.
            values[digest]['audit_prior_id']=prior
    for label,c in anchors.items():add('control_'+label,parent=c)
    for spec in specs.values():
        family='robust_regression' if len(spec['windows'])==1 else 'multi_horizon_consensus'
        add(family,dict(score=spec['name']))
    fixed_scores=(h['score'],'wls25_v20',score_name('wls',(20,40,60),'mean',3),score_name('wls',(20,40,60),'rank',3))
    for score,context,multiple in product(fixed_scores,('current20','prior20','prior60','prior_max20_60'),(1.,1.25,1.5,1.75,2.,2.5)):
        add('prior_risk_scale',dict(score=score,risk_context=context,panic_mode='volatility',panic=multiple))
    for score,threshold in product(fixed_scores,(0.,.03,.04,.05,.06)):
        add('fixed_risk_scale',dict(score=score,panic_mode='fixed',panic=threshold))
    for score,mask,regime,fallback in product(fixed_scores,(0,1,7),('ma180','ma250','all_assets'),('rank','cash')):
        changes=dict(score=score,crash_mask=mask,fallback_mode=fallback)
        if regime=='all_assets':changes['regime']='all_assets'
        else:changes.update(regime='ma',ma=regime)
        add('simplified_branches',changes)
    for score,keep,hold,confirm in product(fixed_scores,(1,2,3),(0,2,3,5),(1,2,3)):
        add('healthy_rank_retention',dict(score=score,buffer_mode='rank',rank_keep=keep,min_hold=hold,switch_confirm=confirm))
    for score,buffer,hold,confirm in product(fixed_scores,(.01,.02,.03,.04),(0,2,3),(1,2,3)):
        add('healthy_momentum_retention',dict(score=score,buffer_mode='momentum',buffer=buffer,
                                             global_buffer=buffer,gold_buffer=buffer,min_hold=hold,switch_confirm=confirm))
    candidates=[];audit={}
    for digest,c in values.items():
        audit[c['id']]=dict(previously_seen=c.pop('audit_prior_id'))
        assert identifier(c)==c['hash']
        candidates.append(c)
    candidates.sort(key=lambda c:c['id'])
    control_ids={key:next(c['id'] for c in candidates if 'control_'+key in c['families']) for key in anchors}
    return dict(schema=1,stage='mechanisms',raw_count=raw,unique_count=len(candidates),
                score_specs=score_specs(),candidates=candidates,controls=control_ids,audit=audit,
                family_memberships=dict(Counter(f for c in candidates for f in c['families'])),
                development=['2014-01-02','2021-12-31'],confirmation=['2022-01-01','2025-12-31'],
                report_only=['2026-01-01','2026-09-24'],clean_oos=False,
                previous_search_scope='9558 unique V10 single-position configurations, adaptive historical research')


def register():
    path=BASE/'registered_candidates.json'
    result=registry()
    if path.exists():
        if json.loads(path.read_text())!=result:raise ValueError('Registry is already frozen; use a new explicitly registered stage')
        return result
    dump(path,result)
    return result


if __name__=='__main__':
    result=register()
    print(json.dumps({k:result[k] for k in ('raw_count','unique_count','family_memberships','controls')},indent=2))
