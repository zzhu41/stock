"""Mechanism-first finite stage, registered without loading strategy returns."""
from collections import Counter
from copy import deepcopy
from itertools import product
from .schema import baseline, identifier, STOCK, GLOBAL


def canonical(config):
    c=deepcopy(config)
    c['stock_pool']=sorted(c['stock_pool']);c['global_pool']=sorted(c['global_pool'])
    if c['buffer_mode']!='momentum':c.update(global_buffer=-1.,gold_buffer=-1.,trend_buffer=0.)
    if c['buffer_mode']=='rank':c['buffer']=0.
    else:c['rank_keep']=1
    if c['regime'] in ('always_bull','all_assets'):
        c.update(ma='ma250',regime_hyst=0.,regime_confirm=1,bear_entry=.07,breadth_threshold=.5)
    if c['regime']!='breadth':c['breadth_threshold']=.5
    if c['regime']=='breadth':c['regime_hyst']=0.
    if c['fallback_mode']=='cash':c['fallback_floor']=-9.9
    if not c['crash_mask']:
        b=baseline()
        for key in ('crash_lock','deep_mom','deep_below','relaxed_mom','volume_below','volume_ratio','crash_guard','crash_pick','crash_stock_only'):
            c[key]=b[key]
    if c['overheat']>=9:c['fast_mom']=5
    return c


def stage_one():
    unique={};raw=0
    def add(family,changes):
        nonlocal raw
        c=baseline();c.update(changes);c=canonical(c);raw+=1
        h=identifier(c)
        if h in unique:
            unique[h]['families']=sorted(set(unique[h]['families'])|{family})
        else:
            unique[h]=dict(c,id='vd_'+h[:20],hash=h,families=[family],parents=[],stage='mechanisms')
    add('baseline',{})
    add('baseline_v91',dict(crash_mask=1))
    add('baseline_v9',dict(crash_mask=1,global_buffer=.02,gold_buffer=.02))
    # Alternate rank estimators, each compared with exactly the baseline gates.
    for family,w,vol in product(('wls','logwls','ols'),(10,15,20,25,30,35,40,50,60,90,120),(20,40,60)):
        add('rank_estimator',dict(score='%s%d_v%d'%(family,w,vol)))
    for w in (10,15,20,25,30,35,40,50,60,90,120):
        for suffix in ('raw','r2'):add('rank_estimator',dict(score='wls%d_%s'%(w,suffix)))
    for w in (10,20,40,60,120,240):
        for suffix in ('raw','v20','v40','v60'):add('rank_estimator',dict(score='mom%d_%s'%(w,suffix)))
    for windows in ((15,25,40),(20,40,60),(10,20,40),(25,30),(20,25,30),(20,60),(40,60,120)):
        add('rank_consensus',dict(score='blend_'+'_'.join(map(str,windows))))
    for w in (20,40,60):add('rank_smoothing',dict(score='wls%d_smooth3'%w))
    anchors=('wls20_v20','wls25_v20','wls30_v20','blend_20_25_30','blend_15_25_40')
    for score in anchors:
        for mom in (10,15,20,30,40,60):add('momentum_horizon',dict(score=score,mom=mom,exit_mom=mom))
        for buffer in (0.,.01,.02,.03,.04,.06,.08):
            add('rotation_momentum',dict(score=score,buffer=buffer,global_buffer=-1.,gold_buffer=-1.))
        for buffer in (0.,1.,2.,3.,5.,8.,12.):add('rotation_score',dict(score=score,buffer_mode='score_gap',buffer=buffer))
        for buffer in (.05,.10,.20,.30,.50):add('rotation_relative_score',dict(score=score,buffer_mode='score_relative',buffer=buffer))
        for rank in (1,2,3):add('rotation_rank_retention',dict(score=score,buffer_mode='rank',rank_keep=rank))
        for days in (2,3,5,10):
            add('healthy_minimum_hold',dict(score=score,min_hold=days))
            add('healthy_rotation_confirmation',dict(score=score,switch_confirm=days))
        for threshold in (.03,.05,.08):add('healthy_trend_buffer',dict(score=score,trend_buffer=threshold))
        for exit_mom,floor in product((10,20,30,40,60),(0.,-.01,-.02,-.03,-.05)):
            add('exit_hysteresis',dict(score=score,exit_mom=exit_mom,exit_floor=floor))
        for bull,bear in product((0.,.01,.02),(.0,.03,.05,.07,.10)):
            add('absolute_entry',dict(score=score,bull_entry=bull,bear_entry=bear))
        for panic in (0.,.02,.03,.04,.05,.06,.08,.10):add('panic_threshold',dict(score=score,panic=panic))
        for multiple in (1.5,2.,2.5,3.,4.):add('panic_volatility_scaled',dict(score=score,panic_mode='volatility',panic=multiple))
        for stop in (.05,.08,.10,.15,.20):add('trailing_exit',dict(score=score,trail_stop=stop))
        for cooldown in (1,2,3,5):add('panic_reentry_cooldown',dict(score=score,panic_cooldown=cooldown))
        for threshold,fast in product((.20,.30,.40,.50,.60,9.9),(3,5,10)):
            add('overheat_exit',dict(score=score,overheat=threshold,fast_mom=fast))
        for mode in ('rank','cash','low_vol'):
            for floor in (-9.9,-.05,0.,.02):add('defensive_fallback',dict(score=score,fallback_mode=mode,fallback_floor=floor))
    for ma,confirm,hyst in product(('ma80','ma100','ma120','ma150','ma180','ma200','ma250','ma300','ma360','ema120','ema250'),(1,2,3,5),(0.,.01,.02,.03)):
        add('regime_trend',dict(ma=ma,regime_confirm=confirm,regime_hyst=hyst))
    for mode in ('open_stock','always_bull','all_assets','dual'):
        for score in anchors:add('regime_alternative',dict(regime=mode,score=score))
    for ma,breadth,confirm in product(('ma120','ma200','ma250'),(.4,.5,.6),(1,3,5)):
        add('regime_breadth',dict(regime='breadth',ma=ma,breadth_threshold=breadth,regime_confirm=confirm))
    for mask,lock in product(range(8),(0,1,2,3,5,7,10)):
        add('crash_channels_lock',dict(crash_mask=mask,crash_lock=lock))
    for guard,mask,lock in product(('always','held_negative','candidate_better','held_shock','cash_only'),(1,4,5,7),(3,5,7)):
        add('crash_override_guard',dict(crash_guard=guard,crash_mask=mask,crash_lock=lock))
    for mom,below,lock in product((-.06,-.08,-.10,-.12),(.10,.15,.20,.25,.30),(3,5,7)):
        add('crash_deep_threshold',dict(deep_mom=mom,deep_below=below,crash_lock=lock))
    for ratio,mom,below in product((1.5,2.,2.5,3.),(-.03,-.04,-.05,-.06),(.05,.10,.15,.20)):
        add('crash_volume_threshold',dict(volume_ratio=ratio,relaxed_mom=mom,volume_below=below))
    for pick,scope,mask in product(('score','deepest','least_volatile'),(0,1),(1,4,5,7)):
        add('crash_choice_scope',dict(crash_pick=pick,crash_stock_only=scope,crash_mask=mask))
    pools=[
        ('broad',('159915','510300','510500'),GLOBAL),
        ('broad_plus_small',('159915','510300','510500','563300'),GLOBAL),
        ('sector_extended',STOCK+('159928','512010','512070','512800','512880'),GLOBAL),
        ('dividend_extended',STOCK+('510880','515080','515100'),GLOBAL),
        ('global_extended',STOCK,GLOBAL+('513030','513520')),
        ('commodity_bond_extended',STOCK,GLOBAL+('159985','511010')),
        ('macro_balanced',('510300','510500'),('513100','513030','513520','159985','511010')),
    ]
    for name,stocks,globals_ in pools:
        for score,mask in product(anchors,(0,1,5,7)):
            add('economic_pool_'+name,dict(stock_pool=list(stocks),global_pool=list(globals_),score=score,crash_mask=mask))
    values=list(unique.values())
    return dict(stage='mechanisms',raw_count=raw,unique_count=len(values),candidates=values,
                family_memberships=dict(Counter(f for c in values for f in c['families'])),
                selection_period='2014-2025 known history',report_only='2026 previously seen; not clean OOS')
