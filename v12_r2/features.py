"""Pure score lanes for a finite fixed-window R2 family; no old cache writes."""
from copy import deepcopy
import numpy as np
from v12.diagnostic_features import make_view,INVALID
from v12.exante_features import view_for_config
from v12.consensus_features import RISK_ASSETS


def lane_name(window,smooth):
    if (window,smooth)==(25,1):return 'wls25_v20'
    if (window,smooth)==(20,3):return 'wls20_smooth3'
    return 'v12r2_wls%d_s%d'%(window,smooth)


def build(arrays,meta,histories,specs):
    """Append requested score definitions; preserve original features/lanes."""
    out={k:np.array(v,copy=True,order='C') for k,v in arrays.items()};m=deepcopy(meta)
    for window,smooth in sorted(set((s['window'],s['smooth']) for s in specs)):
        name=lane_name(window,smooth)
        if name in m['score_names']:continue
        v,dm,c=make_view(arrays,meta,histories,dict(score='wls25_v20',ma='ma180'),window=window,smooth=smooth,ma_window=180)
        score=v['scores'][dm['score_names'].index(c['score'])].copy()
        for ai,code in enumerate(m['assets']):
            if code not in RISK_ASSETS:score[:,ai]=INVALID
        out['scores']=np.ascontiguousarray(np.concatenate((out['scores'],score[None,:,:])))
        order=np.argsort(-score,axis=1,kind='stable').astype(np.int32)
        out['orders']=np.ascontiguousarray(np.concatenate((out['orders'],order[None,:,:])))
        m['score_names'].append(name)
    m['r2_score_specs']=[dict(window=w,smooth=s,name=lane_name(w,s)) for w,s in sorted(set((x['window'],x['smooth']) for x in specs))]
    return out,m


def effective_config(record,lag=0):
    c=deepcopy(record['config']);spec=record['feature_spec']
    c.update(score=lane_name(spec['window'],spec['smooth']),ma='ma%d'%spec['ma_window'],lag=lag)
    return c


def view(arrays,meta,record):
    return view_for_config(arrays,meta,effective_config(record))
