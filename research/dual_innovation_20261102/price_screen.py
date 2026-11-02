"""Read-only market data research. Prices/returns end at 2026-10-30 close.

Latest quote is retained separately; cap at cutoff is quote cap * cutoff/quote price.
No investment recommendation is produced by the numerical gates alone.
"""
import json, time, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd
import numpy as np

BASE=Path(__file__).resolve().parent
RAW=BASE/'price_raw'; RAW.mkdir(exist_ok=True)
END='2026-10-30'

def fetch(sym):
    path=RAW/(sym+'.json')
    if path.exists(): return sym,json.loads(path.read_text())
    for attempt in range(3):
        try:
            r=requests.get('https://web.ifzq.gtimg.cn/appstock/app/fqkline/get',params={'param':sym+',day,2025-01-01,'+END+',640,qfq'},timeout=25)
            r.raise_for_status(); j=r.json()
            if sym not in j.get('data',{}): raise ValueError(str(j)[:200])
            path.write_text(json.dumps(j,ensure_ascii=False)); return sym,j
        except Exception:
            if attempt==2: raise
            time.sleep(attempt+1)

def bars(sym,j):
    n=j['data'][sym]; a=n.get('qfqday') or n.get('day') or []
    df=pd.DataFrame([x[:6] for x in a],columns=['date','open','close','high','low','volume'])
    df=df.set_index('date').apply(pd.to_numeric,errors='coerce')
    return df.loc[df.index<=END]

def main():
    rows=json.loads((BASE/'financial_raw'/'raw_exact_2026-09-30.json').read_text())
    universe={x['SECURITY_CODE']:x for x in rows if x['SECURITY_CODE'].startswith(('30','688','689'))}
    symbols=[('sz' if c.startswith('30') else 'sh')+c for c in universe]
    benchmarks=['sz399006','sh000688','sh000300']
    data={}; errors=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        tasks={ex.submit(fetch,s):s for s in benchmarks+symbols}
        for i,f in enumerate(as_completed(tasks)):
            try:
                s,j=f.result(); data[s]=j
            except Exception as e: errors.append({'symbol':tasks[f],'error':str(e)})
            if (i+1)%100==0: print('histories',i+1,'/',len(tasks),flush=True)
    frames={s:bars(s,j) for s,j in data.items()}
    out=[]
    for sym,df in frames.items():
        if len(df)<2: continue
        c=df.close; ret=c.pct_change(fill_method=None); last=c.iloc[-1]
        q=data[sym]['data'][sym].get('qt',{}).get(sym,[])
        def qnum(i):
            try:return float(q[i])
            except:return np.nan
        row={'code':sym[2:],'symbol':sym,'name':q[1] if len(q)>1 else '', 'history_rows':len(df),'date':df.index[-1],'close':last,
             'quote_price':qnum(3),'quote_time':q[30] if len(q)>30 else '',
             'cap_yi_cutoff_estimate':qnum(45)*last/qnum(3),'quote_cap_yi':qnum(45)}
        for n in [5,20,60,120,250]:
            row['ret_'+str(n)]=last/c.iloc[-n-1]-1 if len(c)>n else np.nan
            row['dd_'+str(n)]=last/c.tail(n).max()-1 if len(c)>=n else np.nan
        row['peak_close_250']=c.tail(250).max(); row['up_to_peak_250']=row['peak_close_250']/last-1
        for n in [5,10,20,60]: row['ma'+str(n)]=c.tail(n).mean()
        delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
        row['rsi14']=float((100-100/(1+gain/loss)).iloc[-1]);row['vol60']=ret.tail(60).std()*math.sqrt(244)
        row['volume5_20']=df.volume.tail(5).mean()/df.volume.tail(20).mean()
        quote_scale=qnum(37)*10000/(qnum(6)*qnum(3)) if qnum(6)>0 and qnum(3)>0 else np.nan
        row['volume_multiplier']=100 if quote_scale>10 else 1
        row['liquidity20_proxy_yi']=(df.volume*df.close*row['volume_multiplier']).tail(20).median()/1e8
        row['low5']=df.low.tail(5).min();row['high5']=df.high.tail(5).max()
        bench='sz399006' if sym.startswith('sz30') else 'sh000688'
        for label,b in [('own',bench),('star','sh000688')]:
            br=frames[b].close.pct_change(fill_method=None)
            # Align closes before returns; do not treat post-suspension multi-day
            # returns as a one-session observation. Window is 120 benchmark days.
            sr=c.reindex(frames[b].index).pct_change(fill_method=None)
            aligned=pd.concat([sr,br],axis=1,keys=['s','b']).tail(120).dropna()
            row[label+'_n120']=len(aligned)
            row[label+'_beta120']=aligned.s.cov(aligned.b)/aligned.b.var() if len(aligned)>80 else np.nan
        out.append(row)
    pd.DataFrame(out).to_csv(str(BASE/'price_metrics.csv'),index=False,encoding='utf-8-sig')
    meta={'cutoff':END,'universe_source':'Eastmoney RPT_F10_FINANCE_MAINFINADATA 2026Q3','requested_stocks':len(symbols),'returned':len(data),'errors':errors,'basis':'Tencent forward-adjusted OHLC; 250-session closing peak; Beta against own-board benchmark; quote snapshots separately dated. Cutoff capitalization inferred from snapshot capitalization and price ratio; A-equivalent total cap may differ from A+H market cap. Liquidity is an estimated median turnover proxy.'}
    (BASE/'price_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));print(json.dumps(meta,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
