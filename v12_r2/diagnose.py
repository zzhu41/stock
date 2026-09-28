"""Existing C4/C/H 2021 losses only; no new-grid candidate performance."""
from bisect import bisect_left,bisect_right
import json
import numpy as np

from .data import BASE,START,END,SCENARIOS,pin_inputs,load_base,load_npz,controls,read,dump,sha
from .features import build,view,effective_config
from v12.reference import run_reference,DatedValues,Policy
from v10_deep.scan import metrics


def reason_rows(record,result,arrays,meta):
    config=effective_config(record);data=DatedValues(arrays,meta,config);model=Policy(data,config)
    lookup={d:i for i,d in enumerate(meta['dates'])};entry=0;out=[]
    for i,row in enumerate(result['trace']):
        day=lookup[row['date']];held=result['trace'][i-1]['holding'] if i else None
        if not row['filled']:continue
        if held is None:reason='initial_entry'
        elif row['crash_requested']:reason='crash_override'
        elif row['panic']:reason='panic_exit'
        elif not data.informed(day,held):reason='held_signal_missing'
        elif data.value(day,'510300',config['ma'])<=0 and held in model.stocks:reason='bear_stock_exit'
        elif data.value(day,held,'mom20')<=0:reason='momentum_exit'
        elif data.value(day,held,'mom20')>config['overheat'] and data.value(day,held,'mom5')<=0:reason='overheat_exit'
        else:reason='healthy_rotation'
        values={k:data.value(day,held,k) if held else None for k in ('ret1','mom5','mom20','vol20')}
        values={k:(float(v) if v is not None and np.isfinite(v) else None) for k,v in values.items()}
        out.append(dict(date=row['date'],old_holding=held,new_holding=row['holding'],reason=reason,
                        old_holding_age=i-entry,panic=bool(row['panic']),crash=bool(row['crash_requested']),indicators=values))
        entry=i
    return out


def run():
    target=BASE/'diagnosis/2021.json'
    if target.exists():return json.loads(target.read_text())
    pin_inputs();arrays,meta,histories=load_base();records=controls()
    enriched,m=build(arrays,meta,histories,[r['feature_spec'] for r in records.values()])
    saved=load_npz('results/sensitivity/paths.npz');saved_meta=read('v12/results/sensitivity/path_metadata.json')
    reg=read('v12/results/sensitivity/registered_candidates.json')
    by_source={source:r['id'] for r in reg['records'] for source in r['source_ids']}
    results,trades={},{}
    for name,record in records.items():
        v=view(enriched,m,record);config=effective_config(record)
        result=run_reference(v,m,config,START,END,.0001)
        cid=record['source_id'] if name=='c4' else by_source[record['source_id']]
        j=saved_meta['ids'].index(cid)
        for key in ('returns','holdings','summary'):
            if not np.array_equal(result[key],saved['close_1bp__'+key][j]):raise AssertionError('Frozen reference changed: '+name+' '+key)
        results[name]=result;trades[name]=reason_rows(record,result,v,m)
        dump(BASE/'diagnosis'/('%s_trade_reasons.json'%name),trades[name])
    dates=results['c4']['dates'];lo,hi=bisect_left(dates,'2021-01-01'),bisect_right(dates,'2021-12-31')
    out=dict(year=2021,period=[dates[lo],dates[hi-1]],known_history=True,source='Pinned continuous frozen paths; no year reset',
             controls={},comparisons={},frozen_inputs_sha256=sha(BASE/'frozen_inputs.json'))
    for name,result in results.items():
        swaps=np.r_[False,result['holdings'][1:]!=result['holdings'][:-1]]
        fees=float(swaps[lo:hi].sum()*np.log1p(-.0002))
        logs=np.log1p(result['returns'][lo:hi])
        out['controls'][name]=dict(metrics=metrics(result['returns'][lo:hi]),switches=int(swaps[lo:hi].sum()),
            fee_log_growth=fees,market_log_growth=float(logs.sum()-fees),
            trade_reasons={reason:sum(r['date'][:4]=='2021' and r['reason']==reason for r in trades[name])
                           for reason in sorted({r['reason'] for r in trades[name]})})
    for benchmark in ('c','h'):
        ours,other=results['c4'],results[benchmark]
        differences=np.log1p(ours['returns'])-np.log1p(other['returns'])
        def detail(i):
            return dict(date=dates[i],log_excess=float(differences[i]),c4_return=float(ours['returns'][i]),
                other_return=float(other['returns'][i]),
                c4_exposure=m['assets'][ours['holdings'][i-1]] if i and ours['holdings'][i-1]>=0 else None,
                other_exposure=m['assets'][other['holdings'][i-1]] if i and other['holdings'][i-1]>=0 else None,
                c4_close_target=ours['trace'][i]['holding'],other_close_target=other['trace'][i]['holding'])
        episodes=[];begin=None
        for i in range(lo,hi+1):
            disagree=i<hi and i>0 and ours['holdings'][i-1]!=other['holdings'][i-1]
            if disagree and begin is None:begin=i
            if not disagree and begin is not None:
                end=i-1
                episodes.append(dict(start=dates[begin],end=dates[end],days=end-begin+1,
                    log_excess=float(differences[begin:end+1].sum()),
                    c4_return=float(np.prod(1+ours['returns'][begin:end+1])-1),
                    other_return=float(np.prod(1+other['returns'][begin:end+1])-1),
                    c4_trades=[r for r in trades['c4'] if dates[max(0,begin-1)]<=r['date']<=dates[end]],
                    other_trades=[r for r in trades[benchmark] if dates[max(0,begin-1)]<=r['date']<=dates[end]]))
                begin=None
        out['comparisons'][benchmark]=dict(net_log_excess=float(differences[lo:hi].sum()),
            fee_log_excess=out['controls']['c4']['fee_log_growth']-out['controls'][benchmark]['fee_log_growth'],
            worst_days=[detail(i) for i in sorted(range(lo,hi),key=lambda i:differences[i])[:12]],
            exposure_disagreement_days=sum(ours['holdings'][i-1]!=other['holdings'][i-1] for i in range(lo,hi)),
            disagreement_episodes=sorted(episodes,key=lambda e:e['log_excess']))
    dump(target,out)
    lines=['# C4的2021损失诊断（选型前）','','按2014起连续账户切片，收益归前一收盘持仓；换仓日收盘目标不是当日收益承担者。','',
           '| 模型 | 2021累计 | 2021回撤 | 换仓 | 费用log贡献 |','|---|---:|---:|---:|---:|']
    for name,x in out['controls'].items():lines.append('| %s | %.4f%% | %.4f%% | %d | %.6f |'%(name,100*x['metrics']['total_return'],100*x['metrics']['max_dd'],x['switches'],x['fee_log_growth']))
    for name,x in out['comparisons'].items():
        lines+=['','## C4相对'+name,'','净log超额 %.6f，费用log差 %.6f；有%d个观察日持仓暴露不同。'%(x['net_log_excess'],x['fee_log_excess'],x['exposure_disagreement_days']),
                '','| 不同持仓连续段 | C4累计 | 对照累计 | log差 |','|---|---:|---:|---:|']
        for ep in x['disagreement_episodes'][:6]:lines.append('| %s—%s | %.3f%% | %.3f%% | %.5f |'%(ep['start'],ep['end'],100*ep['c4_return'],100*ep['other_return'],ep['log_excess']))
    lines+=['','完整日期、前日持仓、交易原因和指标见JSON。归因不证明某一参数的独立因果效应；本轮不写2021专用分支、不删除亏损日期。','']
    (BASE/'diagnosis/REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({k:dict(return_2021=v['metrics']['total_return'],switches=v['switches']) for k,v in out['controls'].items()},ensure_ascii=False))
    return out


if __name__=='__main__':run()
