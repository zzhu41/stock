"""Append-only R2 audit/statistics/reporting; cannot replace frozen selection."""
import argparse
from bisect import bisect_left,bisect_right
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .data import BASE,ROOT,START,END,SCENARIOS,BLOCKS,dump,sha,verify_inputs
from .features import effective_config,view
from .research import load_results,identity,GRID,stamp
from .diagnose import reason_rows
from v12.reference import run_reference
from v11.diagnostics import white_style_test,advantage_concentration


def roles_for_report(registry,selection):
    roles=dict(registry['controls'])
    if selection['primary']:roles['primary']=selection['primary']
    if selection['balanced_reference']:roles['balanced']=selection['balanced_reference']
    roles.update(selection['exploratory'])
    # Explanatory intersection, NOT another selection tier or relaxed winner.
    joint=[r['id'] for r in registry['records'] if r['selectable'] and
           all(selection['checks'][r['id']]['balanced'][k] for k in ('full_cagr','drawdown','recovery_2021'))]
    for i,cid in enumerate(sorted(joint)):roles['main_three_gates_%d'%(i+1)]=cid
    return roles


def prepare():
    verify_inputs();registry,evaluation,selection=load_results();roles=roles_for_report(registry,selection)
    definition=dict(source_sha256=sha(Path(__file__)),dependencies={name:sha(ROOT/name) for name in
        ('v11/diagnostics.py','v12/reference.py','v10_deep/reference.py')},
        frozen_selection_sha256=sha(BASE/'results/selection.json'),frozen_paths_sha256=sha(BASE/'results/paths.npz'),
        registry_sha256=sha(BASE/'registered_candidates.json'),roles=roles,
        role_semantics='controls, predeclared qualified/exploratory results, and all records satisfying three main gates only; no promotion',
        statistics=dict(candidate_count=862,draws=2000,block_lengths=[20,60],seed=20260928,method='stationary'),
        neighborhood='Only adjacent registered numerical values; other registered risk contexts; no new model runs',
        ablation='Restore one differing C4 axis using existing registered rows only',no_reselection=True)
    path=BASE/'results/post_registration.json'
    if path.exists():
        if json.loads(path.read_text())['definition']!=definition:raise ValueError('Post-analysis definition changed')
    else:dump(path,dict(registered_at=stamp(),definition=definition))
    return registry,evaluation,selection,roles


def load_features():
    receipt=json.loads((BASE/'results/feature_receipt.json').read_text())
    if sha(BASE/'results/features.npz')!=receipt['array_sha256'] or sha(BASE/'results/feature_metadata.json')!=receipt['metadata_sha256']:
        raise ValueError('Prepared R2 features changed')
    with np.load(BASE/'results/features.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    return arrays,json.loads((BASE/'results/feature_metadata.json').read_text())


def audit():
    registry,evaluation,selection,roles=prepare();out=BASE/'results/audit'
    if (out/'receipt.json').exists():return json.loads((out/'receipt.json').read_text())
    arrays,meta=load_features();records={r['id']:r for r in registry['records']}
    pm=json.loads((BASE/'results/path_metadata.json').read_text());proofs=[];paths={};reasons={}
    with np.load(BASE/'results/paths.npz',allow_pickle=False) as saved:
        for cid in sorted(set(roles.values())):
            record=records[cid];v=view(arrays,meta,record);i=pm['ids'].index(cid)
            for name,lag,fee in SCENARIOS:
                result=run_reference(v,meta,effective_config(record,lag),START,END,fee)
                if result['dates']!=pm['dates']:raise AssertionError('Reference dates differ')
                error={}
                for field in ('returns','holdings','summary'):
                    target=saved[name+'__'+field][i]
                    if not np.array_equal(result[field],target):raise AssertionError('Reference changed: '+cid+' '+name+' '+field)
                    error[field]=0.;paths[cid+'__'+name+'__'+field]=result[field]
                proofs.append(dict(id=cid,scenario=name,observations=len(result['dates']),exact=True,max_error=error))
                if name=='close_1bp':reasons[cid]=[r for r in reason_rows(record,result,v,meta) if r['date'][:4]=='2021']
            print('R2 independent reference exact',cid,flush=True)
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'reference_paths.npz',**paths)
    dump(out/'2021_trade_reasons.json',reasons)
    receipt=dict(completed_at=stamp(),passed=True,roles=roles,cases=proofs,paths_sha256=sha(out/'reference_paths.npz'),
        source_sha256=sha(Path(__file__)),selection_unchanged_sha256=sha(BASE/'results/selection.json'),no_reselection=True)
    dump(out/'receipt.json',receipt);return receipt


def statistics():
    registry,evaluation,selection,roles=prepare();pm=json.loads((BASE/'results/path_metadata.json').read_text())
    ids=[r['id'] for r in registry['records'] if r['selectable']];positions=[pm['ids'].index(cid) for cid in ids]
    reference=pm['ids'].index(registry['controls']['c4']);out=BASE/'results/statistics';results={}
    with np.load(BASE/'results/paths.npz',allow_pickle=False) as saved:
        for name,unused,unused in SCENARIOS:
            values=saved[name+'__returns'];results[name]={}
            for block in (20,60):
                path=out/(name+'_'+str(block)+'.json')
                if path.exists():result=json.loads(path.read_text())
                else:
                    result=white_style_test(values[positions],values[reference],block_length=block,draws=2000,
                        seed=20260928,method='stationary',candidate_ids=ids,expected_candidate_count=862,batch_size=32)
                    dump(path,result)
                results[name][str(block)]=result
                print('R2 whole-family statistics',name,block,'p',result['p_value'],flush=True)
        main=saved['close_1bp__returns'];lo,hi=bisect_left(pm['dates'],'2021-01-01'),bisect_right(pm['dates'],'2021-12-31')
        concentration={cid:dict(full=advantage_concentration(main[pm['ids'].index(cid)],main[reference],pm['dates']),
            year_2021=advantage_concentration(main[pm['ids'].index(cid),lo:hi],main[reference,lo:hi],pm['dates'][lo:hi]))
            for cid in sorted(set(roles.values())-{registry['controls']['c4']})}
    dump(out/'concentration.json',concentration)
    dump(out/'receipt.json',dict(completed_at=stamp(),family_size=862,block_lengths=[20,60],draws=2000,
        p_values={name:{b:x['p_value'] for b,x in blocks.items()} for name,blocks in results.items()},
        selection_unchanged_sha256=sha(BASE/'results/selection.json'),no_reselection=True,clean_oos=False,
        limitations='Conditional full-family test on repeatedly researched history; conservative zero centering, not an overfit probability or clean OOS.'))
    return results


def neighbors(registry,record):
    records={identity(r['config'],r['feature_spec']):r['id'] for r in registry['records']}
    center=record['id'];items=[]
    def add(axis,label,c,s):
        cid=records.get(identity(c,s))
        if cid is None:raise AssertionError('Post-analysis tried an unregistered neighbor')
        if cid!=center and cid not in [x['id'] for x in items]:items.append(dict(axis=axis,label=label,id=cid))
    c,s=record['config'],record['feature_spec']
    for axis in ('window','smooth','panic','min_hold'):
        source=s if axis in s else c;levels=GRID[axis];i=levels.index(source[axis])
        for j in (i-1,i+1):
            if 0<=j<len(levels):
                nc,ns=deepcopy(c),deepcopy(s);(ns if axis in s else nc)[axis]=levels[j]
                add(axis,str(levels[j]),nc,ns)
    nc=deepcopy(c)
    uniform=nc['global_buffer']==.02 and nc['gold_buffer']==.02
    nc.update(buffer=.02,global_buffer=.03 if uniform else .02,gold_buffer=.03 if uniform else .02)
    add('buffer','original' if uniform else 'uniform2',nc,deepcopy(s))
    for context in GRID['risk_context']:
        if context!=c.get('risk_context','current20'):add('risk_context',context,dict(c,risk_context=context),deepcopy(s))
    return items


def local_diagnostics(registry,evaluation,selection,roles):
    records={r['id']:r for r in registry['records']};lookup={r['id']:r for r in evaluation['rows']}
    c4=records[registry['controls']['c4']];keys={identity(r['config'],r['feature_spec']):r['id'] for r in registry['records']}
    out={}
    # When no tier qualifies, these are explanatory predeclared extremes / the
    # disclosed three-gate intersection, not replacement selections.
    for cid in sorted(set(roles.values())-set(registry['controls'].values())):
        record=records[cid];group=neighbors(registry,record);ablations=[]
        for axis in ('window','smooth','panic','min_hold','buffer','risk_context'):
            c,s=deepcopy(record['config']),deepcopy(record['feature_spec'])
            if axis in s:s[axis]=c4['feature_spec'][axis]
            elif axis=='buffer':
                for key in ('buffer','global_buffer','gold_buffer'):c[key]=c4['config'][key]
            elif axis=='risk_context':c[axis]=c4['config'].get(axis,'current20')
            else:c[axis]=c4['config'][axis]
            changed=keys[identity(c,s)]
            if changed!=cid:ablations.append(dict(axis=axis,id=changed))
        center=lookup[cid]['scenarios'];values=[]
        for row in group+ablations:
            s=lookup[row['id']]['scenarios']
            row['delta']=dict(cagr_pp=100*(s['close_1bp']['full']['cagr']-center['close_1bp']['full']['cagr']),
                return_2021_pp=100*(s['close_1bp']['yearly']['2021']['total_return']-center['close_1bp']['yearly']['2021']['total_return']),
                return_2026_pp=100*(s['close_1bp']['yearly']['2026']['total_return']-center['close_1bp']['yearly']['2026']['total_return']))
        values=[x['delta']['cagr_pp'] for x in group]
        out[cid]=dict(neighbors=group,restore_one_C4_axis=ablations,neighbor_cagr_delta_median_pp=float(np.median(values)),
            neighbor_cagr_delta_q25_pp=float(np.percentile(values,25)),no_reselection=True)
    return out


def report():
    registry,evaluation,selection,roles=prepare();lookup={r['id']:r for r in evaluation['rows']};records={r['id']:r for r in registry['records']}
    stats=json.loads((BASE/'results/statistics/receipt.json').read_text());audit_receipt=json.loads((BASE/'results/audit/receipt.json').read_text())
    local=local_diagnostics(registry,evaluation,selection,roles);dump(BASE/'results/local_diagnostics.json',local)
    profile=dict(primary=selection['primary'],balanced_reference=selection['balanced_reference'],roles=roles,
        records={cid:records[cid] for cid in sorted(set(roles.values()))},deployed=False,clean_oos=False,
        status='No re-selection from explanatory diagnostics',selection_sha256=sha(BASE/'results/selection.json'))
    dump(BASE/'profiles.json',profile)
    import csv
    with (BASE/'results/comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['role','id','scenario','cagr','max_dd','return_2021','return_2026','switches'])
        for role,cid in roles.items():
            for name,_,_ in SCENARIOS:
                s=lookup[cid]['scenarios'][name];writer.writerow([role,cid,name,s['full']['cagr'],s['full']['max_dd'],s['yearly']['2021']['total_return'],s['yearly']['2026']['total_return'],s['switches']])
    lines=['# V12-R2：2021能改善，但未找到同时守住全部目标的升级','',
        '这轮864个固定格点、合计866条记录（862可选、4控制）已完整结束。严格层与事前公开的折衷层均为0合格；不放宽门槛，不扩网格，不把最高2021者替换为主策略。2021、2026都参与已知历史优化，没有新增样本外。','',
        '## 2021为什么只有6.04%','',
        'C4与C在2021各换仓29次，费用差为零，只在10个收益暴露日不同。四报价平滑让普通轮动略迟，继而产生不同风险退出路径：2/24才买入有色，而C在2/23已买、2/24因急跌退出；C4随后承受2/25—26有色跌幅。4/6、6/18转纳指也各迟一天。相对H还有1月创业板/沪深300仓位差。详见[选型前诊断](diagnosis/REPORT.md)。','',
        '## 固定对照与公开解释项','',
        '下列探索项均未通过本轮资格，不是替代赢家。年化/回撤为全期；2021和2026列为当年累计，2026仅至09-24。','',
        '| 角色 | 年化 | 最大回撤 | 2021累计 | 2026累计 |','|---|---:|---:|---:|---:|']
    for role,cid in roles.items():
        s=lookup[cid]['scenarios']['close_1bp'];lines.append('| %s | %.4f%% | %.4f%% | %.4f%% | %.4f%% |'%(role,100*s['full']['cagr'],100*s['full']['max_dd'],100*s['yearly']['2021']['total_return'],100*s['yearly']['2026']['total_return']))
    new=[r for r in registry['records'] if r['selectable']]
    recovery_c=sum(lookup[r['id']]['scenarios']['close_1bp']['yearly']['2021']['total_return']>=selection['balanced_target_2021'] for r in new)
    recovery_h=sum(lookup[r['id']]['scenarios']['close_1bp']['yearly']['2021']['total_return']>=selection['strict_target_2021'] for r in new)
    lines+=['','173/16条分别恢复至C/H的2021水平。仅修复某一年确实能做到，但不能据此推定整体更好。' if (recovery_c,recovery_h)==(173,16) else '\n恢复至C/H的2021水平分别%d/%d条。'%(recovery_c,recovery_h),
        '','### 主要折衷（不晋级）','']
    for role,cid in roles.items():
        if role in registry['controls']:continue
        r=records[cid];c=r['config'];sp=r['feature_spec']
        lines+=['`%s`：`%s`；WLS%d、%d报价平滑、%s×%.2f、健康持有%d日、%s缓冲。'%(role,cid,sp['window'],sp['smooth'],c.get('risk_context','current20'),c['panic'],c['min_hold'],'原2/3/3%' if c['global_buffer']==.03 else '统一2%'),'',
            '| 情景 | 年化 | 回撤 | 2021累计 | 2026累计 |','|---|---:|---:|---:|---:|']
        for name,_,_ in SCENARIOS:
            s=lookup[cid]['scenarios'][name];lines.append('| %s | %.3f%% | %.3f%% | %.3f%% | %.3f%% |'%(name,100*s['full']['cagr'],100*s['full']['max_dd'],100*s['yearly']['2021']['total_return'],100*s['yearly']['2026']['total_return']))
        lines+=['','未过折衷门槛：'+', '.join(k for k,v in selection['checks'][cid]['balanced'].items() if not v)+'。','']
    lines+=['## 执行敏感性与过拟合边界','',
        'C4的2021在主同收盘口径为6.04%，延迟一天反而为29.99%；其它改动可能把同收盘2021拉高，却损害延迟情景。这不是手续费问题，也不能拿一个年份、一个时钟恢复就宣布更稳健。实际14:50报价、溢价、冲击、整手和现金分红到账日仍未模拟。','',
        '| 全862可选族相对C4：White-style p | 20日块 | 60日块 |','|---|---:|---:|']
    for name,p in stats['p_values'].items():lines.append('| %s | %.4f | %.4f |'%(name,p['20'],p['60']))
    lines+=['','每项2000次stationary区块、同一时序抽样作用于全族；零中心最大值检验可能保守，且只条件于本轮已知历史家族，未校正此前全部自适应研究。p值不是过拟合概率，不证明未来盈利。','',
        '解释项的局部扰动全部来自已登记864格，未新跑网格外点。完整逐轴邻域及恢复C4消融见[local_diagnostics.json](results/local_diagnostics.json)，优势日集中见[concentration.json](results/statistics/concentration.json)。不从这些诊断重选。','',
        '## 复现','',
        '全部4控制×4情景与V12保存路径逐值一致。追加独立Python复算%d条角色去重情景，returns/holdings/summary全部exact。67项明确的旧模型/数据/证据保持哈希不变，没有锁死或修改生产推送。'%len(audit_receipt['cases']),
        '','- [注册协议](PROTOCOL.md) / [完整登记](registered_candidates.json) / [冻结选择](results/selection.json)',
        '- [完整日收益矩阵](results/paths.npz) / [逐角色四压力表](results/comparison.csv) / [独立复算](results/audit/receipt.json)',
        '- `python3.8 -B -m v12_r2.cli backtest c4`；primary/balanced为空时CLI明确拒绝，不自动换成探索冠军。',
        '','本轮提供的是有限搜索失败及可解释的收益交换，不发布新的合格实盘版本，不接每日推送。','']
    (BASE/'REPORT.md').write_text('\n'.join(lines))
    (BASE/'README.md').write_text('# V12-R2研究档案\n\n[报告](REPORT.md) · [事前协议](PROTOCOL.md) · [2021诊断](diagnosis/REPORT.md)\n\n严格与公开折衷均无合格升级；保留完整866条已知历史路径与解释项，生产版本不变。\n\n```bash\npython3.8 -B -m v12_r2.cli backtest c4\npython3.8 -B -m v12_r2.cli backtest c --lag 1 --fee-bp 11\npython3.8 -B -m v12_r2.cli signal h --end 2026-09-24\n```\n\n只读历史查询，不拉行情、不写账户、不下单。2026是部分年度，且已参与优化。\n')
    plot(registry,evaluation,roles)
    files={str(p.relative_to(BASE)):sha(p) for p in BASE.rglob('*') if p.is_file() and 'cache' not in p.parts and '__pycache__' not in p.parts and p.name!='completion.json'}
    dump(BASE/'completion.json',dict(completed_at=stamp(),primary=selection['primary'],balanced_reference=selection['balanced_reference'],
        file_sha256=files,no_reselection=True,deployed=False,clean_oos=False,old_inputs_verified=len(verify_inputs()['sha256'])))


def plot(registry,evaluation,roles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pm=json.loads((BASE/'results/path_metadata.json').read_text());lo=bisect_left(pm['dates'],'2021-01-01');hi=bisect_right(pm['dates'],'2021-12-31')
    with np.load(BASE/'results/paths.npz',allow_pickle=False) as z:r=z['close_1bp__returns']
    figure,axes=plt.subplots(1,2,figsize=(13,4.5))
    shown={}
    for role,cid in roles.items():
        if role in ('simple','highest_2021'):continue
        if cid in shown:continue
        shown[cid]=role;i=pm['ids'].index(cid)
        axes[0].plot(np.cumprod(1+r[i]),label=role,lw=1.1)
        axes[1].plot(np.cumprod(1+r[i,lo:hi]),label=role,lw=1.1)
    axes[0].set_yscale('log');axes[0].set_title('Full frozen history: ideal close NAV')
    axes[1].set_title('2021 contribution: includes first-day return')
    for ax in axes:ax.set_xlabel('Observed session');ax.legend(fontsize=7);ax.grid(alpha=.25)
    figure.tight_layout();figure.savefig(BASE/'results/comparison.png',dpi=150);plt.close(figure)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('audit','statistics','report','all'));a=p.parse_args()
    if a.command in ('audit','all'):audit()
    if a.command in ('statistics','all'):statistics()
    if a.command in ('report','all'):report()
