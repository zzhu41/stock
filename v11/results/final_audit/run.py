"""Reproduce the final read-only V11 review from frozen result paths.

No candidate selection, parameter changes, portfolio messages or production
writes. Outputs are confined to this final_audit directory. Existing saved
returns are independently re-marked from TR quotes; only the four named frozen
policies are replayed to recover actual decision/crash traces.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

from bisect import bisect_left, bisect_right
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from v11.data import BASE, ROOT, START, END, DEV_END, CONFIRM_END, dump, sha
from v11.features import build, risk_view, ORIGINAL_RISK
from v11.diagnostics import event_cluster_diagnostics, advantage_concentration
from v10_deep.reference import run_reference, DatedValues

OUT = Path(__file__).resolve().parent
DISPLAY_ROLES = ("v9", "v91", "v92", "v10_h", "simple_reference", "v11_a", "v11_b")
TRACE_ROLES = ("v10_h", "v11_a", "v11_b", "simple_reference")
PERIODS = dict(development=(START, DEV_END), confirmation=("2022-01-01", CONFIRM_END),
               report_only_2026=("2026-01-01", END), full=(START, END))


def read(path):
    return json.loads(Path(path).read_text())


def expect(path, expected):
    if sha(path) != expected:
        raise AssertionError("Input fingerprint differs: " + str(path))


def stats(values):
    values = [float(v) for v in values]
    if not values or any(not math.isfinite(v) or v <= -1 for v in values):
        raise ValueError("Invalid saved daily returns")
    nav, peak, drawdown = 1., 1., 0.
    for value in values:
        nav *= 1 + value
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1)
    return dict(sessions=len(values), total_return=nav-1, cagr=nav**(244./len(values))-1, max_dd=drawdown)


def independent_gates(row, benchmark):
    a, b = row['scenarios'], benchmark['scenarios']
    c, h = a['close_1bp']['confirmation'], b['close_1bp']['confirmation']
    tests = dict(return_floor=c['cagr'] >= max(.30,.9*h['cagr']), main_drawdown=c['max_dd'] >= h['max_dd']-.02)
    for scenario in ('close_11bp','lag1_11bp'):
        c, h = a[scenario]['confirmation'], b[scenario]['confirmation']
        tests[scenario+'_growth'] = c['cagr'] >= h['cagr']-1e-12
        tests[scenario+'_drawdown'] = c['max_dd'] >= h['max_dd']-.02
    return dict(passed=all(tests.values()),checks=tests)


def reprice(holdings, dates, quote_index, assets, fee):
    """Independent saved-holding account: mark old leg, then switch at this close."""
    nav, previous_holding, mark = 1., -1, None
    result = []
    for i, (date, held) in enumerate(zip(dates, holdings)):
        held = int(held)
        before = nav
        old_code = assets[previous_holding] if previous_holding >= 0 else None
        new_code = assets[held] if held >= 0 else None
        old_quote = quote_index.get(old_code, {}).get(date)
        if i and previous_holding >= 0 and old_quote is not None:
            nav *= old_quote / mark
            mark = old_quote
        if held != previous_holding:
            if previous_holding >= 0 and old_quote is None:
                raise AssertionError("Saved path sells an unquoted asset: " + date)
            if new_code is None or date not in quote_index.get(new_code, {}):
                raise AssertionError("Saved path buys an unquoted asset: " + date)
            if i:
                nav *= 1-2*fee
            mark = quote_index[new_code][date]
        previous_holding = held
        result.append(nav / before - 1)
    return np.asarray(result)


def trace_details(reference, view, meta, config):
    data = DatedValues(view, meta, config)
    positions = {date:i for i,date in enumerate(meta['dates'])}
    previous_holding, previous_entry, previous_lock = None, None, -1
    traces, trades, crashes = [], [], []
    early_exits, locked_panics = [], []
    for item in reference['trace']:
        item = deepcopy(item)
        day = positions[item['date']]
        age = None if previous_entry is None else day - previous_entry
        was_locked = day < previous_lock
        item.update(previous_holding=previous_holding, holding_age_before=age, locked_before_decision=was_locked)
        if was_locked:
            if item['filled'] or item['holding'] != previous_holding:
                raise AssertionError("A saved crash lock was bypassed")
            if item['panic']:
                locked_panics.append(item['date'])
        if item['filled']:
            channels = []
            if item['crash_requested']:
                code = item['holding']
                m5, depth, volume = data.value(day,code,'mom5'), data.value(day,code,'ma250'), data.value(day,code,'volume_ratio')
                if config['crash_mask']&1 and m5 <= config['deep_mom'] and depth < -config['deep_below']:
                    channels.append('deep')
                if config['crash_mask']&2 and data.fear[day] and m5 <= config['relaxed_mom'] and depth < -.20:
                    channels.append('qvix')
                if config['crash_mask']&4 and volume >= config['volume_ratio'] and m5 <= config['relaxed_mom'] and depth < -config['volume_below']:
                    channels.append('volume')
                if not channels or item['lock_until'] != day+config['crash_lock']:
                    raise AssertionError("Filled crash trigger/lock not explained by frozen features")
                classification = 'crash_override'
            elif previous_holding is None:
                classification = 'initial_entry'
            elif item['panic']:
                classification = 'panic_exit'
            elif not data.informed(day,previous_holding):
                classification = 'cash_or_unavailable_indicator_rotation'
            else:
                bull = data.value(day,'510300',config['ma']) > 0
                momentum = data.value(day,previous_holding,'mom20')
                if not bull and previous_holding in config['stock_pool']:
                    classification = 'bear_regime_exit'
                elif momentum <= config['exit_floor']:
                    classification = 'nonpositive_momentum_exit'
                elif momentum > config['overheat'] and data.value(day,previous_holding,'mom5') <= 0:
                    classification = 'overheat_exit'
                else:
                    classification = 'healthy_rotation'
            if age is not None and age < config['min_hold']:
                if classification == 'healthy_rotation':
                    raise AssertionError("Healthy rotation bypassed the registered minimum hold")
                early_exits.append(dict(date=item['date'],age=age,reason=classification))
            trade=dict(date=item['date'],from_code=previous_holding,to_code=item['holding'],
                       holding_age_before=age,classification=classification,panic=item['panic'],
                       crash_requested=item['crash_requested'],filled=True,channels=channels,
                       switch_fee_factor=1. if not trades and item['date']==reference['dates'][0] else .9998,
                       lock_until_index=item['lock_until'])
            trades.append(trade)
            if item['crash_requested']:
                crashes.append(trade)
            previous_entry=day
        item['filled_crash']=bool(item['filled'] and item['crash_requested'])
        traces.append(item)
        previous_holding,previous_lock=item['holding'],item['lock_until']
    return dict(trace=traces,trades=trades,crashes=crashes,
                early_exits=early_exits,panic_days_inside_crash_lock=locked_panics)


def main():
    report=BASE/'results/report_2026'; confirmation=BASE/'results/confirmation'
    evaluation,registration,metadata=read(report/'evaluation.json'),read(report/'registration.json'),read(report/'path_metadata.json')
    prior_eval,prior_registration,prior_meta=read(confirmation/'evaluation.json'),read(confirmation/'registration.json'),read(confirmation/'path_metadata.json')
    dates,ids,roles=metadata['dates'],metadata['ids'],evaluation['roles']
    assert dates[0]==START and dates[-1]==END and prior_meta['dates'][-1]==CONFIRM_END
    expect(report/'registration.json',evaluation['registration_sha256']);expect(report/'paths.npz',evaluation['paths_sha256'])
    expect(report/'paths.npz',metadata['sha256'])
    expect(confirmation/'registration.json',prior_eval['registration_sha256']);expect(confirmation/'paths.npz',prior_eval['paths_sha256'])
    expect(confirmation/'paths.npz',prior_meta['sha256'])
    assert registration['finalists']==prior_registration['finalists'] and roles==prior_eval['roles']
    assert ids==prior_meta['ids']==[c['id'] for c in registration['finalists']['candidates']]
    for directory,record in ((report,evaluation),(confirmation,prior_eval)):
        for name,digest in record['sources'].items():expect(BASE/name,digest)
    configs={c['id']:c for c in registration['finalists']['candidates']}
    rows={row['id']:row for row in evaluation['rows']};prior_rows={row['id']:row for row in prior_eval['rows']}
    watched=[report/name for name in ('evaluation.json','registration.json','path_metadata.json','paths.npz')]
    watched += [confirmation/name for name in ('evaluation.json','registration.json','path_metadata.json','paths.npz')]
    watched += [Path(__file__),ROOT/'v10_deep/reference.py',ROOT/'v10_deep/native.cpp',BASE/'diagnostics.py',BASE/'features.py']
    with np.load(report/'paths.npz',allow_pickle=False) as source:
        paths={name:source[name] for name in source.files}
    with np.load(confirmation/'paths.npz',allow_pickle=False) as source:
        for name in source.files:
            if name.endswith('__returns') or name.endswith('__holdings'):
                np.testing.assert_array_equal(source[name],paths[name][:,:len(prior_meta['dates'])])
    arrays,meta=build(registration['finalists']['score_specs'])
    assert meta['fingerprints']==registration['feature_fingerprints']==prior_registration['feature_fingerprints']
    watched += [Path(meta['cache_paths']['arrays']),Path(meta['cache_paths']['metadata'])]
    manifest_path=ROOT/'v10_h_close/corrected_manifest.json';manifest=read(manifest_path);watched.append(manifest_path)
    quote_index={}
    for code in list(ORIGINAL_RISK)+['511880']:
        path=ROOT/'v10_h_close/corrected_snapshots'/(code+'.csv');expect(path,manifest['assets'][code]['sha256']);watched.append(path)
        with path.open(newline='') as stream:quote_index[code]={r[0]:float(r[2]) for r in csv.reader(stream) if r}
    fingerprints={str(path.relative_to(ROOT)):sha(path) for path in watched}
    maximum_return_error=maximum_metric_error=0.;repriced=0;metric_checks=0
    period_positions={name:(bisect_left(dates,begin),bisect_right(dates,end)) for name,(begin,end) in PERIODS.items()}
    year_positions={year:(bisect_left(dates,year+'-01-01'),bisect_right(dates,year+'-12-31')) for year in sorted({d[:4] for d in dates})}
    for scenario,unused_lag,fee in registration['scenarios']:
        for index,cid in enumerate(ids):
            saved=paths[scenario+'__returns'][index];held=paths[scenario+'__holdings'][index]
            rebuilt=reprice(held,dates,quote_index,meta['assets'],fee)
            error=float(np.max(np.abs(saved-rebuilt)));maximum_return_error=max(maximum_return_error,error)
            np.testing.assert_array_equal(saved,rebuilt);repriced+=1
            changed=np.r_[False,held[1:]!=held[:-1]]
            for period,(lo,hi) in period_positions.items():
                computed=stats(saved[lo:hi]);expected=rows[cid]['scenarios'][scenario][period]
                for key,value in computed.items():
                    error=abs(value-expected[key]);maximum_metric_error=max(maximum_metric_error,error)
                    if not np.isclose(value,expected[key],rtol=1e-11,atol=1e-12):raise AssertionError('Period metric differs: '+cid+' '+scenario+' '+period+' '+key)
                    metric_checks+=1
                assert int(changed[lo:hi].sum())==expected['switches']
            for year,(lo,hi) in year_positions.items():
                computed=stats(saved[lo:hi]);expected=rows[cid]['scenarios'][scenario]['yearly'][year]
                for key,value in computed.items():
                    error=abs(value-expected[key]);maximum_metric_error=max(maximum_metric_error,error)
                    if not np.isclose(value,expected[key],rtol=1e-11,atol=1e-12):raise AssertionError('Yearly metric differs')
                    metric_checks+=1
    independent_verdicts={role:independent_gates(rows[roles[role]],rows[roles['v10_h']]) for role in ('v11_a','v11_b')}
    assert independent_verdicts==evaluation['confirmation_verdicts']==prior_eval['confirmation_verdicts']
    trace_output,trades_output,crash_output,rule_details={},{},{},{}
    for role in TRACE_ROLES:
        cid=roles[role];config=configs[cid];index=ids.index(cid)
        view=risk_view(arrays,meta,config['risk_context'],score=config['score'])
        result=run_reference(view,meta,config,start=START,end=END,fee=.0001)
        for field in ('returns','holdings','summary'):
            np.testing.assert_array_equal(result[field],paths['close_1bp__'+field][index])
        assert result['dates']==dates
        detailed=trace_details(result,view,meta,config)
        trace_output[role]=dict(candidate_id=cid,rows=detailed['trace'])
        trades_output[role]=dict(candidate_id=cid,rows=detailed['trades'])
        crash_output[role]=detailed['crashes']
        rule_details[role]=dict(min_hold=config['min_hold'],crash_lock=config['crash_lock'],
            filled_crash_count=len(detailed['crashes']),early_exits_under_min_hold=detailed['early_exits'],
            panic_days_inside_crash_lock=detailed['panic_days_inside_crash_lock'])
        print('Independent full reference+trace verified:',role,flush=True)
    paired={};h=paths['close_1bp__returns'][ids.index(roles['v10_h'])]
    h_hold=paths['close_1bp__holdings'][ids.index(roles['v10_h'])]
    for role in ('v11_a','v11_b'):
        returns=paths['close_1bp__returns'][ids.index(roles[role])]
        holding=paths['close_1bp__holdings'][ids.index(roles[role])]
        events=sorted({trade['date'] for trade in crash_output[role]+crash_output['v10_h']})
        periods={}
        for period,(lo,hi) in period_positions.items():
            a=rows[roles[role]]['scenarios']['close_1bp'][period];b=rows[roles['v10_h']]['scenarios']['close_1bp'][period]
            excess=np.log1p(returns[lo:hi])-np.log1p(h[lo:hi]);net=float(excess.sum())
            fees=(a['switches']-b['switches'])*math.log1p(-.0002)
            order=sorted(range(lo,hi),key=lambda i:float(np.log1p(returns[i])-np.log1p(h[i])))
            def item(i):
                return dict(date=dates[i],selected_return=float(returns[i]),benchmark_return=float(h[i]),
                            log_difference=float(np.log1p(returns[i])-np.log1p(h[i])),
                            selected_exposure=meta['assets'][holding[i-1]] if i and holding[i-1]>=0 else None,
                            benchmark_exposure=meta['assets'][h_hold[i-1]] if i and h_hold[i-1]>=0 else None)
            periods[period]=dict(net_log_excess=net,fee_log_excess=fees,market_log_excess=net-fees,
                different_closing_holdings=int(np.count_nonzero(holding[lo:hi]!=h_hold[lo:hi])),
                worst_relative_days=[item(i) for i in order[:5]],best_relative_days=[item(i) for i in order[-5:][::-1]])
        cluster=event_cluster_diagnostics(returns,h,dates,events,pre=5,post=20,merge_gap=20)
        selected_dates={t['date'] for t in crash_output[role]};h_dates={t['date'] for t in crash_output['v10_h']}
        cluster['events']=[dict(date=day,selected_trigger=day in selected_dates,benchmark_trigger=day in h_dates) for day in events]
        paired[role]=dict(periods=periods,advantage_concentration=advantage_concentration(returns,h,dates),crash_event_clusters=cluster)
    comparison={role:dict(id=roles[role],scenarios=rows[roles[role]]['scenarios']) for role in DISPLAY_ROLES}
    parameter_changes={role:{key:dict(benchmark=configs[roles['v10_h']].get(key),candidate=value)
                            for key,value in configs[roles[role]].items() if key not in ('id','hash','families','parents','stage')
                            and value!=configs[roles['v10_h']].get(key)} for role in ('v11_a','v11_b')}
    outputs=[]
    for filename,value in (('reference_traces.json',trace_output),('trades.json',trades_output)):
        dump(OUT/filename,value);outputs.append(filename)
    with (OUT/'daily.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);columns=['date']
        for role in DISPLAY_ROLES:columns += [role+'_return',role+'_closing_holding']
        for role in TRACE_ROLES:columns += [role+'_filled_crash']
        writer.writerow(columns)
        for day,date in enumerate(dates):
            row=[date]
            for role in DISPLAY_ROLES:
                i=ids.index(roles[role]);held=int(paths['close_1bp__holdings'][i,day])
                row += [repr(float(paths['close_1bp__returns'][i,day])),meta['assets'][held] if held>=0 else '']
            row += [int(trace_output[role]['rows'][day]['filled_crash']) for role in TRACE_ROLES]
            writer.writerow(row)
    outputs.append('daily.csv')
    for name,expected in fingerprints.items():expect(ROOT/name,expected)
    audit=dict(completed_at=datetime.now(timezone.utc).isoformat(),period=[dates[0],dates[-1]],passed=True,
        no_reselection=True,clean_oos=False,comparison=comparison,parameter_changes_from_h=parameter_changes,
        confirmation_verdicts=independent_verdicts,
        checks=dict(quote_repriced_paths=repriced,maximum_daily_return_error=maximum_return_error,
                    recomputed_metric_values=metric_checks,maximum_metric_absolute_error=maximum_metric_error,
                    confirmation_prefix_exact=True,reference_roles=list(TRACE_ROLES),reference_days=len(dates),
                    exact_reference_returns_holdings_summary=True),
        simple_and_crash_semantics=dict(
            rule='min_hold only delays healthy ordinary ranking rotations; panic, nonpositive momentum, bear-regime, overheat exits and new crash overrides can bypass it. An existing 5-session crash lock overrides even panic until the fifth following market session.',
            code_locations=['v10_deep/reference.py:172','v10_deep/reference.py:230','v10_deep/native.cpp:92','v10_deep/native.cpp:119'],
            measured=rule_details),paired=paired,input_sha256=fingerprints,
        artifacts={name:sha(OUT/name) for name in outputs},
        findings=[
            'No accounting, fee, cutoff, prefix or reference/native implementation mismatch found in the tested frozen outputs.',
            'Both independently frozen A and B fail the unchanged 2022-2025 confirmation gates; 2026 outcomes do not replace that verdict.',
            'A combines a 3-session healthy minimum hold with tighter 1% buffers; it is not a minimum-hold-only change and does not reduce total turnover.',
            'B additionally changes ranking to the unsmoothed 10/20/40 equal-score mixture; its better 2026 outcome cannot rescue failed confirmation.',
            'The fee decomposition and crash windows describe the saved paths; they are not causal mechanism isolation or independent event samples.',
            'Simple-reference min_hold2 remains a previously known comparison; it has not been selected as a new low-overfitting V11 winner.'
        ])
    dump(OUT/'audit.json',audit)
    lines=['# V11最终独立审查','','**A、B均未通过冻结的2022–2025确认门槛；未发现能解释退化的计费、截止日或实现错配。**','',
        '所有区间沿2014开始的连续持仓路径切片，不重新建仓。没有后期选型，2026只作事后报告。','',
        '|版本|全程年化|全程最大回撤|2022–2025年化|2026截至09-24累计|全程换仓|','|---|---:|---:|---:|---:|---:|']
    for role in DISPLAY_ROLES:
        r=comparison[role]['scenarios']['close_1bp'];a,c,y=r['full'],r['confirmation'],r['report_only_2026']
        lines.append('|%s|%.2f%%|%.2f%%|%.2f%%|%.2f%%|%d|'%(role,a['cagr']*100,a['max_dd']*100,c['cagr']*100,y['total_return']*100,a['switches']))
    lines += ['', '核验：%d条保存路径按独立TR价格/持仓/费用重计，逐日收益最大误差%.3g；%d个年度/分段指标重算通过；A/B/H/simple各3097日独立reference与保存的持仓、收益、10项汇总逐值一致。'%(repriced,maximum_return_error,metric_checks),'',
        'A只比H多了健康轮动至少持有3日和统一1%缓冲。确认段换仓110次，高于H的102次；B为116次。因此不能把这组改动描述为整体减少换仓。A确认段主费年化56.86%，略低于0.9×H=57.22%的门槛；两种高费场景也落后。B确认段主费年化46.72%，收益和回撤六项门槛均未通过。','',
        'min_hold2/3不是普遍禁止提前离场：只约束健康排名轮动。急跌、趋势/体制退出、新抄底覆盖仍可提前换仓；已有抄底5日锁仓保留原优先级，期内即使panic也继续持有。实际低于min_hold的离场与锁仓内panic日期已保存在审计JSON。','',
        '抄底事件来自reference实际crash_requested且filled的轨迹。A/H、B/H分别取触发日期并集，以相邻20观察日以内合并事件簇，记录前5、簇内、后20观察日的配对对数收益。触发当天收益属于旧持仓；各簇周边窗口可能重叠，不能相加、当独立危机样本或解释为因果收益。']
    for role in ('v11_a','v11_b'):
        p=paired[role]['periods']['confirmation'];e=paired[role]['crash_event_clusters']
        lines += ['', '%s确认段净对数超额%.6f，其中额外费用贡献%.6f、持仓市场收益差%.6f；全程真实抄底并集%d日、合并%d簇。'%(role,p['net_log_excess'],p['fee_log_excess'],p['market_log_excess'],e['unique_event_dates'],e['cluster_count'])]
    lines += ['', '完整逐年/八种费用时点对照、失败门槛、真实交易和事件窗口见audit.json、reference_traces.json、trades.json、daily.csv。本审计只核实现与已知历史，不证明未来表现，也不将simple或2026较优的B改推为新赢家。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(OUT/'audit.json',flush=True)


if __name__=='__main__':main()
