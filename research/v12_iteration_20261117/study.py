"""Bounded mechanism study. Writes exclusively beside this file; no live calls."""
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from data_bundle import load_dataset
from v10_deep.scan import metrics

SCENARIOS = [('close_1bp', 0, .0001), ('close_11bp', 0, .0011),
             ('lag1_1bp', 1, .0001), ('lag1_11bp', 1, .0011)]
BLOCKS = [('2014_2017', '2014-01-02', '2017-12-31'),
          ('2018_2020', '2018-01-01', '2020-12-31'),
          ('2021_2023', '2021-01-01', '2023-12-31')]
CANDIDATES = [
    ('baseline', {}, False),
    ('no_crash_control', {'config_changes': {'crash_mask': 0}}, False),
    ('crash_half', {'crash_weight': .5}, True),
    ('crash_75', {'crash_weight': .75}, True),
    ('cash_buffer', {'fallback_hysteresis': .02}, True),
    ('defensive_lowvol', {'defensive_lowvol': True}, True),
    ('half_cash', {'crash_weight': .5, 'fallback_hysteresis': .02}, True),
    ('half_defensive', {'crash_weight': .5, 'defensive_lowvol': True}, True),
    ('cash_defensive', {'fallback_hysteresis': .02, 'defensive_lowvol': True}, True),
    ('half_cash_defensive', {'crash_weight': .5, 'fallback_hysteresis': .02, 'defensive_lowvol': True}, True),
    ('three_quarter_defensive', {'crash_weight': .75, 'defensive_lowvol': True}, True),
    ('three_quarter_cash_defensive', {'crash_weight': .75, 'fallback_hysteresis': .02, 'defensive_lowvol': True}, True),
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(name, obj):
    def scalar(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                       allow_nan=False, default=scalar) + '\n')

def register():
    definition = dict(
        candidate_specs=[dict(name=n, spec=s, selectable=q) for n,s,q in CANDIDATES],
        development=['2014-01-02', '2023-12-31'],
        historical_validation=['2024-01-01', '2025-12-31'],
        report_only_2026=['2026-01-01', 'latest_completed'],
        postrelease_cold_start='2026-09-29', scenarios=SCENARIOS, blocks=BLOCKS,
        qualification=dict(
            broad_upgrade='Development close CAGR >= baseline, drawdown at least 1pp smaller; all common stress/block gates pass.',
            risk_efficiency='Development close CAGR >= 95% baseline and drawdown <= 90% baseline; all common stress/block gates pass.',
            common='Each close11bp and lag1_11bp CAGR >= 95% baseline and drawdown <= baseline; each of three development block CAGRs >= baseline minus 5pp in both stresses.',
            validation='2024-25 close and both stress CAGRs >= baseline minus 3pp, drawdown <= baseline plus 2pp. Validation can veto frozen nomination, never replace it.',
            nomination='Prefer broad_upgrade, then risk_efficiency. Rank passing candidates by worst stress-block CAGR excess, then close CAGR, then fewer switches. No passing candidate means no qualified upgrade.',
            diagnostic='If no qualification, separately show highest relative close Calmar among selectable candidates; never call it qualified.'),
        semantics=dict(crash_weight='Fixed risk/cash units from crash entry until next target change; no daily rebalance.',
                       cash_buffer='Fallback only: existing risk asset eligible at MOM20 >= 0; new one requires MOM20 > 2%; else cash. Regular entrants/crash rules retained.',
                       defensive='512890 may participate in bearish normal competition and fallback; retain original stock classification for crash logic.',
                       costs='Legacy-compatible turnover at pre-fee NAV, not exact post-fee self-financing transaction costs.'),
        constraints=['No original source/config/account changes.',
                     'No parameter search beyond these twelve predefined specifications.',
                     'All historical periods have previously been seen/researched; no clean out-of-sample claim.',
                     'Postrelease behavior informed hypotheses; its replay is diagnostic, not independent validation.',
                     'Latest completed closes only; no synthetic 14:50 fills or opening-price claim.',
                     'No new universe members, leverage, or short positions.'],
        clean_oos=False, deployed=False)
    path=HERE/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        assert old['definition']==json.loads(json.dumps(definition)), 'Registered protocol differs'
    else:
        dump('protocol.json',dict(registered_at=datetime.now(timezone.utc).isoformat(), definition=definition))

def summarize(result, full=False):
    dates=result['dates']; r=result['returns']
    def part(lo,hi):
        a,b=bisect_left(dates,lo),bisect_right(dates,hi)
        return metrics(r[a:b]) if b>a else None
    out=dict(total=metrics(r), switches=int(result['summary'][3]),
             blocks={n:part(a,b) for n,a,b in BLOCKS},
             yearly={y:part(y+'-01-01',y+'-12-31') for y in sorted({d[:4] for d in dates})})
    if full:
        out.update(development=part('2014-01-02','2023-12-31'),
                   validation=part('2024-01-01','2025-12-31'),
                   historical_frozen=part('2014-01-02','2026-09-24'),
                   pre_release_2026=part('2026-01-01','2026-09-24'),
                   ytd2026=part('2026-01-01', dates[-1]),
                   postrelease_continuation=part('2026-09-29', dates[-1]))
    return out

def choose(rows):
    base=rows['baseline']; checks={}; ranked={}
    for name,spec,selectable in CANDIDATES:
        if not selectable: continue
        row=rows[name]; m=row['close_1bp']['total']; b=base['close_1bp']['total']
        common={}; differences=[]
        for scenario in ['close_11bp','lag1_11bp']:
            x,y=row[scenario],base[scenario]
            common[scenario+'_cagr']=x['total']['cagr']>=.95*y['total']['cagr']
            common[scenario+'_dd']=abs(x['total']['max_dd'])<=abs(y['total']['max_dd'])+1e-12
            for block,_,_ in BLOCKS:
                gap=x['blocks'][block]['cagr']-y['blocks'][block]['cagr']
                common[scenario+'_'+block]=gap>=-.05
                differences.append(gap)
        broad=dict(common, nominal_cagr=m['cagr']>=b['cagr'],
                   nominal_dd=abs(m['max_dd'])<=abs(b['max_dd'])-.01)
        balanced=dict(common, nominal_cagr=m['cagr']>=.95*b['cagr'],
                      nominal_dd=abs(m['max_dd'])<=.9*abs(b['max_dd']))
        checks[name]=dict(broad_upgrade=broad,risk_efficiency=balanced)
        ranked[name]=dict(worst_stress_block_gap=min(differences), cagr=m['cagr'],
                          calmar=m['calmar'], switches=row['close_1bp']['switches'])
    def key(n):
        x=ranked[n]; return (-x['worst_stress_block_gap'],-x['cagr'],x['switches'],n)
    qualified={tier:sorted([n for n,c in checks.items() if all(c[tier].values())],key=key)
               for tier in ['broad_upgrade','risk_efficiency']}
    pool=qualified['broad_upgrade'] or qualified['risk_efficiency']
    diagnostic=max(ranked,key=lambda n:(ranked[n]['calmar'],ranked[n]['cagr'],n))
    return dict(nominee=pool[0] if pool else None, qualified=qualified, checks=checks,
                ranks=ranked, diagnostic_only=diagnostic, clean_oos=False,
                frozen_at=datetime.now(timezone.utc).isoformat())

def run():
    from engine import run_portfolio
    register()
    if (HERE/'selection.json').exists(): raise RuntimeError('Study already selected; outputs are sealed')
    arrays,meta,cfg=load_dataset()
    dump('run_manifest.json',dict(started_at=datetime.now(timezone.utc).isoformat(),
         input_sha256={n:sha(HERE/n) for n in ['protocol.json','dataset.npz','dataset_metadata.json',
                                             'study.py','engine.py','data_bundle.py']},
         completed_through=meta['dates'][-1]))
    rows={}; development_paths={}
    for name,spec,_ in CANDIDATES:
        rows[name]={}
        for scenario,lag,fee in SCENARIOS:
            result=run_portfolio(arrays,meta,cfg,spec,start='2014-01-02',end='2023-12-31',fee=fee,lag=lag)
            rows[name][scenario]=summarize(result)
            development_paths[name+'__'+scenario]=result['returns']
        print('DEVELOPMENT',name,rows[name]['close_1bp']['total']['cagr'],rows[name]['close_1bp']['total']['max_dd'],flush=True)
    dump('development.json', rows)
    selection=choose(rows);dump('selection.json', selection)
    print('FROZEN NOMINATION',selection['nominee'],'diagnostic',selection['diagnostic_only'],flush=True)
    full={}; paths={}; post={}; train_n=len(next(iter(development_paths.values())))
    for name,spec,_ in CANDIDATES:
        full[name]={};post[name]={}
        for scenario,lag,fee in SCENARIOS:
            result=run_portfolio(arrays,meta,cfg,spec,start='2014-01-02',fee=fee,lag=lag)
            np.testing.assert_array_equal(result['returns'][:train_n],development_paths[name+'__'+scenario])
            full[name][scenario]=summarize(result,True)
            for field in ['returns','holdings','weights']:
                paths[name+'__'+scenario+'__'+field]=result[field]
            cold=run_portfolio(arrays,meta,cfg,spec,start='2026-09-29',fee=fee,lag=lag)
            post[name][scenario]=dict(metrics=metrics(cold['returns']), switches=int(cold['summary'][3]),
                                      dates=cold['dates'], trace=cold['trace'])
        print('EVALUATED',name,'post',post[name]['close_1bp']['metrics']['total_return'],flush=True)
    validation={}
    for name,spec,selectable in CANDIDATES:
        if not selectable: continue
        gates={}
        for scenario in ['close_1bp','close_11bp','lag1_11bp']:
            m,b=full[name][scenario]['validation'],full['baseline'][scenario]['validation']
            gates[scenario+'_cagr']=m['cagr']>=b['cagr']-.03
            gates[scenario+'_dd']=abs(m['max_dd'])<=abs(b['max_dd'])+.02
        validation[name]=gates
    chosen=selection['nominee']
    verdict=dict(nominee=chosen,validation=validation,
                 validated_nominee=chosen if chosen and all(validation[chosen].values()) else None,
                 clean_oos=False,deployed=False,
                 reason='Historical mechanism study; clean future validation remains required.',
                 final_date=meta['dates'][-1],train_prefix_exact=True)
    dump('evaluation.json',dict(results=full,verdict=verdict))
    dump('postrelease.json',post)
    dump('path_metadata.json',dict(dates=result['dates'],assets=meta['assets'],scenarios=SCENARIOS))
    np.savez_compressed(HERE/'paths.npz',**paths)
    original=json.loads((HERE/'original_code_manifest.json').read_text())['files']
    changed=[name for name,h in original.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
    dump('preservation_audit.json',dict(original_source_count=len(original), changed=changed,
          all_original_sources_unchanged=not changed,checked_at=datetime.now(timezone.utc).isoformat()))
    assert not changed, changed
    print('VERDICT',verdict['validated_nominee'],flush=True)

if __name__=='__main__':
    if '--register-only' in sys.argv: register()
    else: run()
