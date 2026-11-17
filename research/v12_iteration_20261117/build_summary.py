"""Make portable research tables/figures from sealed study outputs."""
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LABELS={
 'baseline': 'V12-R2', 'no_crash_control':'No dip buying',
 'crash_half':'50% dip allocation', 'crash_75':'75% dip allocation',
 'cash_buffer':'Cash buffer', 'defensive_lowvol':'Low-vol defensive access',
 'half_cash':'50% dip + cash', 'half_defensive':'50% dip + defensive',
 'cash_defensive':'Cash + defensive', 'half_cash_defensive':'50% dip + cash + defensive',
 'three_quarter_defensive':'75% dip + defensive',
 'three_quarter_cash_defensive':'75% dip + cash + defensive'}

def read(name):return json.loads((HERE/name).read_text())

def main():
    evaluation=read('evaluation.json');results=evaluation['results'];post=read('postrelease.json')
    rows=[]
    for name,scenarios in results.items():
        for scenario,s in scenarios.items():
            for period in ['development','validation','historical_frozen','pre_release_2026','ytd2026','total']:
                m=s[period]
                rows.append(dict(candidate=name,scenario=scenario,period=period,**m))
            rows.append(dict(candidate=name,scenario=scenario,period='postrelease_cold_start',
                             **post[name][scenario]['metrics']))
    with (HERE/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    names=['baseline','crash_75','crash_half','cash_buffer','defensive_lowvol','half_defensive','cash_defensive']
    fig,axes=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1,1]})
    colors=['#174b75' if n=='baseline' else '#bf623c' if n=='defensive_lowvol' else '#7b949e' for n in names]
    y=np.arange(len(names))
    axes[0].barh(y,[100*results[n]['close_1bp']['historical_frozen']['cagr'] for n in names],color=colors)
    axes[0].set_yticks(y);axes[0].set_yticklabels([LABELS[n] for n in names]);axes[0].invert_yaxis()
    axes[0].set_title('2014-01-02 to 2026-09-24');axes[0].set_xlabel('Annualized return (%)')
    for i,n in enumerate(names):
        m=results[n]['close_1bp']['historical_frozen'];axes[0].text(100*m['cagr']+.5,i,f"{100*m['cagr']:.1f}%",va='center',fontsize=9)
    axes[0].set_xlim(0,65)
    values=[100*post[n]['close_1bp']['metrics']['total_return'] for n in names]
    axes[1].barh(y,values,color=colors);axes[1].set_yticks(y);axes[1].set_yticklabels([]);axes[1].invert_yaxis()
    axes[1].set_title('2026-09-29 to 2026-11-16');axes[1].set_xlabel('Cumulative return (%)')
    axes[1].axvline(0,color='#555',linewidth=.8);axes[1].set_xlim(-8,8)
    for i,v in enumerate(values):axes[1].text(v+(.2 if v>=0 else -.2),i,f'{v:+.2f}%',va='center',ha='right' if v<0 else 'left',fontsize=9)
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('V12 mechanism study: recent gains do not establish a stronger strategy',fontsize=13)
    fig.text(.5,.015,'Completed TR closes; ideal close fills; 1 bp per side. All periods are previously seen research data.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(HERE/'comparison.png',dpi=160);plt.close(fig)
    summary=dict(status='no_qualified_upgrade',nominee=evaluation['verdict']['nominee'],
                 validated_nominee=evaluation['verdict']['validated_nominee'],
                 independent_validation=read('validation_extra.json'),
                 interpretation=[
                   'Only recent-period results favor bearish low-vol defensive access; 2024-2025 validation materially deteriorates.',
                   'Lower dip allocation reduces historical CAGR substantially while the overall maximum drawdown improves little.',
                   'Cash entry/exit buffers improve some drawdowns but lose too much return under the registered criteria.',
                   'No expanded parameter search was performed after failure; no production upgrade is authorized by these results.'],
                 research_cli='python3.8 -B research/v12_iteration_20261117/candidate.py --variant defensive_lowvol --start 2026-09-29',
                 methodological_reference='https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf',
                 clean_oos=False,production_modified=False)
    (HERE/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    original=read('original_code_manifest.json')['files']
    changed=[n for n,h in original.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    assert not changed, changed
    (HERE/'preservation_audit.json').write_text(json.dumps(dict(original_source_count=len(original),
        changed=changed,all_original_sources_unchanged=True),indent=2)+'\n')
    print('Wrote comparison.csv, comparison.png, summary.json; all',len(original),'original source files unchanged')

if __name__=='__main__':main()
