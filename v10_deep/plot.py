"""Export dated research curves and annual comparisons from frozen paths."""
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from .data import BASE


def main():
    p=BASE/'results/refinements';e=json.loads((p/'evaluation.json').read_text());selection=e['selection']
    paths=json.loads((p/'selected_paths.json').read_text());profiles=json.loads((BASE/'profiles.json').read_text())
    simple=profiles['variants']['simple']['config']['id']
    paths[simple]=json.loads((BASE/'results/mechanisms/selected_paths.json').read_text())[simple]
    curves=[(selection['controls']['v9'],'v9','#aeb7c0'),(selection['controls']['v9.1'],'v9.1','#8397ad'),
            (selection['controls']['v9.2'],'v9.2','#254c78'),(simple,'Single-change reference','#19837d'),
            (selection['primary'],'V10-H frozen candidate','#ce513d')]
    dates=[datetime.strptime(d,'%Y-%m-%d') for d in e['dates']]
    fig,(ax,dd)=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for n,label,color in curves:
        nav=np.cumprod(1+np.asarray(paths[n]['returns']))
        ax.plot(dates,nav,label=label,color=color,linewidth=1.5)
        draw=nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1
        dd.plot(dates,draw,color=color,linewidth=1)
    ax.set_yscale('log');ax.set_ylabel('Growth of 1 (log scale)');dd.set_ylabel('Drawdown');dd.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='upper left',frameon=False)
    for panel in (ax,dd):
        panel.grid(True,alpha=.15);panel.spines['top'].set_visible(False);panel.spines['right'].set_visible(False)
        panel.axvspan(datetime(2026,1,1),dates[-1],color='#edcf92',alpha=.25)
    fig.suptitle('V10-H research | Corrected data, same-close execution | 2014-2026',fontsize=13,x=.08,ha='left')
    fig.text(.08,.935,'1 bp per side, original first-session convention, instant free dividend reinvestment. No leverage.',fontsize=9,color='#555')
    fig.text(.08,.02,'Shaded: 2026 report-only (already seen). Historical improvement is not validated future outperformance.',fontsize=9,color='#555')
    fig.subplots_adjust(top=.90,bottom=.10,left=.08,right=.98,hspace=.08)
    fig.savefig(BASE/'results/comparison.png',dpi=160);fig.savefig(BASE/'results/comparison.pdf');plt.close(fig)
    rows={r['id']:r for r in e['rows']};h=selection['primary'];b=selection['controls']['v9.2'];years=sorted(rows[h]['metrics']['yearly'])
    x=np.arange(len(years));fig,ax=plt.subplots(figsize=(11,4))
    ax.bar(x-.18,[rows[b]['metrics']['yearly'][y] for y in years],.36,color='#254c78',label='v9.2')
    ax.bar(x+.18,[rows[h]['metrics']['yearly'][y] for y in years],.36,color='#ce513d',label='V10-H')
    ax.set_xticks(x);ax.set_xticklabels([y+('*' if y=='2026' else '') for y in years]);ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.set_title('Annual returns | *2026 through September 24 only')
    fig.tight_layout();fig.savefig(BASE/'results/annual.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
