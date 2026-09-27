"""Render frozen numeric evidence, without changing any selected configuration."""
import csv
import json
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from .data import BASE, sha, dump, stamp
from .finalists import PARENTS

LABELS = {'v9':'V9','v91':'V9.1','v92':'V9.2','simple':'V9.2+','h':'V10-H',
          'balanced_a':'V12-A 探索','balanced_b':'V12-B 探索','balanced_c':'V12-C 探索'}
ORDER = ('v9','v91','v92','simple','h','balanced_a','balanced_b','balanced_c')
SCENARIOS = ('close_1bp','close_5bp','close_11bp','close_21bp','lag1_1bp','lag1_5bp','lag1_11bp','lag1_21bp')


def read(path):
    return json.loads((BASE/path).read_text())


def pct(value):
    return '%.2f%%' % (100*value)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def csv_write(path,headers,rows):
    with (BASE/path).open('w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream);writer.writerow(headers);writer.writerows(rows)


def render():
    document = read('results/pressure/base/evaluation.json')
    completion = read('results/pressure/completion.json')
    for name,expected in completion['output_sha256'].items():
        if sha(BASE/'results/pressure'/name) != expected:
            raise ValueError('Pressure evidence changed: '+name)
    profiles = read('profiles.json')['variants']
    rows = {r['id']:r for r in document['rows']}
    by_role = {role:rows[profiles[role]['id']]['scenarios'] for role in ORDER}
    full_headers = ('策略','全期累计收益','全期年化','最大回撤','2026累计收益')
    full = [(LABELS[role],pct(v['full']['total_return']),pct(v['full']['cagr']),
             pct(v['full']['max_dd']),pct(v['tail']['total_return']))
            for role in ORDER for v in [by_role[role]['close_1bp']]]
    csv_write('results/full_comparison.csv',('role','id','total_return','cagr','max_dd','return_2026'),
        [(role,profiles[role]['id'],v['full']['total_return'],v['full']['cagr'],v['full']['max_dd'],v['tail']['total_return'])
         for role in ORDER for v in [by_role[role]['close_1bp']]])
    years = sorted(by_role['h']['close_1bp']['yearly'])
    annual = [(year+'*' if year == '2026' else year,)+tuple(pct(by_role[role]['close_1bp']['yearly'][year]['total_return']) for role in ORDER)
              for year in years]
    csv_write('results/annual_returns.csv',('year',)+ORDER,
        [(year,)+tuple(by_role[role]['close_1bp']['yearly'][year]['total_return'] for role in ORDER) for year in years])
    csv_write('results/yearly_metrics.csv',('role','year','start','end','sessions','cumulative_return','annualized_244','max_dd'),
        [(role,year,v['start'],v['end'],v['sessions'],v['total_return'],v['cagr'],v['max_dd'])
         for role in ORDER for year,v in by_role[role]['close_1bp']['yearly'].items()])
    csv_write('results/execution_stress.csv',('role','scenario','cagr','max_dd','return_2026','turnover_equivalent'),
        [(role,name,v['full']['cagr'],v['full']['max_dd'],v['tail']['total_return'],v['turnover_equivalent'])
         for role in ORDER for name,v in by_role[role].items()])
    stress = [(name,)+tuple(pct(by_role[role][name]['full']['cagr'])+' / '+pct(by_role[role][name]['full']['max_dd'])
                            for role in ('h','balanced_a','balanced_b','balanced_c')) for name in SCENARIOS]
    cash_csv,cash_table = [],[]
    for case in ('base','cash_lower','cash_upper','official_515100'):
        case_rows = {r['id']:r for r in read('results/pressure/'+case+'/evaluation.json')['rows']}
        cash_table.append((case,)+tuple(pct(case_rows[profiles[role]['id']]['scenarios']['close_1bp']['full']['cagr'])
                          for role in ('h','balanced_a','balanced_b','balanced_c')))
        for role in ORDER:
            for name,v in case_rows[profiles[role]['id']]['scenarios'].items():
                cash_csv.append((case,role,name,v['full']['cagr'],v['full']['max_dd'],v['tail']['total_return']))
    csv_write('results/cash_precision.csv',('cash_scenario','role','execution','cagr','max_dd','return_2026'),cash_csv)
    tests = []
    for name in ('close_1bp','close_11bp','lag1_1bp','lag1_11bp'):
        values = read('results/family_diagnostics/'+name+'.json')['tests']
        tests.append((name,'%.6f' % values['20']['p_value'],'%.6f' % values['60']['p_value']))
    leaders,checks = [],{}
    for stage in ('main','consensus','exante'):
        choice = read('results/'+stage+'/selection.json')
        if choice['primary'] is not None or choice['qualified_count'] != 0:
            raise ValueError('Frozen no-qualification status changed')
        evaluation = {r['id']:r for r in read('results/'+stage+'/evaluation.json')['rows']}
        for label in ('top_return','top_tail','least_drawdown'):
            cid = choice[label];v = evaluation[cid]['scenarios']['close_1bp']
            leaders.append((stage,label,cid,pct(v['full']['cagr']),pct(v['full']['max_dd']),pct(v['tail']['total_return'])))
        for role,cid in PARENTS.items():
            if cid in choice['checks']:
                checks.setdefault(role,dict(stage=stage,id=cid,checks=choice['checks'][cid]))
    dump(BASE/'results/final_qualification.json',dict(qualified_primary=None,parents=checks,
        all_stages_qualified_count=0,no_reselection=True))
    segments = [
        '# V12 冻结数值附表',
        '数据为2014-01-02—2026-09-24，共3097个交易日。主口径：修正总回报、理想同收盘、单边1bp、首日免费、244日年化。所有 V12 行均为未通过全部资格的探索对照。',
        '## 全期对比',table(full_headers,full),
        '## 每年累计收益',
        '完整年份列示当年实际复利收益；2026*只到9月24日，不能当全年收益或保证。若需要244日折算年化，见 yearly_metrics.csv；年度收益取连续账户切片，不逐年重启账户。',
        table(('年份',)+tuple(LABELS[role] for role in ORDER),annual),
        '## 费用与一个收盘延迟',
        '单元格为全期年化 / 最大回撤。close是理想同收盘，lag1是延迟一个收盘，不是下一开盘或14:50。bp为单边；非初始全换仓扣两边费用。',
        table(('情景','V10-H','V12-A 探索','V12-B 探索','V12-C 探索'),stress),
        '## 分红现金金额敏感性',
        '表内为同收盘1bp的全期年化。lower/upper是已识别分红事件的现金金额区间，不是策略收益上下界；official_515100替换一笔已核官方金额，但该票不在这些固定原池内。完整四情景和2026结果见 cash_precision.csv。',
        table(('分红情景','V10-H','V12-A 探索','V12-B 探索','V12-C 探索'),cash_table),
        '## 全候选族检验',
        '完整 A∪B∪C 的1807项，配对平稳区块重采样；每个情景/区块长度2000次同步重采样、固定seed20260928。p值不是过拟合概率，也没有覆盖本项目前期全部搜索和自适应造候选过程。',
        table(('情景','p：20日块','p：60日块'),tests),
        '## 每轮公开的极值对照',
        '只用于展示取舍，不是晋级策略，也没有被拿来替换失败主选择。相同ID跨阶段只计一个配置。',
        table(('阶段','角色','配置ID','年化','最大回撤','2026累计'),leaders),
        '![同收盘净值与回撤](results/equity_drawdown.png)',
    ]
    (BASE/'RESULTS.md').write_text('\n\n'.join(segments)+'\n',encoding='utf-8')
    metadata = read('results/pressure/base/path_metadata.json')
    with np.load(BASE/'results/pressure/base/paths.npz',allow_pickle=False) as archive:
        daily = archive['close_1bp__returns']
    dates = [datetime.strptime(day,'%Y-%m-%d') for day in metadata['dates']]
    fig,(ax,dx) = plt.subplots(2,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for role,label in (('v92','V9.2'),('simple','V9.2+'),('h','V10-H'),('balanced_b','V12-B exploratory'),('balanced_c','V12-C exploratory')):
        values = daily[metadata['ids'].index(profiles[role]['id'])]
        nav = np.cumprod(1.+values);peak = np.maximum.accumulate(np.r_[1.,nav])[1:]
        ax.plot(dates,nav,label=label,linewidth=1.35)
        dx.plot(dates,(nav/peak-1.)*100,linewidth=1.1)
    ax.set_yscale('log');ax.set_ylabel('Wealth (start = 1; log scale)')
    ax.legend(loc='upper left',fontsize=8);ax.grid(alpha=.2)
    ax.set_title('Known-history comparison, 2014-01-02 to 2026-09-24\nIdeal same-close / 1bp per leg; V12 candidates did not qualify')
    dx.set_ylabel('Drawdown (%)');dx.grid(alpha=.2)
    fig.tight_layout();fig.savefig(BASE/'results/equity_drawdown.png',dpi=150);plt.close(fig)
    output_names = ('RESULTS.md','results/full_comparison.csv','results/annual_returns.csv','results/yearly_metrics.csv',
        'results/execution_stress.csv','results/cash_precision.csv','results/final_qualification.json','results/equity_drawdown.png')
    inputs = ['profiles.json','results/pressure/completion.json','results/pressure/base/path_metadata.json',
              'results/pressure/base/paths.npz']
    inputs += ['results/pressure/'+case+'/evaluation.json' for case in ('base','cash_lower','cash_upper','official_515100')]
    inputs += ['results/'+stage+'/'+name+'.json' for stage in ('main','consensus','exante') for name in ('selection','evaluation')]
    inputs += ['results/family_diagnostics/'+name+'.json' for name in ('close_1bp','close_11bp','lag1_1bp','lag1_11bp')]
    dump(BASE/'results/render_receipt.json',dict(rendered_at=stamp(),source_sha256=sha(BASE/'render.py'),
        input_sha256={n:sha(BASE/n) for n in inputs},output_sha256={n:sha(BASE/n) for n in output_names},
        candidate_selection_performed=False))


if __name__ == '__main__':
    render()
