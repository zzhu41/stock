"""Render the complete frozen research result; never choose or rerun a model."""
import csv
from datetime import datetime
import json

import numpy as np

from .data import BASE, START, END, dump, sha
from .scan import SCENARIOS
from .confirmation import PRESSURES

ROLES=('v9','v91','v92','v10_h','v11_a','v11_b')
LABELS=dict(v9='V9',v91='V9.1',v92='V9.2',v10_h='V10-H',v11_a='V11-A',v11_b='V11-B')


def read(name):return json.loads((BASE/name).read_text())
def pct(value):return '%.2f%%'%(100*value)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def csv_file(name,headers,rows):
    with (BASE/'results'/name).open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)


def figures(report,metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dates=[datetime.strptime(d,'%Y-%m-%d') for d in metadata['dates']]
    colors={'v10_h':'#2563eb','v11_a':'#9b59b6','v11_b':'#159570','v92':'#8893a1'}
    with np.load(BASE/'results/report_2026/paths.npz',allow_pickle=False) as arrays:
        fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        for role in ('v92','v11_a','v10_h','v11_b'):
            index=metadata['ids'].index(report['roles'][role]);r=arrays['close_1bp__returns'][index]
            nav=np.cumprod(1+r);peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
            axes[0].plot(dates,nav,label=LABELS[role],color=colors[role],lw=1.6)
            axes[1].plot(dates,100*(nav/peak-1),color=colors[role],lw=1.1)
        for ax in axes:
            ax.axvline(datetime(2022,1,1),color='#777',ls='--',alpha=.7)
            ax.axvline(datetime(2026,1,1),color='#aaa',ls=':',alpha=.7)
            ax.grid(alpha=.18)
        axes[0].set_yscale('log');axes[0].set_ylabel('NAV (log scale, initial = 1)')
        axes[1].set_ylabel('Drawdown (%)');axes[0].legend(loc='upper left',ncol=4)
        axes[0].set_title('V11 frozen research | corrected total return | ideal same-close | 1 bp per side')
        fig.text(.10,.018,'2014–2021 development | 2022–2025 confirmation | 2026 report only. All history previously examined; neither V11 candidate passed confirmation.',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(BASE/'results/equity_drawdown.png',dpi=160);plt.close(fig)


def main():
    report=read('results/report_2026/evaluation.json');metadata=read('results/report_2026/path_metadata.json')
    if sha(BASE/'results/report_2026/paths.npz')!=report['paths_sha256']:raise ValueError('Final path hash differs')
    mapping={row['id']:row for row in report['rows']}
    selected={role:mapping[report['roles'][role]] for role in ROLES}
    tests=read('results/combined_development_diagnostics/diagnostics.json')
    neighborhoods=read('results/b_neighborhood/comparison.json')
    reselection=read('results/reselection/evaluation.json')
    leave_one_out=read('results/leave_one_out/evaluation.json')
    audit=read('results/final_audit/audit.json')
    # These are research outputs, never inputs to a new parameter choice.
    main_rows=[];numeric=[]
    for role in ROLES:
        m=selected[role]['scenarios']['close_1bp']['full']
        main_rows.append([LABELS[role],pct(m['total_return']),pct(m['cagr']),pct(m['max_dd']),m['switches']])
        numeric.append([role,m['total_return'],m['cagr'],m['max_dd'],m['switches'],m['sessions']])
    csv_file('full_comparison.csv',['version','total_return','cagr','max_dd','switches','sessions'],numeric)
    years=sorted(selected['v10_h']['scenarios']['close_1bp']['yearly'])
    annual=[[year]+[pct(selected[role]['scenarios']['close_1bp']['yearly'][year]['total_return']) for role in ROLES] for year in years]
    csv_file('annual_returns.csv',['year']+list(ROLES),[[year]+[selected[role]['scenarios']['close_1bp']['yearly'][year]['total_return'] for role in ROLES] for year in years])
    pressures=[];pressure_csv=[]
    for name,lag,fee in PRESSURES:
        row=[('延迟1日' if lag else '同收盘')+' / 单边%d bp'%round(fee*10000)]
        for role in ('v10_h','v11_a','v11_b'):
            m=selected[role]['scenarios'][name]['full'];row.append(pct(m['cagr'])+' / '+pct(m['max_dd']))
            pressure_csv.append([role,name,m['cagr'],m['max_dd'],m['total_return'],m['switches']])
        pressures.append(row)
    csv_file('execution_stress.csv',['version','scenario','cagr','max_dd','total_return','switches'],pressure_csv)
    period_rows=[]
    for period,label in (('development','开发 2014–2021'),('confirmation','确认 2022–2025'),('report_only_2026','2026至09-24累计')):
        metric='total_return' if period=='report_only_2026' else 'cagr'
        period_rows.append([label]+[pct(selected[role]['scenarios']['close_1bp'][period][metric]) for role in ('v10_h','v11_a','v11_b')])
    confirm_rows=[]
    for name,unused_lag,unused_fee in SCENARIOS:
        confirm_rows.append([name]+[pct(selected[role]['scenarios'][name]['confirmation']['cagr'])+' / '+pct(selected[role]['scenarios'][name]['confirmation']['max_dd']) for role in ('v10_h','v11_a','v11_b')])
    neighborhood_rows=[]
    for key in ('H','A','B'):
        full=neighborhoods['full_neighborhoods'][key];matched=neighborhoods['matched_comparison'][key]
        neighborhood_rows.append([key,'%d/%d'%(full['existing_H_gate_passed'],full['neighbor_count']),
            '%d/%d'%(matched['existing_H_gate_passed'],matched['neighbor_count']),
            '%.2f pp'%(100*full['worst_paired_pressure_block_delta_to_center']['q25']),
            '%.2f pp'%(100*matched['worst_paired_pressure_block_delta_to_center']['q25'])])
    p_rows=[]
    for name,unused_lag,unused_fee in SCENARIOS:
        p_rows.append([name]+['%.4f'%tests['scenarios'][name]['tests'][str(block)]['p_value'] for block in (20,60)])
    walk_rows=[]
    for name,unused_lag,unused_fee in SCENARIOS:
        walk_rows.append([name]+[pct(reselection['streams'][role][name]['periods']['report_2018_2025']['cagr'])+' / '+pct(reselection['streams'][role][name]['periods']['report_2018_2025']['max_dd']) for role in ('H','A','B')])
    fold_rows=[[fold['execution_period'][0][:4]+'–'+fold['execution_period'][1][:4]]+
               [pct(fold['execution'][role]['close_1bp']['cagr']) for role in ('H','A','B')] for fold in reselection['folds']]
    loo_rows=[]
    for row in leave_one_out['rows']:
        loo_rows.append([row['excluded'] or '完整池']+[pct(row['roles'][role]['scenarios']['close_1bp']['full']['metrics']['cagr']) for role in ('h','a','b')])
    lines=[
        '# V11 冻结研究结果',
        '',
        '**产出了可复现的 V11-A / V11-B 研究版本，但两者均未通过预先登记的后期确认门槛，暂不替换 V10-H。** B 的全期回撤更小，值得保留观察；这不等于已证明过拟合更低。参数与结果保持冻结，没有在确认失败后改推后期赢家。',
        '',
        '## 口径与总成绩',
        '',
        '2014-01-02—2026-09-24，修正总回报、理想同收盘成交、单边1 bp、244观察日年化，首个总评价日免费建仓。不加杠杆、原11只ETF池、单只持仓。这里沿用 V10-H 年化53.31%的已修正口径，不混用旧加法前复权的V9.2“50%+”。所有年份此前都被研究过，分段确认和历史重选也不是全新的样本外。',
        '',
        'H本身也经过既往全历史搜索，并不是事前独立的基准。V11未过相对H的门槛，不能反过来证明H未来一定优于V11，或H已没有过拟合。这里能确定的是：本轮尚未拿到足以支持“高收益且更低过拟合升级”的证据。',
        '',table(['版本','总收益','年化','最大回撤','实际换仓次数'],main_rows),
        '', '![资金曲线与回撤](results/equity_drawdown.png)',
        '', '## 两个候选具体改了什么',
        '',
        '- A：保留 H 的20日WLS评分及3日平滑，将三类动量缓冲统一为1%，普通健康轮动最短持有3个后续观察日。',
        '- B：在同样1%缓冲和3日健康持有基础上，将评分换成10/20/40日WLS风险调整斜率的固定等权均值，不再额外平滑。仍是单只ETF持仓，不是三条资金曲线免费组合。',
        '- 两者保留H的MA180体制、波动退出、原抄底通道与5日锁仓等规则。健康最短持有仅延后普通轮动，不能解释成抄底锁仓期间可无条件风险退出。1%缓冲更敏感，抵消了最短持有的降频作用，实际换仓次数并未下降。',
        '', '## 冻结后的确认',
        '',table(['阶段','V10-H','V11-A','V11-B'],period_rows),
        '', '前两行是区间年化，2026是截至09-24的累计收益。2026在候选冻结及确认结束后才生成，不参与救回失败模型。',
        '',table(['2022–2025 情景：年化 / 最大回撤','V10-H','V11-A','V11-B'],confirm_rows),
        '',
        'A通过三项回撤要求，但收益门槛和两项成本/延迟收益要求失败；B六项要求均失败。门槛在跑后期前已写好：主年化至少30%且不低于同段H的90%，主回撤不恶化超过2个百分点，高费用及高费用+延迟的年化不少于H、回撤不恶化超过2个百分点。开发期更强，并没有确保后期持续领先。',
        '', '## 逐年收益',
        '',table(['年份']+[LABELS[r] for r in ROLES],annual),
        '', '均为该年累计收益。2026不是完整年度；各年包含年界首个交易日的旧持仓涨跌和真实换仓费用，没有年度免费重启。',
        '', '## 执行压力：全期年化 / 最大回撤',
        '',table(['时钟与成本','V10-H','V11-A','V11-B'],pressures),
        '',
        '1 bp=0.01%，完整换仓计两边。延迟1日表示前一收盘信号在下一收盘成交，并非下一开盘，也不是把收益数组简单后移。B全期压力回撤改善，但未在后期确认继续取得收益优势。真实14:50报价、现金分红到账日、交易单位与跨境ETF溢价成交仍未建模；几十万元资金不会自动消除信号时点偏差。',
        '',
        '主口径H全期最大回撤落在2026-07-24，B的最大回撤落在2022-07-01。确认段B的回撤其实比H更深（19.47%对17.00%）；因此全期回撤改善也不能单独代替跨阶段稳健性证据。',
        '', '## 搜索与过拟合诊断',
        '',
        'A登记593个原始提案、去重586个配置；其中17个已有历史研究签名。B是看到A开发结果后明确登记的一次有限自适应组合扩展，不能冒充最初即登记；A∪B共有714个配置。另有邻域、消融、8个长期趋势控制及每折重选试验，714不是全部尝试数。V10此前还试过9558个唯一单持仓配置，不能忽略这段研究历史。',
        '',
        'OLS-log、Huber、Theil–Sen、加权log趋势，多期限均分数/均排名，事前波动尺度，固定退出，去牛熊划分，关抄底，现金回退，持有/确认规则都已纳入登记枚举。原池固定用于主选型，删票只用于稳定性诊断，没有据此再挑赢家池。论文方向及具体实现差异见 [LITERATURE.md](LITERATURE.md)。',
        '',
        '714全族同步时间区块White-style检验，2000次重采样，相对同情景H；下面是两种平均区块长度的p值。它是条件于本族的未学生化检验，不能解释为过拟合概率，也没有校正整个旧项目或B组件生成的适应性。',
        '',table(['情景','20日区块 p','60日区块 p'],p_rows),
        '',
        '没有显著的全族优势证据。开发期同收盘1bp情景中，B最大的10个正优势日贡献了正log超额的33.98%、净log超额的228.03%；去掉这10天的固定路径算术诊断会转为负超额。超过100%来自其它日期抵消，这不是“总利润有228%来自10天”，也不是可执行的删日策略。',
        '',
        '邻域全部来自开发期。既报告全部有效扰动，也报告A/B/H共同的8个扰动，不能只展示有利方向。最后两列是邻居相对自身中心的最差压力子段年化差的下四分位。',
        '',table(['中心','完整邻域过H门槛','共同8扰动过H门槛','完整自身差Q25','共同自身差Q25'],neighborhood_rows),
        '',
        'B更多邻居仍过H门槛，但相对自身中心的下降反而更大，不能据此说B位于更平坦的参数区域。长期趋势8个简化控制在开发期年化约3.38%—12.66%，不满足本轮收益目标；这仅反映本ETF池、长仓、固定槽位和调仓方式，不否定原文的多空期货策略。',
        '', '## 历史重选：连续账户2018–2025',
        '',table(['执行情景：年化 / 最大回撤','固定H','A选择过程','B选择过程'],walk_rows),
        '',table(['各执行段同收盘1bp年化','固定H','A选择过程','B选择过程'],fold_rows),
        '',
        '四种压力下，A/B整个重选过程年化均落后固定H。按同收盘1bp的四个两年执行段，A赢1/4、B赢0/4；B开发阶段的局部组合优势未能在重选过程复现。2021截止的重选准确恢复原先冻结的A/B候选，排除了使用不同选择器造成的错配。',
        '',
        '训练终点2017/2019/2021/2023，每次执行随后两年；B在每折根据当时A训练结果重新生成组件，没有回填2021选出的组件。所有账户从2014共同H初始化、2018开始实际切模型，持仓、估值价、年龄和锁仓连续保留，只在真实换仓时收费。完整分折选择、边界和交易见 [reselection/evaluation.json](results/reselection/evaluation.json)。这诊断选择过程的可重复性，不是承诺今后动态调参。',
        '', '## 删票与正确性',
        '',
        '逐一取消7只股票ETF和2只跨境ETF的交易资格；A/B/H相同剔除、保留黄金及货币ETF，并继续保留原基准指标。不是删除该资产所有信息通道。所有结果及相对自身完整池、配对H的差异见 [留一法完整记录](results/leave_one_out/evaluation.json)。这些结果未用于更换候选池或参数。',
        '',table(['删除交易资格：全期同收盘1bp年化','V10-H','V11-A','V11-B'],loo_rows),
        '',
        '删除纳指ETF 513100 后三者收益均明显下降，显示共同资产依赖，不只发生在新候选。确认2022–2025，B在全部9种删票、四种情景下仍落后配对H；A仅同收盘1bp删513120时微弱领先0.015个百分点，其余均落后。删票结果没有救回确认失败。',
        '',
        '独立P&L重计、原版本复现、原生实现与Python参考逐日对账及抄底真实事件簇见 [最终审计](results/final_audit/REPORT.md)。区间切片包含首日收益、分红口径沿用修正TR。原V9/V10策略与生产推送、账户均未接入V11。',
        '',
        '11个最终配置×8情景共88条路径独立重计，逐日收益误差为0；5984个逐年/分段指标在浮点容差内一致。A/B/H/simple各3097日的参考实现完全一致。原H、A、B都有27个真实抄底成交触发，合并为19个事件簇；其中周边观察窗口有重叠，不视为独立样本。确认期的落后主要来自持仓市场收益差，而不是费用漏算或多算。',
        '', '## 复现与使用',
        '',
        '新克隆先恢复被归档的原始矩阵：',
        '', '```bash', 'python3.8 -B -m v11.artifacts restore',
        'python3.8 -B -m v11.cli backtest b',
        'python3.8 -B -m v11.cli backtest b --lag 1 --fee-bp 11',
        'python3.8 -B -m v11.cli signal b --end 2026-09-24', '```',
        '',
        '`a/b/h`分别为冻结A、B、H对照。signal只给冻结历史日的模型目标，不发推送、不下单、不使用当前实时行情。profiles包含代码、依赖、输入及证据hash；已有结果拒绝覆盖。CSV里的收益/回撤为小数，显示表格才转换为百分比。',
        '',
        '主要产物：[全期CSV](results/full_comparison.csv)、[逐年CSV](results/annual_returns.csv)、[费用/延迟CSV](results/execution_stress.csv)、[冻结候选](profiles.json)、[事前协议](PROTOCOL.md)、[组合扩展协议](COMBINATION_PROTOCOL.md)。'
    ]
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    figures(report,metadata)
    dump(BASE/'results/report_receipt.json',dict(source_sha256=sha(BASE/'report.py'),
        inputs={name:sha(BASE/name) for name in ('results/report_2026/evaluation.json','results/report_2026/path_metadata.json',
        'results/report_2026/paths.npz','results/combined_development_diagnostics/diagnostics.json',
        'results/b_neighborhood/comparison.json','results/reselection/evaluation.json',
        'results/leave_one_out/evaluation.json','results/final_audit/audit.json')},
        outputs={name:sha(BASE/name) for name in ('REPORT.md','results/equity_drawdown.png',
        'results/full_comparison.csv','results/annual_returns.csv','results/execution_stress.csv')},
        candidate_selection_performed=False))


if __name__=='__main__':main()
