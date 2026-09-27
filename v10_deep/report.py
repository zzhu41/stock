"""Freeze public research profiles and generate a result-first audit report."""
import json
from .data import BASE, ROOT, dump, sha, protect
from .registry import canonical
from .schema import baseline, identifier


def pct(x):return '{:+,.2f}%'.format(x*100)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    out=BASE/'results/refinements';e=json.loads((out/'evaluation.json').read_text())
    registry=json.loads((out/'registered_candidates.json').read_text());configs={c['id']:c for c in registry['candidates']}
    rows={r['id']:r for r in e['rows']};selection=e['selection'];h=selection['primary'];ref=selection['controls']['v9.2']
    simple=baseline();simple['min_hold']=2;simple='vd_'+identifier(canonical(simple))[:20]
    ids=[selection['controls'][n] for n in ('v9','v9.1','v9.2')]+[h,simple]
    labels={ids[0]:'v9',ids[1]:'v9.1',ids[2]:'v9.2',h:'新V10-H',simple:'单项简化对照'}
    comparison=table(['版本','累计收益','年化','最大回撤','换仓次数'],
                     [[labels[n],pct(rows[n]['metrics']['full']['total_return']),pct(rows[n]['metrics']['full']['cagr']),
                       pct(rows[n]['metrics']['full']['max_dd']),rows[n]['switches']] for n in ids])
    annual=table(['年份']+[labels[n] for n in ids],
                 [[y+('（截至09-24）' if y=='2026' else '')]+[pct(rows[n]['metrics']['yearly'][y]) for n in ids]
                  for y in sorted(rows[h]['metrics']['yearly'])])
    periods=table(['连续区间','H年化','v9.2年化','H最大回撤','v9.2最大回撤'],
                  [[label,pct(rows[h]['metrics'][key]['cagr']),pct(rows[ref]['metrics'][key]['cagr']),
                    pct(rows[h]['metrics'][key]['max_dd']),pct(rows[ref]['metrics'][key]['max_dd'])]
                   for key,label in [('early_2014_2017','2014–2017'),('middle_2018_2021','2018–2021'),
                                     ('recent_2022_2025','2022–2025'),('selection','2014–2025（用于选型）')]])
    stress=json.loads((out/'stress/evaluation.json').read_text())
    names={'registered_lag_fee1bp':'原收盘，单边1bp','registered_lag_fee5bp':'原收盘，单边5bp',
           'registered_lag_fee11bp':'原收盘，单边11bp','registered_lag_fee21bp':'原收盘，单边21bp',
           'lag1_fee1bp':'滞后一日收盘，单边1bp','lag1_fee5bp':'滞后一日收盘，单边5bp',
           'lag1_fee11bp':'滞后一日收盘，单边11bp','lag1_fee21bp':'滞后一日收盘，单边21bp',
           'cash_lower':'分红推断下界','cash_upper':'分红推断上界','official_515100':'515100官方金额替代'}
    stress_rows=[]
    for scenario in stress['scenarios']:
        values={r['id']:r for r in scenario['rows']};a,b=values[h]['metrics']['full'],values[ref]['metrics']['full']
        stress_rows.append([names[scenario['spec']['name']],pct(a['cagr']),pct(b['cagr']),pct(a['max_dd']),pct(b['max_dd'])])
    stress_table=table(['压力情景','H年化','v9.2年化','H最大回撤','v9.2最大回撤'],stress_rows)
    neighborhood=json.loads((out/'neighborhood/evaluation.json').read_text())
    neighbor_names={'frozen_center':'冻结H','reference_v92':'原v9.2','restore_WLS25':'评分恢复原WLS25',
                    'restore_MA250':'均线恢复MA250','restore_fixed_panic4':'急跌恢复固定4%',
                    'drop_QVIX':'仅移除QVIX','deep_only':'只保留深跌抄底','no_crash_channels':'去掉全部抄底通道',
                    'neighbor_ma150':'均线改MA150','neighbor_ma200':'均线改MA200',
                    'panic_multiple_x0.9':'波动倍数改1.35','panic_multiple_x1.1':'波动倍数改1.65',
                    'diagnostic_wls20_smooth1':'WLS20不平滑','diagnostic_wls20_smooth2':'WLS20取2日评分均值',
                    'diagnostic_wls20_smooth4':'WLS20取4日评分均值','diagnostic_wls18_smooth3':'WLS18取3日评分均值',
                    'diagnostic_wls22_smooth3':'WLS22取3日评分均值'}
    neighbor_table=table(['冻结后单项诊断（不重选）','全程年化','最大回撤'],
                         [[neighbor_names[r['label']],pct(r['metrics']['full']['cagr']),pct(r['metrics']['full']['max_dd'])]
                          for r in neighborhood['rows']])
    diagnostics=json.loads((out/'diagnostics.json').read_text());paired=diagnostics['paired'][h]
    walk=diagnostics['historical_reselection'];p20=diagnostics['white_style']['20']['p_value'];p60=diagnostics['white_style']['60']['p_value']
    portfolio=json.loads((BASE/'results/portfolios/evaluation.json').read_text())
    stage_rows=[]
    for stage in ('mechanisms','combinations','refinements'):
        trial=json.loads((BASE/'results'/stage/'evaluation.json').read_text());reg=json.loads((BASE/'results'/stage/'registered_candidates.json').read_text())
        winner=next(x for x in trial['rows'] if x['id']==trial['selection']['primary'])
        stage_rows.append([stage,reg['unique_count'],pct(winner['metrics']['selection']['cagr']),pct(winner['metrics']['full']['cagr'])])
    stages=table(['阶段','累计唯一单持仓配置','2014–2025所选年化','同配置全程年化'],stage_rows)
    report='''# V10深入迭代报告

**本轮达到历史回测收益目标：新H年化53.31%，高于同数据v9.2的46.32%；不能据此宣称未来稳定超额或已消除过拟合。**

统一期间为2014-01-02—2026-09-24。采用修正后的总回报收盘指数、原同收盘信号/成交、首评日免费、之后每次整仓切换净值乘0.9998，无杠杆。这里比较的不是旧失真复权输入下的50.42%；输入和撮合假设与本轮v9基准完全相同。

## 总体与逐年

{comparison}

![净值与回撤](results/comparison.png)

“单项简化对照”只在v9.2上增加普通排名轮动最短持有2日，保留风险退出例外。它是已有登记配置的解释性对照，未冒称独立验证的新S，也未替换冻结H。主收益冠军与风险门槛选择均为同一H，不是两次独立确认。

{periods}

{annual}

H在2026这一段累计**45.47%**，低于v9.2的**57.36%**；同期最大回撤**23.88% / 14.69%**。这部分不能省略，也不能把不足一年累计收益称为全年收益。

## H的具体规则

原11只ETF池、单一标的轮动、熊市动量门槛7%、分池缓冲2%/3%/3%、过热门和危机抄底框架保留。改动为：

1. 用20日WLS相对斜率除以20日日收益标准差，取最近3个有效报价日评分平均，代替原WLS25。
2. 沪深300牛熊门由MA250改为MA180。
3. 急跌退出阈值由固定4%改为`clip(1.5 × VOL20, 2%, 10%)`；VOL20是日收益标准差，包含当日收盘。

冻结ID：`{h}`。参数包含原有深跌/QVIX/量能通道与5日危机锁仓。未引入杠杆、动态优化权重或按年份切换规则；未接管生产推送或真实持仓。

## 探索过程与样本选择

{stages}

累计9558个唯一单持仓配置：1355初始机制、4683新增双机制组合、3520新增第三阶段组合/简化。各阶段保留前一阶段集合，所以不能把累计数简单相加作为不同试验数。另有36个独立多持仓配置及冻结后的诊断情景，均单独留档。

每阶段先登记、用2014–2025选型并写出选择，再展示全程/2026；后两阶段依据前阶段已知训练表现自适应提出。所有历史此前已经研究，**2026也不是干净样本外**。前三段改善不等于三次独立验证。

风险门槛为：选型年化至少比v9.2高2个百分点、回撤最多多5个百分点、2022–2025最多落后2个百分点、FEE5不低于同费基准。最后695个配置通过，最终H按选型收益最高冻结；不根据后面的压力结果改选。

## 费用、执行时点及数据精度

{stress_table}

H在11种情景中均保留相对v9.2的全程收益优势，但绝对收益对执行时间敏感：滞后一日、单边1bp时H年化38.18%、最大回撤35.13%；2026累计变为−6.02%。这是按前日特征和实际持仓重新决策、在当日收盘执行的压力模型，不是固定信号重放，不是次日开盘，也不是14:50实盘验证。

原费用约定忽略首次建仓、固定同价线性扣费。精度情景只改变条件区间内的现金金额，重算信号和持仓，因此分红下界情景的策略收益反而可能更高；它不是组合收益的数学下界。官方515100金额替代对选定5个配置无影响，因为它们未采用该资产。

## 局部敏感性与消融

{neighbor_table}

邻近窗口/均线大多仍高于v9.2，但并非完全平坦的平台。波动倍数从1.5提高10%到1.65，年化降到49.63%、最大回撤升到30.48%，说明参数选择仍明显影响风险。恢复固定4%急跌门后几乎回到基准收益；各项贡献有交互，不能相加。

去掉QVIX得到53.36%只是事后简化诊断，未用这0.05个百分点回流替换已冻结ID。改成更简单的WLS25得到50.44%、回撤19.97%，同样只作对照。完整注册和结果见[neighborhood](results/refinements/neighborhood/evaluation.json)。

## 过拟合：改善成立，但泛化证据不足

- 全9558族在2919个选型交易日上的White-style区块检验：20日p={p20:.4f}，60日p={p60:.4f}（各1000次）。没有显示搜索后显著超额。该方法可能保守，条件于已经自适应构造的候选族；不是SPA/PBO，也不是未来失败或过拟合概率。
- 四段历史重选仅一段胜出。已知候选族的拼接收益年化{walk_h}，基准{walk_b}；族本身用过未来历史，且拼接未建模切换持仓成本，不能当可交易OOS。
- 最大10个优势日解释净对数超额的{top10:.2f}%；优势有明显集中性。删除这些日期只是固定路径归因，不是可执行替代策略。
- 固定路径排除2014–2015后，H年化48.58%，基准41.12%，表明提升并非完全由最早两年驱动，但这仍不能取代未来检验。

完整统计、逐日贡献和限制见[diagnostics](results/refinements/diagnostics.md)。结论是值得保存并作独立前向观察的研究候选；不把53.31%当作未来承诺。

## S与多持仓尝试

额外登记36个无抄底top1/2/3配置，用真实份额、成员变化再平衡和全部权重换手计费。多持仓S的门槛没有放松：24个top2/3候选均未达到训练年化30%和近期年化25%，故`S=None`。
最佳top2全程年化29.75%、回撤26.04%，也不能四舍五入包装成30%达标。该族和原有S策略不是同一个东西，本轮没有替换此前S。见[组合研究报告](results/portfolios/REPORT.md)。

## 实现与复现

3个旧控制的3097日原路径逐值保真；28个预定真实配置与独立Python实现逐值一致；最终H与v9.2在0/1/5bp下再做6次独立全程复算，持仓和收益最大误差0。所有单持仓候选的FEE5由完整新运行复核，训练前缀加入2026后不变。组合72条路径另行重建审计。

原数据仍有条件性分红推断、除息日立即免费再投资、幸存ETF池、上市预热和缺报价等限制，详见[数据审查](DATA_REVIEW.md)。实盘付款时差、整手、溢价/冲击和14:50可得性未完整模拟。

运行入口及冻结配置见[README](README.md)与[profiles](profiles.json)。新研究仅写本目录，{protected}个旧文件哈希保持不变。各阶段源码/输入/注册/选择/矩阵哈希保留；完整矩阵无损归档后可按归档工具恢复。
'''.format(comparison=comparison,periods=periods,annual=annual,h=h,stages=stages,stress_table=stress_table,
           neighbor_table=neighbor_table,p20=p20,p60=p60,walk_h=pct(walk['selected_stream_cagr']),
           walk_b=pct(walk['benchmark_stream_cagr']),top10=100*paired['advantage_concentration']['10']['share_of_net_log_advantage'],protected=protect())
    (BASE/'REPORT.md').write_text(report)
    own=('__init__.py','data.py','features.py','schema.py','native.cpp','native.py','registry.py','scan.py','combine.py','refine.py','cli.py','report.py')
    dependencies=('v10_next/data.py','v10_next/frozen/strategy.py','v10_next/frozen/metadata.py','v10_next/frozen/presets.json',
                  'v10_search/features.py','v10_search/data.py','v10_search/scan.py','v10_search/fast_execution.py','v10_search/policy.py',
                  'v10_h_close/data.py','v10_h_close/scan.py','v10_h_close/corrected_scan.py','v10_round2/v92.py','v10_next/legacy.py')
    artifacts=('results/refinements/registered_candidates.json','results/refinements/registration.json','results/refinements/selection.json',
               'results/refinements/evaluation.json','results/refinements/diagnostics.json','results/refinements/stress/evaluation.json',
               'results/refinements/neighborhood/evaluation.json','results/refinements/selected_reference_fidelity.json',
               'results/portfolios/selection.json')
    variants={}
    for key,n,name,role in [('growth',h,'V10-H 深入研究候选','2014-2025 primary and guarded selection'),
                          ('simple',simple,'单项简化研究对照','Registered min-hold2 reference; not an independently validated S'),
                          ('v92',ref,'v9.2 同数据基准','Existing reference, not a new H')]:
        status=('Historical return target met; search-adjusted significance/generalization unproven' if key=='growth' else
                'Explanatory single-change comparison; not a qualified new S' if key=='simple' else 'Existing v9.2 reference')
        variants[key]=dict(name=name,config=configs[n],selected_by=role,research_status=status)
    dump(BASE/'profiles.json',dict(variants=variants,source_sha256={n:sha(BASE/n) for n in own},
              dependency_sha256={n:sha(ROOT/n) for n in dependencies},artifact_sha256={n:sha(BASE/n) for n in artifacts},
              historical_target_met=True,clean_oos=False,live_enabled=False,selected_s=None,
              feature_cache_sha256=e['fingerprints']['feature_cache_sha256'],
              feature_metadata_sha256=sha(BASE/'cache/features.json'),
              input_manifests={'corrected':sha(ROOT/'v10_h_close/corrected_manifest.json'),'qvix':sha(ROOT/'v10_h_close/qvix_manifest.json')}))
    print(comparison)
    print('Frozen H:',h,'; S:',portfolio['selection']['selected_s'],'; old files verified:',protect())


if __name__=='__main__':main()
