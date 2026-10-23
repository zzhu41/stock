"""Build a self-contained, searchable local research report and readable CSV."""
import html
import json
import shutil
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
NOTES = [
 ('300604','长川科技','重点：设备','半导体测试设备；历史Beta在多个窗口较稳定。前三季净利预告17.15—18.15亿元，同比增长98%—110%。','验收、存货和回款；半年经营现金流/利润约0.35，估值也不低。','https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12653220&stockid=300604'),
 ('300223','君正股份（原北京君正）','重点：芯片设计','存储与计算业务；半年净利11.98亿元，扣非11.66亿元，经营现金9.59亿元。','存储售价、库存成本优势能否延续；不把当前高毛利永续化。','https://disc.static.szse.cn/disc/disk03/finalpage/2026-08-29/86ae93e8-5f48-4c6f-b6b9-a9daaac9d7c4.PDF'),
 ('688019','安集科技','重点：材料','CMP材料、先进封装电镀液等；历史弹性与鼎龙接近，当前滚动估值较低。','半年扣非利润仅增长19%，低于归母42%；H股融资及产品放量需跟踪。','https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12552860&stockid=688019'),
 ('688183','生益电子','重点：PCB','半年净利11.11亿元、经营现金9.17亿元；前三季预告18.69—19.52亿元，增长68%—75%。','扩产、客户资本开支及折旧；与华正/兴森同属电子硬件主题，不能视为完全分散。','https://static.cninfo.com.cn/finalpage/2026-10-17/1225608856.PDF'),
 ('001389','广合科技','PCB同组替代','半年净利9.56亿元、经营现金9.37亿元；前三季预告14.5—15亿元，增长100%—107%。','海外客户、泰国产能及新股本；并无数据证明生益全面优于广合。','https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12641383&stockid=001389'),
 ('688766','普冉股份','更进攻：并购与存储','较高且跨窗口稳定的历史Beta；半年净利8.27亿元、经营现金9.45亿元。','SHM新增并表贡献约5.08亿元归母利润；2D NAND周期、供应链及整合风险，不能将1929%增速全视为内生。','https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12503056&stockid=688766'),
 ('300475','香农芯创','更进攻：存储周期','半年净利36.42亿元、经营现金57.31亿元；筛选分数高，现金与利润均有支撑。','分销属性、资产负债率约74%、存储价格周期；低PE可能反映高峰利润，不能简单视为低估。','https://static.cninfo.com.cn/finalpage/2026-08-28/1225518636.PDF'),
 ('300054','鼎龙股份','材料备选','CMP材料业务放量；前三季净利预告8.4—8.6亿元，增长62%—66%。','部分增长涉及并表与业务结构调整；估值高于安集，需更强增长兑现。','https://disc.static.szse.cn/download/disc/disk03/finalpage/2026-10-09/b1d39cd1-0baf-41c7-afc1-c01b9e3e18f0.PDF'),
 ('603186','华正新材','已有持仓：保留研究','前三季净利8亿元，扣非7.93亿元、经营现金5.59亿元；盈利改善有实绩支撑。','CCL涨价周期和回款；季度高利润不能直接乘四。','https://static.cninfo.com.cn/finalpage/2026-10-23/1225615477.PDF'),
 ('002436','兴森科技','已有持仓：拐点待兑现','前三季净利预告4—4.3亿元；CSP、mSAP和宜兴业务改善。近期弱势不等于没有反弹弹性。','FCBGA规模盈利仍待兑现；actual TTM约270倍，按Q3预告替换后滚动情景仍约135—145倍。','https://static.cninfo.com.cn/finalpage/2026-10-14/1225602695.PDF'),
 ('688668','鼎通科技','光互连备选','半年盈利与现金匹配较好，高速连接器有业务依据。','估值、客户订单和行业资本开支；历史下跌捕获也偏高。','https://static.cninfo.com.cn/finalpage/2026-08-05/1225457964.PDF'),
 ('688372','伟测科技','测试备选','利润增长与经营现金流有支撑。','半年经营现金5.47亿元，但购建长期资产支出17.35亿元；重资产扩产与融资压力。','https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12569182&stockid=688372'),
 ('300620','光库科技','下调优先级','利润增长并不弱。','滚动PE较高，现金流为负；不能仅凭光通信题材判为最优。','https://static.cninfo.com.cn/finalpage/2026-10-23/1225616009.PDF'),
 ('688322','奥比中光','主题观察','机器人视觉主题可以产生独立于指数的行情。','较长窗口Beta不高、估值高、半年利润下滑；近期大涨不能证明当前买入还可再涨50%。','https://static.cninfo.com.cn/finalpage/2026-08-22/1225493290.PDF'),
]


def fmt(value, places=2, pct=False):
    if pd.isna(value):
        return '—'
    return ('%.*f' % (places, value * 100 if pct else value)) + ('%' if pct else '')


def main():
    data = pd.read_csv(str(BASE / 'all_technology_metrics.csv'), dtype={'code': str})
    meta = json.loads((BASE / 'analysis_manifest.json').read_text(encoding='utf-8'))
    universe = json.loads((BASE / 'universe_manifest.json').read_text(encoding='utf-8'))
    indexed = data.set_index('code')
    note_rows, selected_rows = [], []
    for code, name, role, thesis, risk, source in NOTES:
        r = indexed.loc[code]
        selected_rows.append(dict(code=code, name=name, role=role, thesis=thesis, risk=risk,
                                  source=source, **{k: r[k] for k in [
                                      'price','market_cap_yi','pe_ttm_calculated','pe_deduct_ttm',
                                      'drawdown_250','star50_beta_60','star50_beta_120','star50_beta_250',
                                      'cash_profit_ratio_actual','cash_profit_ratio_ttm']}))
        cells = [name + ' ' + code, role, fmt(r.pe_ttm_calculated, 1), fmt(r.pe_deduct_ttm, 1),
                 fmt(r.drawdown_250, 1, True), fmt(r.star50_beta_60), fmt(r.star50_beta_120),
                 fmt(r.star50_beta_250), fmt(r.cash_profit_ratio_actual)]
        note_rows.append('<tr>' + ''.join('<td>'+html.escape(c)+'</td>' for c in cells) + '</tr>')
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(str(BASE / 'reviewed_candidates.csv'), index=False, encoding='utf-8-sig')
    (BASE / 'reviewed_notes.json').write_text(json.dumps(selected_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    column_map = {'code':'代码','name':'名称','industry':'行业','scope':'研究范围',
                  'trading_date':'报价日期','price':'价格','market_cap_yi':'A股价格折算总股本市值_亿元',
                  'report_date':'实绩财报期','net_profit':'本期归母净利_元',
                  'deduct_profit_actual':'本期扣非净利_元','operating_cashflow_actual':'本期经营现金_元',
                  'pe_ttm_calculated':'归母PE_TTM','pe_deduct_ttm':'扣非PE_TTM',
                  'drawdown_250':'最多250日收盘高点回撤','observations_250':'250日窗口有效观测',
                  'star50_beta_60':'科创50_Beta60','star50_beta_120':'科创50_Beta120',
                  'star50_beta_250':'科创50_Beta250','star50_up_capture_120':'上涨日捕获倍数120',
                  'star50_down_capture_120':'下跌日捕获倍数120','cash_profit_ratio_actual':'本期经营现金除归母净利',
                  'cash_profit_ratio_ttm':'TTM经营现金除归母净利','return_20':'20交易日涨跌幅',
                  'liquidity_proxy_yi':'20日流动性代理_亿元','primary_quant_gate':'通过基础量化门槛'}
    readable = data[list(column_map)].rename(columns=column_map)
    readable.to_csv(str(BASE / '全量科技筛选.csv'), index=False, encoding='utf-8-sig')
    public_cols = ['code','name','industry','scope','pe_ttm_calculated','pe_deduct_ttm',
                   'drawdown_250','star50_beta_120','cash_profit_ratio_actual','primary_quant_gate']
    payload = data.sort_values('priority_score_heuristic', ascending=False)[public_cols].to_json(orient='records', force_ascii=False)
    payload = payload.replace('</', '<\\/')
    details = ''.join('<p><strong>'+html.escape(n[1]+' '+n[0]+'｜'+n[2])+'</strong><br>'+html.escape(n[3])+
                      '<br>主要限制：'+html.escape(n[4])+' <a href="'+html.escape(n[5], quote=True)+'">公告/来源</a></p>' for n in NOTES)
    summary = ('全A快照5576家；财务匹配5579个代码。核心科技1204家，交叉行业1303家。'
               '风险标记/无成交过滤后2434家；2230家沪深取得日线，其中一只新股仅一根日线，2229家可计算。'
               '204家北交所日线未取得，未将其伪装成完成量化覆盖。基础数据及流动性可比样本%d家，'
               '基础量化入围%d家；其中现金与增长条件更强子集%d家。' %
               (meta['data_eligible'], meta['primary_quant_gate'], meta['growth_cash_primary']))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股科技反弹筛选｜2026-10-23</title><style>
body{font:15px/1.75 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 22px;color:#172536;background:#f6f8fa}
h1{font-size:28px}p{max-width:1100px}a{color:#135eaa}table{border-collapse:collapse;width:100%;background:white;font-size:13px}
th,td{padding:9px 10px;border-bottom:1px solid #e3e8ed;text-align:left;vertical-align:top}th{background:#eaf0f6;position:sticky;top:0}
.scroll{overflow:auto;max-height:660px;margin:16px 0}.note{background:#fff4d5;padding:16px;border-left:4px solid #d99f1e}
input,select{font:inherit;padding:7px;margin:6px;border:1px solid #bdcbd7;border-radius:4px}.muted{color:#53677b}
</style><h1>A股科技反弹筛选｜2026年10月23日</h1>
<p class="note"><strong>结论：</strong>重点研究长川科技、君正股份、安集科技、生益电子；广合科技为PCB同组替代。普冉、香农是更进攻的周期/并购候选。
华正本身仍符合筛选；兴森保留为盈利拐点观察。历史Beta、回撤或低PE都不能证明未来涨50%。最近涨幅没有进入淘汰条件。</p>
<p>__SUMMARY__</p><p class="muted">行情快照：2026-10-23明确交易日。复权日线与历史统计统一截至2026-10-22。
财报采用公告日不晚于10月23日的实绩；Q3预告另列。范围为电子、半导体、通信、计算机、自动化及相关设备材料，交叉行业不等同全部为科技主业。</p>
<p><strong>候选对比</strong>：PE使用滚动归母与扣非利润各自计算；回撤来自最多250个基准交易日的复权收盘高点。Beta相对科创50指数，非ETF。</p>
<div class="scroll"><table><thead><tr><th>公司</th><th>研究定位</th><th>归母PE TTM</th><th>扣非PE TTM</th><th>回撤</th><th>Beta60</th><th>Beta120</th><th>Beta250</th><th>本期现金/利润</th></tr></thead><tbody>__ROWS__</tbody></table></div>
__DETAILS__
<p><strong>如何筛选与解释</strong></p><p>历史收益先对齐基准交易日，再计算相邻日收益；不填补停牌价格。Beta为协方差/基准方差。
上涨、下跌捕获为相应基准日子集的几何平均收益比，描述历史条件表现。基准从2024-09-23至2024-10-08曾涨59.24%，
2025-04-08至2025-05-09涨7.05%；这两个共同窗口仅作情景对照，不代表未来概率。</p>
<p>基础量化门槛：至少100个120日窗口有效收益样本、沪深可交易、20日流动性代理至少3000万元、本期净利至少5000万元、扣非为正、营收增速大于－10%、距最多250日收盘高点回撤至少15%，且Beta120至少1.05或Beta60至少1.15。
短期收益不参与门槛和得分。门槛只是缩小核查范围，未通过的股票仍保留在全量文件，不等于没有上涨机会。</p>
<p>TTM＝2026最新实绩累计＋2025全年－2025同期；现金及扣非都使用精确金额，未使用每股指标代理。最新Q3缺扣非EPS的问题已通过另一批量财务接口补齐。
对存储、CCL不能以峰值季度利润乘四。市值为当前A股价格折算全部股本，非A/H各市场价格合计。
科创成交量的股/手量纲已独立校正。研究优先级得分只用于安排人工核查，没有拟合未来收益，不应直接当买入排序。</p>
<p><strong>预告与实绩的区别：</strong>兴森按已发布实绩PE约270倍；若前三季预告兑现，替换为截至三季度的滚动利润后约135—145倍。
长川同样的预告滚动情景约57—60倍，鼎龙53—54倍、生益32.5—33.7倍、广合30.7—31.6倍。它们不是未来全年预测PE，也不是保证兑现。</p>
<p><strong>全量表检索</strong>（最多展示前200条；完整数据请下载CSV）</p>
<input id="query" placeholder="代码、名称或行业"><label><input id="gate" type="checkbox">只看通过基础量化门槛</label><span id="count"></span>
<div class="scroll"><table><thead><tr><th>代码</th><th>名称</th><th>行业</th><th>归母PE</th><th>扣非PE</th><th>回撤</th><th>Beta120</th><th>现金/利润</th><th>量化门槛</th></tr></thead><tbody id="allrows"></tbody></table></div>
<p><a href="全量科技筛选.csv">中文全量CSV</a> · <a href="all_technology_metrics.csv">全部字段CSV</a> · <a href="reviewed_candidates.csv">人工复核候选</a> ·
<a href="analysis_manifest.json">计算口径</a> · <a href="history_manifest.json">行情获取清单</a> · <a href="validation.json">交叉验证</a> · <a href="guidance_scenarios.csv">预告滚动情景</a></p>
<p class="muted">原始来源：腾讯公开前复权日线与报价；新浪全A证券名单；东方财富RPT_LICO_FN_CPD与RPT_F10_FINANCE_MAINFINADATA批量财报；最终候选再核公司公告。
相关脚本、输入快照与原始日线均保存在本报告所在目录。没有修改账户、下单、发送消息或改动现有策略缓存。</p>
<script>const data=__DATA__;const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=(v,n=2,p=false)=>v===null||v===undefined?'—':(v*(p?100:1)).toFixed(n)+(p?'%':'');
function render(){const q=document.getElementById('query').value.trim().toLowerCase();const gate=document.getElementById('gate').checked;
const rows=data.filter(r=>(!gate||r.primary_quant_gate)&&(!q||[r.code,r.name,r.industry].join(' ').toLowerCase().includes(q)));
document.getElementById('count').textContent=rows.length+' 条匹配';document.getElementById('allrows').innerHTML=rows.slice(0,200).map(r=>'<tr>'+[
r.code,r.name,r.industry,f(r.pe_ttm_calculated,1),f(r.pe_deduct_ttm,1),f(r.drawdown_250,1,true),f(r.star50_beta_120),f(r.cash_profit_ratio_actual),r.primary_quant_gate?'通过':'未通过/数据不足'].map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('');}
document.getElementById('query').addEventListener('input',render);document.getElementById('gate').addEventListener('change',render);render();</script></html>'''
    page = page.replace('__SUMMARY__', html.escape(summary)).replace('__ROWS__',''.join(note_rows)).replace('__DETAILS__',details).replace('__DATA__',payload)
    (BASE / 'report.html').write_text(page, encoding='utf-8')
    # Freeze source acquisition/normalisation scripts alongside the research outputs.
    for src, dst in [('/tmp/tech_screen_access/probe.py','source_quotes.py'),
                     ('/tmp/tech_financial_scan/fetch_financial.py','source_financial.py'),
                     ('/tmp/tech_financial_scan/fetch_exact.py','source_exact_financial.py'),
                     ('/tmp/tech_financial_scan/normalize.py','normalize_financial.py')]:
        shutil.copy2(src, str(BASE / dst))
    print('Report and readable exports saved:', BASE)
    print(selected[['code','name','role','pe_ttm_calculated','drawdown_250','star50_beta_120']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
