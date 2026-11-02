"""Join the reproducible screens and export a reviewable research report."""
from pathlib import Path
import json, html
import pandas as pd

B=Path(__file__).resolve().parent
f=pd.read_csv(B/'financial_screen_all.csv',dtype={'code':str})
p=pd.read_csv(B/'price_metrics.csv',dtype={'code':str})
m=f.merge(p,on=['code','symbol'],suffixes=('','_price'),how='left',validate='one_to_one')
m['pe_deduct']=m.cap_yi_cutoff_estimate*1e8/m.ttm_deduct_profit.where(m.ttm_deduct_profit>0)
m['pe']=m.cap_yi_cutoff_estimate*1e8/m.ttm_net_profit.where(m.ttm_net_profit>0)
m['price_gate']=(m.date=='2026-10-30')&(m.history_rows>=250)&(m.dd_250<=-.30)&(m.ret_20<=-.15)&(m.own_beta120>=1.10)&(m.own_n120>=100)&(m.liquidity20_proxy_yi>=.5)
m['quality_value_gate']=(m.cash_profit_ratio>=.5)&(m.debt_ratio<60)&(m.pe_deduct<=80)&(m.deduct_profit>=1e8)
m['combined_pass']=m.strict_financial_pass&m.price_gate&m.quality_value_gate
m.to_csv(B/'combined_screen.csv',index=False,encoding='utf-8-sig')
m[m.combined_pass].to_csv(B/'quantitative_candidates.csv',index=False,encoding='utf-8-sig')
order=['688008','300604','300223','300394','688766','688183','300408','300548','300811','300373','301183','300684','688110','688002','300308','300054','688019']
focus=m.set_index('code').loc[order].reset_index()
focus.to_csv(B/'reviewed_candidates.csv',index=False,encoding='utf-8-sig')
summary={'financial_records':len(f),'history_records':int(m.close.notna().sum()),'strict_financial':int(m.strict_financial_pass.sum()),'financial_and_price':int((m.strict_financial_pass&m.price_gate).sum()),'combined':int(m.combined_pass.sum()),'unmatched_financial_codes':m.loc[m.close.isna(),'code'].tolist(),'research_date':'2026-11-02','price_cutoff':'2026-10-30','not_backtest':True,'warning':'Heuristic research filters; not optimized or validated return predictions. Cash flow is consolidated, while profit is attributable to parent. A+H total capitalization is A-price-equivalent.'}
(B/'screen_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

evidence={
'300548':('更进攻备选：集中度和订单口径需留意','本板块Beta1.87，扣非环比+44.02%。45亿长协履行至2030年末，仅为估计金额且净额法按毛利确认，公司称预计不构成重大业绩影响，不能拿来证明Q4爆发。最大客户较集中；现金/归母约1，但现金/合并净利润约0.62。','https://static.cninfo.com.cn/finalpage/2026-07-27/1225443592.PDF'),
'300811':('增产备选：有具体新产能，但爬坡待跟踪','9月公告确认一体成型电感产线验收并大规模投入生产，达产规划约2亿颗/年，在手订单同比增长约400%；不能直接等同Q4完全达产。经营现金流同比下滑、资本开支需留意；USITC对部分下游VPD厂商的337调查存在潜在影响，铂科不是被调查对象，实际订单影响未核实。','https://static.cninfo.com.cn/finalpage/2026-09-07/1225551755.PDF'),
'300373':('较低估值备选','Q3收入、归母、扣非均改善，扣非TTM约25.7倍；未核到优于核心候选的公司Q4量化交付指引，因此保留在量化名单而不升为首选。','https://static.cninfo.com.cn/finalpage/2026-10-28/1225631689.PDF'),
'301183':('光器件进攻备选','量化弹性较好，已有扩产与光器件批量交付；Q4量化交付仍未核实，关键材料依赖、应收和约72倍扣非TTM估值限制首选地位。','https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12667071&stockid=301183'),
'300684':('单季环比很高，但需处理季节性','Q3扣非环比约105%，同比仅约15.7%。不能把消费电子三季度旺季环比增长自动外推为四季度再翻倍；未核到更强公司Q4量化指引。','https://static.cninfo.com.cn/finalpage/2026-10-30/1225641222.PDF'),
'688008':('首选研究：Q4订单证据最强','公司披露截至10/26预计Q4交付互连芯片订单25.0—27.9亿元，对比Q3同业务20.47亿元高22.1%—36.3%。订单不是确认收入。前三季现金流28.98亿元、负债率6.91%；归母含投资收益，应看扣非。67倍TTM扣非估值依然不低。','https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12678366&stockid=688008'),
'300604':('设备备选：环比增长整齐','Q3扣非7.95亿元、环比+33.25%。公司称订单充裕；未见量化Q4交付指引。存货61.80亿元、同比较年初+73.74%，研发资本化与验收进度需要跟踪，不能称零风险。','https://static.cninfo.com.cn/finalpage/2026-10-27/1225627248.PDF'),
'300223':('估值备选：Q4涨价而非确定增量','Q3扣非环比+67.28%，TTM扣非估值约21倍。10/27调研预计Q4存储毛利率继续增长，但DRAM产能紧张；这是涨价利润弹性，不能写成确定放量。商誉约30亿元。','https://static.cninfo.com.cn/finalpage/2026-10-28/1225637336.PDF'),
'300394':('进攻备选：弹性大、估值也高','核心物料供应缓解，有源无源同步增产；Q4进一步释放属于推断。CPO逐步提量主要从2027年开始，不应提前计入本季度。','https://static.cninfo.com.cn/finalpage/2026-10-28/1225638543.PDF'),
'688766':('高风险备选：并购与周期须折价','Q3收入约2/3来自SHM；Q2与Q3并表口径可比，但同比不能等同内生增长。剩余股权收购、供应商集中和存储高利润周期构成主要风险；未核到最新量化Q4指引。','https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12680064&stockid=688766'),
'688183':('PCB备选：财务较好，Beta略低于门槛','公司经营公告称订单充足，AI服务器及高速交换机需求增长；新增产能需认证爬坡。Beta1.08略低于1.10机械门槛，因此作为人工备选，未冒充量化通过。','https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12657665'),
'300408':('盈利强，但Q4行业开始分化','Q3扣非环比+72.16%，现金流较好。高容AI MLCC与消费级产品的需求及价格已分化，缺少公司Q4分规格数据，不能机械外推Q3。','https://www.trendforce.com.tw/presscenter/news/20261029-13283.html'),
'688110':('存储周期备选','Q4涨价展望有依据，但79.31%的Q3毛利率处在高周期水平；GPU投资不等于本季经营利润。','https://static.cninfo.com.cn/finalpage/2026-10-27/1225626412.PDF'),
'688002':('基本面候选，未过本次超跌弹性门槛','订单充裕、产能释放和8微米良率改善有公司IR支持；回撤26.2%、本板块Beta0.89，未达到本次高弹性深回撤口径。','https://pdf.dfcfw.com/pdf/H22_AN202610301831229776_1.pdf'),
'300308':('增长强，现金转化降权','收入、扣非环比改善，订单支持继续增长；9M现金/归母仅36%，且有海外政策风险，未过额外现金质量门槛。','https://static.cninfo.com.cn/finalpage/2026-10-30/1225643953.PDF'),
'300054':('材料备选，放宽收入同比阈值才入围','Q3三项环比正，现金流好；9M收入同比7.97%低于10%阈值。Q4新品推向市场不等于当季大额利润。','https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12664677&stockid=300054'),
'688019':('主业不错，归母受投资收益放大','扣非环比+31.67%，按扣非估值约40.7倍。前三季扣非占归母不足70%，因此未通过严格财务门槛；不应使用高额公允价值收益年化。','https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12682270&stockid=688019')}
rows=[]
for _,r in focus.iterrows():
    vals=[r['name']+' '+r.code,f'{r.close:.2f}',f'{r.dd_250:.1%}',f'{r.own_beta120:.2f}',f'{r.q3_revenue_qoq:+.1f}%',f'{r.q3_net_profit_qoq:+.1f}%',f'{r.q3_deduct_profit_qoq:+.1f}%',f'{r.pe_deduct:.1f}',str(bool(r.combined_pass))]
    rows.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in vals)+'</tr>')
cards=''.join('<article><h3>'+html.escape(focus.set_index('code').loc[c,'name']+' '+c+' — '+v[0])+'</h3><p>'+html.escape(v[1])+' <a href="'+v[2]+'">依据</a></p></article>' for c,v in evidence.items())
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双创超跌与Q4增长筛查</title><style>body{font:16px/1.75 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#192b3b}h1,h2,h3{line-height:1.35}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:right;padding:9px;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}th{background:#edf3f8}.box{background:#fff4d6;padding:18px;border-radius:8px}article{border-top:1px solid #ddd;padding:12px 0}a{color:#1263a5}img{width:100%}.scroll{overflow-x:auto}</style>
<h1>双创超跌与四季度增长筛查</h1><p>研究日2026-11-02；行情统一截至10月30日完整收盘，财务纳入截至研究日已披露三季报。本报告不是10月30日可交易时点回测。</p>
<p class="box"><b>按“基本面、环比增长、Q4有交付依据”的组合，优先研究澜起科技；长川为设备备选，君正为较低估值备选，天孚为更高历史弹性备选。</b> 这是一份条件性研究名单，不代表最低点已出现，也没有足够证据承诺新高。若要求同时“极低估值、强弹性、Q4确定放量、无风险”，没有一只完全满足。</p>
<h2>筛选范围与口径</h2><p>覆盖FIN家公司财务、PRICE家公司历史行情。财务严格通过STRICT家，叠加价格与弹性后PRICEPASS家，再加现金流、负债、估值及利润规模条件后FINAL家。6家旧代码无行情，均未过严格财务筛选；不是独立上市公司名册普查。</p>
<p>财务：Q2=H1−Q1，Q3=9M−H1；Q2/Q3归母与扣非均正，Q3收入/归母/扣非均环比增长，9M收入与扣非同比均&gt;10%，经营现金流&gt;0，扣非/归母&gt;70%，排除风险名称。价格：250日收盘高点回撤≥30%，近20日跌幅≥15%，本板块120日Beta≥1.10且至少100有效样本，至少250条历史，20日成交额估算中位数≥0.5亿元。额外质量：9M现金/归母≥0.5、负债率&lt;60%、TTM扣非PE≤80、9M扣非≥1亿元。阈值是研究选择，不是经过回测的收益保证。</p>
<p>前复权历史；回撤相对250交易日最高收盘，非历史最高盘中价。Beta对各自板块指数估算，各板块数值不能完全等价横比。TTM=2026前三季+2025全年−2025前三季，2025数据复用10月23日下载；未自动重述所有并购比较期。市值由报价×总股本回推至10/30，A+H为全部股本按A价估值，用于A股每股PE，而非两地实际市值合计。经营现金是合并口径，归母是母公司股东口径，比值仅作粗筛。</p>
<h2>核心候选</h2><div class="scroll"><table><thead><tr><th>股票</th><th>10/30收盘</th><th>250日回撤</th><th>Beta</th><th>Q3收入环比</th><th>归母环比</th><th>扣非环比</th><th>扣非TTM PE</th><th>量化全通过</th></tr></thead><tbody>ROWS</tbody></table></div><p>排序含人工证据判断，未过量化门槛者明确列为人工备选或排除，未改变门槛使其过关。</p>
<h2>四季度证据与风险</h2>CARDS
<h2>为什么不选几个热门票</h2><p>新易盛Q3收入/归母/扣非环比−12.98%/−19.07%/−23.47%；佰维−29.89%/−62.45%/−51.94%，且9M经营现金流负。胜宏归母−66.40%而扣非+9.99%，主业改善但投资损失不能忽略。中微收入、归母、扣非环比均降。源杰归母−2.54%，严格条件未过。协创负债约85.77%，大额长协以多年交付验收结算，不能当Q4确定收入。对应数字可在全量CSV逐项复核。</p>
<h2>新高需要什么</h2><p>澜起10/30价格158.30元，250日最高收盘315.30元，回去需涨99.2%。以Q3扣非10.05亿元简单乘4得到40.19亿元年化运行水平，当前A股总股本等价估值约48倍，回到该高点则约96倍；除非盈利继续显著增长或市场重新给出更高估值，否则前高不自动构成合理目标。单季年化只是敏感性分析，不是盈利预测。</p>
<h2>执行与失效条件</h2><p>以1—3个月窗口研究。优先等待指数止跌、个股收复10日线并能在回踩中守住，而不是仅因跌幅大满仓买入。澜起10/30的10日线约169.92、20日线179.90、近5日低点157.72元，均为静态参考，需交易当日重新计算，不构成保证支撑。若买入后再次有效跌破新形成的低点，或Q4订单延期、互连毛利率显著下滑，应减少风险，不能无限摊低成本。风险预算示例：愿让单笔交易最多损失总资金1%，设想退出距离8%，则仓位上限约12.5%；跳空和流动性会使实际损失超出计划。</p>
<h2>数据与复现</h2><p><a href="combined_screen.csv">全量合并CSV</a> · <a href="quantitative_candidates.csv">量化通过CSV</a> · <a href="reviewed_candidates.csv">人工重点复核CSV</a> · <a href="financial_manifest.json">财务来源</a> · <a href="price_manifest.json">行情来源与错误</a> · <a href="screen_summary.json">筛选计数</a></p><p>程序：financial_screen.py、price_screen.py、build_report.py。公开接口：东方财富RPT_F10_FINANCE_MAINFINADATA；腾讯前复权日线。原始JSON保留在financial_raw及price_raw。财务报表未审计；本轮没有声称完成所有诉讼、监管、实控人及未来公告的全面风险排除。</p></html>'''
for a,b in [('PRICEPASS',str(summary['financial_and_price'])),('STRICT',str(summary['strict_financial'])),('FINAL',str(summary['combined'])),('FIN',str(summary['financial_records'])),('PRICE',str(summary['history_records'])),('ROWS',''.join(rows)),('CARDS',cards)]:page=page.replace(a,b)
(B/'report.html').write_text(page,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
