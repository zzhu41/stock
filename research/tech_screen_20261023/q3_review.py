"""Add a Q3-vs-Q2 earnings check without changing the original research snapshot."""
import html
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
GUIDANCE = {
    '300604': (17.15, 18.15, 16.67, 17.67),
    '688183': (18.686, 19.517, 18.649, 19.479),
    '001389': (14.5, 15.0, 14.2, 14.7),
    '300054': (8.4, 8.6, 7.8, 8.0),
    '002436': (4.0, 4.3, 4.2, 4.5),
    '300458': (6.4, 6.7, 6.1, 6.3),
}
SOURCES = {
    '300604': 'https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12653220&stockid=300604',
    '688183': 'https://static.cninfo.com.cn/finalpage/2026-10-17/1225608856.PDF',
    '001389': 'https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12641383&stockid=001389',
    '300054': 'https://disc.static.szse.cn/download/disc/disk03/finalpage/2026-10-09/b1d39cd1-0baf-41c7-afc1-c01b9e3e18f0.PDF',
    '002436': 'https://static.cninfo.com.cn/finalpage/2026-10-14/1225602695.PDF',
    '300458': 'https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12638227&stockid=300458',
    '603186': 'https://static.cninfo.com.cn/finalpage/2026-10-23/1225615477.PDF',
    '688368': 'https://www.cfi.net.cn/p20261022001385.html',
    '300620': 'https://static.cninfo.com.cn/finalpage/2026-10-23/1225616009.PDF',
    '688313': 'https://static.cninfo.com.cn/finalpage/2026-10-22/1225614249.PDF',
}


def classify(lo, hi, base):
    if pd.isna(lo) or pd.isna(hi) or pd.isna(base):
        return '未核实Q3'
    if base <= 0:
        return '基期亏损或为零，单独核查'
    if hi < base:
        return '环比下降'
    if lo >= base:
        return '环比不降'
    return '区间跨越零增长，未确定'


def main():
    frames = {}
    for period in ['2026-03-31', '2026-06-30', '2026-09-30']:
        p = '/tmp/tech_screen_financial/exact_' + period + '.csv'
        frames[period] = pd.read_csv(p, dtype={'code': str}).set_index('code')
    shutil.copy2('/tmp/tech_screen_financial/exact_2026-03-31.csv',
                 str(BASE / 'inputs' / 'exact_2026-03-31.csv'))
    pool = pd.read_csv(str(BASE / 'all_technology_metrics.csv'), dtype={'code': str})
    q1, h1, ytd = [frames[x] for x in ['2026-03-31','2026-06-30','2026-09-30']]
    records = []
    for _, stock in pool.iterrows():
        code = stock.code
        r = {'code': code, 'name': stock['name'], 'primary_quant_gate': stock.primary_quant_gate,
             'source_type': '未核实Q3', 'source_url': SOURCES.get(code, '')}
        for metric in ['net_profit', 'deduct_profit']:
            first = q1[metric].get(code, np.nan)
            half = h1[metric].get(code, np.nan)
            r['q2_' + metric] = half - first
            lo, hi = np.nan, np.nan
            if code in ytd.index:
                lo = hi = ytd.loc[code, metric] - half
                r['source_type'] = '正式三季报'
            elif code in GUIDANCE:
                g = GUIDANCE[code]
                low, high = g[:2] if metric == 'net_profit' else g[2:]
                lo, hi = low * 1e8 - half, high * 1e8 - half
                r['source_type'] = '前三季预告推算'
            r['q3_' + metric + '_low'] = lo
            r['q3_' + metric + '_high'] = hi
            base = r['q2_' + metric]
            r[metric + '_qoq_low'] = (lo / base - 1) * 100 if base > 0 else np.nan
            r[metric + '_qoq_high'] = (hi / base - 1) * 100 if base > 0 else np.nan
            r[metric + '_status'] = classify(lo, hi, base)
        if r['net_profit_status'] == r['deduct_profit_status'] == '环比不降':
            r['strict_both_pass'] = True
            r['review_status'] = '归母及扣非均不降：通过本项'
        else:
            r['strict_both_pass'] = False
            if '环比下降' in [r['net_profit_status'], r['deduct_profit_status']]:
                r['review_status'] = '至少一项环比下降：暂移出严格主名单'
            else:
                r['review_status'] = '等待核实：不能视作通过'
        records.append(r)
    out = pd.DataFrame(records)
    out.to_csv(str(BASE / 'q3_review_all.csv'), index=False, encoding='utf-8-sig')
    order = ['300604','688183','001389','300054','603186','002436','300458','688368','300620','688313',
             '300223','688019','688766','300475']
    chosen = out.set_index('code').loc[order].reset_index()
    chosen.to_csv(str(BASE / 'q3_review_candidates.csv'), index=False, encoding='utf-8-sig')
    p = out[out.primary_quant_gate]
    counts = p.review_status.value_counts().to_dict()
    meta = {'basis': 'Q2=H1-Q1; Q3=9M-H1. Forecast intervals remain intervals; nonpositive Q2 denominators reviewed separately.',
            'original_quant_pool': len(p), 'counts': counts,
            'guidance_checked_codes': list(GUIDANCE),
            'unknown_note': 'Unverified does not prove no disclosure; six guidance issuers were explicitly checked, not all guidance issuers across the market.',
            'date': '2026-10-23'}
    (BASE / 'q3_review_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    def rate(row, name):
        lo, hi = row[name+'_qoq_low'], row[name+'_qoq_high']
        if pd.isna(lo): return '待验证'
        return ('%+.1f%%' % lo) if abs(lo-hi)<1e-8 else ('%+.1f%% ～ %+.1f%%' % (lo,hi))
    rows = []
    for _, r in chosen.iterrows():
        cells = [r['name']+' '+r.code, r.source_type, rate(r,'net_profit'), rate(r,'deduct_profit'),r.review_status]
        rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in cells)+
                    ('<td><a href="'+html.escape(r.source_url,quote=True)+'">公告/来源</a></td>' if r.source_url else '<td>等待披露核验</td>')+'</tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>三季度环比复核</title><style>body{font:16px/1.7 system-ui;max-width:1300px;margin:30px auto;padding:0 20px;color:#172536}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f7}a{color:#125eaa}.box{background:#fff3d2;padding:15px}</style>
<h1>2026年第三季度单季环比复核</h1><p class="box">若采用“归母和扣非利润环比均不降”的严格条件，长川、生益、鼎龙及已有华正、兴森通过本项（其中预告尚待正式报告确认）。广合、全志移出严格主名单；君正、安集、普冉为Q3待验证，不是已确认下降。通过本项不等于估值、现金流及其他风险检查全部通过。</p>
<p>计算：Q2＝半年累计－第一季度；Q3＝前三季度累计－半年累计。环比用单季金额比较。预告用完整上下限；亏损/零基数不使用普通百分比机械判断。</p>
<table><thead><tr><th>公司</th><th>数据状态</th><th>归母环比</th><th>扣非环比</th><th>严格条件结果</th><th>依据</th></tr></thead><tbody>ROWS</tbody></table>
<p>单季下降可能包含季节性、费用投放、并表或一次性收益基数。为这轮偏进攻交易，暂移出主名单可以降低盈利放缓风险，但不是永久否定公司；未披露者不能按增长放行。</p>
<p><a href="q3_review_candidates.csv">候选明细CSV</a> · <a href="q3_review_all.csv">全池复核CSV</a> · <a href="q3_review_manifest.json">覆盖范围与口径</a> · <a href="report.html">原研究报告</a></p></html>'''
    (BASE / 'q3_review.html').write_text(page.replace('ROWS',''.join(rows)), encoding='utf-8')
    original = BASE / 'report.html'
    body = original.read_text(encoding='utf-8')
    if 'id="q3-update"' not in body:
        marker = '<h1>A股科技反弹筛选｜2026年10月23日</h1>'
        banner = '<p id="q3-update" class="note"><strong>新增复核：</strong><a href="q3_review.html">第三季度单季环比检查</a>已补齐。广合转观察；君正、安集、普冉尚未确认Q3。下文为加入该条件前的原研究结论，请结合更新阅读。</p>'
        original.write_text(body.replace(marker,marker+banner), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(chosen[['code','name','source_type','net_profit_qoq_low','net_profit_qoq_high',
                  'deduct_profit_qoq_low','deduct_profit_qoq_high','review_status']].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
