"""Reproducible public-data financial screen; amounts are CNY, growth rates percent.

Run: python research/dual_innovation_20261102/financial_screen.py
Source: Eastmoney financial main indicator API. 2025 periods reuse the retained
2026-10-23 downloads; 2026 periods are freshly fetched with disclosure cutoff.
This is an accounting screen, not an investment recommendation or risk clearance.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
RAW = BASE / 'financial_raw'
RAW.mkdir(exist_ok=True)
URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
CUTOFF = '2026-11-03'
MAPPING = {'SECURITY_CODE':'code','SECURITY_NAME_ABBR':'name',
 'REPORT_DATE':'report_date','NOTICE_DATE':'notice_date','UPDATE_DATE':'update_date',
 'TOTALOPERATEREVE':'revenue','PARENTNETPROFIT':'net_profit','KCFJCXSYJLR':'deduct_profit',
 'NETCASH_OPERATE_PK':'operating_cashflow','TOTALOPERATEREVETZ':'revenue_growth',
 'PARENTNETPROFITTZ':'profit_growth','KCFJCXSYJLRTZ':'deduct_profit_growth',
 'TOTAL_SHARE':'total_shares','ROEJQ':'roe','XSMLL':'gross_margin','ZCFZL':'debt_ratio',
 'CHZZTS':'inventory_days','YSZKZZTS':'receivable_days',
 'DJD_DPNP_QOQ':'vendor_single_quarter_profit_qoq',
 'DJD_DEDUCTDPNP_QOQ':'vendor_single_quarter_deduct_qoq'}


def get(date, page=1):
    params = {'reportName':'RPT_F10_FINANCE_MAINFINADATA','columns':'ALL',
      'filter':f'(REPORT_DATE=\'{date}\')(SECURITY_TYPE_CODE="058001001")(NOTICE_DATE<\'{CUTOFF}\')',
      'pageSize':500,'pageNumber':page,'sortColumns':'SECURITY_CODE,NOTICE_DATE',
      'sortTypes':'1,-1','source':'WEB','client':'WEB'}
    for attempt in range(3):
        try:
            response = requests.get(URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            if not payload.get('success'):
                raise ValueError(str(payload)[:500])
            return payload
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


def load_period(date):
    dest = RAW / ('raw_exact_' + date + '.json')
    provenance = 'fresh_download'
    if dest.exists():
        rows = json.loads(dest.read_text())
        provenance = ('reused_2026-10-23_download:/tmp/tech_screen_financial/' + dest.name
                      if date.startswith('2025') else 'this_run_cached_fresh_2026-11-02_download')
    elif date.startswith('2025'):
        source = Path('/tmp/tech_screen_financial') / dest.name
        rows = json.loads(source.read_text())
        provenance = 'reused_2026-10-23_download:' + str(source)
        rows = [r for r in rows if r['SECURITY_CODE'].startswith(('30','688','689'))]
        dest.write_text(json.dumps(rows, ensure_ascii=False))
    else:
        first = get(date)
        rows = first['result']['data']
        pages = first['result']['pages']
        with ThreadPoolExecutor(max_workers=4) as executor:
            for payload in executor.map(lambda page:get(date,page),range(2,pages+1)):
                rows += payload['result']['data']
        rows = [r for r in rows if r['SECURITY_CODE'].startswith(('30','688','689'))]
        dest.write_text(json.dumps(rows, ensure_ascii=False))
    frame = pd.DataFrame([{dst:r.get(src) for src,dst in MAPPING.items()} for r in rows])
    frame = frame.sort_values(['code','notice_date'],ascending=[True,False]).drop_duplicates('code')
    for col in ['report_date','notice_date','update_date']:
        frame[col] = frame[col].str[:10]
    frame.to_csv(BASE / ('financial_'+date+'.csv'),index=False,encoding='utf-8-sig')
    print(date, len(frame), provenance, flush=True)
    return frame.set_index('code'), {'date':date,'unique_codes':len(frame),
       'raw_rows':len(rows),'source':provenance,'raw_file':str(dest),
       'retained_file_modified_at':datetime.fromtimestamp(dest.stat().st_mtime).isoformat()}


def main():
    periods = ['2026-03-31','2026-06-30','2026-09-30','2025-09-30','2025-12-31']
    frames, coverage = {}, []
    for date in periods:
        frames[date], info = load_period(date)
        coverage.append(info)
    q1,h1,q3,old9,fy = [frames[date] for date in periods]
    out = q3.copy()
    out['symbol'] = np.where(out.index.str.startswith('30'),'sz','sh') + out.index
    out['has_q1_h1'] = out.index.isin(q1.index) & out.index.isin(h1.index)
    for metric in ['revenue','net_profit','deduct_profit','operating_cashflow']:
        out['q1_'+metric] = q1[metric]
        out['q2_'+metric] = h1[metric] - q1[metric]
        out['q3_'+metric] = q3[metric] - h1[metric]
        out['q3_'+metric+'_qoq'] = (out['q3_'+metric] / out['q2_'+metric].where(out['q2_'+metric] > 0) - 1) * 100
        out['ttm_'+metric] = q3[metric] + fy[metric] - old9[metric]
    out['deduct_ratio'] = out.deduct_profit / out.net_profit.where(out.net_profit > 0)
    out['cash_profit_ratio'] = out.operating_cashflow / out.net_profit.where(out.net_profit > 0)
    out['ttm_cash_profit_ratio'] = out.ttm_operating_cashflow / out.ttm_net_profit.where(out.ttm_net_profit > 0)
    out['ttm_deduct_ratio'] = out.ttm_deduct_profit / out.ttm_net_profit.where(out.ttm_net_profit > 0)
    for metric in ['inventory_days','receivable_days','debt_ratio','gross_margin']:
        out['h1_'+metric] = h1[metric]
        out['prior9_'+metric] = old9[metric]
        out[metric+'_yoy_delta'] = out[metric] - out['prior9_'+metric]
    gates = {
      'clean_name':~out.name.str.contains('ST|退',na=False),
      'quarter_profits_positive':(out.q2_net_profit>0)&(out.q3_net_profit>0)&(out.q2_deduct_profit>0)&(out.q3_deduct_profit>0),
      'revenue_qoq_positive':out.q3_revenue_qoq>0,
      'profit_qoq_positive':out.q3_net_profit_qoq>0,
      'deduct_qoq_positive':out.q3_deduct_profit_qoq>0,
      'revenue_yoy_gt10':out.revenue_growth>10,
      'deduct_yoy_gt10':out.deduct_profit_growth>10,
      'cashflow_positive':out.operating_cashflow>0,
      'deduct_ratio_gt70':out.deduct_ratio>0.7,
    }
    for name,gate in gates.items():
        out['gate_'+name] = gate.fillna(False)
    out['strict_financial_pass'] = out[['gate_'+k for k in gates]].all(axis=1) & out.has_q1_h1
    out['relaxed_financial_pass'] = (out.gate_clean_name & out.gate_quarter_profits_positive &
        out.gate_profit_qoq_positive & out.gate_deduct_qoq_positive & (out.revenue_growth > 0) &
        (out.deduct_profit_growth > 0) & (out.deduct_ratio > .5))
    out['failed_gates'] = out.apply(lambda row:','.join(name for name in gates if not row['gate_'+name]),axis=1)
    # These are review flags, not implicit exclusions or evidence of misconduct.
    out['flag_debt_gt60'] = out.debt_ratio > 60
    out['flag_receivable_days_up30'] = out.receivable_days_yoy_delta > 30
    out['flag_inventory_days_up60'] = out.inventory_days_yoy_delta > 60
    out['flag_ocf_below_half_profit'] = out.cash_profit_ratio < .5
    out['flag_small_q2_deduct_lt10m'] = out.q2_deduct_profit < 1e7
    out['flag_missing_ttm'] = out.ttm_net_profit.isna() | out.ttm_operating_cashflow.isna()
    out['profit_qoq_vendor_delta_pp'] = out.q3_net_profit_qoq - out.vendor_single_quarter_profit_qoq
    out['deduct_qoq_vendor_delta_pp'] = out.q3_deduct_profit_qoq - out.vendor_single_quarter_deduct_qoq
    out['flag_vendor_qoq_discrepancy'] = ((out.profit_qoq_vendor_delta_pp.abs() > .1) |
                                           (out.deduct_qoq_vendor_delta_pp.abs() > .1))
    out.index.name = 'code'
    out.to_csv(BASE/'financial_screen_all.csv',encoding='utf-8-sig')
    strict = out[out.strict_financial_pass].sort_values('deduct_profit',ascending=False)
    strict.to_csv(BASE/'financial_strict.csv',encoding='utf-8-sig')
    out[out.relaxed_financial_pass].to_csv(BASE/'financial_relaxed.csv',encoding='utf-8-sig')
    quality = strict[(strict.cash_profit_ratio >= .5) & (strict.debt_ratio < 60) &
       (strict.deduct_profit >= 1e8) & (strict.q3_net_profit_qoq >= 10) &
       (strict.q3_deduct_profit_qoq >= 10) & ~strict.flag_receivable_days_up30 &
       ~strict.flag_inventory_days_up60 & ~strict.flag_vendor_qoq_discrepancy]
    quality.to_csv(BASE/'financial_quality_focus.csv',encoding='utf-8-sig')
    manifest = {'retrieved_at':datetime.now().isoformat(),'source':URL,'report':'RPT_F10_FINANCE_MAINFINADATA',
      'disclosure_cutoff_exclusive':CUTOFF,'period_coverage':coverage,'q3_dual_board_count':len(out),
      'complete_q1_h1_q3':int(out.has_q1_h1.sum()),'strict_pass':len(strict),
      'relaxed_pass':int(out.relaxed_financial_pass.sum()),
      'quality_focus_pass':len(quality),
      'quality_focus_extra_rules':'9M cash/profit >=0.5; debt <60%; 9M deduct >=100m CNY; Q3 profit and deduct QoQ >=10%; no flagged working-capital deterioration or vendor discrepancy. Missing turnover values do not themselves trigger a flag.',
      'vendor_qoq_comparison':{'profit_pairs':int(out.profit_qoq_vendor_delta_pp.notna().sum()),
       'deduct_pairs':int(out.deduct_qoq_vendor_delta_pp.notna().sum()),
       'discrepancy_above_0_1_percentage_points':int(out.flag_vendor_qoq_discrepancy.sum()),
       'codes_to_review':list(out.index[out.flag_vendor_qoq_discrepancy]),
       'note':'Use cumulative subtraction consistently. Discrepancies can reflect restatements or vendor basis; review original reports before final selection.'},
      'gate_pass_counts':{name:int(gate.sum()) for name,gate in gates.items()},
      'q3_notices':out.notice_date.value_counts().to_dict(),
      'definitions':{'q2':'H1-Q1','q3':'9M-H1','TTM':'2026 9M + 2025 FY - 2025 9M',
       'quarter_qoq':'100 * (Q3/Q2 - 1); NaN where Q2 <= 0',
       'universe':'All financial API records starting 30, 688, or 689; not an independent listing census',
       'screen_limits':'Accounting metrics only; no audit of sanctions, litigation, future demand, delisting or reporting integrity. Prior-period comparatives are not automatically reconciled for restatements.'}}
    (BASE/'financial_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k in ['q3_dual_board_count','complete_q1_h1_q3','strict_pass','relaxed_pass','gate_pass_counts']},ensure_ascii=False,indent=2))
    print(strict[['name','q3_revenue_qoq','q3_net_profit_qoq','q3_deduct_profit_qoq',
       'deduct_profit','revenue_growth','deduct_profit_growth','cash_profit_ratio','debt_ratio']].head(40).to_string())


if __name__ == '__main__':
    main()
