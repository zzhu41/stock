import os,requests,json,csv,time,sys
from concurrent.futures import ThreadPoolExecutor
OUT='/tmp/tech_screen_financial'
BASE='https://datacenter-web.eastmoney.com/api/data/v1/get'
def get(date,page=1):
    p={'reportName':'RPT_F10_FINANCE_MAINFINADATA','columns':'ALL','filter':"(REPORT_DATE='%s')(SECURITY_TYPE_CODE=\"058001001\")(NOTICE_DATE<'2026-10-24')"%date,'pageSize':500,'pageNumber':page,'sortColumns':'SECURITY_CODE,NOTICE_DATE','sortTypes':'1,-1','source':'WEB','client':'WEB'}
    for i in range(3):
        try:
            r=requests.get(BASE,params=p,timeout=25);r.raise_for_status();j=r.json()
            if not j.get('success'):raise RuntimeError(str(j)[:300])
            return j
        except Exception:
            if i==2:raise
            time.sleep(1)
mapping={'SECURITY_CODE':'code','SECURITY_NAME_ABBR':'name','REPORT_DATE':'report_date','NOTICE_DATE':'notice_date','UPDATE_DATE':'update_date','TOTALOPERATEREVE':'revenue','PARENTNETPROFIT':'net_profit','KCFJCXSYJLR':'deduct_profit','NETCASH_OPERATE_PK':'operating_cashflow','NETCASH_INVEST_PK':'investing_cashflow','NETCASH_FINANCE_PK':'financing_cashflow','TOTALOPERATEREVETZ':'revenue_growth','PARENTNETPROFITTZ':'profit_growth','KCFJCXSYJLRTZ':'deduct_profit_growth','TOTAL_SHARE':'total_shares','A_FREE_SHARE':'free_shares','ROEJQ':'roe','ROEKCJQ':'deduct_roe','XSMLL':'gross_margin','ZCFZL':'debt_ratio','CHZZTS':'inventory_days','YSZKZZTS':'receivable_days','FCFF_FORWARD':'fcff_forward','FCFF_BACK':'fcff_back','DJD_DPNP_YOY':'single_quarter_profit_yoy','DJD_DEDUCTDPNP_YOY':'single_quarter_deduct_yoy','DJD_DPNP_QOQ':'single_quarter_profit_qoq','DJD_DEDUCTDPNP_QOQ':'single_quarter_deduct_qoq'}
for date in (sys.argv[1:] or ['2026-06-30','2026-09-30']):
    j=get(date);rows=j['result']['data'];pages=j['result']['pages'];print(date,'pages',pages,flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for p,j in zip(range(2,pages+1),ex.map(lambda p:get(date,p),range(2,pages+1))):
            rows+=j['result']['data'];print(date,'page',p,'rows',len(rows),flush=True)
    with open(OUT+'/raw_exact_'+date+'.json','w') as f:json.dump(rows,f,ensure_ascii=False)
    out=[];seen=set()
    for x in rows:
        if x['SECURITY_CODE'] in seen:continue
        seen.add(x['SECURITY_CODE']);y={v:x.get(k) for k,v in mapping.items()}
        for k in ['report_date','notice_date','update_date']:y[k]=(y[k] or '')[:10]
        y['ocf_profit_ratio']=y['operating_cashflow']/y['net_profit'] if y['net_profit'] and y['net_profit']>0 and y['operating_cashflow'] is not None else None
        y['deduct_ratio']=y['deduct_profit']/y['net_profit'] if y['net_profit'] and y['net_profit']>0 and y['deduct_profit'] is not None else None
        out.append(y)
    with open(OUT+'/exact_'+date+'.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print(date,'raw',len(rows),'unique',len(out),'deduct_missing',sum(x['deduct_profit'] is None for x in out),'cash_missing',sum(x['operating_cashflow'] is None for x in out),flush=True)
