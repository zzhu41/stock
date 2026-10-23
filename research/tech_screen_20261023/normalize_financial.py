import os,json,csv,shutil,glob
from collections import Counter
SRC='/tmp/tech_financial_scan'
OUT='/tmp/tech_screen_financial'
os.makedirs(OUT,exist_ok=True)
mapping={'SECURITY_CODE':'code','SECURITY_NAME_ABBR':'name','BOARD_NAME':'industry','REPORTDATE':'report_date','NOTICE_DATE':'notice_date','EITIME':'entry_time','TOTAL_OPERATE_INCOME':'revenue','PARENT_NETPROFIT':'net_profit','YSTZ':'revenue_growth','SJLTZ':'profit_growth','XSMLL':'gross_margin','WEIGHTAVG_ROE':'roe','BASIC_EPS':'eps','DEDUCT_BASIC_EPS':'deduct_eps','MGJYXJJE':'ocfps','BPS':'bps','SJLHZ':'profit_qoq','YSHZ':'revenue_qoq','TRADE_MARKET':'market','SECUCODE':'secucode'}
allrows=[]
for date in sorted(p.split('financial_')[-1].split('.json')[0] for p in glob.glob(SRC+'/financial_*.json')):
    data=json.load(open(SRC+'/financial_'+date+'.json'))
    shutil.copyfile(SRC+'/financial_'+date+'.json',OUT+'/raw_'+date+'.json')
    kept=[]
    for x in data:
        notice=(x.get('NOTICE_DATE') or '')[:10]
        entry=(x.get('EITIME') or '')
        if notice>'2026-10-23' or entry>'2026-10-23 16:35:00': continue
        y={v:x.get(k) for k,v in mapping.items()}
        for c in ['report_date','notice_date']: y[c]=(y[c] or '')[:10]
        eps=y['eps']; cf=y['ocfps']; dep=y['deduct_eps']
        y['ocf_profit_proxy']=cf/eps if eps is not None and eps>0 and cf is not None else None
        y['deduct_profit_proxy']=y['net_profit']*dep/eps if eps and dep is not None else None
        kept.append(y)
    with open(OUT+'/normalized_'+date+'.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(kept[0])); w.writeheader(); w.writerows(kept)
    allrows+=kept
    print(date,'kept',len(kept),'excluded',len(data)-len(kept),'latestnotice',max(x['notice_date'] for x in kept))
latest={}
for x in allrows:
    if x['report_date']<'2026-01-01': continue
    if x['code'] not in latest or x['report_date']>latest[x['code']]['report_date']: latest[x['code']]=x
with open(OUT+'/latest_asof_2026-10-23.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(latest.values())
print('latest',len(latest),'q3',sum(x['report_date']=='2026-09-30' for x in latest.values()))
print('notice q3 distribution',Counter(x['notice_date'] for x in allrows if x['report_date']=='2026-09-30'))
