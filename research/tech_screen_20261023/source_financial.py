import os,json,time,requests,csv,sys
from concurrent.futures import ThreadPoolExecutor
BASE='https://datacenter-web.eastmoney.com/api/data/v1/get'
OUT='/tmp/tech_financial_scan'
os.makedirs(OUT,exist_ok=True)
def fetch(date,page=1,size=500):
    p={'reportName':'RPT_LICO_FN_CPD','columns':'ALL','filter':"(REPORTDATE='%s')(SECURITY_TYPE_CODE=\"058001001\")"%date,'pageSize':size,'pageNumber':page,'sortColumns':'SECURITY_CODE,NOTICE_DATE','sortTypes':'1,-1','source':'WEB','client':'WEB'}
    r=requests.get(BASE,params=p,timeout=25); r.raise_for_status(); j=r.json()
    if not j.get('success'): raise RuntimeError(str(j)[:300])
    return j
def main(date):
    first=fetch(date); pages=first['result']['pages']; data=first['result']['data']; print(date,'pages',pages,'first',len(data),flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for p,j in zip(range(2,pages+1),ex.map(lambda p:fetch(date,p),range(2,pages+1))):
            data+=j['result']['data']; print(date,'page',p,'rows',len(data),flush=True)
    with open(OUT+'/financial_'+date+'.json','w') as f: json.dump(data,f,ensure_ascii=False)
    keys=list(data[0]) if data else []
    with open(OUT+'/financial_'+date+'.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(data)
    from collections import Counter
    print(date,'rows',len(data),'unique',len(set(x['SECURITY_CODE'] for x in data)),'industries',Counter(x['BOARD_NAME'] for x in data),flush=True)
for date in (sys.argv[1:] or ['2026-06-30','2026-09-30']):
    main(date)
