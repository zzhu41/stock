import json, requests, sys, csv, math, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
out=Path('/tmp/tech_screen_access')
if '--tencent-all' in sys.argv:
 import re
 rows=list(csv.DictReader((out/'universe.csv').open()))
 symbols=[r['symbol'] for r in rows]+['sh000688','sz399006','sh000001']
 chunks=[symbols[i:i+100] for i in range(0,len(symbols),100)]
 rawdir=out/'tencent_raw';rawdir.mkdir(exist_ok=True)
 def fetchqt(i,syms):
  path=rawdir/('batch_%03d.txt'%i)
  for attempt in range(3):
   try:
    r=requests.get('https://qt.gtimg.cn/q='+','.join(syms),headers={'User-Agent':'Mozilla/5.0','Referer':'https://gu.qq.com/'},timeout=20)
    r.raise_for_status();txt=r.content.decode('gbk')
    if 'v_' not in txt: raise ValueError(txt[:100])
    path.write_text(txt)
    data=[]
    for sym,val in re.findall(r'v_(\w+)="([^"]*)";',txt):
     f=val.split('~')
     data.append({'symbol':sym,'fields':f})
    return data
   except Exception:
    if attempt==2:raise
    time.sleep(1+attempt)
 allquotes=[];errors=[]
 with ThreadPoolExecutor(max_workers=3) as ex:
  fs={ex.submit(fetchqt,i,s):i for i,s in enumerate(chunks)}
  for f in as_completed(fs):
   try: allquotes.extend(f.result())
   except Exception as e:errors.append({'batch':fs[f],'error':str(e)})
   if len(allquotes)%1000<100:print('quotes',len(allquotes),flush=True)
 (out/'tencent_fields.json').write_text(json.dumps(allquotes,ensure_ascii=False))
 parsed=[]
 for q in allquotes:
  f=q['fields']
  if len(f)<47:continue
  def val(i):return f[i] if i<len(f) else ''
  def num(i):
   try:return float(val(i))
   except:return None
  stamp=val(30)
  date=stamp[:4]+'-'+stamp[4:6]+'-'+stamp[6:8] if len(stamp)==14 and stamp.isdigit() else ''
  parsed.append({'symbol':q['symbol'],'code':val(2),'name':val(1),'price':num(3),'prev_close':num(4),'open':num(5),'volume_hands':num(6),'timestamp':stamp,'trading_date':date,'change_pct':num(32),'day_high':num(33),'day_low':num(34),'amount_wan':num(37),'turnover_pct':num(38),'pe_tencent_f39':num(39),'float_cap_yi':num(44),'market_cap_yi':num(45),'pb':num(46),'field49':num(49),'field52':num(52),'field53':num(53),'field66':num(66),'field67':num(67),'field_count':len(f)})
 with (out/'tencent_quotes.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,parsed[0].keys());w.writeheader();w.writerows(parsed)
 meta={'requested':len(symbols),'response_records':len(allquotes),'parsed':len(parsed),'errors':errors,'retrieved_at':datetime.now().isoformat(timespec='seconds'),'field_mapping_note':'f39 PE, f44/45 market cap CNY 100m, f52/f53 and 66/67 retained as raw numbered fields pending verification.'}
 (out/'tencent_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 print(json.dumps(meta,ensure_ascii=False),flush=True)
 for q in allquotes:
  if q['symbol'] in ['sh603186','sz002436']:
   print(q['symbol'],list(enumerate(q['fields'])),flush=True)
 sys.exit(0)
if '--fetch-all' in sys.argv:
 base='https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/'
 headers={'User-Agent':'Mozilla/5.0','Referer':'https://finance.sina.com.cn/'}
 r=requests.get(base+'Market_Center.getHQNodeStockCount',params={'node':'hs_a'},headers=headers,timeout=20)
 count=int(r.text.strip('"'))
 print('count',count,flush=True)
 def fetch(page):
  path=out/('sina_page_%03d.json'%page)
  if path.exists(): return json.loads(path.read_text())
  for attempt in range(3):
   try:
    r=requests.get(base+'Market_Center.getHQNodeData',params={'page':page,'num':80,'sort':'symbol','asc':1,'node':'hs_a','symbol':'','_s_r_a':'page'},headers=headers,timeout=20)
    r.raise_for_status(); data=r.json()
    if not isinstance(data,list): raise ValueError(str(data)[:200])
    path.write_text(json.dumps(data,ensure_ascii=False))
    return data
   except Exception:
    if attempt==2: raise
    time.sleep(1+attempt)
 allrows=[]; errors=[]
 with ThreadPoolExecutor(max_workers=3) as ex:
  fs={ex.submit(fetch,p):p for p in range(1,math.ceil(count/80)+1)}
  for f in as_completed(fs):
   try: allrows.extend(f.result())
   except Exception as e: errors.append({'page':fs[f],'error':str(e)})
   if len(allrows)%800==0: print('downloaded',len(allrows),flush=True)
 allrows.sort(key=lambda x:x['symbol'])
 (out/'universe_raw.json').write_text(json.dumps(allrows,ensure_ascii=False))
 fields=['code','symbol','name','industry','price','change_pct','amount_yuan','market_cap_yi','float_cap_yi','pe_sina','pb','turnover_pct','source_ticktime','retrieved_at','trading_date']
 now=datetime.now().isoformat(timespec='seconds')
 with (out/'universe.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fields); w.writeheader()
  for x in allrows:
   def num(key):
    try: return float(x.get(key) or 0)
    except: return None
   w.writerow({'code':x['code'],'symbol':x['symbol'],'name':x['name'],'industry':'','price':num('trade'),'change_pct':num('changepercent'),'amount_yuan':num('amount'),'market_cap_yi':num('mktcap')/10000,'float_cap_yi':num('nmc')/10000,'pe_sina':num('per'),'pb':num('pb'),'turnover_pct':num('turnoverratio'),'source_ticktime':x.get('ticktime'),'retrieved_at':now,'trading_date':''})
 meta={'source':base,'node':'hs_a','expected_count':count,'fetched_count':len(allrows),'unique_codes':len({x['code'] for x in allrows}),'errors':errors,'retrieved_at':now,'price_note':'Sina response gives ticktime but no date; trading_date deliberately blank until quote cross-validation.','pe_note':'Sina per field basis not separately verified; do not assume forward or TTM.','market_cap_original_unit':'CNY 10,000, converted to CNY 100m'}
 (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 print(json.dumps(meta,ensure_ascii=False),flush=True)
 sys.exit(0)
tests=[
 ('em_clist','https://push2.eastmoney.com/api/qt/clist/get',{'pn':1,'pz':5,'po':1,'np':1,'fltt':2,'invt':2,'fid':'f12','fs':'m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048','fields':'f12,f13,f14,f2,f3,f6,f20,f21,f9,f23,f100,f26,f115,f124'}),
 ('sina_clist','https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData',{'page':1,'num':5,'sort':'symbol','asc':1,'node':'hs_a','symbol':'','_s_r_a':'page'}),
]
for name,url,params in tests:
 try:
  r=requests.get(url,params=params,headers={'User-Agent':'Mozilla/5.0','Referer':'https://quote.eastmoney.com/'},timeout=20)
  print(name,r.status_code,r.text[:1800])
  (out/(name+'.json')).write_text(r.text)
 except Exception as e: print(name,type(e).__name__,str(e))
