"""Real frozen seed plus artificial 9/28 completed and 9/29–30 live quotes.

Actual data/policy/ledger/runtime are exercised with temp seed/cache/state.
No real network, outbound messages, or production state writes are permitted.
"""
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import strategy
from v10_live import data
from v12_live import policy,runtime

ROOT=Path(__file__).resolve().parents[1]
PRE,FIRST,SECOND='2026-09-28','2026-09-29','2026-09-30'


class EndToEndTests(unittest.TestCase):
    def test_real_layers_start_at_one_and_track_only_actual_forward_signal_marks(self):
        before={k:deepcopy(v) for k,v in vars(strategy).items() if k.isupper() or k=='_state_bull'}
        with tempfile.TemporaryDirectory(prefix='v12_r2_live_e2e_') as directory:
            base=Path(directory);seed=base/'seed';cache=base/'cache';state_path=base/'signals/R2.json'
            (seed/'results/price_audit').mkdir(parents=True);(seed/'corrected_snapshots').mkdir();(seed/'snapshots').mkdir()
            source=ROOT/'v10_h_close'
            shutil.copy2(source/'corrected_manifest.json',seed/'corrected_manifest.json')
            raw,qfq,quotes={},{},{FIRST:{},SECOND:{}}
            for code in data.CODES:
                name=code+'_raw_2010-01-01_2026-09-24.json'
                shutil.copy2(source/'results/price_audit'/name,seed/'results/price_audit'/name)
                shutil.copy2(source/'corrected_snapshots'/(code+'.csv'),seed/'corrected_snapshots'/(code+'.csv'))
                shutil.copy2(source/'snapshots'/(code+'.csv'),seed/'snapshots'/(code+'.csv'))
                payload=json.loads((seed/'results/price_audit'/name).read_text())
                raw[code]=[(r[0],float(r[1]),float(r[2]),float(r[5])) for r in payload['rows']]
                with (seed/'snapshots'/(code+'.csv')).open() as f:qfq[code]=[(r[0],float(r[1]),float(r[2]),float(r[3])) for r in csv.reader(f) if r]
                previous,volume=raw[code][-1][2],raw[code][-1][3]
                pre_close=round(previous*1.003,3)
                first_open,first_signal,first_close=(round(pre_close*m,3) for m in (1.001,1.005,1.009))
                second_open,second_signal,second_close=(round(first_close*m,3) for m in (1.001,1.007,1.011))
                artificial=[(PRE,previous,pre_close,volume),(FIRST,first_open,first_close,volume*1.1),
                            (SECOND,second_open,second_close,volume*1.2)]
                raw[code].extend(artificial);qfq[code].extend(artificial)
                for day,opening,price,prev in ((FIRST,first_open,first_signal,pre_close),
                                              (SECOND,second_open,second_signal,first_close)):
                    quotes[day][code]=dict(date=day,timestamp=day+' 14:50:00',open=opening,price=price,
                        prev_close=prev,volume=volume*.8,high=max(opening,price),low=min(opening,price))
            calls=[];views=[];decisions=[]
            def fetch(code,start,end):
                calls.append((code,start,end))
                return dict(raw=[r for r in raw[code] if start<=r[0]<=end],qfq=[r for r in qfq[code] if start<=r[0]<=end],
                    sources={'raw':'artificial temporary fixture','qfq':'same-vintage artificial fixture'})
            def build(quotes,date,now=None):
                result=data.build_live_view(quotes,date,now=now,cache_dir=cache,fetch_pair=fetch)
                views.append(result);return result
            def decide(histories,calendar,date,state=None):
                result=policy.decide(histories,calendar,date,state);decisions.append(result);return result
            with patch.object(data,'SEED_DIR',seed), \
                    patch('urllib.request.urlopen',side_effect=AssertionError('Network forbidden')), \
                    patch('v10_live.policy.qvix_state',side_effect=AssertionError('R2 must not read QVIX')):
                first=runtime.run(quotes[FIRST],FIRST,state_path,datetime.fromisoformat(FIRST+'T14:50:30'),build,decide)
                state=json.loads(state_path.read_text());held=state['holding']
                self.assertEqual(state['nav'],1.);self.assertEqual(state['start_date'],FIRST)
                self.assertEqual(state['entry_date'],FIRST);self.assertEqual(state['candidate_id'],policy.CANDIDATE_ID)
                self.assertEqual(state['events'][0]['from'],None);self.assertEqual(state['events'][0]['actions'],[])
                self.assertEqual(state['mark_raw_price'],quotes[FIRST][held]['price'])
                self.assertEqual(state['quote_volume'],quotes[FIRST][held]['volume'])
                self.assertIn('用户指定主推送','\n'.join(first));self.assertIn('建议: 买入','\n'.join(first))
                completed=json.loads((cache/'completed.json').read_text())['state']
                self.assertEqual(completed['last_completed'],PRE)
                state_bytes,cache_bytes=state_path.read_bytes(),(cache/'completed.json').read_bytes()
                repeated=runtime.run({},FIRST,state_path,datetime.fromisoformat(FIRST+'T15:30:00'),
                    lambda *a,**k:self.fail('same day data rebuilt'),lambda *a,**k:self.fail('same day policy rerun'))
                self.assertEqual(repeated,first);self.assertEqual(state_path.read_bytes(),state_bytes)
                self.assertEqual((cache/'completed.json').read_bytes(),cache_bytes)
                runtime.run(quotes[SECOND],SECOND,state_path,datetime.fromisoformat(SECOND+'T14:50:30'),build,decide)
                second=json.loads(state_path.read_text())
                expected=quotes[SECOND][held]['price']/quotes[FIRST][held]['price']
                if second['holding']!=held:expected*=.9998
                self.assertAlmostEqual(second['nav'],expected,places=12)
                self.assertEqual(len(second['events']),2);self.assertEqual(second['start_date'],FIRST)
                self.assertEqual(second['events'][-1]['actions'],[])
                completed=json.loads((cache/'completed.json').read_text())['state']
                self.assertEqual(completed['last_completed'],FIRST)
                self.assertNotEqual(completed['assets'][held]['raw'][-1][2],state['mark_raw_price'])
                self.assertEqual(decisions[-1]['diagnostics']['prior_quote_dates'][held],FIRST)
                self.assertEqual(decisions[-1]['diagnostics']['actual_holding_age'],1)
                self.assertEqual(len(calls),len(data.CODES)*2)
        self.assertEqual(before,{k:v for k,v in vars(strategy).items() if k.isupper() or k=='_state_bull'})


if __name__=='__main__':unittest.main()
