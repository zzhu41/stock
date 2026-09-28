"""Frozen V12-R2 signal fidelity; offline and without production state I/O."""
from bisect import bisect_right
from copy import deepcopy
import csv
from datetime import date,timedelta
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from v12_live import policy,profile
from v12_live.constants import CANDIDATE_ID,CANDIDATE_HASH

ROOT=Path(__file__).resolve().parents[1]
ASSETS=policy.ASSETS
A,B="159915","510500"
DAYS=["2026-09-24","2026-09-28","2026-09-29","2026-09-30","2026-10-09","2026-10-12"]


def mock_snapshot(index,**held_changes):
    rows={code:dict(close=100.,valid=float(code!=policy.CASH),score=0.,bars=300,
        ret1=.01,mom5=.02,mom20=.10,mom60=.2,vol20=.02,ma180=.1,ma250=.1,volume_ratio=1.,
        prior_vol20=.015,prior_vol60=.02) for code in ASSETS}
    rows[B].update(score=100.,mom20=.3)
    rows[A].update(held_changes)
    snapshot=policy.Snapshot(rows,index,False)
    snapshot.prior_quote_dates={code:DAYS[index-1] if index else None for code in ASSETS}
    return snapshot,DAYS[:index+1]


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.profile=profile.load_profile()

    def call(self,index,state,**changes):
        with patch.object(policy,"verified_profile",return_value=deepcopy(self.profile)), \
                patch.object(policy,"snapshot",return_value=mock_snapshot(index,**changes)):
            return policy.decide({},DAYS[:index+1],DAYS[index],state)

    def test_fixed_candidate_and_old_research_status_are_not_reselected(self):
        self.assertEqual(self.profile['record']['id'],CANDIDATE_ID)
        self.assertEqual(self.profile['record']['hash'],CANDIDATE_HASH)
        self.assertEqual(self.profile['config']['panic'],1.4)
        self.assertEqual(self.profile['record']['feature_spec'],dict(window=25,smooth=4,ma_window=180))
        self.assertIsNone(self.profile['research_primary'])
        self.assertFalse(self.profile['research_qualified'])
        original=profile._sha
        for name in ('/v12/reference.py','/v10_h_close/corrected_manifest.json'):
            def corrupted(path):
                return '0'*64 if str(path).endswith(name) else original(path)
            with patch.object(profile,'_sha',side_effect=corrupted):
                with self.assertRaisesRegex(ValueError,'dependency changed'):profile.load_profile()
        with patch.object(profile,'CANDIDATE_ID','v12r2_another_candidate'):
            with self.assertRaisesRegex(ValueError,'status/identity'):profile.load_profile()

    def test_actual_entry_age_two_with_risk_exit_exception_and_prior_sigma_clip(self):
        state=dict(holding=A,entry_date=DAYS[0],last_date=DAYS[0])
        self.assertEqual(self.call(1,state)['target'],A)
        self.assertEqual(self.call(2,state)['target'],B)
        for sigma,threshold in ((.001,.02),(.04,.056),(.2,.10)):
            out=self.call(1,state,vol20=sigma,ret1=-threshold)
            self.assertTrue(out['panic']);self.assertEqual(out['target'],B)
            self.assertAlmostEqual(out['diagnostics']['panic_threshold'],threshold,places=14)
        delayed=dict(holding=A,entry_date=DAYS[1],last_date=DAYS[1])
        self.assertEqual(self.call(2,delayed)['target'],A)

    def test_existing_crash_lock_still_beats_panic_and_expires_at_five_observations(self):
        state=dict(holding=A,entry_date=DAYS[0],last_date=DAYS[0],crash_code=A,crash_trigger_date=DAYS[0])
        for i in (1,2,3,4):
            out=self.call(i,state,ret1=-.20,mom20=-.2)
            self.assertTrue(out['panic']);self.assertEqual(out['target'],A);self.assertTrue(out['lock_active'])
        out=self.call(5,state,ret1=-.20,mom20=-.2)
        self.assertEqual(out['target'],B);self.assertIsNone(out['crash_trigger_date'])

    def test_only_deep_and_volume_channels_and_missing_held_quote_cannot_create_lock(self):
        for mom5,ma250,ratio,label in ((-.09,-.21,1.,'深跌'),(-.05,-.11,2.,'量能恐慌')):
            snapshot,dates=mock_snapshot(2)
            snapshot.indicators[B].update(mom5=mom5,ma250=ma250,volume_ratio=ratio)
            snapshot.fear[2]=True
            with patch.object(policy,'verified_profile',return_value=self.profile),patch.object(policy,'snapshot',return_value=(snapshot,dates)):
                out=policy.decide({},dates,DAYS[2],dict(holding=A,entry_date=DAYS[0]))
                self.assertTrue(out['crash']);self.assertEqual(out['diagnostics']['crash_channels'],[label])
                self.assertFalse(out['diagnostics']['qvix_in_use'])
                snapshot.indicators[A].update(close=float('nan'),valid=0.)
                blocked=policy.decide({},dates,DAYS[2],dict(holding=A,entry_date=DAYS[0]))
                self.assertFalse(blocked['executable']);self.assertFalse(blocked['crash'])
        snapshot,dates=mock_snapshot(2)
        snapshot.indicators[B].update(mom5=-.05,ma250=-.21,volume_ratio=1.)
        snapshot.fear[2]=True  # This would activate only the QVIX channel.
        with patch.object(policy,'verified_profile',return_value=self.profile),patch.object(policy,'snapshot',return_value=(snapshot,dates)):
            out=policy.decide({},dates,DAYS[2],dict(holding=A,entry_date=DAYS[0]))
        self.assertFalse(out['crash'])

    def test_prior_risk_is_previous_own_quote_not_current_or_twice_lagged(self):
        dates=[(date(2024,1,1)+timedelta(days=i)).isoformat() for i in range(310)]
        p=[100.+i*.04+.3*np.sin(i*.23) for i in range(310)]
        rows=[(d,float(x),100.) for d,x in zip(dates,p)]
        first=policy.latest_features(rows,dates[-1],A)
        shock=list(rows);shock[-1]=(dates[-1],rows[-1][1]*.6,100.)
        changed=policy.latest_features(shock,dates[-1],A)
        self.assertEqual(first['vol20'],changed['vol20'])
        self.assertNotEqual(first['score'],changed['score'])
        missing=rows[:-2]+rows[-1:]
        actual=policy.latest_features(missing,dates[-1],A)
        previous=[x[1] for x in missing[:-1]]
        expected=max(policy._legacy_vol(previous,20),policy._conv_vol(previous,60))
        self.assertEqual(actual['vol20'],expected)
        self.assertNotEqual(actual['vol20'],first['vol20'])
        self.assertEqual(policy.latest_features(rows[:269],dates[268],A)['valid'],0.)

    def test_future_values_do_not_change_live_asof_snapshot(self):
        dates=[(date(2024,1,1)+timedelta(days=i)).isoformat() for i in range(300)]
        hs={code:[(d,100+i*.1,100+i*.1,100+i*.1,100+i*.1,100.) for i,d in enumerate(dates)] for code in ASSETS}
        first,_=policy.snapshot(hs,dates,dates[-4])
        for code in ASSETS:hs[code][-3:]=[(d,1.,float('nan'),1.,1.,-1.) for d in dates[-3:]]
        second,_=policy.snapshot(hs,dates,dates[-4])
        self.assertEqual(first.indicators,second.indicators)
        self.assertEqual(first.prior_quote_dates,second.prior_quote_dates)


class FrozenFullPathTests(unittest.TestCase):
    def test_3097_dates_scores_risk_features_targets_and_fee_returns_are_bit_exact(self):
        import strategy
        from v12_r2.data import sha
        from v12.exante_features import view_for_config
        from v12.reference import run_reference
        verified=profile.load_profile()
        release=json.loads((ROOT/'v12_r2/release_receipt.json').read_text())
        for name in ('results/features.npz','results/feature_metadata.json','results/paths.npz','results/path_metadata.json'):
            self.assertEqual(sha(ROOT/'v12_r2'/name),release['file_sha256'][name])
        meta=json.loads((ROOT/'v12_r2/results/feature_metadata.json').read_text())
        with np.load(ROOT/'v12_r2/results/features.npz',allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        config=verified['config'];reference_view=view_for_config(arrays,meta,config)
        expected=run_reference(reference_view,meta,config,'2014-01-02','2026-09-24',.0001)
        saved_meta=json.loads((ROOT/'v12_r2/results/path_metadata.json').read_text())
        record=saved_meta['ids'].index(CANDIDATE_ID)
        with np.load(ROOT/'v12_r2/results/paths.npz',allow_pickle=False) as archive:
            for field in ('returns','holdings','summary'):
                np.testing.assert_array_equal(expected[field],archive['close_1bp__'+field][record])
        histories={};axes={}
        for code in ASSETS:
            with (ROOT/'v10_h_close/corrected_snapshots'/(code+'.csv')).open() as f:
                histories[code]=[(r[0],float(r[2]),float(r[3])) for r in csv.reader(f) if r]
            axes[code]=[r[0] for r in histories[code]]
        fields={name:i for i,name in enumerate(meta['feature_names'])}
        score_idx=meta['score_names'].index(config['score']);date_idx={d:i for i,d in enumerate(meta['dates'])}
        state={};nav=1.;mark=0.;actual_returns=[];actual_holdings=[]
        before={k:deepcopy(v) for k,v in vars(strategy).items() if k.isupper()}
        with patch.object(policy,'verified_profile',return_value=verified):
            for trace in expected['trace']:
                signal_date=trace['date'];day=date_idx[signal_date];indicators={};prior_dates={}
                for code in ASSETS:
                    end=bisect_right(axes[code],signal_date);rows=histories[code][:end]
                    ind=policy.latest_features(rows,signal_date,code);indicators[code]=ind
                    prior_dates[code]=rows[-2][0] if end>1 and rows[-1][0]==signal_date else None
                    ai=meta['assets'].index(code)
                    if code!=policy.CASH and reference_view['features'][day,ai,fields['valid']]>.5:
                        self.assertEqual(ind['score'],arrays['scores'][score_idx,day,ai],(signal_date,code,'score'))
                        for field in ('vol20','ret1','mom5','mom20','mom60','ma180','ma250','volume_ratio'):
                            self.assertEqual(ind[field],reference_view['features'][day,ai,fields[field]],(signal_date,code,field))
                data=policy.Snapshot(indicators,day,False);data.prior_quote_dates=prior_dates
                previous=nav;held=state.get('holding');price=data.price(day,held) if held else float('nan')
                if held and np.isfinite(price):nav*=price/mark;mark=price
                with patch.object(policy,'snapshot',return_value=(data,meta['dates'][:day+1])):
                    result=policy.decide({},meta['dates'],signal_date,state)
                self.assertEqual(result['target'],trace['holding'],signal_date)
                self.assertEqual(result['intended_target'],trace['intended'],signal_date)
                self.assertEqual(result['crash_requested'],trace['crash_requested'],signal_date)
                self.assertEqual(result['panic'],trace['panic'],signal_date)
                if result['executable']:
                    if result['target']!=held:
                        if actual_returns:nav*=.9998
                        mark=data.price(day,result['target']);entry=signal_date
                    else:entry=state['entry_date']
                    state=dict(holding=result['target'],entry_date=entry,last_date=signal_date,
                        crash_trigger_date=result['crash_trigger_date'],crash_code=result['crash_code'])
                actual_returns.append(nav/previous-1.)
                actual_holdings.append(meta['assets'].index(state['holding']))
        self.assertEqual(len(actual_returns),3097)
        np.testing.assert_array_equal(actual_returns,expected['returns'])
        np.testing.assert_array_equal(actual_holdings,expected['holdings'])
        self.assertEqual(before,{k:v for k,v in vars(strategy).items() if k.isupper()})


if __name__=='__main__':unittest.main()
