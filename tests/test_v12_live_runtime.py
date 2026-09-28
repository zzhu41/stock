"""Isolated V12-R2 raw-units transactions and publication-window guards."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock,patch

import shadow_v12_r2
from v12_live import ledger,runtime
from v12_live.constants import ASSETS,CANDIDATE_ID,CANDIDATE_HASH,STRATEGY_ID

DAYS=['2026-09-29','2026-09-30','2026-10-09']
A,B='159915','513100'


def decision(target=A,previous=None,**values):
    return dict(dict(strategy_id=STRATEGY_ID,candidate_id=CANDIDATE_ID,candidate_hash=CANDIDATE_HASH,
        target=target,previous_holding=previous,executable=True,reason='测试观察',crash_trigger_date=None,
        crash_code=None,diagnostics={}),**values)


def fixtures(day,price=10.):
    calendar=[d for d in DAYS if d<=day]
    quotes={code:dict(price=price,open=price,volume=1234.,date=day,timestamp=day+' 14:50:00') for code in ASSETS}
    view=dict(calendar=calendar,histories={code:[(d,price,price,price,price,1234.) for d in calendar] for code in ASSETS},
        raw_histories={code:[(d,price,price,1234.) for d in calendar] for code in ASSETS},
        actions={code:{d:dict(split_ratio=1.,cash_per_old_share=0.) for d in calendar} for code in ASSETS},
        metadata=dict(provisional_date=day,quote_timestamps={code:quotes[code]['timestamp'] for code in ASSETS}))
    return quotes,view


class LedgerTests(unittest.TestCase):
    def initial(self):
        q,v=fixtures(DAYS[0]);return ledger.advance(None,decision(),v,q,DAYS[0])

    def test_first_fill_has_own_identity_nav_one_and_no_pre_entry_entitlement(self):
        q,v=fixtures(DAYS[0],9.)
        v['actions'][A][DAYS[0]]['cash_per_old_share']=1.
        out=ledger.advance(None,decision(),v,q,DAYS[0])
        self.assertEqual(out['nav'],1.);self.assertEqual(out['strategy_id'],STRATEGY_ID)
        self.assertEqual(out['candidate_hash'],CANDIDATE_HASH);self.assertEqual(out['entry_date'],DAYS[0])
        self.assertEqual(out['events'][0]['actions'],[]);self.assertEqual(out['last_return'],0.)
        self.assertEqual(out['quote_volume'],1234.);self.assertIsNone(out['research_primary'])
        self.assertFalse(out['order_submission'])

    def test_split_and_cash_belong_to_old_units_before_switch_not_new_target(self):
        state=self.initial();q,v=fixtures(DAYS[1],4.5)
        v['actions'][A][DAYS[1]].update(split_ratio=2.,cash_per_old_share=1.)
        v['actions'][B][DAYS[1]].update(cash_per_old_share=10.)
        out=ledger.advance(state,decision(B,A),v,q,DAYS[1])
        self.assertAlmostEqual(out['nav'],.9998)
        self.assertAlmostEqual(out['events'][-1]['actions'][0]['cash_received'],.1)
        self.assertEqual(len(out['events'][-1]['actions']),1)
        self.assertEqual(out['holding'],B);self.assertEqual(out['entry_date'],DAYS[1])
        self.assertEqual(state,self.initial())

    def test_skipped_day_and_verified_quote_gap_keep_real_units_without_invented_trades(self):
        state=self.initial();q,v=fixtures(DAYS[2],9.9)
        v['raw_histories'][A][1]=(DAYS[1],9.,9.,1234.)
        v['actions'][A][DAYS[1]]['cash_per_old_share']=1.
        out=ledger.advance(state,decision(A,A),v,q,DAYS[2])
        self.assertAlmostEqual(out['nav'],1.1);self.assertEqual(len(out['events']),2)
        q,v=fixtures(DAYS[2],11.)
        v['raw_histories'][A]=[r for r in v['raw_histories'][A] if r[0]!=DAYS[1]]
        v['actions'][A][DAYS[1]].update(not_observed=True,verification='bracketed_no_action_interval',
            previous_quote_date=DAYS[0],next_quote_date=DAYS[2])
        out=ledger.advance(state,decision(A,A),v,q,DAYS[2])
        self.assertAlmostEqual(out['nav'],1.1)
        v['actions'][A][DAYS[1]]['cash_per_old_share']=.1
        with self.assertRaisesRegex(ValueError,'unsafe unobserved'):ledger.advance(state,decision(),v,q,DAYS[2])

    def test_same_date_foreign_identity_missing_action_and_revision_do_not_mutate_account(self):
        state=self.initial();q,v=fixtures(DAYS[0],999.)
        self.assertEqual(ledger.advance(state,decision(B),v,q,DAYS[0]),state)
        for field,value in (('strategy_id','v10-h'),('candidate_id','vd_d2a02ab14be481f562fd'),('candidate_hash','0'*64)):
            bad=deepcopy(state);bad[field]=value
            with self.assertRaisesRegex(ValueError,'profile mismatch'):ledger.validate_state(bad)
        q,v=fixtures(DAYS[1]);del v['actions'][A][DAYS[1]]
        with self.assertRaisesRegex(ValueError,'unverified corporate action'):ledger.advance(state,decision(),v,q,DAYS[1])
        q,v=fixtures(DAYS[1]);second=ledger.advance(state,decision(B,A),v,q,DAYS[1])
        q,v=fixtures(DAYS[2]);v['actions'][A][DAYS[1]]['cash_per_old_share']=.2
        with self.assertRaisesRegex(ValueError,'previously booked'):ledger.advance(second,decision(B,B),v,q,DAYS[2])
        self.assertEqual(state,self.initial())


class RuntimeTests(unittest.TestCase):
    def test_atomic_first_day_render_and_repeat_keep_one_independent_event(self):
        q,v=fixtures(DAYS[0])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'r2.json'
            first=runtime.run(q,DAYS[0],path,datetime(2026,9,29,14,50),lambda *a,**k:v,lambda *a,**k:decision())
            saved=path.read_bytes();state=json.loads(saved)
            self.assertEqual(state['nav'],1.);self.assertEqual(state['start_date'],DAYS[0])
            self.assertIn('【影子 V12-R2】','\n'.join(first));self.assertIn('建议: 买入 159915','\n'.join(first))
            self.assertIn('数据截止: 2026-09-29','\n'.join(first));self.assertIn('行情时间: 2026-09-29 14:50:00','\n'.join(first))
            never=Mock(side_effect=AssertionError('sealed day recomputed'))
            self.assertEqual(runtime.run({},DAYS[0],path,datetime(2026,9,29,16),never,never),first)
            self.assertEqual(path.read_bytes(),saved)
            self.assertEqual(shadow_v12_r2.saved_block(DAYS[0],path),first)
            self.assertIsNone(shadow_v12_r2.saved_block(DAYS[1],path))
            q,v=fixtures(DAYS[1],10.5)
            lines=runtime.run(q,DAYS[1],path,datetime(2026,9,30,14,50),lambda *a,**k:v,lambda *a,**k:decision(A,A))
            self.assertAlmostEqual(json.loads(path.read_text())['nav'],1.05)
            self.assertIn('建议: 继续持有 159915','\n'.join(lines))

    def test_launch_day_backfill_stale_quotes_and_missing_view_cannot_create_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'r2.json'
            with self.assertRaisesRegex(ValueError,'不补记'):
                runtime.run({},'2026-09-28',path,datetime(2026,9,28,14,50))
            with self.assertRaisesRegex(ValueError,'historical'):
                runtime.run({},DAYS[0],path,datetime(2026,9,30,14,50))
            q,v=fixtures(DAYS[0]);q[A]['timestamp']=DAYS[0]+' 14:40:00'
            with self.assertRaisesRegex(ValueError,'陈旧'):
                runtime.run(q,DAYS[0],path,datetime(2026,9,29,14,50))
            q,v=fixtures(DAYS[0])
            with self.assertRaisesRegex(ValueError,'missing view'):
                runtime.run(q,DAYS[0],path,datetime(2026,9,29,14,50),Mock(side_effect=ValueError('missing view')))
            self.assertFalse(path.exists())

    def test_profile_failure_quote_view_mismatch_and_disk_failure_preserve_old_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'r2.json';q,v=fixtures(DAYS[0])
            runtime.run(q,DAYS[0],path,datetime(2026,9,29,14,50),lambda *a,**k:v,lambda *a,**k:decision())
            saved=path.read_bytes();q,v=fixtures(DAYS[1])
            with self.assertRaisesRegex(ValueError,'profile changed'):
                runtime.run(q,DAYS[1],path,datetime(2026,9,30,14,50),lambda *a,**k:v,Mock(side_effect=ValueError('profile changed')))
            bad=deepcopy(v);bad['raw_histories'][A][-1]=(DAYS[1],10.,10.,999.)
            with self.assertRaisesRegex(ValueError,'differs from quote volume'):
                runtime.run(q,DAYS[1],path,datetime(2026,9,30,14,50),lambda *a,**k:bad,lambda *a,**k:decision())
            with patch.object(runtime,'atomic_json',side_effect=OSError('disk full')):
                with self.assertRaises(OSError):runtime.run(q,DAYS[1],path,datetime(2026,9,30,14,50),lambda *a,**k:v,lambda *a,**k:decision())
            self.assertEqual(path.read_bytes(),saved)

    def test_elapsed_clock_and_fsync_boundary_reject_late_commit_and_lock_is_nonblocking(self):
        q,v=fixtures(DAYS[0]);moment=[datetime(2026,9,29,14,50)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'r2.json'
            def fsync(unused):moment[0]=datetime(2026,9,29,14,54)
            with patch.object(runtime,'runtime_clock',return_value=lambda:moment[0]),patch('v10_live.runtime.os.fsync',side_effect=fsync):
                with self.assertRaisesRegex(ValueError,'陈旧'):
                    runtime.run(q,DAYS[0],path,build_view=lambda *a,**k:v,decide=lambda *a,**k:decision())
            self.assertFalse(path.exists())
            with runtime.locked(path):
                with self.assertRaises(BlockingIOError):runtime.run(q,DAYS[0],path,datetime(2026,9,29,14,50))

    def test_timeout_bridge_recovers_only_committed_same_identity_lines(self):
        with patch.object(shadow_v12_r2.subprocess,'run',side_effect=subprocess.TimeoutExpired('worker',60)), \
                patch.object(shadow_v12_r2,'saved_block',return_value=['committed card']):
            self.assertEqual(shadow_v12_r2.block({},DAYS[0]),['committed card'])
        with patch.object(shadow_v12_r2.subprocess,'run',side_effect=subprocess.TimeoutExpired('worker',60)), \
                patch.object(shadow_v12_r2,'saved_block',return_value=None):
            lines=shadow_v12_r2.block({},DAYS[0])
            self.assertIn('本次无有效建议','\n'.join(lines))
            self.assertIn('行情时间: 未核验',lines[-2])


if __name__=='__main__':unittest.main()
