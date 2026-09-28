"""Three-version staged generation: failed publication cannot advance accounts."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import signal_daily
import signal_store

NOW = datetime(2026, 9, 28, 14, 50, 30)
DAY = '2026-09-28'
CODE = '513100'


def account():
    return dict(valuation_version=4, account_id='v9.2', holding=CODE, nav=1., units=.1, cash=0.,
                mark_raw_price=10., last_date=DAY, start_date=DAY, entry_date=DAY, lock_code=None,
                quote_timestamp=DAY+' 14:50:00', saved_lines=['【影子 V9.2】',
                '  数据截止: '+DAY, '  影子持仓: 513100 | 虚拟净值 1.0 | 建议: 513100 纳指ETF | fixture'],
                events=[dict(date=DAY,to=CODE,nav=1.,baseline=True,trade=False,price=10.,**{'from':None})])


class V10DailyIntegrationTests(unittest.TestCase):
    def setup_job(self, stack, directory):
        quotes={c:dict(price=10., date=DAY, timestamp=DAY+' 14:50:00', open=10., volume=100.)
                for c in signal_daily.UNIVERSE}
        stack.enter_context(patch.object(signal_daily,'SIGNAL_DIR',directory))
        fetch=stack.enter_context(patch.object(signal_daily,'fetch_realtime',return_value=quotes))
        def prepare(unused_quotes,date,stage,now=None):
            self.assertEqual(date,DAY)
            with self.assertRaises(BlockingIOError):
                with signal_store.file_lock(Path(directory)/'shadow_v92.lock'):
                    self.fail('real account must remain locked while staging')
            state=account()
            (Path(stage)/'shadow_v92.json').write_text(json.dumps(state))
            return dict(lines=state['saved_lines']+['【影子 V9.2+】','信号状态: 计算失败，本次无有效建议',
                        '【影子 V10-H】','信号状态: 计算失败，本次无有效建议'],
                        successful_accounts=['shadow_v92.json'])
        worker=stack.enter_context(patch.object(signal_daily,'prepare_versions',side_effect=prepare))
        stack.enter_context(patch('sys.stdout',new_callable=io.StringIO))
        return quotes,fetch,worker,prepare

    def test_three_version_journal_commits_accounts_and_duplicate_does_not_trade(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            quotes,fetch,worker,unused=self.setup_job(stack,directory)
            oldlock=Path(directory)/'crash_lock.json';oldlock.write_text('{"legacy":true}')
            self.assertIsNone(signal_daily.main(now=NOW))
            saved,journal=signal_store.load_saved(directory)
            self.assertEqual(journal['schema'],2)
            self.assertEqual(journal['account_states']['shadow_v92.json'],account())
            self.assertEqual(json.loads((Path(directory)/'shadow_v92.json').read_text()),account())
            self.assertEqual(oldlock.read_text(),'{"legacy":true}')
            self.assertIn('策略版本: V12-R2 | V9.2 | V9.2+ | V10-H',saved)
            self.assertEqual(journal['primary_version'],'V12-R2')
            self.assertIsNone(journal['target'])
            self.assertNotIn('★ 建议:',saved);self.assertNotIn('0906',saved)
            self.assertTrue(signal_store.signal_info(saved,now=NOW)['actionable'])
            fetch.side_effect=AssertionError('duplicate fetch');worker.side_effect=AssertionError('duplicate trade')
            self.assertIsNone(signal_daily.main(now=NOW))
            self.assertEqual(signal_store.load_journal(directory),journal)
            self.assertFalse(list(Path(directory).glob('.signal-stage-*')))

    def test_failed_canonical_commit_leaves_account_and_legacy_lock_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.setup_job(stack,directory)
            path=Path(directory)/'shadow_v92.json';path.write_text('{"old":"preserved"}')
            real_write=signal_store.atomic_json
            def fail(path,value,**kwargs):
                if Path(path).name=='daily_state.json':raise OSError('disk full')
                real_write(path,value,**kwargs)
            stack.enter_context(patch.object(signal_store,'atomic_json',side_effect=fail))
            with self.assertRaises(OSError):signal_daily.main(now=NOW)
            self.assertEqual(path.read_text(),'{"old":"preserved"}')
            self.assertFalse((Path(directory)/'daily_state.json').exists())
            self.assertFalse((Path(directory)/'latest.txt').exists())

    def test_worker_timeout_after_stage_write_cannot_create_a_real_account(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            unused,fetch,worker,prepare=self.setup_job(stack,directory)
            def timeout(*args,**kwargs):
                prepare(*args,**kwargs)
                raise TimeoutError('staged, but no canonical commit')
            worker.side_effect=timeout
            with self.assertRaises(TimeoutError):signal_daily.main(now=NOW)
            self.assertFalse((Path(directory)/'shadow_v92.json').exists())
            self.assertFalse((Path(directory)/'daily_state.json').exists())

    def test_quote_expiring_during_staging_or_window_end_cannot_commit(self):
        for finished in (NOW.replace(minute=54),NOW.replace(minute=55)):
            with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                unused,fetch,worker,prepare=self.setup_job(stack,directory)
                current=[NOW]
                stack.enter_context(patch('v10_live.runtime.runtime_clock',return_value=lambda:current[0]))
                def slow(*args,**kwargs):
                    value=prepare(*args,**kwargs);current[0]=finished;return value
                worker.side_effect=slow
                with self.assertRaises(ValueError):signal_daily.main(now=NOW)
                self.assertFalse((Path(directory)/'shadow_v92.json').exists())
                self.assertFalse((Path(directory)/'daily_state.json').exists())

    def test_failed_account_projection_recovers_from_canonical_without_retrading(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            unused,fetch,worker,prepare=self.setup_job(stack,directory)
            real_write=signal_store.atomic_text
            def fail(path,text,**kwargs):
                if Path(path).name=='shadow_v92.json':raise OSError('projection failure')
                real_write(path,text,**kwargs)
            with patch.object(signal_store,'atomic_text',side_effect=fail):
                self.assertIsNone(signal_daily.main(now=NOW))
            self.assertFalse((Path(directory)/'shadow_v92.json').exists())
            self.assertEqual(signal_store.load_journal(directory)['account_states']['shadow_v92.json'],account())
            fetch.side_effect=AssertionError('must recover rather than regenerate')
            self.assertIsNone(signal_daily.main(now=NOW))
            self.assertEqual(json.loads((Path(directory)/'shadow_v92.json').read_text()),account())
            self.assertEqual(worker.call_count,1)

    def test_canonical_fsync_crossing_freshness_boundary_cannot_commit(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.setup_job(stack,directory)
            current=[NOW]
            stack.enter_context(patch('v10_live.runtime.runtime_clock',return_value=lambda:current[0]))
            actual=signal_store.os.fsync
            def slow_fsync(fd):
                actual(fd)
                current[0]=NOW.replace(minute=54)
            stack.enter_context(patch.object(signal_store.os,'fsync',side_effect=slow_fsync))
            with self.assertRaises(ValueError):signal_daily.main(now=NOW)
            self.assertFalse((Path(directory)/'daily_state.json').exists())
            self.assertFalse((Path(directory)/'shadow_v92.json').exists())

    def test_unprojected_checkpoint_survives_next_day_when_that_version_fails(self):
        from v10_live.ledger import advance, CANDIDATE_ID
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            quotes,fetch,worker,prepare=self.setup_job(stack,directory)
            broken=Path(directory)/'shadow_v92.json';actual=signal_store.atomic_text
            def write(path,text,**kwargs):
                if Path(path)==broken:raise OSError('persistent account projection failure')
                actual(path,text,**kwargs)
            stack.enter_context(patch.object(signal_store,'atomic_text',side_effect=write))
            signal_daily.main(now=NOW)
            self.assertFalse(broken.exists())
            next_day='2026-09-29'
            fresh={c:dict(q,date=next_day,timestamp=next_day+' 14:50:00') for c,q in quotes.items()}
            fetch.return_value=fresh
            def prepare_next(unused,date,stage,now=None):
                self.assertEqual(json.loads((Path(stage)/'shadow_v92.json').read_text()),account())
                view=dict(calendar=[next_day],actions={CODE:{next_day:dict(split_ratio=1.,cash_per_old_share=0.)}})
                decision=dict(candidate_id=CANDIDATE_ID,target=CODE,executable=True)
                h=advance(None,decision,view,fresh,next_day)
                h['saved_lines']=['【影子 V10-H】','影子持仓: 513100 | 建议: 513100 纳指ETF | fixture']
                (Path(stage)/'shadow_v10.json').write_text(json.dumps(h))
                return dict(lines=['【影子 V9.2】','信号状态: 计算失败，本次无有效建议']+h['saved_lines'],
                            successful_accounts=['shadow_v10.json'])
            worker.side_effect=prepare_next
            signal_daily.main(now=NOW.replace(day=29))
            journal=signal_store.load_journal(directory)
            self.assertEqual(journal['updated_accounts'],['shadow_v10.json'])
            self.assertEqual(journal['account_states']['shadow_v92.json'],account())
            self.assertEqual(journal['account_states']['shadow_v10.json']['last_date'],next_day)
            self.assertFalse(broken.exists())

    def test_outside_window_and_no_success_never_publish(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            unused,fetch,worker,prepare=self.setup_job(stack,directory)
            with self.assertRaises(ValueError):signal_daily.main(now=NOW.replace(hour=15))
            fetch.assert_not_called()
            worker.side_effect=None;worker.return_value=dict(lines=['failure'],successful_accounts=[])
            with self.assertRaisesRegex(ValueError,'均无有效建议'):signal_daily.main(now=NOW)
            self.assertFalse((Path(directory)/'daily_state.json').exists())

    def test_changed_account_checksum_or_unknown_account_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.setup_job(stack,directory);signal_daily.main(now=NOW)
            path=Path(directory)/'daily_state.json';r=json.loads(path.read_text())
            r['account_states']['shadow_v92.json']['nav']=100
            path.write_text(json.dumps(r))
            with self.assertRaises(ValueError):signal_store.load_journal(directory)
            with self.assertRaises(ValueError):
                signal_store.publish('x',DAY,None,None,directory,account_states={'../outside.json':account()})


if __name__=='__main__':unittest.main()
