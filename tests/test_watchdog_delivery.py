"""Delivery monitoring tests: temporary files and mocked probes/alarms only."""
from datetime import datetime
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import watchdog


TODAY = '2026-09-28'
NOW = datetime(2026, 9, 28, 15, 20)
PREVIOUS = '2026-09-24'


class WatchdogDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.signals = self.root / 'signals'; self.signals.mkdir()
        (self.root / 'data').mkdir(); (self.root / 'calendars').mkdir()
        (self.root / 'calendars/sse.json').write_text(json.dumps({
            '2026': {'closed_ranges': [['2026-09-25', '2026-09-27']]}}))
        (self.root / 'data/510300.csv').write_text(PREVIOUS + ',100,100,10\n')
        (self.root / 'data/qvix50.csv').write_text(TODAY + ',20\n')
        self.original_probe = watchdog._probe_benchmark
        self.patches = [patch.object(watchdog, 'BASE', str(self.root)),
                        patch.object(watchdog.store, 'ROOT', self.root),
                        patch.object(watchdog, '_probe_benchmark', return_value=[(PREVIOUS, 100.), (TODAY, 103.)]),
                        patch.object(watchdog.live_utils, 'send_dingtalk', return_value=True),
                        patch('sys.stdout', new_callable=io.StringIO)]
        values = [p.start() for p in self.patches]
        self.probe, self.alarm = values[2], values[3]
        for p in self.patches: self.addCleanup(p.stop)

    def write(self, name, value):
        (self.signals / name).write_text(json.dumps(value, ensure_ascii=False))

    def success(self, day=TODAY, generated='14:50:30', quote='14:50:00'):
        text = ('动量轮动信号 | 生成 %s %s | 数据截止 %s\n行情时间: %s %s\n'
                '★ 建议: 买入 513100 纳指ETF\n' % (day, generated, day, day, quote))
        identity = watchdog.store.digest(text)
        journal = dict(schema=1, date=day, signal_id=identity, text=text,
                       target='513100', crash_lock=None, committed_at=day+'T'+generated)
        journal['checksum'] = watchdog.store.record_digest(journal)
        self.write('daily_state.json', journal)
        self.write('generation_status.json', dict(date=day, status='ready', signal_id=identity,
                                                 finished_at=day+'T14:51:00'))
        receipt = dict(schema=1, date=day, status='sent', signal_id=identity,
                       confirmed_at=day+'T14:51:01', attempts=[dict(status='sent', errcode=0)])
        self.write('delivery.json', receipt)
        return journal, receipt

    def log(self):
        return (self.signals / 'watchdog.log').read_text()

    def test_yesterday_daily_cache_is_normal_with_verified_current_signal_and_delivery(self):
        self.success()
        result = watchdog.main(now=NOW)
        self.assertEqual(result['status'], 'OK')
        self.assertEqual(result['source_comparison']['compared_dates'], [PREVIOUS])
        self.assertTrue(result['canonical_verified'])
        self.assertEqual(result['alerts'], [])
        self.probe.assert_called_once_with(TODAY)
        self.alarm.assert_not_called()
        self.assertIn(' OK', self.log())

    def test_today_partial_close_is_never_compared_to_final_source_close(self):
        self.success()
        (self.root / 'data/510300.csv').write_text(PREVIOUS + ',100,100,10\n' + TODAY + ',100,80,100\n')
        self.probe.return_value = [(PREVIOUS, 100.), (TODAY, 120.)]
        result = watchdog.main(now=NOW)
        self.assertEqual(result['status'], 'OK')
        self.assertNotIn(TODAY, result['source_comparison']['compared_dates'])

    def test_same_date_text_and_mtime_do_not_replace_canonical_commit(self):
        (self.signals / (TODAY+'.txt')).write_text('fresh-looking file')
        (self.signals / 'latest.txt').write_text('fresh-looking file')
        result = watchdog.main(now=NOW)
        self.assertEqual(result['status'], 'FAILED')
        self.assertTrue(any('canonical' in a for a in result['alerts']))
        self.alarm.assert_called_once()

    def test_failed_running_or_missing_generation_cannot_be_ok(self):
        for status in ('failed', 'running', 'missing'):
            journal, _ = self.success()
            if status == 'missing':
                (self.signals/'generation_status.json').unlink()
            else:
                self.write('generation_status.json', dict(date=TODAY, status=status, signal_id=journal['signal_id']))
            with self.subTest(status=status):
                result = watchdog.main(now=NOW)
                self.assertEqual(result['status'], 'FAILED')
                self.assertTrue(any('生成记录' in a for a in result['alerts']))

    def test_sent_receipt_must_match_exact_committed_signal_and_date(self):
        for changed in ({'signal_id': 'old-signal'}, {'date': PREVIOUS}, {'attempts': []}, {'attempts': 123},
                        {'attempts': [{'status': 'sent', 'errcode': False}]}):
            _, receipt = self.success(); receipt.update(changed); self.write('delivery.json', receipt)
            with self.subTest(changed=changed):
                result = watchdog.main(now=NOW)
                self.assertEqual(result['status'], 'FAILED')
                self.assertTrue(any('发送回执' in a for a in result['alerts']))

    def test_uncertain_failed_pending_and_expired_delivery_are_alerted(self):
        for status in ('uncertain', 'failed', 'pending', 'sending', 'expired'):
            _, receipt = self.success(); receipt['status'] = status; self.write('delivery.json', receipt)
            with self.subTest(status=status):
                result = watchdog.main(now=NOW)
                self.assertEqual(result['status'], 'FAILED')
                self.assertTrue(any(status in a for a in result['alerts']))

    def test_corrupt_canonical_or_stale_at_generation_quote_is_not_success(self):
        for mode in ('digest', 'old_quote', 'outside_window'):
            if mode == 'digest':
                journal, _ = self.success(); journal['text'] += 'tampered'; self.write('daily_state.json', journal)
            elif mode == 'old_quote':
                self.success(quote='10:00:00')
            else:
                self.success(generated='14:49:00', quote='14:48:50')
            with self.subTest(mode=mode):
                result = watchdog.main(now=NOW)
                self.assertEqual(result['status'], 'FAILED')
                self.assertFalse(result['canonical_verified'])

    def test_both_sources_failed_with_successful_delivery_is_degraded_not_ok(self):
        self.success()
        (self.root/'data/510300.csv').unlink()
        self.probe.side_effect = subprocess.TimeoutExpired('probe', 6)
        result = watchdog.main(now=NOW)
        self.assertEqual(result['status'], 'DEGRADED')
        self.assertEqual(result['source_comparison']['compared_dates'], [])
        self.assertNotIn(' OK', self.log())
        # Flaky probes get one retry and degrade to log-only notes, never a page.
        self.assertEqual(self.probe.call_count, 2)
        self.alarm.assert_not_called()
        self.assertTrue(any('探测失败' in n for n in result['notes']))

    def test_zero_completed_common_dates_is_explicitly_degraded(self):
        self.success()
        self.probe.return_value = [(TODAY, 103.)]
        result = watchdog.main(now=NOW)
        self.assertEqual(result['status'], 'DEGRADED')
        self.assertTrue(any('对比为0' in n for n in result['notes']))
        self.alarm.assert_not_called()

    def test_alarm_transport_failure_does_not_erase_monitor_failure(self):
        self.alarm.side_effect = OSError('mock alarm unavailable')
        result = watchdog.main(now=NOW)
        self.assertEqual(result['status'], 'FAILED')
        self.assertIn('告警发送未确认', self.log())
        self.assertNotIn(' OK', self.log())

    def test_historical_divergence_remains_a_failed_health_check(self):
        self.success()
        self.probe.return_value = [(PREVIOUS, 102.), (TODAY, 103.)]
        result = watchdog.main(now=NOW)
        self.assertEqual(result['status'], 'FAILED')
        self.assertTrue(any('偏差' in a for a in result['alerts']))

    def test_published_weekday_holiday_is_skip_without_source_or_alarm_calls(self):
        result = watchdog.main(now=datetime(2026, 9, 25, 15, 20))
        self.assertEqual(result['status'], 'SKIP')
        self.probe.assert_not_called(); self.alarm.assert_not_called()
        self.assertIn(' SKIP', self.log()); self.assertNotIn(' OK', self.log())

    def test_unknown_calendar_and_failed_source_cannot_be_called_a_holiday(self):
        self.probe.side_effect = subprocess.TimeoutExpired('probe', 6)
        result = watchdog.main(now=datetime(2027, 1, 4, 15, 20))
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertIsNone(result['calendar'])
        self.assertTrue(any('请补充日历' in a for a in result['alerts']))
        self.assertNotIn(' SKIP', self.log()); self.assertNotIn(' OK', self.log())

    def test_unknown_calendar_without_today_evidence_stays_unknown_even_if_history_matches(self):
        self.probe.return_value = [(PREVIOUS, 100.)]
        result = watchdog.main(now=datetime(2027, 1, 4, 15, 20))
        self.assertEqual(result['source_comparison']['status'], 'OK')
        self.assertEqual(result['status'], 'UNKNOWN')

    def test_unknown_year_with_today_canonical_confirms_open_but_requests_calendar_update(self):
        self.success(day='2027-01-04')
        (self.root/'data/qvix50.csv').write_text('2027-01-04,20\n')
        result = watchdog.main(now=datetime(2027, 1, 4, 15, 20))
        self.assertTrue(result['observed_open'])
        self.assertEqual(result['status'], 'DEGRADED')
        self.assertTrue(any('已凭当日信号' in a for a in result['alerts']))

    def success_with_holding(self, code='513100', close=2.0):
        """Schema-2 journal (pre-PRIMARY_START_DATE) carrying one held asset."""
        journal, receipt = self.success()
        accounts = {'shadow_v92.json': dict(last_date=TODAY, holding=code)}
        journal.update(schema=2, account_states=accounts,
                       updated_accounts=['shadow_v92.json'])
        journal.pop('primary_version', None)
        journal['checksum'] = watchdog.store.record_digest(journal)
        self.write('daily_state.json', journal)
        (self.root / ('data/%s.csv' % code)).write_text(PREVIOUS + ',%s,%s,10\n' % (close, close))
        return journal

    def test_held_asset_close_is_cross_checked_and_divergence_pages(self):
        self.success_with_holding(close=2.0)
        with patch.object(watchdog, '_probe', return_value=[(PREVIOUS, 2.10)]) as held_probe:
            result = watchdog.main(now=NOW)
        held_probe.assert_called_once_with('513100', TODAY)
        self.assertEqual(result['status'], 'FAILED')
        self.assertTrue(any('持仓513100' in a and '偏差' in a for a in result['alerts']))
        self.alarm.assert_called_once()

    def test_held_asset_matching_closes_keep_the_day_ok(self):
        self.success_with_holding(close=2.0)
        with patch.object(watchdog, '_probe', return_value=[(PREVIOUS, 2.0)]):
            result = watchdog.main(now=NOW)
        self.assertEqual(result['status'], 'OK')
        self.assertEqual(result['source_comparison']['assets']['513100']['status'], 'OK')
        self.alarm.assert_not_called()

    def test_held_asset_probe_failure_is_a_note_not_a_page(self):
        self.success_with_holding(close=2.0)
        with patch.object(watchdog, '_probe',
                          side_effect=subprocess.TimeoutExpired('probe', 6)) as held_probe:
            result = watchdog.main(now=NOW)
        self.assertEqual(held_probe.call_count, 2)  # one retry, then degrade
        self.assertEqual(result['status'], 'OK')
        self.assertTrue(any('持仓513100' in n for n in result['notes']))
        self.alarm.assert_not_called()

    def test_probe_has_one_process_deadline_not_eleven_serial_network_timeouts(self):
        # Bypass the normal mocked probe only while subprocess itself is mocked.
        with patch.object(watchdog.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 6)) as runner:
            with self.assertRaises(subprocess.TimeoutExpired):
                self.original_probe(TODAY)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(runner.call_args.kwargs['timeout'], 6)
            self.assertIn('--probe', runner.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
