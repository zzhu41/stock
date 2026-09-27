"""Isolated live TR tests: no network, real cache writes or production messages."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v10_live import data as live


SEED_DATES = ['2026-09-16', '2026-09-17', '2026-09-18', '2026-09-21',
              '2026-09-22', '2026-09-23', '2026-09-24']
PRICES = [100., 101., 99., 100., 102., 101., 103.]


def seed_fixture():
    raw = {c: [(d, p, p, 100.) for d, p in zip(SEED_DATES, PRICES)] for c in live.CODES}
    tr = {c: [(d, p / 100, p / 100, 100.) for d, p in zip(SEED_DATES, PRICES)] for c in live.CODES}
    return dict(end=SEED_DATES[-1], manifest_hash='frozen-seed', raw=raw, tr=tr,
                assets={c: dict(cash_events=[], split_events=[]) for c in live.CODES})


def quotes(date='2026-09-25', price=106., opening=104.):
    return {c: dict(date=date, timestamp=date + ' 14:50:00', price=price, open=opening, volume=200.)
            for c in live.CODES}


class LiveDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.seed = seed_fixture()
        self.saved_seed = deepcopy(self.seed)
        self.seed_patch = patch.object(live, '_load_seed', return_value=self.seed)
        self.seed_patch.start()
        self.addCleanup(self.seed_patch.stop)

    def pairs(self, date='2026-09-25'):
        out = {}
        for c in live.CODES:
            raw = list(self.seed['raw'][c])
            if date == '2026-09-28':
                raw.append(('2026-09-25', 104., 105., 120.))
            raw.append((date, 104., 104.5, 180.))
            q = list(raw)
            q[-1] = (date, 104., 105.5, 190.)
            out[c] = dict(raw=raw, qfq=q)
        return out

    def run_view(self, pairs=None, q=None, date='2026-09-25', **kwargs):
        pairs = self.pairs(date) if pairs is None else pairs
        q = quotes(date) if q is None else q
        now = datetime.strptime(date + ' 14:50:02', '%Y-%m-%d %H:%M:%S')
        return live.build_live_view(q, date, now=now, cache_dir=self.directory,
                                   fetch_pair=lambda code, start, end: deepcopy(pairs[code]), **kwargs)

    def test_live_tr_uses_main_quote_not_asynchronous_daily_closes(self):
        v = self.run_view()
        for c in live.CODES:
            self.assertAlmostEqual(v['histories'][c][-1][2], 1.06)
            self.assertEqual(len(v['histories'][c][-1]), 6)
            self.assertEqual(v['histories'][c][-1][5], 200.)
            self.assertEqual(v['raw_histories'][c][-1][2], 106.)
            self.assertEqual(v['actions'][c]['2026-09-25']['cash_per_old_share'], 0.)
        self.assertEqual(self.seed, self.saved_seed)
        state = json.loads((self.directory / 'completed.json').read_text())['state']
        self.assertEqual(state['last_completed'], '2026-09-24')
        self.assertTrue(all(not a['raw'] and not a['actions'] for a in state['assets'].values()))

    def test_today_dividend_uses_stable_open_and_is_not_added_twice(self):
        pairs = self.pairs()
        code = '159915'
        pairs[code]['qfq'][:-1] = [(d, o - 1, c - 1, v) for d, o, c, v in pairs[code]['raw'][:-1]]
        first = self.run_view(pairs)
        self.assertAlmostEqual(first['actions'][code]['2026-09-25']['cash_per_old_share'], 1.)
        self.assertAlmostEqual(first['histories'][code][-1][2], 1.07)
        updated = quotes(price=107.)
        second = self.run_view(pairs, updated)
        self.assertAlmostEqual(second['histories'][code][-1][2], 1.08)
        self.assertEqual(len(first['histories'][code]), len(second['histories'][code]))
        self.assertEqual(first['histories'][code][:-1], second['histories'][code][:-1])

    def test_explicit_split_changes_new_volume_units_without_rewriting_seed(self):
        pairs = self.pairs(); q = quotes(); code = '159915'
        pairs[code]['qfq'][:-1] = [(d, o / 2, c / 2, v) for d, o, c, v in pairs[code]['raw'][:-1]]
        pairs[code]['raw'][-1] = ('2026-09-25', 52., 52.5, 180.)
        pairs[code]['qfq'][-1] = ('2026-09-25', 52., 52.7, 190.)
        q[code].update(open=52., price=53.)
        with self.assertRaises(live.LiveDataError):
            self.run_view(pairs, q)
        view = self.run_view(pairs, q, split_overrides={code: {'2026-09-25': 2}})
        self.assertAlmostEqual(view['histories'][code][-1][2], 1.06)
        self.assertEqual(view['histories'][code][-1][5], 100.)
        self.assertEqual(view['actions'][code]['2026-09-25']['split_ratio'], 2.)
        self.assertEqual([r[5] for r in view['histories'][code][:-1]], [100.] * len(SEED_DATES))

    def test_catchup_commits_only_completed_rows_and_includes_zero_actions(self):
        self.run_view()
        view = self.run_view(date='2026-09-28')
        state = json.loads((self.directory / 'completed.json').read_text())['state']
        self.assertEqual(state['last_completed'], '2026-09-25')
        for c in live.CODES:
            self.assertEqual([r[0] for r in state['assets'][c]['raw']], ['2026-09-25'])
            self.assertEqual(state['assets'][c]['raw'][0][2], 105.)
            self.assertEqual(set(view['actions'][c]), {'2026-09-25', '2026-09-28'})
            self.assertAlmostEqual(view['histories'][c][-1][2], 1.06)

    def test_missing_today_history_and_raw_revision_fail_closed(self):
        for mode in ('missing_today', 'revised_raw'):
            pairs = self.pairs('2026-09-28')
            if mode == 'missing_today':
                pairs['159915']['raw'].pop(); pairs['159915']['qfq'].pop()
            elif mode == 'revised_raw':
                r = pairs['159915']['raw'][1]
                pairs['159915']['raw'][1] = (r[0], r[1], r[2] + .001, r[3])
            with self.subTest(mode=mode), self.assertRaises(live.LiveDataError):
                self.run_view(pairs, date='2026-09-28')
        self.assertFalse((self.directory / 'completed.json').exists())

    def test_all_eleven_quotes_including_cash_must_be_fresh_and_positive(self):
        for mode in ('cash_missing', 'stale', 'future', 'zero', 'skew'):
            q = quotes()
            if mode == 'cash_missing': del q['511880']
            elif mode == 'stale': q['159915']['timestamp'] = '2026-09-25 14:00:00'
            elif mode == 'future': q['159915']['timestamp'] = '2026-09-25 15:00:00'
            elif mode == 'zero': q['159915']['price'] = 0
            else: q['159915']['timestamp'] = '2026-09-25 14:48:00'
            with self.subTest(mode=mode), self.assertRaises(live.LiveDataError): self.run_view(q=q)
        with self.assertRaises(live.LiveDataError): self.run_view(date='2026-09-27')

    def test_download_elapsed_time_invalidates_previously_fresh_quote(self):
        with patch.object(live, '_fetch_all', return_value=self.pairs()), \
                patch.object(live.time, 'monotonic', side_effect=[0., 180.]), \
                self.assertRaisesRegex(live.LiveDataError, '陈旧'):
            self.run_view()
        self.assertFalse((self.directory / 'completed.json').exists())

    def test_atomic_failure_preserves_previous_completed_cache(self):
        self.run_view()
        path = self.directory / 'completed.json'; before = path.read_bytes()
        with patch.object(live.os, 'replace', side_effect=OSError('rename unavailable')), self.assertRaises(OSError):
            self.run_view(date='2026-09-28')
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.directory.glob('.completed-*')), [])

    def alias_pairs(self):
        pairs = self.pairs()
        for code in live.DAY_ALIAS_CODES:
            pairs[code].update(qfq_alias='day', qfq_provider_version='16',
                              qfq_quote_info=dict(date='2026-09-25', timestamp='2026-09-25 14:50:00',
                                                  prev_close=103., open=104.))
        return pairs

    def test_whitelisted_no_action_day_alias_is_explicitly_validated(self):
        view = self.run_view(self.alias_pairs())
        for c in live.DAY_ALIAS_CODES:
            self.assertEqual(view['metadata']['receipts'][c]['qfq_representation'], 'validated_no_known_action_day_alias')
            self.assertEqual(view['actions'][c]['2026-09-25']['cash_per_old_share'], 0.)

    def test_day_alias_rejects_action_history_different_completed_quotes_or_reference(self):
        for mode in ('seed_action', 'override', 'prices', 'qt_date', 'qt_previous', 'version'):
            pairs = self.alias_pairs(); kwargs = {}; code = '563300'
            if mode == 'seed_action': self.seed['assets'][code]['cash_events'] = [dict(date='2026-09-22', cash_per_old_share=.01)]
            elif mode == 'override': kwargs['split_overrides'] = {code: {'2026-09-25': 2}}
            elif mode == 'prices':
                d,o,c,v = pairs[code]['qfq'][0]; pairs[code]['qfq'][0] = (d,o-.01,c-.01,v)
            elif mode == 'qt_date': pairs[code]['qfq_quote_info']['date'] = '2026-09-24'
            elif mode == 'qt_previous': pairs[code]['qfq_quote_info']['prev_close'] = 102.
            else: pairs[code]['qfq_provider_version'] = 'unknown'
            with self.subTest(mode=mode), self.assertRaises(live.LiveDataError): self.run_view(pairs, **kwargs)
            self.seed['assets'][code]['cash_events'] = []

    def test_provider_day_alias_extracts_only_audited_quote_fields(self):
        code='563300'; symbol='sh'+code; raw=self.pairs()[code]['raw']
        six=[[d,str(o),str(c),str(max(o,c)),str(min(o,c)),str(v)] for d,o,c,v in raw]
        qt=['']*31;qt[4]='103';qt[5]='104';qt[30]='20260925145000'
        bodies=[dict(data={symbol:dict(day=six)}), dict(data={symbol:dict(day=six,qt={symbol:qt},version=16)})]
        class Response:
            def __init__(self, body): self.body=body
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,*args): return json.dumps(self.body).encode()
        with patch.object(live.urllib.request,'urlopen',side_effect=[Response(x) for x in bodies]):
            pair=live._fetch_pair(code,'2026-09-16','2026-09-25')
        self.assertEqual(pair['qfq_alias'],'day')
        self.assertEqual(pair['qfq_quote_info']['prev_close'],103.)
        self.assertEqual(pair['qfq_provider_version'],'16')


if __name__ == '__main__': unittest.main()
