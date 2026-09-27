"""Recovery across absent bars, without fake prices or dated cash assumptions."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v10_live import data as live
from v10_live.ledger import advance, CANDIDATE_ID
from tests.test_v10_live_data import seed_fixture, quotes


HELD = '159915'
TAIL = ['2026-09-25', '2026-09-28', '2026-09-29', '2026-09-30']
PRICE = dict(zip(TAIL, (105., 106., 107., 110.)))


def decision(code=HELD):
    return dict(target=code, candidate_id=CANDIDATE_ID, executable=True,
                reason='synthetic', crash_trigger_date=None, crash_code=None)


class GapRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache = Path(self.tmp.name)
        self.seed = seed_fixture()
        self.original_seed = deepcopy(self.seed)
        self.patch = patch.object(live, '_load_seed', return_value=self.seed)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def pairs(self, date, gap=True, dividend=False, split=False):
        pairs = {}
        for code in live.CODES:
            raw = list(self.seed['raw'][code])
            for day in TAIL:
                if day > date or (gap and code == HELD and day == '2026-09-28'):
                    continue
                p = PRICE[day] / (2 if split and code == HELD and day >= '2026-09-29' else 1)
                raw.append((day, p, p, 400. if split and code == HELD and day >= '2026-09-29' else 200.))
            adjusted = []
            for day, o, c, v in raw:
                if code == HELD and day < '2026-09-29':
                    if dividend:
                        o, c = o - 1, c - 1
                    elif split:
                        o, c = o / 2, c / 2
                adjusted.append((day, o, c, v))
            pairs[code] = dict(raw=raw, qfq=adjusted)
        return pairs

    def run_view(self, date, pairs=None, split=False):
        q = quotes(date, price=PRICE[date], opening=PRICE[date])
        if split:
            q[HELD].update(price=PRICE[date] / 2, open=PRICE[date] / 2, volume=400.)
        source = self.pairs(date, split=split) if pairs is None else pairs
        view = live.build_live_view(q, date,
            now=datetime.strptime(date + ' 14:50:02', '%Y-%m-%d %H:%M:%S'),
            cache_dir=self.cache, fetch_pair=lambda code, start, end: deepcopy(source[code]),
            split_overrides={HELD: {'2026-09-29': {'ratio': 2, 'ratio_exact': '2'}}} if split else None)
        return view, q

    def state(self):
        return json.loads((self.cache / 'completed.json').read_text())['state']

    def test_resume_and_following_day_recover_without_filling_missing_bar(self):
        first, _ = self.run_view('2026-09-29')
        cache = self.state()
        self.assertEqual(cache['schema'], 2)
        self.assertEqual(cache['last_completed'], '2026-09-28')
        self.assertEqual(cache['assets'][HELD]['last_observed_completed'], '2026-09-25')
        self.assertEqual(cache['assets']['510300']['last_observed_completed'], '2026-09-28')
        self.assertEqual(cache['calendar'], ['2026-09-25', '2026-09-28'])
        for rows in (first['histories'][HELD], first['raw_histories'][HELD]):
            self.assertNotIn('2026-09-28', [r[0] for r in rows])
        gap = first['actions'][HELD]['2026-09-28']
        self.assertTrue(gap['not_observed'])
        self.assertEqual(gap['verification'], 'bracketed_no_action_interval')
        self.assertEqual(gap['previous_quote_date'], '2026-09-25')
        self.assertEqual(gap['next_quote_date'], '2026-09-29')
        self.assertEqual((gap['split_ratio'], gap['cash_per_old_share']), (1., 0.))
        self.assertAlmostEqual(first['histories'][HELD][-1][2], 1.07)
        self.assertEqual(first['histories'][HELD][-1][5], 200.)
        second, _ = self.run_view('2026-09-30')
        self.assertAlmostEqual(second['histories'][HELD][-1][2], 1.10)
        self.assertEqual(self.state()['assets'][HELD]['last_observed_completed'], '2026-09-29')
        self.assertEqual(set(second['actions'][HELD]), set(TAIL))
        self.assertEqual(self.seed, self.original_seed)

    def test_one_day_gap_after_latest_cached_date_uses_asset_observed_watermark(self):
        # No post-seed quote before the gap: this previously required the asset
        # to quote on the benchmark's latest completed date and never recovered.
        p = self.pairs('2026-09-29')
        for kind in ('raw', 'qfq'):
            p[HELD][kind] = [r for r in p[HELD][kind] if r[0] != '2026-09-25']
        self.run_view('2026-09-29', p)
        self.assertEqual(self.state()['assets'][HELD]['last_observed_completed'], self.seed['end'])
        following = self.pairs('2026-09-30')
        for kind in ('raw', 'qfq'):
            following[HELD][kind] = [r for r in following[HELD][kind] if r[0] != '2026-09-25']
        v, _ = self.run_view('2026-09-30', following)
        self.assertAlmostEqual(v['histories'][HELD][-1][2], 1.10)

    def test_units_ledger_carries_confirmed_zero_action_gap_without_a_price(self):
        initial, q = self.run_view('2026-09-25')
        account = advance(None, decision(), initial, q, '2026-09-25')
        view, q = self.run_view('2026-09-29')
        resumed = advance(account, decision(), view, q, '2026-09-29')
        self.assertAlmostEqual(resumed['units'], 1 / 105.)
        self.assertAlmostEqual(resumed['nav'], 107 / 105.)
        self.assertEqual(resumed['events'][-1]['actions'], [])
        next_view, q = self.run_view('2026-09-30')
        after = advance(resumed, decision(), next_view, q, '2026-09-30')
        self.assertAlmostEqual(after['nav'], 110 / 105.)
        self.assertEqual(after['switches'], 0)

    def test_unknown_cross_gap_dividend_does_not_get_assigned_to_resume_day(self):
        self.run_view('2026-09-25')
        path = self.cache / 'completed.json'; before = path.read_bytes()
        for date in ('2026-09-29', '2026-09-30'):
            with self.subTest(date=date), self.assertRaisesRegex(live.LiveDataError, '跨缺报价区间'):
                self.run_view(date, self.pairs(date, dividend=True))
            self.assertEqual(path.read_bytes(), before)

    def test_explicit_split_on_first_new_unit_quote_preserves_wealth_and_volume(self):
        view, q = self.run_view('2026-09-25')
        account = advance(None, decision(), view, q, '2026-09-25')
        pairs = self.pairs('2026-09-29', split=True)
        with self.assertRaises(live.LiveDataError):
            self.run_view('2026-09-29', pairs)
        resumed, q = self.run_view('2026-09-29', split=True)
        self.assertAlmostEqual(resumed['histories'][HELD][-1][2], 1.07)
        self.assertEqual(resumed['histories'][HELD][-1][5], 200.)
        self.assertEqual(resumed['actions'][HELD]['2026-09-29']['split_ratio'], 2.)
        self.assertEqual(resumed['actions'][HELD]['2026-09-28']['split_ratio'], 1.)
        updated = advance(account, decision(), resumed, q, '2026-09-29')
        self.assertAlmostEqual(updated['units'], 2 / 105.)
        self.assertAlmostEqual(updated['nav'], 107 / 105.)
        final, q = self.run_view('2026-09-30', split=True)
        settled = advance(updated, decision(), final, q, '2026-09-30')
        self.assertAlmostEqual(settled['nav'], 110 / 105.)
        self.assertEqual(self.state()['assets'][HELD]['actions']['2026-09-29']['split_ratio'], 2.)

    def test_schema1_cache_is_migrated_without_resetting_observed_history(self):
        self.run_view('2026-09-28', self.pairs('2026-09-28', gap=False))
        old = self.state(); old['schema'] = 1; old.pop('calendar')
        for a in old['assets'].values():
            a.pop('last_observed_completed')
        before = deepcopy(old)
        live._write_state(self.cache / 'completed.json', old)
        self.run_view('2026-09-29')
        now = self.state()
        self.assertEqual(now['schema'], 2)
        for code in live.CODES:
            self.assertEqual(now['assets'][code]['raw'][:1], before['assets'][code]['raw'])

    def test_backfilled_previously_absent_raw_bar_requires_explicit_reconciliation(self):
        self.run_view('2026-09-29')
        path = self.cache / 'completed.json'; before = path.read_bytes()
        with self.assertRaisesRegex(live.LiveDataError, '补回'):
            self.run_view('2026-09-30', self.pairs('2026-09-30', gap=False))
        self.assertEqual(path.read_bytes(), before)

    def test_ledger_rejects_unverified_or_cash_bearing_gap_marker(self):
        first, q = self.run_view('2026-09-25')
        account = advance(None, decision(), first, q, '2026-09-25')
        view, q = self.run_view('2026-09-29')
        for change in ({'cash_per_old_share': .1}, {'split_ratio': 2.}, {'verification': 'assumed'}):
            broken = deepcopy(view)
            broken['actions'][HELD]['2026-09-28'].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'unsafe unobserved'):
                advance(account, decision(), broken, q, '2026-09-29')

    def test_schema2_cache_requires_marker_for_every_absent_calendar_day(self):
        self.run_view('2026-09-29')
        cache = self.state(); cache['assets'][HELD]['actions']['2026-09-28'].pop('not_observed')
        live._write_state(self.cache / 'completed.json', cache)
        with self.assertRaisesRegex(live.LiveDataError, '缺报价日'):
            self.run_view('2026-09-30')


if __name__ == '__main__':
    unittest.main()
