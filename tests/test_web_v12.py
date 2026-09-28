"""Pinned R2 web export and read-only forward-card integration."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import signal_store as store
import web_app
import web_build
import web_signals
import web_v12


class FrozenWebV12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = [web_v12.BASE / p for p in ('release_receipt.json', 'profiles.json',
            'results/paths.npz', 'results/audit/reference_paths.npz', 'results/selection.json')]
        cls.before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in cls.files}
        with patch('urllib.request.urlopen', side_effect=AssertionError('No network')):
            cls.version = web_v12.export_versions()['v12-r2']

    def test_frozen_path_metrics_and_trade_mapping(self):
        v = self.version
        self.assertAlmostEqual(v['metrics']['ann'], 55.26937725744602, places=12)
        self.assertAlmostEqual(v['metrics']['max_dd'], -20.55529108380264, places=12)
        self.assertEqual(v['metrics']['days'], 3097)
        self.assertEqual(v['metrics']['switches'], 310)
        self.assertEqual(len(v['trades']), 311)
        self.assertEqual(v['daily'][0], ['2014-01-02', 1., v['daily'][0][2]])
        np.testing.assert_array_equal(np.cumprod(1 + np.asarray(v['daily_returns'])), [r[1] for r in v['daily']])
        index = {r[0]: i for i, r in enumerate(v['daily'])}
        for date, before, target, nav in v['trades']:
            i = index[date]
            self.assertEqual(v['daily'][i][1:], [nav, target])
            self.assertEqual(before, v['daily'][i-1][2] if i else None)
        json.dumps(v, allow_nan=False)

    def test_deployment_choice_does_not_rewrite_unqualified_research_or_inputs(self):
        meta = self.version['metadata']
        self.assertEqual(meta['deployment_choice'], 'user_selected')
        self.assertFalse(meta['research_qualified'])
        self.assertIsNone(meta['research_primary'])
        self.assertFalse(meta['clean_oos'])
        self.assertFalse(meta['shadow_nav_included'])
        self.assertEqual(meta['candidate_id'], web_v12.CANDIDATE_ID)
        self.assertEqual(self.before, {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in self.files})

    def test_year_boundaries_and_arbitrary_interval_include_first_daily_return(self):
        v = self.version
        for year, expected in v['period_metrics']['yearly'].items():
            days = [r for r in v['daily'] if r[0][:4] == year]
            metrics, annual, unused = web_app.range_summary(v, days)
            self.assertAlmostEqual(metrics['total_ret'], expected['total_return'] * 100, places=10)
            self.assertAlmostEqual(annual[0][1], expected['total_return'] * 100, places=10)
        days = v['daily'][20:23]
        metrics, unused, base = web_app.range_summary(v, days)
        self.assertEqual(base, v['daily'][19][1])
        self.assertAlmostEqual(metrics['total_ret'], (np.prod(1 + np.asarray(v['daily_returns'][20:23])) - 1) * 100)
        self.assertEqual(web_app.range_summary(v, v['daily'])[0]['ann'], v['metrics']['ann'])

    def test_tampered_sealed_file_fails_before_any_array_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            shutil.copyfile(web_v12.BASE / 'release_receipt.json', path / 'release_receipt.json')
            (path / 'profiles.json').write_text('{}')
            with patch.object(web_v12, 'BASE', path), patch.object(web_v12.np, 'load') as loader:
                with self.assertRaisesRegex(ValueError, 'input changed'):
                    web_v12.export_versions()
                loader.assert_not_called()

    def test_fast_append_preserves_old_curves_without_fetching_or_rebuilding(self):
        original = dict(start='2014-01-01', names={}, benchmarks={},
                        versions={'v10-h': {'daily': [['old', 42., '511880']]}, 'v9.2-tr': {'preserved': True}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'web_cache.json'; path.write_text(json.dumps(original))
            with patch.object(web_build, 'CACHE', str(path)), \
                    patch.object(web_v12, 'export_versions', return_value={'v12-r2': deepcopy(self.version)}), \
                    patch.object(web_build, 'fetch_history', side_effect=AssertionError('No history fetch')):
                web_build.build_v12_only()
            after = json.loads(path.read_text())
        self.assertEqual(after['versions']['v10-h'], original['versions']['v10-h'])
        self.assertEqual(after['versions']['v9.2-tr'], original['versions']['v9.2-tr'])
        self.assertEqual(after['default_version'], 'v12-r2')

    def test_default_api_version_and_comparison_use_corrected_tr(self):
        base = deepcopy(self.version); base['label'] = 'H same basis'
        payload = dict(start='2014-01-01', names={}, benchmarks={}, versions={'v12-r2': self.version, 'v10-h': base})
        with patch.object(web_app, 'v12_signal', return_value={'status': 'not_started'}):
            response = web_app.series_response(payload, {'compare': ['v10-h']})
        self.assertEqual(response['version'], 'v12-r2')
        self.assertEqual(response['version_meta']['default_compare'], 'v10-h')
        self.assertIsNone(response['comparison_warning'])
        self.assertIsNone(response['signal'].get('target'))
        self.assertEqual(response['metrics']['ann'], self.version['metrics']['ann'])
        self.assertEqual({trade[4] for trade in response['trades']}, {'建仓', '换仓'})

    def test_missing_r2_curve_never_promotes_h_signal_to_primary_card(self):
        payload = dict(versions={'v10-h': deepcopy(self.version)})
        with patch.object(web_app, 'v12_signal', return_value={'status': 'not_started'}) as current, \
                patch.object(web_app, 'v10_signal', side_effect=AssertionError('No legacy advice fallback')):
            result = web_app.primary_signal(payload)
        current.assert_called_once_with(None, now=None)
        self.assertEqual(result['status'], 'not_started')


class ReadonlyR2CardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name); self.now = datetime(2026, 9, 28, 14, 51)
        self.state = dict(version=1, strategy_id='v12-r2', candidate_id=web_v12.CANDIDATE_ID,
            last_date='2026-09-28', start_date='2026-09-28', holding='513100', nav=1.,
            mark_raw_price=2.3, quote_volume=120000., quote_timestamp='2026-09-28 14:50:00')
        self.text = '\n'.join(['动量轮动信号 | 生成 2026-09-28 14:50:10 | 数据截止 2026-09-28',
            '策略版本: V12-R2 | V9.2 | V9.2+ | V10-H',
            '行情时间: 2026-09-28 14:50:00', '【影子 V12-R2】用户指定主推送 · 虚拟跟踪不下单',
            '信号状态: 盘中影子试算', '建议: 513100 纳指ETF'])

    def journal(self, state=None, updated=None, text=None):
        states = {'shadow_v12_r2.json': deepcopy(state or self.state)}
        return dict(date='2026-09-28', text=text or self.text, account_states=states, primary_version=store.PRIMARY_VERSION,
                    updated_accounts=updated if updated is not None else list(states), signal_id='synthetic')

    def call(self, journal=None, when=None, validator=None):
        with patch.object(store, 'load_saved', return_value=((journal or {}).get('text', ''), journal)), \
                patch.object(web_signals, '_validate_v12_state', side_effect=validator):
            return web_signals.v12_signal(None, self.directory, when or self.now)

    def test_missing_state_has_no_preview_target_or_backtest_nav(self):
        v = dict(daily=[['2026-09-24', 266.3055, '513100']])
        result = web_signals.v12_signal(v, self.directory, self.now)
        self.assertEqual(result['status'], 'not_started')
        for field in ('nav', 'target', 'holding', 'preview'): self.assertIsNone(result[field])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_old_generation_failure_before_r2_start_is_not_an_r2_failure(self):
        path = self.directory / 'generation_status.json'
        path.write_text(json.dumps(dict(date='2026-09-28', status='failed')))
        before = web_signals.v12_signal(None, self.directory, datetime(2026, 9, 28, 16))
        self.assertEqual(before['status'], 'not_started')
        self.assertEqual(before['reason'], '尚无有效信号，等待首次生成')
        self.assertIsNone(before['generation_status'])
        self.assertIsNone(before['target'])
        path.write_text(json.dumps(dict(date='2026-09-29', status='failed')))
        after = web_signals.v12_signal(None, self.directory, datetime(2026, 9, 29, 16))
        self.assertEqual(after['status'], 'failed')
        self.assertIsNone(after['target'])
        # A dated R2-specific failure is still meaningful before the first
        # scheduled operating day, unlike an unrelated old global failure.
        failed = self.call(self.journal(text=self.text + '\n信号状态: 本次无有效建议，计算失败'))
        self.assertEqual(failed['status'], 'failed')

    def test_canonical_checkpoint_wins_over_stale_physical_projection(self):
        physical = deepcopy(self.state); physical.update(nav=99., holding='518880')
        path = self.directory / 'shadow_v12_r2.json'; path.write_text(json.dumps(physical))
        before = path.read_bytes()
        result = self.call(self.journal())
        self.assertEqual(result['nav'], 1.)
        self.assertEqual(result['target']['code'], '513100')
        self.assertTrue(result['actionable'])
        self.assertEqual(result['quote_price'], 2.3)
        self.assertEqual(result['quote_volume'], 120000.)
        self.assertEqual(path.read_bytes(), before)

    def test_old_three_version_signal_or_carried_account_cannot_make_current_r2_advice(self):
        journal = self.journal(text=self.text.replace('V12-R2', 'V10-H'))
        self.assertFalse(self.call(journal)['actionable'])
        self.assertIsNone(self.call(journal)['target'])
        journal = self.journal(updated=['shadow_v10.json'])
        self.assertFalse(self.call(journal)['actionable'])
        (self.directory / 'shadow_v12_r2.json').write_text(json.dumps(self.state))
        self.assertFalse(self.call(None)['actionable'])

    def test_failure_stale_quote_or_corrupt_account_never_displays_executable_target(self):
        failure = self.call(self.journal(text=self.text + '\n信号状态: 本次无有效建议，计算失败'))
        self.assertEqual(failure['status'], 'failed'); self.assertIsNone(failure['target'])
        stale = self.call(self.journal(), when=datetime(2026, 9, 28, 15))
        self.assertFalse(stale['actionable']); self.assertIsNone(stale['target'])
        invalid = self.call(self.journal(), validator=ValueError('Wrong candidate or NAV'))
        self.assertEqual(invalid['status'], 'failed'); self.assertIsNone(invalid['nav'])

    def test_real_ledger_validator_and_canonical_journal_start_at_one_readonly(self):
        from v12_live import ledger
        day = '2026-09-29'
        q = {'513100': dict(price=2.3, open=2.31, volume=120000., date=day, timestamp=day + ' 14:50:00')}
        intent = dict(candidate_id=ledger.CANDIDATE_ID, candidate_hash=ledger.CANDIDATE_HASH,
                      strategy_id=ledger.STRATEGY_ID, executable=True, target='513100', reason='offline synthetic')
        state = ledger.advance(None, intent, dict(calendar=[day]), q, day)
        text = self.text.replace('2026-09-28', day)
        record, errors = store.publish(text, day, '513100', None, self.directory,
            account_states={'shadow_v12_r2.json': state}, primary_version=store.PRIMARY_VERSION)
        self.assertEqual(errors, [])
        # A failed compatibility projection cannot replace the canonical NAV.
        (self.directory / 'shadow_v12_r2.json').write_text('{}')
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        result = web_signals.v12_signal(None, self.directory, datetime(2026, 9, 29, 14, 51))
        self.assertTrue(result['actionable']); self.assertEqual(result['nav'], 1.)
        self.assertEqual(result['target']['code'], '513100')
        self.assertEqual(result['quote_volume'], 120000.)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})
        record['checksum'] = 'corrupt'
        (self.directory / 'daily_state.json').write_text(json.dumps(record))
        invalid = web_signals.v12_signal(None, self.directory, datetime(2026, 9, 29, 14, 51))
        self.assertEqual(invalid['status'], 'failed'); self.assertIsNone(invalid['target'])
        self.assertIsNone(invalid['nav'])


@unittest.skipUnless(shutil.which('node'), 'Node is needed for browser JavaScript check')
class BrowserR2Tests(unittest.TestCase):
    def test_frontend_defaults_and_waiting_card_do_not_leak_historical_buy_target(self):
        html = (Path(__file__).resolve().parents[1] / 'web/index.html').read_text()
        script = html.split('<script>')[1].split('</script>')[0]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'page.js'; source.write_text(script)
            subprocess.run(['node', '--check', str(source)], check=True, capture_output=True)
        self.assertIn('version: "v12-r2", compare: "v10-h"', script)
        self.assertLess(html.index('id="shadowCard"'), html.index('id="stats"'))
        begin = script.index('function renderShadow('); end = script.index('\nfunction segColor', begin)
        js = '''const nodes={}; const $=key=>nodes[key]||(nodes[key]={innerHTML:""});
const finite=v=>typeof v==="number"&&Number.isFinite(v);
const esc=v=>String(v==null?"":v).replace(/</g,"&lt;");
const textList=v=>Array.isArray(v)?v:[];
''' + script[begin:end] + '''
renderShadow({label:"V12-R2 主推送",version:"v12-r2",status:"not_started",actionable:false,nav:null,preview:null});
if(!nodes['#shadowContent'].innerHTML.includes('等待首次生成')||nodes['#shadowContent'].innerHTML.includes('513100'))throw Error('Invalid waiting card');
renderShadow({label:"V12-R2 主推送",version:"v12-r2",status:"active",actionable:true,target:{code:"513100",name:"纳指ETF"},holding:{code:"513100"},nav:1,start_date:"2026-09-28",state_date:"2026-09-28",quote_price:2.3,quote_volume:120000});
if(!nodes['#shadowContent'].innerHTML.includes('<strong>513100 纳指ETF</strong>'))throw Error('Target not bold');
'''
        subprocess.run(['node', '-e', js], check=True, capture_output=True)


if __name__ == '__main__': unittest.main()
