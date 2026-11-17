"""Independent engine audit: archived parity, prefix causality, units/cash.

Writes only validation_extra.json. No original data/code mutations or network.
"""
from bisect import bisect_right
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from research.v12_iteration_20261117 import engine
from research.v12_iteration_20261117.data_bundle import load_dataset
from v12.reference import run_reference

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefix(arrays, meta, stop):
    out = {k: v[:, :stop].copy() if k in ('scores', 'orders') else v[:stop].copy()
           for k, v in arrays.items()}
    m = deepcopy(meta)
    m['dates'] = m['dates'][:stop]
    m['shape'] = list(out['features'].shape)
    return out, m


def synthetic_units(config, missing=False):
    dates = ['2026-01-%02d' % n for n in range(5, 10)]
    assets = ['159915', '511880', '510300']
    cube = np.ones((5, 3, 2), dtype=np.float64)
    cube[:, 0, 0] = [10., 11., 9., 8., 8.]
    cube[:, 1, 0] = [100., np.nan if missing else 101., 102., 103., 104.]
    cube[:, 2, 0] = 10.
    meta = dict(assets=assets, dates=dates, feature_names=['close', 'valid'], score_names=[config['score']])
    arrays = dict(features=cube, scores=np.ones((1, 5, 3)), fear=np.zeros(5))

    class ScriptedPolicy:
        def __init__(self, data, cfg, spec):
            self.rotation = SimpleNamespace(candidate=None, count=0, clear=lambda: None)
            self.cooldown_until = {code: -1 for code in assets}
            self.locked_exit_requested = False
            self.fallback_used = False

        def intent(self, day, account, can_sell):
            cash_day = 1 if missing else 3
            return ('159915' if day < cash_day else '511880'), False, day == 0

        def after_fill(self, account):
            pass

    with patch.object(engine, 'ResearchPolicy', ScriptedPolicy):
        r = engine.run_portfolio(arrays, meta, config, {'crash_weight': .5}, fee=.001)
    t = r['trace']
    np.testing.assert_allclose(r['units'][0], [.05, .005, 0.], rtol=0., atol=1e-16)
    if not missing:
        np.testing.assert_array_equal(r['units'][0], r['units'][1])
        np.testing.assert_array_equal(r['units'][0], r['units'][2])
        np.testing.assert_allclose([x['nav'] for x in t[:4]], [1., 1.055, .96, .9142], rtol=0., atol=3e-16)
        np.testing.assert_allclose(r['weights'][1, 0], .55/1.055, rtol=0., atol=3e-16)
        np.testing.assert_allclose(r['weights'][2, 0], .45/.96, rtol=0., atol=3e-16)
        np.testing.assert_allclose(t[3]['cost'], .0008, rtol=0., atol=1e-16)
        np.testing.assert_allclose(r['units'][3, 1], .9142/103, rtol=0., atol=1e-16)
        np.testing.assert_array_equal(r['units'][3], r['units'][4])
    else:
        assert t[1]['filled'] is False and t[1]['blocked_missing'] == ['511880']
        np.testing.assert_array_equal(r['units'][0], r['units'][1])
        np.testing.assert_allclose(t[1]['nav'], 1.05, rtol=0., atol=3e-16)
        assert t[2]['filled'] is True and t[2]['holding'] == '511880'
        np.testing.assert_allclose(t[2]['pretrade_nav'], .96, rtol=0., atol=3e-16)
        np.testing.assert_allclose(t[2]['cost'], .0009, rtol=0., atol=1e-16)
        np.testing.assert_allclose(t[2]['nav'], .9591, rtol=0., atol=3e-16)
    return dict(passed=True, missing_cash=missing,
                nav=[x['nav'] for x in t], risk_weights=r['weights'][:, 0].tolist(),
                fees=r['costs'].tolist(), units=r['units'].tolist())


def main():
    engine_hash = sha(HERE/'engine.py')
    bundle_hash = sha(HERE/'dataset.npz')
    arrays, meta, config = load_dataset()
    report = dict(engine_sha256=engine_hash, dataset_sha256=bundle_hash,
                  dataset_end=meta['dates'][-1], baseline_scenarios={}, prefix_checks=[])
    for label, lag, fee in [('close_1bp', 0, .0001), ('close_11bp', 0, .0011),
                             ('lag1_1bp', 1, .0001), ('lag1_11bp', 1, .0011)]:
        result = engine.run_portfolio(arrays, meta, config, start='2014-01-02', end='2026-09-24', lag=lag, fee=fee)
        ref = run_reference(arrays, meta, dict(config, lag=lag), '2014-01-02', '2026-09-24', fee)
        np.testing.assert_array_equal(result['holdings'], ref['holdings'])
        np.testing.assert_allclose(result['returns'], ref['returns'], rtol=0., atol=2e-14)
        np.testing.assert_allclose(result['summary'], ref['summary'], rtol=2e-12, atol=2e-12)
        for a, b in zip(result['trace'], ref['trace']):
            for field in ['signal_date', 'intended', 'holding', 'panic', 'crash_requested', 'filled', 'lock_until']:
                assert a[field] == b[field], (label, a['date'], field)
        report['baseline_scenarios'][label] = dict(passed=True, identical_holdings=True,
            identical_decision_trace=True, return_max_abs_error=float(np.max(abs(result['returns']-ref['returns']))),
            final_nav_abs_error=float(abs(result['summary'][0]-ref['summary'][0])))

    specs = dict(baseline={}, half={'crash_weight': .5},
                 half_cash_defensive={'crash_weight': .5, 'fallback_hysteresis': .02, 'defensive_lowvol': True})
    for name, spec in specs.items():
        for lag in (0, 1):
            full = engine.run_portfolio(arrays, meta, config, spec, start='2014-01-02', lag=lag)
            for boundary in ['2020-12-31', '2023-12-31', '2026-09-24']:
                stop = bisect_right(meta['dates'], boundary)
                a, m = prefix(arrays, meta, stop)
                short = engine.run_portfolio(a, m, config, spec, start='2014-01-02', lag=lag)
                n = len(short['dates'])
                for field in ['returns', 'holdings', 'weights', 'units', 'turnover', 'costs']:
                    np.testing.assert_array_equal(short[field], full[field][:n])
                report['prefix_checks'].append(dict(spec=name, lag=lag, truncated_after=m['dates'][-1],
                                                     observations=n, bit_exact=True))
            # Fixed units remain untouched on all no-fill days. Portfolio
            # weights sum to one and marks determine NAV, including cash drift.
            checks = 0
            for i, trace in enumerate(full['trace']):
                if i and not trace['filled']:
                    np.testing.assert_array_equal(full['units'][i], full['units'][i-1])
                    checks += 1
                if trace['units']:
                    np.testing.assert_allclose(sum(trace['weights'].values()), 1., rtol=0., atol=2e-14)
                if trace['filled'] and not trace['initial_session_free']:
                    np.testing.assert_allclose(trace['nav'], trace['pretrade_nav']-trace['cost'], rtol=0., atol=2e-12)
            report.setdefault('fixed_unit_checks', []).append(dict(spec=name, lag=lag, unchanged_unit_days=checks, passed=True))
    report['synthetic_half_units'] = synthetic_units(config)
    report['synthetic_missing_cash'] = synthetic_units(config, missing=True)
    assert sha(HERE/'engine.py') == engine_hash, 'Engine changed during audit; rerun'
    assert sha(HERE/'dataset.npz') == bundle_hash, 'Bundle changed during audit; rerun'
    report['passed'] = True
    report['limitations'] = [
        'Engine NAV uses unit accounting, so baseline P&L is numerically equivalent, not bit-exact; targets and decision traces are exact.',
        'Prefix tests cover baseline, half-weight, and combined half/cash/defensive specifications at lag0 and lag1.',
        'Dataset itself preserves frozen effective feature prefix bit-exact and constructs extension from own-quote prefixes.',
        'Fee tests validate documented legacy pre-fee turnover approximation, not exact brokerage self-financing costs.',
        'TR index cash units model virtual reinvestment; they are not literal broker cash balances or lot-sized holdings.',
    ]
    (HERE/'validation_extra.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k,v in report.items() if k not in ('prefix_checks', 'fixed_unit_checks')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
