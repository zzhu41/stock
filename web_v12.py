"""Read-only export of the user-selected R2 path; never rerun or reseal research."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent / 'v12_r2'
ROOT = BASE.parent
CANDIDATE_ID = 'v12r2_08b522c2e1aa291f054d'
RELEASE_SHA256 = 'f05425fd1e4f599f780d89d785ce93074e1d3af2fe7ea4c8832ad7e9058582eb'
VERSION_ID = 'v12-r2'
LABEL = 'V12-R2 主推送'
SCENARIO = 'close_1bp'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def _read_pinned(name, pins):
    path = BASE / name
    if sha(path) != pins[name]: raise ValueError('Frozen R2 web input changed: ' + name)
    return json.loads(path.read_text(encoding='utf-8'))


def _metrics(returns):
    r = np.asarray(returns, dtype=np.float64)
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1., nav])[1:]
    std = float(r.std()); ann = float(np.expm1(np.log1p(r).sum() * 244 / len(r)))
    dd = float(np.min(nav / peak - 1))
    return dict(sessions=len(r), nav=float(nav[-1]), total_return=float(nav[-1] - 1),
                cagr=ann, max_dd=dd, volatility=std * 244 ** .5,
                sharpe=float(r.mean() / std * 244 ** .5) if std else 0.,
                calmar=ann / abs(dd) if dd else 0.)


def _web_metrics(full, switches):
    return dict(nav=full['nav'], total_ret=100 * full['total_return'], ann=100 * full['cagr'],
                max_dd=100 * full['max_dd'], sharpe=full['sharpe'], calmar=full['calmar'],
                volatility=100 * full['volatility'], days=full['sessions'], switches=int(switches))


def _benchmarks(dates, frozen_inputs):
    manifest_path = 'v10_h_close/corrected_manifest.json'
    if sha(ROOT / manifest_path) != frozen_inputs['sha256'][manifest_path]:
        raise ValueError('Frozen corrected benchmark manifest changed')
    manifest = json.loads((ROOT / manifest_path).read_text())
    result = {}
    for code, name in (('510300', '沪深300ETF'), ('518880', '黄金ETF')):
        path = ROOT / 'v10_h_close/corrected_snapshots' / (code + '.csv')
        if sha(path) != manifest['assets'][code]['sha256']:
            raise ValueError('Frozen corrected benchmark changed: ' + code)
        with path.open(newline='') as stream:
            closes = {r[0]: float(r[2]) for r in csv.reader(stream) if r}
        initial = mark = closes[dates[0]]; previous = 1.; daily, returns = [], []
        for date in dates:
            mark = closes.get(date, mark)
            nav = mark / initial
            daily.append([date, nav]); returns.append(nav / previous - 1); previous = nav
        full = _metrics(returns)
        result[code] = dict(label=name + '（修正总回报买入持有）', daily=daily,
            daily_returns=returns, metrics=_web_metrics(full, 0),
            metadata=dict(basis='corrected_tr_same_close', kind='research',
                          fees_charged=0., shadow_nav_included=False))
    return result


def export_versions():
    if sha(BASE / 'release_receipt.json') != RELEASE_SHA256:
        raise ValueError('Frozen R2 release receipt changed')
    pins = json.loads((BASE / 'release_receipt.json').read_text())['file_sha256']
    names = ('profiles.json', 'results/selection.json', 'results/evaluation.json',
             'results/path_metadata.json', 'results/feature_metadata.json',
             'results/audit/receipt.json', 'frozen_inputs.json')
    items = {name: _read_pinned(name, pins) for name in names}
    for name in ('results/paths.npz', 'results/audit/reference_paths.npz'):
        if sha(BASE / name) != pins[name]: raise ValueError('Frozen R2 web input changed: ' + name)
    profile, choice = items['profiles.json'], items['results/selection.json']
    pm, features = items['results/path_metadata.json'], items['results/feature_metadata.json']
    evaluation, audit = items['results/evaluation.json'], items['results/audit/receipt.json']
    if (profile['roles']['highest_cagr'] != CANDIDATE_ID
            or choice['exploratory']['highest_cagr'] != CANDIDATE_ID
            or profile['primary'] is not None or choice['primary'] is not None
            or choice['balanced_reference'] is not None):
        raise ValueError('User deployment choice must not rewrite frozen qualification')
    if (pm['sha256'] != pins['results/paths.npz']
            or evaluation['paths_sha256'] != pm['sha256']
            or choice['evaluation_sha256'] != pins['results/evaluation.json']
            or not audit['passed'] or not audit['no_reselection']
            or audit['selection_unchanged_sha256'] != pins['results/selection.json']
            or audit['paths_sha256'] != pins['results/audit/reference_paths.npz']):
        raise ValueError('Frozen R2 path/audit chain differs')
    dates, assets = pm['dates'], features['assets']
    if (dates != sorted(set(dates)) or dates[0] != '2014-01-02' or dates[-1] != '2026-09-24'
            or [d for d in features['dates'] if dates[0] <= d <= dates[-1]] != dates):
        raise ValueError('Frozen R2 observation dates differ')
    index = pm['ids'].index(CANDIDATE_ID)
    with np.load(BASE / 'results/paths.npz', allow_pickle=False) as archive:
        saved = {field: archive[SCENARIO + '__' + field][index].copy()
                 for field in ('returns', 'holdings', 'summary')}
    with np.load(BASE / 'results/audit/reference_paths.npz', allow_pickle=False) as archive:
        for field, values in saved.items():
            if not np.array_equal(values, archive[CANDIDATE_ID + '__' + SCENARIO + '__' + field]):
                raise ValueError('Frozen R2 independent reference differs: ' + field)
    cases = [case for case in audit['cases'] if case['id'] == CANDIDATE_ID and case['scenario'] == SCENARIO]
    if len(cases) != 1 or not cases[0]['exact'] or cases[0]['observations'] != len(dates):
        raise ValueError('Missing exact independent R2 path audit')
    returns, indexes = saved['returns'], saved['holdings']
    if (returns.shape != (len(dates),) or indexes.shape != returns.shape
            or not np.isfinite(returns).all() or (returns <= -1).any() or returns[0] != 0
            or not np.equal(indexes, np.floor(indexes)).all()
            or (indexes < -1).any() or (indexes >= len(assets)).any()):
        raise ValueError('Invalid frozen R2 daily path')
    holdings = [assets[int(i)] if i >= 0 else None for i in indexes]
    navs = np.cumprod(1 + returns)
    daily = [[day, float(nav), code] for day, nav, code in zip(dates, navs, holdings)]
    trades = [[day, holdings[i - 1] if i else None, code, float(navs[i])]
              for i, (day, code) in enumerate(zip(dates, holdings))
              if code is not None and (i == 0 or code != holdings[i - 1])]
    row = next(r for r in evaluation['rows'] if r['id'] == CANDIDATE_ID)['scenarios'][SCENARIO]
    computed = _metrics(returns)
    if (any(not np.isclose(computed[k], value, rtol=1e-12, atol=1e-12) for k, value in row['full'].items())
            or len(trades) - 1 != row['switches'] or int(saved['summary'][3]) != row['switches']):
        raise ValueError('Frozen R2 daily path differs from published metrics or turnover')
    metadata = dict(basis='corrected_tr_same_close', kind='research', frozen=True,
        candidate_id=CANDIDATE_ID, config_sha256=profile['records'][CANDIDATE_ID]['hash'],
        period=dict(start=dates[0], end=dates[-1]), data_as_of=dates[-1], clock='same_close',
        fee_per_side=.0001, first_session_free=True, annualization_sessions=244,
        shadow_nav_included=False, clean_oos=False, execution_price_provided=False,
        deployment_choice='user_selected', research_primary=None, research_balanced_reference=None,
        research_qualified=False, research_status='Frozen strict and balanced qualifications failed; user chose deployment separately',
        holding_convention='当日收盘换仓后模型持仓，当天收益主要归此前持仓',
        initial_entry_in_trades=True, switches_exclude_first_session=True,
        crash_markers_available=False, verified_crash_count=row['crashes'],
        warnings=['已知历史回测，截至2026-09-24；不含前向账户净值。',
                  '同收盘理想成交与14:50观察信号不同，尚未证明过拟合更少。',
                  '按用户选择作为主推送；原研究严格及折衷门槛均未全部通过。'],
        provenance=dict(release_receipt_sha256=RELEASE_SHA256,
                        selected_paths_sha256=pins['results/paths.npz'],
                        path_metadata_sha256=pins['results/path_metadata.json'],
                        fidelity_receipt_sha256=pins['results/audit/receipt.json']),
        trace_verification='Saved daily returns/holdings/summary equal frozen independent reference; crash dates are not exported')
    version = dict(id=VERSION_ID, label=LABEL, daily=daily, daily_returns=returns.tolist(),
        trades=trades, crash_buys=[], metrics=_web_metrics(row['full'], row['switches']),
        full_metrics=deepcopy(row['full']), period_metrics=deepcopy(row), metadata=metadata,
        benchmarks=_benchmarks(dates, items['frozen_inputs.json']),
        metrics_convention=dict(initial_nav=1., returns_aligned_with_daily=True,
            first_return_included=True, annualization_sessions=244, drawdown_initial_peak=1.,
            interval_baseline='Previous observed NAV; use 1 only at first research session'))
    return {VERSION_ID: version}


if __name__ == '__main__':
    print(json.dumps(export_versions(), ensure_ascii=False, separators=(',', ':'), allow_nan=False))
