"""Cross-sectional research, not a forecast or trading system.

Historical windows end 2026-10-22; the separately verified quote is 2026-10-23.
No current-day or recent-period return is a positive screening gate or score term.
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
END = '2026-10-22'
BENCHMARKS = {'star50': 'sh000688', 'chinext': 'sz399006', 'csi300': 'sh000300'}
EVENTS = [('2024-09-23', '2024-10-08'), ('2025-04-08', '2025-05-09')]


def load_bars(symbol):
    p = BASE / 'history' / (symbol + '.csv')
    if not p.exists():
        return None
    frame = pd.read_csv(str(p), index_col='date')
    frame = frame.rename(columns={'volume_lots': 'volume_raw'})
    return frame.loc[frame.index <= END]


def return_series(close):
    return close / close.shift(1) - 1


def regression(stock, bench, n):
    b = bench.iloc[-n:].values.astype(float)
    s = stock.reindex(bench.index[-n:]).values.astype(float)
    valid = np.isfinite(b) & np.isfinite(s)
    s, b = s[valid], b[valid]
    answer = {'n': len(s)}
    if len(s) < max(40, int(n * .8)) or np.var(b, ddof=1) <= 0:
        return answer
    beta = np.cov(s, b, ddof=1)[0, 1] / np.var(b, ddof=1)
    alpha = s.mean() - beta * b.mean()
    answer.update(beta=float(beta), correlation=float(np.corrcoef(s, b)[0, 1]),
                  residual_vol=float(np.std(s - alpha - beta * b, ddof=1) * np.sqrt(244)))
    for label, flag in [('up_capture', b > 0), ('down_capture', b < 0)]:
        ss, bb = s[flag], b[flag]
        if len(ss) >= 15 and (ss > -1).all():
            denominator = np.expm1(np.log1p(bb).mean())
            answer[label] = float(np.expm1(np.log1p(ss).mean()) / denominator)
    return answer


def historical_metrics(symbol, benchmarks):
    bars = load_bars(symbol)
    if bars is None or len(bars) < 2:
        return {'symbol': symbol, 'history_error': 'missing_history'}
    # Align BEFORE calculating returns, so a suspension cannot turn a multi-day
    # stock return into a falsely matched one-day benchmark observation.
    close = bars.close.reindex(benchmarks['star50'].index)
    daily = return_series(close)
    out = {'symbol': symbol, 'history_rows': len(bars), 'history_start': bars.index[0],
           'history_end': bars.index[-1], 'history_close': close.iloc[-1],
           'missing_benchmark_days120': int(close.tail(121).isna().sum()),
           'median_volume_raw20': bars.volume_raw.tail(20).median()}
    for n in [5, 20, 60, 120, 250]:
        if len(close) > n:
            out['return_%d' % n] = close.iloc[-1] / close.iloc[-n - 1] - 1
            out['vol_%d' % n] = daily.tail(n).std() * np.sqrt(244)
        out['observations_%d' % n] = int(close.tail(n).notna().sum())
        if close.tail(n).notna().sum() >= min(n, 60):
            peak = close.tail(n).max()
            out['drawdown_%d' % n] = close.iloc[-1] / peak - 1
            out['recovery_to_peak_%d' % n] = peak / close.iloc[-1] - 1
    for label, benchmark in benchmarks.items():
        for n in ([60, 120, 250] if label == 'star50' else [120]):
            result = regression(daily, return_series(benchmark.close), n)
            out.update({'%s_%s_%d' % (label, k, n): v for k, v in result.items()})
    for start, end in EVENTS:
        key = 'event_' + start.replace('-', '') + '_' + end.replace('-', '')
        if start in close.index and end in close.index:
            out[key] = close.loc[end] / close.loc[start] - 1
    return out


def main():
    pool = pd.read_csv(str(BASE / 'technology_universe.csv'), dtype={'code': str})
    quote = pd.read_csv('/tmp/tech_screen_access/tencent_quotes.csv', dtype={'code': str})
    quote = quote[quote.asset_type == 'stock'].set_index('symbol')
    for column in ['price', 'change_pct', 'market_cap_yi', 'float_cap_yi', 'pb', 'trading_date']:
        pool[column] = pool.symbol.map(quote[column])
    pool['pe_tencent_reference'] = pool.symbol.map(quote.pe_tencent_f39)
    # Median traded shares multiplied by the current price is an explicitly labelled
    # liquidity proxy, not reconstructed exact historical RMB turnover.
    benchmarks = {k: load_bars(v) for k, v in BENCHMARKS.items()}
    assert all(v is not None and v.index[-1] == END for v in benchmarks.values())
    records = []
    for i, symbol in enumerate(pool.symbol):
        records.append(historical_metrics(symbol, benchmarks))
        if (i + 1) % 400 == 0:
            print('computed histories', i + 1, '/', len(pool), flush=True)
    metrics = pd.DataFrame(records)
    merged = pool.merge(metrics, on='symbol', how='left', validate='one_to_one')
    # Tencent STAR daily/quote volume is generally shares, ordinary A shares lots.
    # Infer the multiplier from independently reported RMB turnover and raw volume.
    quote_scale = quote.amount_yuan_exact_f35 / (quote.volume_hands * quote.price)
    inferred_scale = pd.Series(np.where(quote_scale > 10, 100.0, 1.0), index=quote.index)
    merged['volume_unit_multiplier'] = merged.symbol.map(inferred_scale)
    merged['quote_volume_scale_ratio'] = merged.symbol.map(quote_scale)
    merged['liquidity_proxy_yi'] = (merged.median_volume_raw20 * merged.volume_unit_multiplier *
                                    merged.price / 1e8)
    years = {}
    for date in ['2025-12-31', '2025-06-30', '2025-09-30']:
        src = '/tmp/tech_screen_financial/normalized_' + date + '.csv'
        shutil.copy2(src, str(BASE / 'inputs' / ('financial_' + date + '.csv')))
        years[date] = pd.read_csv(src, dtype={'code': str}).set_index('code')
    merged['prior_full_year_profit'] = merged.code.map(years['2025-12-31'].net_profit)
    merged['prior_full_year_revenue'] = merged.code.map(years['2025-12-31'].revenue)
    for metric in ['net_profit', 'revenue']:
        comparison = np.where(merged.report_date == '2026-09-30',
                              merged.code.map(years['2025-09-30'][metric]),
                              merged.code.map(years['2025-06-30'][metric]))
        merged[metric + '_ttm'] = merged[metric] + merged.code.map(years['2025-12-31'][metric]) - comparison
    merged['pe_ttm_calculated'] = merged.market_cap_yi * 1e8 / merged.net_profit_ttm.where(merged.net_profit_ttm > 0)
    merged['ps_ttm_calculated'] = merged.market_cap_yi * 1e8 / merged.revenue_ttm.where(merged.revenue_ttm > 0)
    merged['deduct_ratio_proxy'] = merged.deduct_eps / merged.eps.where(merged.eps > 0)
    exact = {}
    for date in ['2025-12-31', '2025-06-30', '2025-09-30', '2026-06-30', '2026-09-30']:
        src = '/tmp/tech_screen_financial/exact_' + date + '.csv'
        shutil.copy2(src, str(BASE / 'inputs' / ('exact_' + date + '.csv')))
        exact[date] = pd.read_csv(src, dtype={'code': str}).set_index('code')
    for metric in ['deduct_profit', 'operating_cashflow', 'debt_ratio', 'inventory_days',
                   'receivable_days', 'deduct_profit_growth']:
        merged[metric + '_actual'] = np.where(
            merged.report_date == '2026-09-30',
            merged.code.map(exact['2026-09-30'][metric]),
            merged.code.map(exact['2026-06-30'][metric]))
    for metric in ['deduct_profit', 'operating_cashflow']:
        comparison = np.where(merged.report_date == '2026-09-30',
                              merged.code.map(exact['2025-09-30'][metric]),
                              merged.code.map(exact['2025-06-30'][metric]))
        merged[metric + '_ttm'] = (merged[metric + '_actual'] +
                                  merged.code.map(exact['2025-12-31'][metric]) - comparison)
    merged['cash_profit_ratio_actual'] = merged.operating_cashflow_actual / merged.net_profit.where(merged.net_profit > 0)
    merged['deduct_ratio_actual'] = merged.deduct_profit_actual / merged.net_profit.where(merged.net_profit > 0)
    merged['cash_profit_ratio_ttm'] = merged.operating_cashflow_ttm / merged.net_profit_ttm.where(merged.net_profit_ttm > 0)
    merged['deduct_ratio_ttm'] = merged.deduct_profit_ttm / merged.net_profit_ttm.where(merged.net_profit_ttm > 0)
    merged['pe_deduct_ttm'] = merged.market_cap_yi * 1e8 / merged.deduct_profit_ttm.where(merged.deduct_profit_ttm > 0)
    merged['deduct_missing_requires_review'] = merged.deduct_profit_actual.isna()
    merged['data_eligible'] = ((merged.history_end == END) & (merged.star50_n_120 >= 100) &
                               merged.user_board_eligible & (merged.liquidity_proxy_yi >= .3))
    merged['profitable_floor'] = ((merged.net_profit >= 5e7) &
                                  ((merged.deduct_profit_actual > 0) | merged.deduct_profit_actual.isna()) &
                                  (merged.revenue_growth > -10))
    merged['deep_retracement'] = merged.drawdown_250 <= -.15
    merged['historical_elasticity'] = ((merged.star50_beta_120 >= 1.05) |
                                      (merged.star50_beta_60 >= 1.15))
    merged['growth_cash_filter'] = ((merged.net_profit >= 8e7) &
                                    (merged.revenue_growth >= 10) &
                                    (merged.profit_growth >= 30) &
                                    (merged.cash_profit_ratio_actual >= .5) &
                                    ((merged.deduct_ratio_actual >= .65) | merged.deduct_ratio_actual.isna()))
    merged['recent_runup_flag'] = merged.return_20 > .3
    # Transparent prioritisation only: no return forecasts, target fitting, or backtest selection.
    merged['ttm_net_margin'] = merged.net_profit_ttm / merged.revenue_ttm
    sector = merged.groupby('industry')
    margin_rank = sector.ttm_net_margin.rank(pct=True).fillna(.5)
    growth_revenue_rank = sector.revenue_growth.rank(pct=True).fillna(.5)
    growth_profit_rank = sector.profit_growth.rank(pct=True).fillna(.5)
    pe_rank = sector.pe_ttm_calculated.rank(pct=True).fillna(1)
    ps_rank = sector.ps_ttm_calculated.rank(pct=True).fillna(1)
    quality = (.35 * margin_rank + .35 * merged.cash_profit_ratio_ttm.clip(0, 1.5).fillna(.75) / 1.5 +
               .30 * merged.deduct_ratio_ttm.clip(0, 1).fillna(.5))
    growth = .55 * growth_revenue_rank + .45 * growth_profit_rank
    value = .70 * (1 - pe_rank) + .30 * (1 - ps_rank)
    elastic = merged.star50_beta_120.clip(.5, 2.5).fillna(.5).sub(.5) / 2
    merged['priority_score_heuristic'] = 100 * (.35 * quality + .30 * growth + .20 * value + .15 * elastic)
    merged['primary_quant_gate'] = (merged.data_eligible & merged.profitable_floor &
                                     merged.deep_retracement & merged.historical_elasticity)
    merged.to_csv(str(BASE / 'all_technology_metrics.csv'), index=False, encoding='utf-8-sig')
    shortlisted = merged[merged.primary_quant_gate].sort_values('priority_score_heuristic', ascending=False)
    shortlisted.to_csv(str(BASE / 'quantitative_candidates.csv'), index=False, encoding='utf-8-sig')
    # A separate list keeps unproven turnarounds and newly listed stocks visible.
    exceptions = merged[~merged.primary_quant_gate & merged.user_board_eligible].copy()
    exceptions.to_csv(str(BASE / 'outside_primary_gate.csv'), index=False, encoding='utf-8-sig')
    benchmark_events = {}
    for name, frame in benchmarks.items():
        benchmark_events[name] = {a + '_' + b: float(frame.close.loc[b] / frame.close.loc[a] - 1)
                                  for a, b in EVENTS if a in frame.index and b in frame.index}
    meta = {'history_end': END, 'quote_date': '2026-10-23', 'pool': len(merged),
            'history_success': int(merged.history_rows.notna().sum()),
            'same_cutoff_histories': int((merged.history_end == END).sum()),
            'data_eligible': int(merged.data_eligible.sum()),
            'profitable_floor': int((merged.data_eligible & merged.profitable_floor).sum()),
            'primary_quant_gate': len(shortlisted),
            'core_primary': int((shortlisted.scope == 'core').sum()),
            'adjacent_primary': int((shortlisted.scope != 'core').sum()),
            'growth_cash_primary': int(shortlisted.growth_cash_filter.sum()),
            'exact_deduct_missing': int(merged.deduct_profit_actual.isna().sum()),
            'exact_cash_missing': int(merged.operating_cashflow_actual.isna().sum()),
            'benchmark_events': benchmark_events,
            'definitions': {'beta': 'OLS covariance(stock,benchmark)/variance(benchmark); same benchmark dates, no missing-price filling',
                            'capture': 'ratio of geometric mean daily returns on benchmark up/down days; descriptive, not a forecast',
                            'ttm_profit': 'latest actual 2026 YTD + actual 2025 FY - corresponding actual 2025 YTD; forecasts excluded. Market cap is A-price times all shares, not combined A/H market value.',
                            'cash_proxy': 'Legacy per-share approximations are retained for audit only, not used in gates or scores. Exact cash and deducted profits fetched from RPT_F10_FINANCE_MAINFINADATA replace them.',
                            'score': '35% quality (TTM margin, exact TTM cash/profit, exact TTM deducted ratio) + 30% sector-relative growth + 20% sector-relative trailing valuation + 15% beta; prioritisation only. Missing exact deducted profit is unknown, not a loss; requires manual verification.',
                            'drawdown': 'Close-based, adjusted, at most N benchmark trading days. Use observations_N to distinguish shorter listed histories.',
                            'liquidity': 'Median raw daily volume over 20 days * inferred shares-per-volume-unit * current price; approximate RMB turnover, not exact historical turnover. STAR generally uses shares; other A shares generally lots.',
                            'latest_return': 'Not used as a gate or score; 20-day run-up is a risk flag only'}}
    (BASE / 'analysis_manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    cols = ['code','name','industry','priority_score_heuristic','market_cap_yi','pe_ttm_calculated',
            'drawdown_250','star50_beta_120','star50_up_capture_120','star50_down_capture_120',
            'revenue_growth','profit_growth','cash_profit_ratio_actual','deduct_ratio_actual']
    print(shortlisted[cols].head(65).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
