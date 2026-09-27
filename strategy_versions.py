# -*- coding: utf-8 -*-
"""Explicit research configurations; importing this module performs no I/O.

These configurations fix parameters, not historical code or market data. Store
the code revision and data hashes with results when reproducing an old run.
v9.2 additionally needs v10.lab's fz25_cv hooks and is not a backtest-only preset.
"""
import argparse
import csv
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


# Every behavior argument is explicit so future engine defaults cannot silently
# change an archived version. Data, the evaluation interval and costs are logged
# separately by the research runner. Changing this list requires review.
_COMMON = dict(
    buffer=0.02, overheat=0.40, bull_vote=False,
    circuit_dd=0.0, circuit_days=10, skip_days=0, use_high52=False,
    min_hold=0, overheat_lookback=1, panic_drop=0.04,
    global_ma_filter=False, global_mom60_filter=False,
    pos_frac_min=0.0, max_cap=9.9, use_downside_vol=False,
    vol_target=0.0, vol_target_lock=False, mom_main=20, ma_bull=250,
    dual_mom=False, score_mom=0, ma_slope=False, rank_exit_n=0,
    score_plain_mom=False, leverage=1.0, borrow_rate=0.06, never_empty=True,
    exit_cooldown=0, abs_stop=0.0, macd_filter=False, macd_exit=False,
    kdj_nochase=False, kdj_dip_buy=False, rsi_nochase=False, ma_align=False,
    macd_dif_pos=False, kdj_golden=False, kdj_dead_exit=False,
    rsi_gt50=False, rsi_lt50_exit=False, above_ma20=False, ma20_rising=False,
    bias_nochase=9.9, pctb_nochase=9.9, cci_nochase=9999.0,
    kdj_nc_thr=100.0, rsi_nc_thr=80.0, score_tech_mix=False,
    ne_min_mom=-9.9, ne_bull_only=False, exit_mom_floor=0.0,
    mom_decay=9.9, donchian_n=0, er_buffer_on=False, kaufman_thr=0.30,
    rel_buffer_on=False, vol_panic_on=False, signal_histories=None,
    w_mom=False, vol_ratio_max=0.0, enter_mom_min=0.0,
    score_square=False, score_wls=True, trail_stop=0.0, trail_cool=0,
    vol_in_min=0.0, vol_in_max=99.0, vol_score_mix=False,
    score_smooth=1, enter_vol_max=9.9, bear_enter_mom=0.07,
    safe_min_vol=False, vol_buffer=0.0, high_vol_half=0.0, dd_guard=0.0,
    score_r2=False, wls60_mix=0.0, vol_ewm=False, bull_dual=False,
    vol_days=20, bear_open_stock=False, crash_mom5=-0.08, crash_lock=5,
    crash_stock_only=False, crash_below_ma=0.20, crash_stop=0.0,
    crash_pick="score", crash_alloc=1.0, score_tstat=False,
    lead_exit=False, zscore_crash=False, zscore_accel=False,
    bull_hyst=0.0, bull_confirm=1, wls_adaptive=False, resid_penalty=False,
    crash_dyn_unlock=False, crash_tp=0.0, trend_buf_on=False,
    slope_days_buf=0, premium_guard=0.0, premium_data=None,
)
V9 = MappingProxyType(dict(_COMMON, pool_buffer=MappingProxyType({})))
V91 = MappingProxyType(dict(_COMMON, pool_buffer=MappingProxyType(
    {"stock": 0.02, "global": 0.03, "gold": 0.03})))
VERSIONS = MappingProxyType({"v9": V9, "v9.1": V91})

STOCK_CODES = ("159915", "588080", "510300", "510500", "563300", "512400", "512890")
GLOBAL_CODES = ("513100", "513120")
BASE_CODES = STOCK_CODES + GLOBAL_CODES + ("518880", "511880")


def _copy_value(value):
    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    return value


def backtest_kwargs(version, **overrides):
    """Return an independent mutable copy; reject misspelled experiment knobs."""
    params = _copy_value(VERSIONS[version])
    unknown = set(overrides) - set(params)
    if unknown:
        raise ValueError("Unknown strategy parameters: %s" % ", ".join(sorted(unknown)))
    params.update({key: _copy_value(value) for key, value in overrides.items()})
    return params


def load_local_histories(data_dir=None):
    """Read the eleven local CSV files, without refreshes or network access."""
    root = Path(data_dir) if data_dir is not None else Path(__file__).parent / "data"
    histories = {}
    for code in BASE_CODES:
        with (root / (code + ".csv")).open(newline="", encoding="utf-8") as f:
            histories[code] = [
                (r[0], float(r[1]), float(r[2]), float(r[3]))
                for r in csv.reader(f) if r
            ]
    return histories


def main():
    parser = argparse.ArgumentParser(description="Run an explicit version using local CSV data only")
    parser.add_argument("version", choices=tuple(VERSIONS))
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="9999")
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    import backtest
    import strategy
    # The CLI owns its process; freeze the baseline pool as well as parameters.
    strategy.STOCK_POOL = list(STOCK_CODES)
    strategy.GLOBAL_POOL = list(GLOBAL_CODES)
    histories = load_local_histories(args.data_dir)
    calendar = [r[0] for r in histories["510300"]]
    result = backtest.backtest(histories, calendar, start=args.start, end=args.end,
                               **backtest_kwargs(args.version))
    backtest.report(result, "%s | %s — %s | local CSV" % (
        args.version, result["daily"][0][0], result["daily"][-1][0]))
    print("Costs: single side %.4f%%; engine annualization: %d trading days" %
          (backtest.FEE * 100, backtest.TRADING_DAYS))
    print("This is an in-sample replay with the current engine, not out-of-sample evidence.")


if __name__ == "__main__":
    main()
