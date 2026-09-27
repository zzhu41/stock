"""Fast reproduction of the original ideal same-close backtest clock.

This is deliberately the old research convention, not a tradable close-fill
claim and not the later units/cash same-close counterfactual. Each session:
mark yesterday's holding at today's close, ask the policy for today's target,
reject switches without both required closes, charge (1 - 2*fee), and establish
the new close mark. The first evaluation session is free, matching backtest.py.

The policy owns the original integer-index crash lock and must only start one
when can_sell is true and its dated candidate has a current price. Failed
ordinary switches are discarded; no deferred order is carried into tomorrow.
"""
import math

from v10_search.fast_execution import _bounds, prepare


TRADING_DAYS = 244


def run(frame, policy, start, end, fee=.0001, capture_daily="arrays", capture_trades=False):
    """policy(global_index, holding, holding_days, can_sell) -> code or None.

    None retains the current holding; an initially empty account remains cash.
    Cash ETF codes are ordinary full-allocation assets. No leverage, fractional
    weights, open-price fills or slippage are supported by this reproduction.

    arrays returns navs/returns/holdings. True also records daily triples and
    requested targets. trades, if captured, uses the original four-item tuple
    and omits the first free entry. switches/trade_count follow that convention;
    entry_count separately counts every actual position change.
    """
    fee = float(fee)
    if not math.isfinite(fee) or not 0 <= fee < .5:
        raise ValueError("Single-side fee must be finite and in [0, .5)")
    if capture_daily not in (False, True, "arrays"):
        raise ValueError("capture_daily must be False, True or 'arrays'")
    low, high = _bounds(frame, start, end)
    dates, prices = frame.dates, frame.closes
    full = capture_daily is True
    keep_arrays = bool(capture_daily)
    navs, returns, holdings = ([], [], []) if keep_arrays else (None, None, None)
    daily, signals, blocked = ([], [], []) if full else (None, None, None)
    trades = [] if capture_trades else None
    holding, mark_close, holding_days = None, None, 0
    nav, peak, max_dd, previous_nav = 1.0, 1.0, 0.0, 1.0
    switches = entry_count = blocked_count = missing_held_days = 0
    total_fees = 0.0
    n_returns, mean, m2 = 0, 0.0, 0.0
    year, year_base, yearly = frame.years[low], 1.0, {}
    fee_factor = 1 - 2 * fee

    for i in range(low, high + 1):
        current = prices[holding][i] if holding is not None else None
        can_sell = holding is None or current is not None
        if not can_sell:
            missing_held_days += 1
        if i > low:
            if holding is not None and current is not None:
                if mark_close is None or mark_close <= 0:
                    raise ArithmeticError("A held asset has no valid previous close mark")
                nav *= current / mark_close
                mark_close = current
            holding_days += 1

        requested = policy(i, holding, holding_days, can_sell)
        target = holding if requested is None else requested
        if target is not None and (not isinstance(target, str) or target not in frame.codes):
            raise ValueError("Policy target must be a known asset code or None")
        if target != holding and (not can_sell or prices[target][i] is None):
            blocked_count += 1
            if full:
                blocked.append(dict(date=dates[i], holding=holding, requested_target=target,
                                    can_sell=can_sell, can_buy=prices[target][i] is not None))
            target = holding
        if target != holding:
            entry_count += 1
            if i > low:
                switches += 1
                total_fees += nav * (2 * fee)
                nav *= fee_factor
                if capture_trades:
                    trades.append((dates[i], holding, target, nav))
            holding = target
            mark_close = prices[holding][i]
            holding_days = 0
        if not math.isfinite(nav) or nav <= 0:
            raise ArithmeticError("NAV is not finite and positive")
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1)
        ret = nav / previous_nav - 1
        if i > low:
            n_returns += 1
            delta = ret - mean
            mean += delta / n_returns
            m2 += delta * (ret - mean)
        if frame.years[i] != year:
            yearly[year] = previous_nav / year_base - 1
            year_base, year = previous_nav, frame.years[i]
        previous_nav = nav
        if keep_arrays:
            navs.append(nav)
            returns.append(ret)
            holdings.append(holding)
        if full:
            daily.append((dates[i], nav, holding))
            signals.append((dates[i], requested))

    yearly[year] = nav / year_base - 1
    sessions = high - low + 1
    years = sessions / TRADING_DAYS
    ann = nav ** (1 / years) - 1
    std = math.sqrt(max(0.0, m2 / n_returns)) if n_returns else 0.0
    return dict(nav=nav, ann=ann, max_dd=max_dd,
                sharpe=mean / std * math.sqrt(TRADING_DAYS) if std else 0.0,
                calmar=ann / abs(max_dd) if max_dd else 0.0,
                switches=switches, trade_count=switches, entry_count=entry_count,
                sw_per_year=switches / years, sessions=sessions, yearly=yearly,
                navs=navs, returns=returns, holdings=holdings, daily=daily, trades=trades,
                signal_target=signals,
                diagnostics=dict(blocked_switch_days=blocked_count, blocked_signals=blocked,
                                 missing_held_days=missing_held_days, total_fees=total_fees,
                                 first_session_entry_fee=0.0, switch_fee_factor=fee_factor,
                                 order_carryover=False, price_clock="original_same_close"),
                final_state=dict(holding=holding, holding_days=holding_days, mark_close=mark_close,
                                 nav=nav, date=dates[high]))
