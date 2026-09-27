"""Continuous-path metrics and exact daily PnL attribution; no model selection."""
import math
from collections import defaultdict


YEAR = 244
CASH_ETF = "511880"


def summarize(result, start="0000", end="9999"):
    daily = result["daily"]
    chosen = [i for i, row in enumerate(daily) if start <= row[0] <= end]
    if not chosen:
        raise ValueError("Empty performance interval")
    first, last = chosen[0], chosen[-1]
    initial = daily[first - 1][1] if first else 1.0
    previous, peak, max_dd = initial, 1.0, 0.0
    returns, exposure, money, uninvested = [], [], [], []
    for i in chosen:
        _, nav, weights = daily[i]
        returns.append(nav / previous - 1)
        previous = nav
        value = nav / initial
        peak = max(peak, value)
        max_dd = min(max_dd, value / peak - 1)
        exposure.append(sum(w for c, w in weights.items() if c != CASH_ETF))
        money.append(weights.get(CASH_ETF, 0))
        uninvested.append(max(0.0, 1 - sum(weights.values())))
    n = len(chosen)
    factor = daily[last][1] / initial
    ann = factor ** (YEAR / n) - 1
    mean = sum(returns) / n
    sd = (sum((r - mean) ** 2 for r in returns) / n) ** .5
    trades = [t for t in result["trades"] if start <= t["date"] <= end]
    years = sorted({daily[i][0][:4] for i in chosen})
    yearly = {}
    for y in years:
        ids = [i for i in chosen if daily[i][0].startswith(y)]
        base = daily[ids[0] - 1][1] if ids[0] else 1.0
        yearly[y] = daily[ids[-1]][1] / base - 1
    cumulative = math.prod(1 + r for r in yearly.values())
    if not math.isclose(cumulative, factor, rel_tol=1e-11):
        raise AssertionError("Annual boundaries lost or duplicated a return")
    diag = result["diagnostics"]
    return dict(start=daily[first][0], end=daily[last][0], sessions=n,
                nav_factor=factor, total_return=factor - 1, cagr=ann,
                max_dd=max_dd, volatility=sd * YEAR ** .5,
                sharpe=mean / sd * YEAR ** .5 if sd else 0.0,
                calmar=ann / abs(max_dd) if max_dd < 0 else None,
                rebalance_days=len(trades), rebalances_per_year=len(trades) * YEAR / n,
                fill_legs=sum(len(t["fills"]) for t in trades),
                turnover=sum(t["turnover"] for t in trades),
                turnover_per_year=sum(t["turnover"] for t in trades) * YEAR / n,
                average_nonmoney_exposure=sum(exposure) / n,
                average_money_etf_weight=sum(money) / n,
                average_uninvested_cash_weight=sum(uninvested) / n,
                deferred_rebalances=sum(start <= r["date"] <= end for r in diag["deferred_rebalances"]),
                missing_held_sessions=sum(start <= r["date"] <= end for r in diag["missing_held_bars"]),
                yearly=yearly)


def attribution(result, histories):
    """Rebuild actual fills, split old-unit overnight/new-unit intraday PnL.

    Per-day arithmetic contributions are scaled by log(1+r)/r, giving an exact
    additive log-growth decomposition across assets and transaction costs.
    This is descriptive accounting, not causal removal of an asset or rule.
    """
    bars = {c: {r[0]: r for r in rows} for c, rows in histories.items()}
    trades = {t["date"]: t for t in result["trades"]}
    units, marks, cash, previous_nav = {}, {}, 1.0, 1.0
    log_by_asset = defaultdict(float)
    held_sessions = defaultdict(int)
    daily_returns = []
    for date, expected_nav, _ in result["daily"]:
        pnl = defaultdict(float)
        for code, quantity in units.items():
            row = bars[code].get(date)
            if row:
                pnl[code] += quantity * (row[1] - marks[code])
        trade = trades.get(date)
        if trade:
            for fill in trade["fills"]:
                code = fill["code"]
                units[code] = units.get(code, 0) + fill["units"] * (1 if fill["side"] == "buy" else -1)
                if abs(units[code]) < 1e-10:
                    units.pop(code)
                cash += fill["cash_flow"]
            pnl["transaction_costs"] -= trade["commission"] + trade["slippage_cost"]
        for code, quantity in units.items():
            row = bars[code].get(date)
            if row:
                pnl[code] += quantity * (row[2] - row[1])
                marks[code] = row[2]
            held_sessions[code] += 1
        nav = cash + sum(q * marks[c] for c, q in units.items())
        if not math.isclose(nav, expected_nav, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("Fill-based NAV attribution mismatch on " + date)
        r = expected_nav / previous_nav - 1
        if not math.isclose(sum(pnl.values()) / previous_nav, r, rel_tol=1e-8, abs_tol=1e-9):
            raise AssertionError("Daily PnL attribution mismatch on " + date)
        scale = math.log1p(r) / r if abs(r) > 1e-14 else 1.0
        for c, value in pnl.items():
            log_by_asset[c] += value / previous_nav * scale
        daily_returns.append((date, r))
        previous_nav = expected_nav
    if not math.isclose(sum(log_by_asset.values()), math.log(result["nav"]), abs_tol=1e-8):
        raise AssertionError("Log attribution failed to reconcile")
    positives = sorted(((d, math.log1p(r)) for d, r in daily_returns if r > 0),
                       key=lambda x: x[1], reverse=True)
    total_log = math.log(result["nav"])
    n = len(result["daily"])
    neutralize = []
    for k in (5, 10, 20):
        contribution = sum(v for _, v in positives[:k])
        neutralize.append(dict(days=k,
                               cagr=math.expm1((total_log - contribution) * YEAR / n),
                               share_net_log_growth=contribution / total_log if total_log else None))
    return dict(log_growth_by_asset=dict(log_by_asset), held_sessions=dict(held_sessions),
                neutralize_best_days=neutralize,
                best_days=[dict(date=d, return_value=math.expm1(v)) for d, v in positives[:20]],
                caveat="Exact descriptive log-growth accounting; not causal asset removal or a tradable neutralization.")
