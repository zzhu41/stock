"""Read-only close-price diagnostics, not a replay of 14:50 execution.

Run from the repository root with PYTHONDONTWRITEBYTECODE=1 python ...
Only this research directory receives output. No network or live writes.
"""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from v10_live.data import _load_seed, _read_state, CACHE_DIR
from v10_live.policy import Snapshot
from v12_live.policy import latest_features, decide
from v12_live.profile import load_profile
from v12.reference import Policy, Portfolio


def main():
    seed = _load_seed()
    state = _read_state(CACHE_DIR / 'completed.json', seed)
    config = load_profile()['config']
    all_rows = {
        c: [(r[0], float(r[2]), float(r[3])) for r in seed['tr'][c] + state['assets'][c]['tr']]
        for c in state['assets']}
    dates = [r[0] for r in all_rows['510300'] if r[0] >= '2026-09-29']
    assert dates[-1] == '2026-11-13'
    features = []
    for date in dates:
        features.append({c: latest_features(rows[:bisect_right([r[0] for r in rows], date)], date, c)
                         for c, rows in all_rows.items()})

    def run(changes):
        cfg = dict(config, **changes)
        account = Portfolio()
        trace = []
        for day, (date, indicators) in enumerate(zip(dates, features)):
            data = Snapshot(indicators, day, False)
            model = Policy(data, cfg)
            if day:
                account.nav *= data.price(day, account.holding) / account.mark
                account.mark = data.price(day, account.holding)
                account.age += 1
            previous = account.holding
            target, panic, crash = model.intent(day, account, True)
            changed = target != previous
            locked_exit = model.locked_exit_requested
            if changed:
                if day:
                    account.nav *= 1 - .0002
                    account.switches += 1
                account.holding, account.mark = target, data.price(day, target)
                account.age, account.asset_peak = 0, account.mark
                if crash:
                    account.lock_until = day + cfg['crash_lock']
                    account.crashes += 1
                model.after_fill(account)
            account.nav_peak = max(account.nav_peak, account.nav)
            account.max_dd = min(account.max_dd, account.nav / account.nav_peak - 1)
            trace.append(dict(date=date, holding=target, previous=previous, changed=changed,
                              nav=account.nav, panic=panic, crash=crash, locked_exit=locked_exit,
                              bull=model.regime.current, lock_until=account.lock_until,
                              target_mom20=data.value(day, target, 'mom20')))
        return dict(changes=changes, total_return=account.nav-1, max_drawdown=account.max_dd,
                    switches=account.switches, crash_entries=account.crashes, trace=trace)

    variants = {
        'baseline_close': {},
        'no_crash': {'crash_mask': 0},
        'locked_panic_exit': {'locked_panic_exit': 1},
        'fallback_mom20_nonnegative': {'fallback_floor': 0.0},
        'no_crash_and_fallback_mom20_nonnegative': {'crash_mask': 0, 'fallback_floor': 0.0},
    }
    results = {name: run(changes) for name, changes in variants.items()}
    # Independent call through the unmodified production adapter confirms the
    # diagnostic executor's baseline targets and panic/crash flags.
    histories = {c: [(d, p, p, p, p, v) for d, p, v in rows] for c, rows in all_rows.items()}
    calendar = [r[0] for r in all_rows['510300']]
    replay_state = None
    for t in results['baseline_close']['trace']:
        decision = decide(histories, calendar, t['date'], replay_state)
        assert decision['target'] == t['holding'], (t, decision)
        assert decision['panic'] == t['panic'] and decision['crash'] == t['crash']
        replay_state = dict(holding=decision['target'], last_date=t['date'],
                            entry_date=t['date'] if t['changed'] else replay_state['entry_date'],
                            crash_trigger_date=decision['crash_trigger_date'],
                            crash_code=decision['crash_code'])
    results['baseline_close']['verified_against_unmodified_live_decide'] = True
    live = json.loads((ROOT / 'signals/shadow_v12_r2.json').read_text())
    events = {e['date']: e for e in live['events']}
    results['baseline_close']['holding_disagreements_with_live'] = [
        dict(date=t['date'], close=t['holding'], live=events[t['date']]['to'])
        for t in results['baseline_close']['trace'] if t['holding'] != events[t['date']]['to']]
    benchmark = {}
    for code in all_rows:
        prices = [f[code]['close'] for f in features]
        peak = prices[0]
        dd = 0
        for p in prices:
            peak = max(peak, p)
            dd = min(dd, p/peak-1)
        benchmark[code] = dict(total_return=prices[-1]/prices[0]-1, max_drawdown=dd)
    output = dict(
        start=dates[0], end=dates[-1], observations=len(dates),
        source_sha256=hashlib.sha256((CACHE_DIR/'completed.json').read_bytes()).hexdigest(),
        methodology='Completed total-return closes; same-close ideal fills; 2bp per switch; first entry free. Features use trailing own observations only. Same cold start date as live; live uses 14:50 prices and differs.',
        limitations=['Ex post diagnostic ablations, not new out-of-sample qualification.',
                     'No actual intraday fills, quote spread, slippage, lot size, or QDII premium gate.',
                     'Variants selected as structural hypotheses, no parameter search.',
                     'Only 29 observations / 28 earning intervals; no annualization.',
                     'Snapshot policy is reconstructed each day, matching live adaptation; approved profile stateful confirmations/cooldown are neutral.'],
        variants=results, benchmarks_close=benchmark,
        actual_live_same_end=dict(nav=events[dates[-1]]['nav'], total_return=events[dates[-1]]['nav']-1),
    )
    path = Path(__file__).with_name('close_ablation.json')
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'trace'} for k,v in results.items()}, ensure_ascii=False, indent=2))
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
