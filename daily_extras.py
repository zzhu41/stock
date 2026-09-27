"""Prepare exactly three strategy cards in an isolated staging directory."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys


def collect(payload):
    from v10_live.data import build_live_view
    from v10_live import runtime
    from v10_live import policy as h_policy
    from v92_plus_live import runtime as plus_runtime
    from v92_plus_live import policy as plus_policy
    import premium
    import shadow_0906  # QVIX data provider only; the 0906 account is not run.
    import shadow_v92
    import strategy
    quotes, date = payload['quotes'], payload['date']
    stage = Path(payload['state_dir']).resolve()
    if not stage.is_dir() or stage == Path(__file__).resolve().parent / 'signals':
        raise ValueError('Strategy worker requires an isolated staging directory')
    initial = datetime.fromisoformat(payload['now']) if payload.get('now') else None
    clock = runtime.runtime_clock(initial)
    pool = ThreadPoolExecutor(max_workers=3)
    futures = {
        'view': pool.submit(build_live_view, quotes, date, now=clock()),
        'qvix': pool.submit(shadow_0906.qvix_state, date),
        'premium': pool.submit(premium.signal_block, quotes=quotes),
    }
    output, successful = [], []
    try:
        output.extend(futures['premium'].result(timeout=18))
    except Exception:
        output.append('  QDII 溢价: 本次获取失败，请另行核验')
    try:
        futures['qvix'].result(timeout=22)
        qvix_ready = True
    except Exception:
        qvix_ready = False
    try:
        view = futures['view'].result(timeout=45)
        view_error = None
    except Exception as exc:
        view, view_error = None, '原始价格/分红数据核验失败(%s)' % type(exc).__name__
    finally:
        pool.shutdown(wait=False)
    # Freeze one local QVIX input for all three. A timed-out updater may still
    # finish in the background; it must not change later versions mid-bundle.
    qvix_path = stage / 'qvix50.csv'
    try:
        raw_qvix = Path(shadow_0906.QVIX_CSV).read_bytes() if qvix_ready else b''
    except OSError:
        raw_qvix = b''
    qvix_path.write_bytes(raw_qvix)
    qvix_info = h_policy.qvix_state(date, path=qvix_path)
    qvix = (qvix_info['z'], qvix_info['value'], qvix_info['date'],
            qvix_info['active'], qvix_info['note'])

    def checked_view(*args, **kwargs):
        if view is None:
            raise ValueError(view_error)
        return view

    def run_v92():
        checked_view()
        # All three use the same verified TR prices and normalized volumes.
        histories = {c: [(r[0], r[2], r[2], r[5]) for r in rows]
                     for c, rows in view['histories'].items()}
        table = strategy.rank(histories, on_date=date)
        previous = shadow_v92.STATE_FILE, shadow_v92.TRADES_FILE
        try:
            shadow_v92.STATE_FILE = str(stage / 'shadow_v92.json')
            shadow_v92.TRADES_FILE = str(stage / 'shadow_v92_trades.csv')
            return shadow_v92.block(table, histories, {c: q['price'] for c, q in quotes.items()},
                                   signal_date=date, quotes=quotes, action_view=view, qvix=qvix)
        finally:
            shadow_v92.STATE_FILE, shadow_v92.TRADES_FILE = previous

    def decide_h(*args, **kwargs):
        return h_policy.decide(*args, qvix_path=qvix_path, **kwargs)

    def decide_plus(*args, **kwargs):
        return plus_policy.decide(*args, qvix_path=qvix_path, **kwargs)

    jobs = (
        ('V9.2', 'shadow_v92.json', run_v92),
        ('V9.2+', 'shadow_v92_plus.json', lambda: plus_runtime.run(
            quotes, date, state_path=stage / 'shadow_v92_plus.json', now=clock(), build_view=checked_view, decide=decide_plus)),
        ('V10-H', 'shadow_v10.json', lambda: runtime.run(
            quotes, date, state_path=stage / 'shadow_v10.json', now=clock(), build_view=checked_view, decide=decide_h)),
    )
    for label, name, compute in jobs:
        try:
            runtime.validate_snapshot(quotes, date, clock())
            lines = compute()
            runtime.validate_snapshot(quotes, date, clock())
            output.extend(lines)
            successful.append(name)
        except Exception as exc:
            output.extend(['-' * 56, '【影子 %s】虚拟跟踪不下单' % label,
                           '  信号状态: 计算失败，本次无有效建议',
                           '  数据说明: ' + ' '.join(str(exc).splitlines())[:200]])
    return dict(lines=output, successful_accounts=successful)


if __name__ == '__main__':
    try:
        print(json.dumps(collect(json.load(sys.stdin)), ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__}))
        raise SystemExit(1)
