"""Prepare the user-selected V12-R2 main card and three comparison accounts."""
from datetime import datetime
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time

from daily_diagnostics import clean_detail, exception_detail, safe_message

INPUT_LIMITS = {'view': 45., 'qvix': 22., 'premium': 18.}
INPUT_BUDGET_SECONDS = 45.


def _input_child(function, args, kwargs, path, name):
    """Publish a complete local result; never forward private exception text."""
    try:
        payload = dict(ok=True, value=function(*args, **kwargs))
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        encoded = json.dumps(dict(ok=False, error=type(exc).__name__,
                                  diagnostic=exception_detail(exc, 'input.' + name)),
                             ensure_ascii=False, allow_nan=False)
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _stop_input(process):
    """Reap a worker even when its network thread ignores ordinary shutdown."""
    if process.is_alive():
        process.terminate()
        process.join(.1)
    if process.is_alive():
        process.kill()
    process.join(.1)
    if process.is_alive():
        raise RuntimeError('Input worker did not stop after SIGKILL')
    process.close()


def _input_failure(exc, name):
    return dict(ok=False, error=type(exc).__name__,
                diagnostic=exception_detail(exc, 'input.' + name))


def _collect_inputs(jobs, stage, budget_seconds=INPUT_BUDGET_SECONDS, limits=None):
    """One monotonic deadline; optional fetchers cannot delay process exit.

    Linux fork is used by the existing Python3.8 cron worker before it starts
    any threads. Each child may use threads internally, but can be terminated
    as a whole. Results use atomic JSON files instead of a Pipe/Queue, whose
    partial frame or feeder thread could itself block shutdown.
    """
    limits = dict(INPUT_LIMITS if limits is None else limits)
    if (not math.isfinite(budget_seconds) or budget_seconds <= 0
            or set(jobs) != set(INPUT_LIMITS) or set(limits) != set(jobs)
            or any(not math.isfinite(v) or v <= 0 for v in limits.values())):
        raise ValueError('Invalid shared input deadlines')
    context = multiprocessing.get_context('fork')
    started = time.monotonic()
    deadlines = {name: started + min(budget_seconds, limits[name]) for name in jobs}
    processes, results = {}, {}
    with tempfile.TemporaryDirectory(prefix='.network-inputs-', dir=str(stage)) as directory:
        paths = {name: Path(directory) / (name + '.json') for name in jobs}
        try:
            for name, (function, args, kwargs) in jobs.items():
                process = context.Process(target=_input_child,
                    args=(function, args, kwargs, paths[name], name), daemon=True)
                process.start()
                processes[name] = process
            pending = set(jobs)
            while pending:
                for name in list(pending):
                    process = processes[name]
                    if time.monotonic() >= deadlines[name]:
                        results[name] = _input_failure(TimeoutError('共享输入获取超过预定时限'), name)
                    elif paths[name].is_file():
                        try:
                            value = json.loads(paths[name].read_text(encoding='utf-8'))
                            if not isinstance(value, dict) or type(value.get('ok')) is not bool:
                                raise ValueError('Invalid worker result')
                            results[name] = value
                        except Exception as exc:
                            results[name] = _input_failure(exc, name)
                    elif not process.is_alive():
                        results[name] = _input_failure(
                            RuntimeError('输入子进程退出但未留下完整结果，退出码%s' % process.exitcode), name)
                    else:
                        continue
                    _stop_input(process)
                    del processes[name]
                    pending.remove(name)
                if 'view' in results and not results['view']['ok']:
                    # Without the verified common price/action view, no
                    # strategy can update. Do not wait on optional sources.
                    for name in pending:
                        results[name] = _input_failure(
                            RuntimeError('必需价格视图失败，取消剩余可选数据获取'), name)
                    break
                if pending:
                    remaining = min(deadlines[name] for name in pending) - time.monotonic()
                    if remaining > 0:
                        time.sleep(min(.02, remaining))
        finally:
            for process in processes.values():
                _stop_input(process)
    return results


def _qvix_input(date):
    import shadow_0906
    shadow_0906.qvix_state(date)
    path = Path(shadow_0906.QVIX_CSV)
    return path.read_text(encoding='utf-8') if path.is_file() else ''


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
    inputs = _collect_inputs({
        'view': (build_live_view, (quotes, date), dict(now=clock())),
        'qvix': (_qvix_input, (date,), {}),
        'premium': (premium.signal_block, (), dict(quotes=quotes)),
    }, stage)
    output, successful, diagnostics = [], [], []
    for name in ('view', 'qvix', 'premium'):
        if not inputs[name]['ok']:
            detail = inputs[name].get('diagnostic')
            if not isinstance(detail, dict):
                detail = dict(phase='input.' + name, error_type=inputs[name].get('error', 'InputError'),
                              message='共享输入获取失败')
            diagnostics.append(clean_detail(detail, 'input.' + name))
    premiums = inputs['premium'].get('value')
    if inputs['premium']['ok'] and isinstance(premiums, list) and all(isinstance(v, str) for v in premiums):
        output.extend(premiums)
    else:
        output.append('  QDII 溢价: 本次获取失败，请另行核验')
    view = inputs['view'].get('value') if inputs['view']['ok'] else None
    if not isinstance(view, dict):
        view = None
        if inputs['view']['ok']:
            diagnostics.insert(0, exception_detail(ValueError('共享价格视图结构无效'), 'input.view'))
    view_detail = next((detail for detail in diagnostics if detail['phase'] == 'input.view'), None)
    view_error = '原始价格/分红数据核验失败: ' + (view_detail['message'] if view_detail else '共享价格视图不可用')
    # The QVIX child returns its exact completed CSV. A slow/failed child is
    # stopped and all three versions use the same empty (disabled) snapshot.
    qvix_path = stage / 'qvix50.csv'
    raw_qvix = inputs['qvix'].get('value') if inputs['qvix']['ok'] else ''
    qvix_path.write_text(raw_qvix if isinstance(raw_qvix, str) else '', encoding='utf-8')
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

    def run_r2():
        # Isolate import/frozen-configuration failures to this version too.
        from v12_live import runtime as r2_runtime
        return r2_runtime.run(quotes, date, state_path=stage / 'shadow_v12_r2.json',
                              now=clock(), build_view=checked_view)

    jobs = (
        ('V12-R2', 'shadow_v12_r2.json', run_r2),
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
            detail = exception_detail(exc, 'strategy.' + label)
            diagnostics.append(detail)
            output.extend(['-' * 56, '【影子 %s】虚拟跟踪不下单' % label,
                           '  信号状态: 计算失败，本次无有效建议',
                           '  数据说明: ' + safe_message(detail['message'], 200)])
    return dict(lines=output, successful_accounts=successful, diagnostics=diagnostics)


if __name__ == '__main__':
    try:
        print(json.dumps(collect(json.load(sys.stdin)), ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__, 'diagnostic': exception_detail(exc, 'daily_extras')},
                         ensure_ascii=False, allow_nan=False))
        raise SystemExit(1)
