# -*- coding: utf-8 -*-
"""Generate main V12-R2 plus three comparison cards in one account commit.

Only the 14:50–14:55 window may produce a new dated bundle. Strategy work runs
against temporary account copies. The canonical daily journal is the commit
point for BOTH the versioned message and every successful virtual account.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

from market_data import UNIVERSE, fetch_history, fetch_realtime
import signal_store
from daily_diagnostics import DiagnosticError, GenerationInputError, child_failure, emit

BASE = os.path.dirname(os.path.abspath(__file__))
SIGNAL_DIR = os.path.join(BASE, 'signals')
ACCOUNT_FILES = signal_store.ACCOUNT_FILES


def fetch_histories():
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(zip(UNIVERSE, pool.map(fetch_history, UNIVERSE)))


def prepare_versions(quotes, signal_date, stage_dir, now=None):
    payload = dict(quotes=quotes, date=signal_date, state_dir=str(stage_dir))
    if now is not None:
        payload['now'] = now.isoformat()
    result = subprocess.run([sys.executable, '-B', '-m', 'daily_extras'], cwd=BASE,
        input=json.dumps(payload, allow_nan=False), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    if result.returncode:
        raise DiagnosticError(child_failure(result.stdout,result.stderr,'prepare_versions',result.returncode))
    try:
        value = json.loads(result.stdout)
    except (TypeError, ValueError):
        raise DiagnosticError(child_failure(result.stdout,result.stderr,'prepare_versions',result.returncode))
    if (not isinstance(value, dict) or not isinstance(value.get('lines'), list) or
            not all(isinstance(line, str) for line in value['lines']) or
            not isinstance(value.get('successful_accounts'), list) or
            len(set(value['successful_accounts'])) != len(value['successful_accounts']) or
            set(value['successful_accounts']) - set(ACCOUNT_FILES)):
        raise ValueError('多版本准备进程未返回完整结果；本轮不提交账户')
    return value


def staged_accounts(directory, names, date, now):
    from shadow_account import validate as validate_v92
    from v10_live.ledger import validate_state as validate_h
    from v92_plus_live.ledger import validate_state as validate_plus
    validators = {'shadow_v92.json': lambda state: validate_v92(state, 'v9.2'),
                  'shadow_v92_plus.json': validate_plus, 'shadow_v10.json': validate_h}
    states = {}
    for name in names:
        state = signal_store.read_json(Path(directory) / name)
        if not isinstance(state, dict) or state.get('last_date') != date:
            raise ValueError('策略没有准备当日账户: ' + name)
        if name == signal_store.PRIMARY_ACCOUNT:
            # Optional main-version code must not make older comparison-only
            # batches fail merely because its import/preparation failed.
            from v12_live.ledger import validate_state as validate_r2
            validate_r2(state)
        else:
            validators[name](state)
        if not state.get('saved_lines') or not all(isinstance(s, str) for s in state['saved_lines']):
            raise ValueError('策略缺少封存卡片: ' + name)
        stamp = datetime.strptime(state['quote_timestamp'], '%Y-%m-%d %H:%M:%S')
        if stamp.strftime('%Y-%m-%d') != date or not -5 <= (now - stamp).total_seconds() <= 180:
            raise ValueError('准备账户使用的行情已过期: ' + name)
        states[name] = state
    if not states:
        raise ValueError('全部版本均无有效建议，保留原账户并等待重试')
    return states


def main(now=None):
    directory = Path(SIGNAL_DIR)
    with ExitStack() as stack:
        stack.enter_context(signal_store.file_lock(directory / 'generation.lock'))
        # Standalone shadow commands cannot advance an account between its
        # snapshot and the canonical multi-account commit.
        for name in sorted(ACCOUNT_FILES):
            stack.enter_context(signal_store.file_lock((directory / name).with_suffix('.lock')))
        return generate(now=now)


def generate(now=None):
    from v10_live.runtime import runtime_clock, validate_snapshot
    clock = runtime_clock(now)
    current = clock()
    if not signal_store.execution_window(current):
        raise ValueError('正式信号仅在14:50–14:55生成；其它时段请查看历史信号')
    date, directory = current.strftime('%Y-%m-%d'), Path(SIGNAL_DIR)
    journal = signal_store.load_journal(directory)
    if journal and journal['date'] > date:
        raise ValueError('已有信号日期晚于当前时钟，禁止倒退覆盖')
    if journal:
        signal_store.repair_projections(journal, directory, accounts_locked=True)
        if journal['date'] == date:
            print(journal['text'])
            return signal_store.primary_target(journal)
    quotes = fetch_realtime(codes=list(UNIVERSE), detailed=True)
    validate_snapshot(quotes, date, clock())
    with tempfile.TemporaryDirectory(prefix='.signal-stage-', dir=str(directory)) as name:
        stage = Path(name)
        for filename in ACCOUNT_FILES + ('shadow_v92_trades.csv',):
            source = directory / filename
            if journal and filename in journal.get('account_states', {}):
                signal_store.atomic_json(stage / filename, journal['account_states'][filename])
            elif source.is_file():
                shutil.copyfile(str(source), str(stage / filename))
        prepared = prepare_versions(quotes, date, stage, now=clock() if now is not None else None)
        finished = clock()
        validate_snapshot(quotes, date, finished)
        if not prepared['successful_accounts']:
            raise GenerationInputError(dict(phase='prepare_versions',error_type='PreparationError',
                message='全部版本均无有效建议，保留原账户并等待重试',
                causes=prepared.get('diagnostics', [])))
        states = staged_accounts(stage, prepared['successful_accounts'], date, finished)
        # Include any earlier same-day saved account mark in the bundle's age.
        stamp = min([q['timestamp'] for q in quotes.values()] + [s['quote_timestamp'] for s in states.values()])
        lines = ['=' * 56,
                 '动量轮动信号 | 生成 %s | 数据截止 %s' % (finished.strftime('%Y-%m-%d %H:%M:%S'), date),
                 '行情时间: ' + stamp,
                 '策略版本: V12-R2 | V9.2 | V9.2+ | V10-H',
                 '主推送: V12-R2（R2提高年化）；其余版本为对照',
                 '四个版本独立记录；虚拟跟踪不自动下单',
                 '=' * 56] + prepared['lines']
        text = '\n'.join(lines) + '\n'
        # Disk/serialization work must not turn an aged quote into a new signal.
        def validate_commit():
            moment = clock()
            validate_snapshot(quotes, date, moment)
            staged_accounts(stage, prepared['successful_accounts'], date, moment)
        validate_commit()
        target = states.get(signal_store.PRIMARY_ACCOUNT, {}).get('holding')
        # Retain every prior authoritative checkpoint even when that version
        # fails today and its physical projection has not yet been repaired.
        checkpoints = dict(journal.get('account_states', {})) if journal else {}
        checkpoints.update(states)
        record, errors = signal_store.publish(text, date, target, None, directory,
                                             account_states=checkpoints, updated_accounts=list(states),
                                             accounts_locked=True, validator=validate_commit,
                                             primary_version=signal_store.PRIMARY_VERSION)
        if errors:
            print('账户及消息已提交；以下投影待恢复: ' + ', '.join(errors))
    print(text)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warmup', action='store_true', help='Refresh only history caches; never create signals/accounts')
    args = parser.parse_args()
    try:
        fetch_histories() if args.warmup else main()
    except Exception as exc:
        emit(exc, 'warmup' if args.warmup else 'signal_generation', sys.stderr)
        raise SystemExit(1)
