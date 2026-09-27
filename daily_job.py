"""Cron entry: bounded generation, durable status, then acknowledged delivery.

No message is sent on import. `run_job` is dependency-injectable for offline
verification. Production retries the SAME committed signal, not new trades.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys

import signal_store as store

ROOT = Path(__file__).resolve().parent


def run_child(command, timeout):
    child = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True,
                             start_new_session=True)
    try:
        output, error = child.communicate(timeout=timeout)
        # Do not leak URLs, credentials or raw exception payloads from workers.
        if child.returncode:
            raise RuntimeError("任务子进程失败，退出码%d" % child.returncode)
        return output
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # It may have exited between the timeout and the kill.
        child.communicate()
        raise TimeoutError("信号生成超过整体时限")


def run_job(directory=store.SIGNALS, now=None, generate=None, deliver=None):
    fixed_now = now
    clock = lambda: fixed_now or store.now_local()
    now = clock()
    date = now.strftime("%Y-%m-%d")
    directory = Path(directory)
    status_path = directory / "generation_status.json"
    if store.trading_day(date) is False:
        print("每日任务跳过: 交易所休市日")
        return 0
    if not store.execution_window(now):
        print("每日任务停止: 不在14:50–14:55窗口内")
        return 1
    try:
        with store.file_lock(directory / "daily_job.lock"):
            journal = store.load_journal(directory)
            if not journal or journal["date"] != date:
                status = dict(date=date, status="running", started_at=now.isoformat())
                store.atomic_json(status_path, status)
                deadline = datetime.strptime(date + " 14:54:30", "%Y-%m-%d %H:%M:%S")
                seconds = min(90., (deadline - now).total_seconds())
                if seconds <= 0:
                    status.update(status="failed", reason="超过新信号生成截止时间")
                    store.atomic_json(status_path, status)
                    return 1
                try:
                    if generate is None:
                        run_child([sys.executable, "-B", str(ROOT / "signal_daily.py")], seconds)
                    else:
                        generate()
                    journal = store.load_journal(directory)
                    if not journal or journal["date"] != date:
                        raise ValueError("生成进程未提交当日信号")
                except Exception as exc:
                    # A child can finish its atomic journal before losing stdout.
                    journal = store.load_journal(directory)
                    if not journal or journal["date"] != date:
                        status.update(status="failed", reason=type(exc).__name__, finished_at=clock().isoformat())
                        store.atomic_json(status_path, status)
                        print("每日生成失败: " + type(exc).__name__)
                        return 1
                status.update(status="ready", signal_id=journal["signal_id"], finished_at=clock().isoformat())
                store.atomic_json(status_path, status)
            else:
                store.repair_projections(journal, directory)
                store.atomic_json(status_path, dict(date=date, status="ready",
                    signal_id=journal["signal_id"], finished_at=clock().isoformat()))
            if deliver is None:
                # A socket read timeout does not bound DNS or a trickling body.
                # Production delivery gets its own process-group wall deadline.
                remaining = (clock().replace(hour=14, minute=55, second=0, microsecond=0) - clock()).total_seconds()
                if remaining <= 0:
                    return 1
                try:
                    output = run_child([sys.executable, "-B", str(ROOT / "push_signal.py"),
                                        "--directory", str(directory)], min(10., remaining))
                    print(output.strip())
                    return 0
                except Exception as exc:
                    with store.file_lock(directory / "delivery.lock"):
                        receipt_path = directory / "delivery.json"
                        receipt = store.read_json(receipt_path, {})
                        if receipt.get("signal_id") == journal["signal_id"]:
                            if receipt.get("status") == "sent":
                                return 0  # ACK was committed before stdout was lost.
                            if receipt.get("status") == "sending":
                                receipt.update(status="uncertain", updated_at=clock().isoformat())
                                for attempt in receipt.get("attempts", []):
                                    if attempt.get("status") == "sending":
                                        attempt.update(status="uncertain", error="SenderProcessInterrupted")
                                store.atomic_json(receipt_path, receipt)
                    print("推送进程未确认成功: " + type(exc).__name__)
                    return 1
            return deliver()
    except BlockingIOError:
        print("每日任务跳过: 上一次任务仍在执行")
        return 0
    except Exception as exc:
        print("每日任务失败: " + type(exc).__name__)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    if args.warmup:
        if store.trading_day(store.now_local().strftime("%Y-%m-%d")) is False:
            return 0
        try:
            with store.file_lock(store.SIGNALS / "warmup.lock"):
                run_child([sys.executable, "-B", str(ROOT / "signal_daily.py"), "--warmup"], 90)
            return 0
        except Exception as exc:
            print("行情预热未完成: " + type(exc).__name__)
            return 1
    return run_job()


if __name__ == "__main__":
    raise SystemExit(main())
