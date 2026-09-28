"""Offline process-level deadline checks. No quotes, accounts or messages."""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import daily_extras


def immediate(value):
    return value


def slow():
    time.sleep(30)
    return "too late"


def ignore_terminate():
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    return slow()


def fail_private():
    raise ValueError("510300 配对日期缺失 https://example.invalid/?access_token=private-secret token=private-value")


def return_with_lingering_thread():
    pool=ThreadPoolExecutor(max_workers=1)
    pool.submit(time.sleep,30)
    pool.shutdown(wait=False)
    return ["completed task with lingering optional thread"]


class InputDeadlineTests(unittest.TestCase):
    def collect(self,jobs,budget=.15,limits=None):
        original={p.pid for p in multiprocessing.active_children()}
        with tempfile.TemporaryDirectory() as directory:
            started=time.monotonic()
            result=daily_extras._collect_inputs(jobs,Path(directory),budget_seconds=budget,
                                               limits=limits or dict(view=budget,qvix=budget,premium=budget))
            elapsed=time.monotonic()-started
            self.assertFalse(list(Path(directory).glob(".network-inputs-*")))
        self.assertEqual({p.pid for p in multiprocessing.active_children()},original)
        return result,elapsed

    def test_optional_timeout_keeps_verified_view_and_terminates_uncooperative_child(self):
        result,elapsed=self.collect(dict(view=(immediate,({"valid":True},),{}),
            qvix=(ignore_terminate,(),{}),premium=(immediate,(["premium available"],),{})))
        self.assertTrue(result["view"]["ok"])
        self.assertTrue(result["premium"]["ok"])
        self.assertFalse(result["qvix"]["ok"])
        self.assertEqual(result["qvix"]["error"],"TimeoutError")
        self.assertEqual(result["qvix"]["diagnostic"]["phase"],"input.qvix")
        self.assertLess(elapsed,1.5)

    def test_one_common_budget_cannot_be_multiplied_by_three_slow_tasks(self):
        result,elapsed=self.collect({name:(slow,(),{}) for name in ("view","qvix","premium")},budget=.1,
                                    limits=dict(view=30.,qvix=30.,premium=30.))
        self.assertFalse(result["view"]["ok"])
        self.assertTrue(all(not value["ok"] for value in result.values()))
        self.assertLess(elapsed,.8)

    def test_required_view_failure_cancels_optional_work_and_keeps_error_private(self):
        result,elapsed=self.collect(dict(view=(fail_private,(),{}),qvix=(slow,(),{}),
                                        premium=(slow,(),{})),budget=5.)
        self.assertFalse(result["view"]["ok"])
        self.assertEqual(result["view"]["error"],"ValueError")
        detail=result["view"]["diagnostic"]
        self.assertEqual(detail["phase"],"input.view")
        self.assertIn("510300 配对日期缺失",detail["message"])
        self.assertTrue(detail["frames"])
        self.assertNotIn("private-secret",str(result))
        self.assertNotIn("private-value",str(result))
        self.assertTrue(all(not value["ok"] for value in result.values()))
        self.assertLess(elapsed,1.)

    def test_completed_task_with_lingering_thread_is_reaped_after_atomic_result(self):
        result,elapsed=self.collect(dict(view=(immediate,({"valid":True},),{}),
            qvix=(immediate,("",),{}),premium=(return_with_lingering_thread,(),{})),budget=1.)
        self.assertTrue(all(value["ok"] for value in result.values()))
        self.assertEqual(result["premium"]["value"],["completed task with lingering optional thread"])
        self.assertLess(elapsed,.8)

    def test_outer_worker_exits_after_optional_timeout_instead_of_joining_for_30_seconds(self):
        code="""
import tempfile,time
from pathlib import Path
import daily_extras
def fast():return {}
def slow():time.sleep(30)
with tempfile.TemporaryDirectory() as directory:
    result=daily_extras._collect_inputs(dict(view=(fast,(),{}),qvix=(slow,(),{}),premium=(slow,(),{})),
        Path(directory),budget_seconds=.1,limits=dict(view=.1,qvix=.1,premium=.1))
    assert result['view']['ok']
    print('COMPLETE')
"""
        started=time.monotonic()
        done=subprocess.run([sys.executable,"-B","-c",code],cwd=str(Path(daily_extras.__file__).parent),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=3)
        self.assertEqual(done.returncode,0,done.stderr)
        self.assertIn("COMPLETE",done.stdout)
        self.assertLess(time.monotonic()-started,2.)


if __name__=="__main__":unittest.main()
