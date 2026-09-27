# -*- coding: utf-8 -*-
"""V10-H daily shadow bridge. Formatting/query paths never call this worker."""
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE, "signals", "shadow_v10.json")


def saved_block(signal_date):
    """Recover a committed transaction if the worker dies before replying."""
    try:
        from v10_live.ledger import validate_state
        with open(STATE_FILE, encoding="utf-8") as file:
            state = json.load(file)
        validate_state(state)
        lines = state.get("saved_lines")
        if (state["last_date"] == signal_date and isinstance(lines, list) and lines
                and all(isinstance(line, str) for line in lines)):
            return lines
    except Exception:
        pass
    return None


def block(quotes, signal_date):
    """Bound research/data work so a V10 failure cannot stop the old daily card."""
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-m", "v10_live.runtime"], cwd=BASE,
            input=json.dumps({"quotes": quotes, "signal_date": signal_date}, allow_nan=False),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
            timeout=60, check=False)
        if completed.returncode:
            raise ValueError("独立计算进程退出(%s)" % completed.returncode)
        payload = json.loads(completed.stdout)
        if not payload.get("ok"):
            raise ValueError(payload.get("error", "独立计算未完成"))
        lines = payload["lines"]
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            raise ValueError("独立计算结果格式无效")
        return lines
    except subprocess.TimeoutExpired:
        detail = "行情核验超过60秒，本次未生成新建议；请勿沿用旧目标"
    except Exception as exc:
        detail = " ".join(str(exc).splitlines())[:350]
    committed = saved_block(signal_date)
    if committed is not None:
        return committed
    return ["-" * 56, "【影子 V10-H】研究观察 · 虚拟跟踪不下单",
            "  信号状态: 计算失败，本次无有效建议",
            "  数据截止: %s（请求日期，行情未通过核验）" % signal_date,
            "  数据说明: " + detail]
