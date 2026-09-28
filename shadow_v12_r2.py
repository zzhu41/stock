# -*- coding: utf-8 -*-
"""V12-R2 worker bridge; reads never initialize its independent account."""
import json
import os
import subprocess
import sys

BASE=os.path.dirname(os.path.abspath(__file__))
STATE_FILE=os.path.join(BASE,"signals","shadow_v12_r2.json")


def saved_block(signal_date,state_path=None):
    try:
        from v12_live.ledger import validate_state
        with open(state_path or STATE_FILE,encoding="utf-8") as file:state=json.load(file)
        validate_state(state)
        lines=state.get("saved_lines")
        if (state["last_date"]==signal_date and isinstance(lines,list) and lines
                and all(isinstance(line,str) for line in lines)):
            return lines
    except Exception:pass
    return None


def run(quotes,signal_date,state_path=None,now=None,build_view=None,decide=None):
    from v12_live.runtime import run as worker
    return worker(quotes,signal_date,state_path=state_path or STATE_FILE,now=now,build_view=build_view,decide=decide)


def block(quotes,signal_date):
    try:
        completed=subprocess.run([sys.executable,"-B","-m","v12_live.runtime"],cwd=BASE,
            input=json.dumps(dict(quotes=quotes,signal_date=signal_date),allow_nan=False),
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60,check=False)
        if completed.returncode:raise ValueError("独立计算进程退出(%s)"%completed.returncode)
        payload=json.loads(completed.stdout)
        if not payload.get("ok"):raise ValueError(payload.get("error","独立计算未完成"))
        lines=payload["lines"]
        if not isinstance(lines,list) or not lines or not all(isinstance(x,str) for x in lines):
            raise ValueError("独立计算返回格式错误")
        return lines
    except subprocess.TimeoutExpired:detail="行情核验超过60秒，本次无新建议"
    except Exception as exc:detail=" ".join(str(exc).splitlines())[:350]
    committed=saved_block(signal_date)
    if committed is not None:return committed
    return ["-"*56,"【影子 V12-R2】用户指定主推送 · 虚拟跟踪不下单",
        "  信号状态: 本次无有效建议，计算失败",
        "  数据截止: %s（请求日期，行情未通过核验）"%signal_date,
        "  行情时间: 未核验","  数据说明: "+detail]
