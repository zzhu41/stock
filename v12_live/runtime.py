"""One atomic V12-R2 observation, optionally staged by the shared daily worker."""
import json
import math
from pathlib import Path
import sys

from .constants import ASSETS,STRATEGY_ID,FIRST_LIVE_DATE
from .ledger import advance,validate_state
from v10_live.runtime import locked,atomic_json,runtime_clock,validate_snapshot

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"signals/shadow_v12_r2.json"


def validate_view(view,quotes,signal_date):
    if (not view.get("calendar") or view["calendar"][-1]!=signal_date
            or view.get("metadata",{}).get("provisional_date")!=signal_date):
        raise ValueError("V12-R2 shared view date mismatch")
    for code in ASSETS:
        raw=view.get("raw_histories",{}).get(code,[])
        tr=view.get("histories",{}).get(code,[])
        if not raw or not tr or raw[-1][0]!=signal_date or tr[-1][0]!=signal_date:
            raise ValueError("V12-R2 missing current validated history: "+code)
        for index,key in ((1,"open"),(2,"price"),(3,"volume")):
            if not math.isclose(float(raw[-1][index]),float(quotes[code][key]),rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("V12-R2 shared view differs from quote "+key+": "+code)
        if view["metadata"].get("quote_timestamps",{}).get(code)!=quotes[code]["timestamp"]:
            raise ValueError("V12-R2 quote/view timestamps differ: "+code)
        action=view.get("actions",{}).get(code,{}).get(signal_date)
        if not isinstance(action,dict) or action.get("not_observed"):
            raise ValueError("V12-R2 current corporate action is unverified: "+code)


def render(state,view):
    from market_data import UNIVERSE
    decision=state["last_decision"];code=state["holding"]
    old=decision.get("previous_holding",decision.get("diagnostics",{}).get("previous_holding"))
    action=("继续持有" if old==code else "买入")+" "+code+" "+UNIVERSE[code][0]
    lines=["-"*56,"【影子 V12-R2】用户指定主推送 · 虚拟跟踪不下单",
        "  信号状态: 盘中独立观察（非自动下单）",
        "  数据截止: %s | 行情时间: %s"%(state["last_date"],state["quote_timestamp"]),
        "  影子持仓: %s | 虚拟净值 %.4f (自%s) | 建议: %s | %s"%(
            code,state["nav"],state["start_date"],action,decision["reason"]),
        "  策略规则: WLS25四报价均值 / MA180 / 前报价波动较大值×1.4 / 健康轮动最短2日",
        "  抄底规则: 深跌与量能通道；不使用QVIX；原5日锁仓仍优先于急跌退出",
        "  跟踪说明: 独立账户从首次有效窗口起算；原始份额记账，分红按虚拟即时再投资"]
    if old and old!=code:
        lines.insert(5,"  换仓动作: 卖出 %s %s，买入 %s %s"%(old,UNIVERSE[old][0],code,UNIVERSE[code][0]))
    return lines


def run(quotes,signal_date,state_path=None,now=None,build_view=None,decide=None,observe=False):
    """Only fresh current-window observations; never backfill the launch day.

    Explicit now is a test hook. Live callers leave it absent; monotonic elapsed
    time is rechecked before settlement and immediately before atomic replace.
    observe=True skips only the execution-window assertion (see validate_snapshot)
    for read-only estimates staged outside production accounts.
    """
    clock=runtime_clock(now)
    if not observe and signal_date!=clock().strftime("%Y-%m-%d"):
        raise ValueError("V12-R2 refuses historical forward-account updates")
    if signal_date<FIRST_LIVE_DATE:
        raise ValueError("V12-R2从%s起等待新的有效窗口，不补记上线当天交易"%FIRST_LIVE_DATE)
    path=Path(STATE if state_path is None else state_path)
    with locked(path):
        state=json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        validate_state(state)
        if state and signal_date==state["last_date"]:
            lines=state.get("saved_lines")
            if not isinstance(lines,list) or not lines or not all(isinstance(x,str) for x in lines):
                raise ValueError("V12-R2 sealed daily card missing; duplicate fill refused")
            return lines
        if state and signal_date<state["last_date"]:raise ValueError("V12-R2 backdated update refused")
        validate_snapshot(quotes,signal_date,clock(),observe=observe)
        if build_view is None:
            from v10_live.data import build_live_view
            build_view=build_live_view
        if decide is None:
            from .policy import decide as policy_decide
            decide=policy_decide
        current=build_view(quotes,signal_date,now=clock())
        validate_view(current,quotes,signal_date)
        decision=decide(current["histories"],current["calendar"],signal_date,state=state)
        validate_snapshot(quotes,signal_date,clock(),observe=observe)
        updated=advance(state,decision,current,quotes,signal_date)
        updated["data_metadata"]=current["metadata"]
        updated["saved_lines"]=render(updated,current)
        validate_snapshot(quotes,signal_date,clock(),observe=observe)
        atomic_json(path,updated,validator=lambda:validate_snapshot(quotes,signal_date,clock(),observe=observe))
        return updated["saved_lines"]


def main():
    try:
        args=json.load(sys.stdin)
        result=dict(ok=True,lines=run(args["quotes"],args["signal_date"]))
    except Exception as exc:result=dict(ok=False,error="%s: %s"%(type(exc).__name__,exc))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":main()
