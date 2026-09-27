"""Single daily transaction for the V10-H shadow account; no outbound messages."""
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile

from .ledger import advance, validate_state

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "signals/shadow_v10.json"


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        # A duplicate scheduler must not wait behind another network request.
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def atomic_json(path, state):
    encoded = json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def render(state, view):
    from market_data import UNIVERSE
    decision = state["last_decision"]
    code = state["holding"]
    stamp = str(state["quote_timestamp"])
    closed = len(stamp) == 19 and stamp[11:] >= "15:00:00"
    mode = "收盘后影子记录" if closed else "盘中影子试算"
    lines = ["-" * 56, "【影子 V10-H】研究观察 · 虚拟跟踪不下单",
             "  信号状态: %s（独立前向账户）" % mode,
             "  数据截止: %s | 行情时间: %s" % (state["last_date"], state["quote_timestamp"]),
             "  影子持仓: %s | 虚拟净值 %.4f (自%s) | 建议: %s %s | %s" % (
                 code, state["nav"], state["start_date"], code, UNIVERSE[code][0], decision["reason"]),
             "  策略规则: WLS20评分取3日均值 / MA180 / 波动急跌退出",
             "  数据说明: 独立总回报指标；原始价格记账；%s" % (
                 "使用收盘后有效报价" if closed else "盘中价格和量能尚未收盘")]
    qvix = decision.get("diagnostics", {}).get("qvix", {})
    # Preserve the policy's human-readable missing-current-date warning.
    if isinstance(qvix, dict):
        detail = qvix.get("note") or qvix.get("warning") or qvix.get("reason")
        if not detail:
            detail = "当日恐慌渠道%s" % ("激活" if qvix.get("fear", qvix.get("active", False)) else "未激活")
        lines.append("  QVIX " + str(detail))
    else:
        lines.append("  QVIX " + str(qvix))
    lines.append("  跟踪说明: 净值从首次有效信号起算；分红按虚拟即时再投资记账")
    return lines


def run(quotes, signal_date, state_path=STATE, now=None, build_view=None, decide=None):
    """Call only from signal generation. Queries use the saved text, never this."""
    now = now or datetime.now()
    if signal_date != now.strftime("%Y-%m-%d"):
        raise ValueError("V10-H cannot start/update a forward account with historical quotes")
    state_path = Path(state_path)
    with locked(state_path):
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        validate_state(state)
        if state and signal_date == state["last_date"]:
            if not isinstance(state.get("saved_lines"), list):
                raise ValueError("V10-H saved daily card missing; refusing a duplicate fill")
            return state["saved_lines"]
        if state and signal_date < state["last_date"]:
            raise ValueError("V10-H refuses backdated forward updates")
        if build_view is None:
            from .data import build_live_view
            build_view = build_live_view
        if decide is None:
            from .policy import decide as policy_decide
            decide = policy_decide
        view = build_view(quotes, signal_date, now=now)
        decision = decide(view["histories"], view["calendar"], signal_date, state=state)
        updated = advance(state, decision, view, quotes, signal_date)
        updated["data_metadata"] = view["metadata"]
        updated["saved_lines"] = render(updated, view)
        # State, mark, action entitlement, virtual fill and card commit together.
        atomic_json(state_path, updated)
        return updated["saved_lines"]


def main():
    try:
        args = json.load(sys.stdin)
        result = {"ok": True, "lines": run(args["quotes"], args["signal_date"])}
    except Exception as exc:
        result = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
