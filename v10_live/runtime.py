"""Single daily transaction for the V10-H shadow account; no outbound messages."""
from contextlib import contextmanager
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from .ledger import advance, validate_state

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "signals/shadow_v10.json"


def runtime_clock(now=None):
    """Naive Asia/Shanghai clock plus elapsed work, matching signal_store clocks."""
    from .data import _clock
    started = time.monotonic()
    reference = _clock(now).replace(tzinfo=None)
    return lambda: reference + timedelta(seconds=max(0., time.monotonic() - started))


def validate_snapshot(quotes, signal_date, now):
    """A cached/injected data view cannot bypass the independent commit guard."""
    from .data import _clock, _quotes
    import signal_store
    moment = _clock(now)
    if not signal_store.execution_window(moment):
        raise ValueError("影子更新须在交易日14:50–14:55窗口内，停止本次记账")
    _quotes(quotes, signal_date, moment)
    return moment


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


def atomic_json(path, state, validator=None):
    encoded = json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(encoded)
            file.flush()
            os.fsync(file.fileno())
        # Encoding/fsync can also cross a freshness/window boundary. Leave the
        # old account intact if the final check fails before atomic replace.
        if validator is not None:
            validator()
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


def run(quotes, signal_date, state_path=None, now=None, build_view=None, decide=None):
    """Compute one dated account update at the explicitly supplied state path.

    The daily worker supplies an isolated staging path. Only the main daily
    journal transaction promotes that prepared state to production. Queries
    use saved text, and same-day calls return the existing card without trades.
    """
    clock = runtime_clock(now)
    started_at = clock()
    if signal_date != started_at.strftime("%Y-%m-%d"):
        raise ValueError("V10-H cannot start/update a forward account with historical quotes")
    state_path = Path(STATE if state_path is None else state_path)
    with locked(state_path):
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        validate_state(state)
        if state and signal_date == state["last_date"]:
            if not isinstance(state.get("saved_lines"), list):
                raise ValueError("V10-H saved daily card missing; refusing a duplicate fill")
            return state["saved_lines"]
        if state and signal_date < state["last_date"]:
            raise ValueError("V10-H refuses backdated forward updates")
        validated_at = validate_snapshot(quotes, signal_date, clock())
        if build_view is None:
            from .data import build_live_view
            build_view = build_live_view
        if decide is None:
            from .policy import decide as policy_decide
            decide = policy_decide
        view = build_view(quotes, signal_date, now=validated_at)
        decision = decide(view["histories"], view["calendar"], signal_date, state=state)
        validate_snapshot(quotes, signal_date, clock())
        updated = advance(state, decision, view, quotes, signal_date)
        updated["data_metadata"] = view["metadata"]
        updated["saved_lines"] = render(updated, view)
        # State, mark, action entitlement, virtual fill and card commit together.
        validator = lambda: validate_snapshot(quotes, signal_date, clock())
        validator()
        atomic_json(state_path, updated, validator=validator)
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
