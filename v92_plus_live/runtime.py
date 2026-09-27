"""Independent V9.2+ daily shadow transaction. No outbound messages/orders."""
import json
import math
from pathlib import Path
import sys

from .ledger import advance, validate_state
from v10_live.runtime import locked, atomic_json

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "signals/shadow_v92_plus.json"


def render(state, view):
    from market_data import UNIVERSE
    decision, code = state["last_decision"], state["holding"]
    stamp = str(state["quote_timestamp"])
    mode = "收盘后影子记录" if len(stamp) == 19 and stamp[11:] >= "15:00:00" else "盘中影子试算"
    lines = ["-" * 56, "【影子 V9.2+】研究观察 · 虚拟跟踪不下单",
             "  信号状态: %s（独立前向账户）" % mode,
             "  数据截止: %s | 行情时间: %s" % (state["last_date"], stamp),
             "  影子持仓: %s | 虚拟净值 %.4f (自%s) | 建议: %s %s | %s" % (
                 code, state["nav"], state["start_date"], code, UNIVERSE[code][0], decision["reason"]),
             "  策略规则: V9.2 + 健康轮动最短持有2日；风险退出例外；原抄底锁仓保留",
             "  数据说明: WLS25 / MA250 / 固定4%急跌；总回报指标、原始份额记账"]
    qvix = decision.get("diagnostics", {}).get("qvix", {})
    detail = qvix.get("note") or ("当日恐慌渠道已激活" if qvix.get("active") else "当日恐慌渠道未激活")
    lines += ["  QVIX " + detail,
              "  跟踪说明: 净值从首次有效信号起算；分红按虚拟即时再投资记账"]
    return lines


def validate_view(view, quotes, signal_date):
    """An injected shared view must describe exactly these current quotes."""
    from .policy import ASSETS
    if (not view.get("calendar") or view["calendar"][-1] != signal_date
            or view.get("metadata", {}).get("provisional_date") != signal_date):
        raise ValueError("V9.2+ shared view date mismatch")
    for code in ASSETS:
        raw = view.get("raw_histories", {}).get(code, [])
        tr = view.get("histories", {}).get(code, [])
        if not raw or raw[-1][0] != signal_date or not tr or tr[-1][0] != signal_date:
            raise ValueError("V9.2+ missing same-day validated history: " + code)
        if (not math.isclose(float(raw[-1][2]), float(quotes[code]["price"]), rel_tol=1e-12)
                or view["metadata"].get("quote_timestamps", {}).get(code) != quotes[code]["timestamp"]):
            raise ValueError("V9.2+ shared view differs from supplied quotes: " + code)
        action = view.get("actions", {}).get(code, {}).get(signal_date)
        if not isinstance(action, dict) or action.get("not_observed"):
            raise ValueError("V9.2+ missing verified current corporate action: " + code)


def run(quotes, signal_date, state_path=STATE, now=None, build_view=None, decide=None):
    """New dates require fresh validated inputs; sealed dates are read-only.

    state_path may point into the caller's batch staging directory. This module
    commits only that one independent account; the parent owns batch publishing.
    """
    from v10_live.runtime import runtime_clock, validate_snapshot
    clock = runtime_clock(now)
    if signal_date != clock().strftime("%Y-%m-%d"):
        raise ValueError("V9.2+ refuses historical forward-account updates")
    state_path = Path(state_path)
    with locked(state_path):
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
        validate_state(state)
        if state and signal_date == state["last_date"]:
            lines = state.get("saved_lines")
            if not isinstance(lines, list) or not lines or not all(isinstance(x, str) for x in lines):
                raise ValueError("V9.2+ sealed daily card missing; refusing duplicate fill")
            return lines
        if state and signal_date < state["last_date"]:
            raise ValueError("V9.2+ refuses backdated forward updates")
        validate_snapshot(quotes, signal_date, clock())
        if build_view is None:
            from v10_live.data import build_live_view
            build_view = build_live_view
        if decide is None:
            from .policy import decide as policy_decide
            decide = policy_decide
        view = build_view(quotes, signal_date, now=clock())
        validate_view(view, quotes, signal_date)
        decision = decide(view["histories"], view["calendar"], signal_date, state=state)
        validate_snapshot(quotes, signal_date, clock())
        updated = advance(state, decision, view, quotes, signal_date)
        updated["data_metadata"] = view["metadata"]
        updated["saved_lines"] = render(updated, view)
        validate_snapshot(quotes, signal_date, clock())
        atomic_json(state_path, updated,
                    validator=lambda: validate_snapshot(quotes, signal_date, clock()))
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
