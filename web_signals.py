"""Read-only V10 shadow status for the dashboard; never compute or book trades."""
from pathlib import Path
from datetime import datetime

import signal_store as store
from market_data import UNIVERSE
from v10_live.ledger import validate_state


def asset(code):
    return {"code": code, "name": UNIVERSE.get(code, (code,))[0]} if code else None


def v10_signal(version=None, directory=None, now=None):
    directory = Path(directory) if directory is not None else store.SIGNALS
    now = now or store.now_local()
    today = now.strftime("%Y-%m-%d")
    daily = (version or {}).get("daily", [])
    preview = dict(date=daily[-1][0], **asset(daily[-1][2])) if daily and daily[-1][2] else None
    result = dict(status="historical_preview" if preview else "not_started", data_date=None,
                  quote_timestamp=None, target=None, holding=None, nav=None,
                  start_date=None, state_date=None, reason="前向影子跟踪尚未启动，等待下一次有效日信号",
                  warnings=["独立虚拟账户，不下单；历史回测净值不计入前向账户"],
                  preview=preview, delivery_status=None, generation_status=None,
                  actionable=False)
    text, journal, info = "", None, {}
    try:
        text, journal = store.load_saved(directory)
        info = store.signal_info(text, now)
    except Exception:
        result["warnings"].append("每日信号记录不可用；本页不会自动生成或沿用操作指令")
    try:
        generation = store.read_json(directory / "generation_status.json", {})
        delivery = store.read_json(directory / "delivery.json", {})
        if generation.get("date") == today:
            result["generation_status"] = generation.get("status")
        if delivery.get("signal_id") == (journal or {}).get("signal_id") and delivery.get("date") == today:
            result["delivery_status"] = delivery.get("status")
    except Exception:
        result["warnings"].append("生成或推送状态无法核验")
    snapshot_date = (journal or {}).get("date") or info.get("date")
    lines, in_v10 = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("【影子"):
            in_v10 = stripped.startswith("【影子 V10-H】")
        if in_v10 or ("影子 V10-H" in stripped and "计算失败" in stripped):
            lines.append(stripped)
    failed = snapshot_date == today and any("计算失败" in line or "无有效建议" in line for line in lines)
    try:
        state = store.read_json(directory / "shadow_v10.json")
        if state is not None:
            validate_state(state)
            date = state["last_date"]
            if datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d") != date or date > today:
                raise ValueError("Invalid shadow date")
            result.update(status="stale", data_date=date, state_date=date,
                          quote_timestamp=state.get("quote_timestamp"),
                          holding=asset(state["holding"]), nav=state["nav"],
                          start_date=state["start_date"], reason=state.get("last_decision", {}).get("reason", "最近已保存影子记录"))
            published = date == snapshot_date and bool(info.get("valid")) and bool(lines)
            if published and not failed:
                result["target"] = asset(state["holding"])
                try:
                    stamp = datetime.strptime(state.get("quote_timestamp", ""), "%Y-%m-%d %H:%M:%S")
                    fresh = stamp.strftime("%Y-%m-%d") == date and -5 <= (now - stamp).total_seconds() <= 300
                except (TypeError, ValueError):
                    fresh = False
                result["actionable"] = bool(info.get("actionable")) and date == today and fresh
                result["status"] = "active" if result["actionable"] else "stale"
            else:
                result["warnings"].append("账户记录与已发布日信号未对齐；持仓仅供跟踪，不提供当前建议")
            if not result["actionable"]:
                result["warnings"].append("显示最近保存记录，并非当前买入指令")
    except Exception:
        result.update(status="failed", target=None, holding=None, nav=None,
                      reason="前向账户校验失败，暂停显示持仓及净值")
        result["warnings"].append("请核查账户记录；网页不会重置或修补账本")
    if failed or result["generation_status"] == "failed":
        result.update(status="failed", target=None, actionable=False,
                      reason="今日信号生成失败或V10行情未通过核验，暂无新的有效建议")
    return result
