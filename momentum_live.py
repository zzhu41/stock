# -*- coding: utf-8 -*-
"""On-demand "动量" estimate for the DingTalk query path.

Two modes, picked by freshness:
  live  — today IS the quote's trading day but the published bundle is older:
          recompute every version in a throwaway staging directory through
          daily_extras.collect(observe_only=True). No production account or
          journal is ever written; only the 14:50–14:55 commit-window
          assertion is skipped, all quote/history/corporate-action validation
          stays in force.
  saved — the published bundle already covers the latest quote date (after
          close, weekends): restate the authoritative journal checkpoints.

Protocol: stdin ignored; stdout is exactly one JSON object
{"ok", "mode", "markdown", ...}. Any fatal error exits 1 with a sanitized
{"ok": False, ...}; the caller then falls back to push_signal saved rendering.
"""
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import signal_store
from market_data import UNIVERSE, fetch_realtime

SIGNAL_DIR = Path(BASE) / "signals"

VERSION_ROWS = (
    ("shadow_v12_r2.json", "V12-R2(主)"),
    ("shadow_v92.json", "V9.2"),
    ("shadow_v92_plus.json", "V9.2+"),
    ("shadow_v10.json", "V10-H"),
)

# A chat answer cannot wait for the 14:50 defaults (view 45s / qvix 22s / premium 18s).
# qvix keeps its full 22s budget: its own fetch timeout is 20s, so any smaller
# budget would silently disable the panic-dip channel whenever the API is slow.
# Inputs run concurrently, so a generous qvix budget does not lengthen the reply.
LIVE_INPUT_LIMITS = {"view": 25.0, "qvix": 22.0, "premium": 10.0}


def _etf(code):
    entry = UNIVERSE.get(code)
    return entry[0] if entry else str(code)


def _cell(text, limit=40):
    text = str(text or "").replace("|", "/").replace("\n", " ").strip()
    return text[:limit - 1] + "…" if limit and len(text) > limit else text


def _state_reason(state):
    decision = state.get("last_decision") or {}
    reason = decision.get("reason") or ""
    if not reason:
        advice = state.get("last_advice") or ""
        reason = advice.split("|", 1)[1].strip() if "|" in advice else ""
    return _cell(reason)


def _row(label, new_state, old_state=None):
    """(label, action, target, reason); action=None marks an unavailable row."""
    if not isinstance(new_state, dict) or not new_state.get("holding"):
        return label, None, None, "等待首次有效信号"
    code = new_state["holding"]
    decision = new_state.get("last_decision") or {}
    previous = decision.get("previous_holding")
    if not previous and isinstance(decision.get("diagnostics"), dict):
        previous = decision["diagnostics"].get("previous_holding")
    if not previous and old_state:
        previous = old_state.get("holding")
    if previous and previous != code:
        action = "换仓 卖%s" % _etf(previous)
    elif previous == code:
        action = "持有"
    elif new_state.get("start_date") and new_state.get("start_date") == new_state.get("last_date"):
        action = "建仓"
    else:
        action = "持有"
    return label, action, "%s %s" % (code, _etf(code)), _state_reason(new_state)


def _failure_note(label, diagnostics):
    phase = "strategy." + label.split("(")[0]
    for detail in diagnostics or []:
        if isinstance(detail, dict) and detail.get("phase") == phase:
            return "估算失败: " + _cell(detail.get("message"), 30)
    return "本次无有效估算"


def render_table(rows, mode, date, quote_label):
    title = "📊 动量实时估算" if mode == "live" else "📊 动量最新信号"
    lines = ["%s | 数据 %s | %s" % (title, date, quote_label), "",
             "| 版本 | 建议 | 买什么 | 原因 |", "|---|---|---|---|"]
    for label, action, target, reason in rows:
        if action is None:
            lines.append("| %s | — | — | %s |" % (label, _cell(reason)))
        else:
            lines.append("| %s | %s | %s | %s |" % (label, action, target, reason or "—"))
    if mode == "live":
        lines += ["", "⚠️ 盘中实时估算，不写入任何账户；正式信号以 14:50 推送为准"]
    return "\n".join(lines)


def _premium_brief(lines):
    parts = []
    for line in lines or []:
        match = re.search(r"(\d{6})\s+\S+\s+溢价\s+([+-][\d.]+%)", line)
        if match:
            parts.append("%s %s%s" % (match.group(1), match.group(2),
                                      "⚠️" if "⚠️" in line else ""))
    return "QDII 溢价: " + " / ".join(parts) if parts else None


def _load_journal():
    try:
        return signal_store.load_journal(SIGNAL_DIR)
    except Exception:
        return None


def _saved_text(journal):
    if isinstance(journal, dict) and isinstance(journal.get("text"), str):
        return journal["text"]
    try:
        return (SIGNAL_DIR / "latest.txt").read_text(encoding="utf-8")
    except Exception:
        return ""


def _saved_date(journal):
    """Journal date, falling back to the legacy latest.txt header."""
    if isinstance(journal, dict) and journal.get("date"):
        return journal["date"]
    match = re.search(r"数据截止\s*(\d{4}-\d{2}-\d{2})", _saved_text(journal)[:400])
    return match.group(1) if match else None


def _physical_state(name):
    path = SIGNAL_DIR / name
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _current_state(journal, name):
    checkpoints = (journal or {}).get("account_states") or {}
    state = checkpoints.get(name)
    return state if isinstance(state, dict) else _physical_state(name)


def saved_payload(journal):
    date = _saved_date(journal) or "未知"
    rows = [_row(label, _current_state(journal, name) or {}) for name, label in VERSION_ROWS]
    markdown = render_table(rows, "saved", date, "14:50收盘信号")
    brief = _premium_brief(_saved_text(journal).splitlines())
    if brief:
        markdown += "\n\n" + brief
    return dict(ok=True, mode="saved", date=date, markdown=markdown)


def live_payload(journal, now):
    """None when the saved bundle is authoritative (post-close / non-trading)."""
    quotes = fetch_realtime(codes=list(UNIVERSE), detailed=True)
    dates = {q["date"] for q in quotes.values()}
    if len(dates) != 1:
        raise ValueError("报价交易日不一致: %s" % ",".join(sorted(dates)))
    quote_date = dates.pop()
    if quote_date != now.strftime("%Y-%m-%d") or signal_store.trading_day(quote_date) is False:
        return None
    if _saved_date(journal) == quote_date:
        return None
    import daily_extras
    with tempfile.TemporaryDirectory(prefix=".observe-", dir=str(SIGNAL_DIR)) as tmp:
        stage = Path(tmp)
        for name, _label in VERSION_ROWS:
            state = _current_state(journal, name)
            if state is not None:
                signal_store.atomic_json(stage / name, state)
        trades = SIGNAL_DIR / "shadow_v92_trades.csv"
        if trades.is_file():
            shutil.copyfile(str(trades), str(stage / "shadow_v92_trades.csv"))
        result = daily_extras.collect(dict(
            quotes=quotes, date=quote_date, state_dir=str(stage),
            now=now.isoformat(), observe_only=True,
            input_limits=dict(LIVE_INPUT_LIMITS)))
        successful = set(result.get("successful_accounts") or [])
        diagnostics = result.get("diagnostics") or []
        rows = []
        for name, label in VERSION_ROWS:
            new_state = None
            path = stage / name
            if name in successful and path.is_file():
                try:
                    new_state = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    new_state = None
            if isinstance(new_state, dict) and new_state.get("holding"):
                rows.append(_row(label, new_state, _current_state(journal, name)))
            else:
                rows.append((label, None, None, _failure_note(label, diagnostics)))
        quote_time = max(q["timestamp"] for q in quotes.values())
        markdown = render_table(rows, "live", quote_date, "行情 " + quote_time)
        brief = _premium_brief(result.get("lines"))
        if brief:
            markdown += "\n\n" + brief
        return dict(ok=True, mode="live", date=quote_date,
                    quote_time=quote_time, markdown=markdown)


def main():
    override = None
    try:
        raw = sys.stdin.read()
        if raw.strip():
            candidate = json.loads(raw)
            if isinstance(candidate, dict) and isinstance(candidate.get("now"), str):
                override = candidate["now"]
    except Exception:
        override = None
    from datetime import datetime
    now = datetime.fromisoformat(override) if override else signal_store.now_local()
    journal = _load_journal()
    payload, live_error = None, None
    try:
        payload = live_payload(journal, now)
    except Exception as exc:
        live_error = type(exc).__name__
    if payload is None:
        payload = saved_payload(journal)
        if live_error:
            payload["live_fallback"] = live_error
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stdout.write(json.dumps(dict(ok=False, error=type(exc).__name__,
                                         message=str(exc)[:200]), ensure_ascii=False) + "\n")
        raise SystemExit(1)
