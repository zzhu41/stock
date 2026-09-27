"""Version-4 raw-unit shadow accounts, with one locked JSON transaction per day.

The JSON event journal is authoritative. CSV is an atomic, repeatable projection;
failure to export it never rolls back or repeats a committed virtual fill.
This module does not fetch market data, submit orders, or send messages.
"""
from contextlib import contextmanager
from copy import deepcopy
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

VERSION = 4


def _positive(value, label):
    try:
        if isinstance(value, bool):
            raise ValueError("boolean")
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("影子账户%s必须为正有限数" % label)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("影子账户%s必须为正有限数" % label)
    return value


@contextmanager
def account_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as stream:
        # A concurrent cron/manual invocation must not race the same account.
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("影子账户正在更新，拒绝并发记账；请稍后读取当天保存信号")
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, state):
    atomic_text(path, json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate(state, account_id):
    if state.get("valuation_version") != VERSION or state.get("account_id") != account_id:
        raise ValueError("影子账户版本或身份不一致")
    nav = _positive(state["nav"], "净值")
    events = state.get("events")
    if not isinstance(events, list):
        raise ValueError("影子账户缺少权威事件流水")
    dates = [event["date"] for event in events]
    if dates != sorted(set(dates)) or not dates or dates[-1] != state.get("last_date"):
        raise ValueError("影子事件日期或最后记账日期不一致")
    last = events[-1]
    if last.get("to") != state.get("holding") or not math.isclose(
            _positive(last["nav"], "事件净值"), nav, rel_tol=1e-10):
        raise ValueError("影子权威事件与持仓/净值不一致")
    if state.get("holding") is None:
        if state.get("units") != 0 or not math.isclose(float(state["cash"]), nav, rel_tol=1e-10):
            raise ValueError("影子现金账户估值不一致")
    else:
        units = _positive(state["units"], "份额")
        price = _positive(state["mark_raw_price"], "原始估值价")
        if state.get("cash") != 0 or not math.isclose(units * price, nav, rel_tol=1e-10):
            raise ValueError("影子份额/原始价格/净值不一致")
    if state.get("lock_code") and state["lock_code"] != state.get("holding"):
        raise ValueError("影子锁仓与已入账持仓不一致")
    if not last["baseline"] and last["from"] is not None:
        settled = state.get("last_settlement_action")
        if not isinstance(settled, dict) or settled.get("code") != last["from"] or settled.get("date") != dates[-1]:
            raise ValueError("影子账户缺少上次已入账的公司行动凭据")
    lines = state.get("saved_lines")
    if not isinstance(lines, list) or not lines or not all(isinstance(line, str) for line in lines):
        raise ValueError("影子账户缺少当天封存卡片")


def migrate(state, signal_date, account_id, trades_path):
    """Preserve the entire old record and old CSV before establishing NAV=1."""
    state = deepcopy(state)
    if state.get("last_date") and state["last_date"] > signal_date:
        raise ValueError("信号日期早于已记账日期，停止影子记账")
    if state.get("valuation_version") == VERSION:
        validate(state, account_id)
        return state
    if state.get("valuation_version") not in (None, 1, 2, 3):
        raise ValueError("未知影子账户版本，拒绝自动重置")
    old = deepcopy(state)
    trigger = state.get("lock_trigger_date") or state.get("trigger_date")
    if state.get("lock_code") and not trigger:
        raise ValueError("旧影子锁仓没有触发日，无法可靠迁移；旧状态保留")
    if state.get("lock_code") and state["lock_code"] != state.get("holding"):
        raise ValueError("旧影子锁仓与持仓不一致，无法可靠迁移")
    archive = dict(state=old, valuation_version=old.get("valuation_version", 1),
                   nav=old.get("nav", 1.0), start_date=old.get("start_date"),
                   last_date=old.get("last_date"), archived_on=signal_date)
    path = Path(trades_path)
    if path.exists():
        raw = path.read_bytes()
        archive.update(trades_csv=raw.decode("utf-8"), trades_sha256=hashlib.sha256(raw).hexdigest())
    if state.get("legacy_performance"):
        state.setdefault("legacy_performance_history", []).append(deepcopy(state["legacy_performance"]))
    state["legacy_performance"] = archive
    if old.get("last_date") or path.exists():
        state["valuation_note"] = (
            "⚠️ 旧口径净值 %.6f（截至%s）及旧流水已归档，不作新口径前向业绩；"
            "%s 起按原始份额/现金分红重新从1记账" %
            (float(old.get("nav", 1.0)), old.get("last_date") or "未记账", signal_date))
    state.update(valuation_version=VERSION, account_id=account_id, nav=1.0, units=0.0, cash=0.0,
                 start_date=signal_date, last_date=None, entry_date=None,
                 mark_raw_price=None, mark_price=None, mark_code=None,
                 mark_anchor_date=None, mark_anchor_close=None, events=[], saved_lines=None,
                 lock_trigger_date=trigger if state.get("lock_code") else None,
                 last_settlement_action=None)
    state.pop("lock_until", None)
    return state


def _quote(quotes, code, signal_date):
    if code is None:
        return None
    if not isinstance(quotes, dict) or not isinstance(quotes.get(code), dict):
        raise ValueError("%s 缺少原始当日报价，停止影子记账" % code)
    quote = quotes[code]
    if quote.get("date") != signal_date:
        raise ValueError("%s 原始报价日期不符，停止影子记账" % code)
    return _positive(quote.get("price"), "原始报价")


def _action(view, code, date):
    action = view.get("actions", {}).get(code, {}).get(date)
    if not isinstance(action, dict):
        raise ValueError("%s %s 缺少已核验公司行动，不能按零分红处理" % (code, date))
    split = _positive(action.get("split_ratio"), "拆分比例")
    cash = float(action.get("cash_per_old_share"))
    if not math.isfinite(cash) or cash < 0:
        raise ValueError("现金分红金额非法")
    missing = bool(action.get("not_observed"))
    if missing and (split != 1 or cash != 0 or action.get("verification") != "bracketed_no_action_interval"):
        raise ValueError("缺报价日没有可信的无公司行动确认，停止影子记账")
    return action, split, cash, missing


def mark(state, quotes, view, signal_date):
    """Settle old units before changing the target; no QFQ price-ratio fallback."""
    if not isinstance(view, dict) or not isinstance(quotes, dict):
        raise ValueError("缺少共享的已核验原始行情/公司行动，停止影子记账")
    calendar = view.get("calendar", [])
    if calendar != sorted(set(calendar)) or signal_date not in calendar:
        raise ValueError("原始行情交易日历与信号日不一致")
    if view.get("metadata", {}).get("signal_date", signal_date) != signal_date:
        raise ValueError("公司行动视图日期与信号日不一致")
    previous_nav, held = float(state["nav"]), state.get("holding")
    details = dict(previous_nav=previous_nav, from_holding=held, actions=[], settlement=None,
                   first_session=state.get("last_date") is None)
    if not state.get("last_date"):
        return details  # No entitlement to any dividend before the new entry.
    if state["last_date"] not in calendar:
        raise ValueError("当前公司行动视图缺上次影子记账日")
    booked = state.get("last_settlement_action")
    if booked:
        _, split, cash, _ = _action(view, booked["code"], booked["date"])
        if any(not math.isclose(value, float(booked[key]), rel_tol=1e-10, abs_tol=1e-12)
               for key, value in (("split_ratio", split), ("cash_per_old_share", cash))):
            raise ValueError("已入账的盘中公司行动被修订，保留原账户并暂停新记账")
    if held is None:
        state["nav"] = _positive(state["cash"], "现金")
        return details
    current_price = _quote(quotes, held, signal_date)
    rows = {row[0]: row for row in view.get("raw_histories", {}).get(held, [])}
    units = _positive(state["units"], "份额")
    for date in calendar:
        if not state["last_date"] < date <= signal_date:
            continue
        action, split, cash, missing = _action(view, held, date)
        if missing:
            if date == signal_date or date in rows:
                raise ValueError("缺报价行动标记与真实行情不一致，不能虚构估值或成交")
            previous, following = action.get("previous_quote_date"), action.get("next_quote_date")
            if previous not in rows or following not in rows or not previous < date < following <= signal_date:
                raise ValueError("缺报价日缺少前后真实报价的无行动核验")
            continue  # Explicitly bracketed no-action gap; no invented mark/DRIP.
        entitlement = units * cash
        units *= split
        if date == signal_date:
            price = current_price
            details["settlement"] = dict(code=held, date=date, split_ratio=split, cash_per_old_share=cash)
        else:
            if date not in rows:
                raise ValueError("%s 缺已完成原始收盘 %s" % (held, date))
            price = _positive(rows[date][2], "已完成原始收盘")
        units += entitlement / price
        if split != 1 or cash:
            details["actions"].append(dict(date=date, code=held, split_ratio=split,
                                           cash_per_old_share=cash, cash_received=entitlement,
                                           reinvest_price=price))
    state["nav"] = _positive(units * current_price, "新净值")
    return details


def settle(state, target, reason, details, quotes, view, signal_date, fee):
    price = _quote(quotes, target, signal_date)
    if target is not None:
        _, _, _, missing = _action(view, target, signal_date)
        if missing:
            raise ValueError("目标当天缺报价，不能虚构成交")
    changed = target != details["from_holding"]
    charged = changed and not details["first_session"]
    if charged:
        state["nav"] *= 1 - fee
    nav = _positive(state["nav"], "成交后净值")
    state.update(holding=target, units=nav / price if target else 0.0, cash=0.0 if target else nav,
                 mark_raw_price=price, mark_price=price, mark_code=target,
                 last_date=signal_date, quote_timestamp=(quotes[target].get("timestamp") if target else None),
                 entry_date=signal_date if changed or details["first_session"] else state["entry_date"],
                 last_settlement_action=details["settlement"],
                 last_return=nav / details["previous_nav"] - 1)
    state["events"].append(dict(date=signal_date, **{"from": details["from_holding"]}, to=target,
                                price=price, nav=nav, trade=charged, fee_fraction=fee if charged else 0.,
                                baseline=details["first_session"], actions=details["actions"], reason=reason))


def project_csv(path, state):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["date", "from", "to", "price", "nav"])
    for event in state["events"]:
        if event["trade"]:
            writer.writerow([event["date"], event["from"] or "", event["to"] or "",
                             "%.4f" % (event["price"] or 0.), "%.6f" % event["nav"]])
    text = output.getvalue()
    path = Path(path)
    previous = None
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            previous = stream.read()
    if previous != text:
        atomic_text(path, text)


def _card_with_projection(state, trades_path):
    lines = list(state["saved_lines"])
    try:
        project_csv(trades_path, state)
    except Exception:
        lines.append("  跟踪说明: ⚠️ 影子交易CSV导出失败；账户JSON已入账，同日重试只重建流水，不重复交易")
    return lines


def cached_lines(path, account_id, signal_date):
    """Read-only timeout recovery; do not refresh data, rewrite CSV or advance NAV."""
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        validate(state, account_id)
        if state["last_date"] == signal_date:
            return list(state["saved_lines"])
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def transact(state_path, trades_path, loader, saver, account_id, signal_date,
             quotes, view, fee, decide, render):
    with account_lock(state_path):
        state = loader()
        if state.get("last_date") and state["last_date"] > signal_date:
            raise ValueError("信号日期早于已记账日期，停止影子记账")
        if state.get("valuation_version") == VERSION:
            validate(state, account_id)
            if state["last_date"] == signal_date:
                return _card_with_projection(state, trades_path)
        state = migrate(state, signal_date, account_id, trades_path)
        details = mark(state, quotes, view, signal_date)
        target, reason = decide(state)
        settle(state, target, reason, details, quotes, view, signal_date, fee)
        state["saved_lines"] = render(state, target, reason)
        validate(state, account_id)
        saver(state)  # Only commit point; no CSV mutation precedes this rename.
        return _card_with_projection(state, trades_path)
