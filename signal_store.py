"""Small standard-library signal journal shared by generation, query and push.

Schema 2 commits the versioned signal AND successful virtual accounts.
Schema 1 retains compatibility with old primary-lock journals. Account/text
files are recoverable projections. Compatible with the bot's Python 3.6.
"""
from contextlib import contextmanager, ExitStack
import csv
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parent
SIGNALS = ROOT / "signals"
TZ = timezone(timedelta(hours=8))
START_TIME, END_TIME = "14:50:00", "14:55:00"
PRIMARY_VERSION = "V12-R2"
PRIMARY_ACCOUNT = "shadow_v12_r2.json"
PRIMARY_START_DATE = "2026-09-29"
ACCOUNT_FILES = (PRIMARY_ACCOUNT, "shadow_v92.json", "shadow_v92_plus.json", "shadow_v10.json")


def primary_target(record):
    """Only today's committed main account supplies the main target.

    An older three-version journal's target meant V9.2. A retained checkpoint
    from a failed main account is also not a new executable recommendation.
    """
    if record.get('primary_version') != PRIMARY_VERSION:
        return None
    if PRIMARY_ACCOUNT not in record.get('updated_accounts', []):
        return None
    state = record.get('account_states', {}).get(PRIMARY_ACCOUNT, {})
    return state.get('holding') if state.get('last_date') == record.get('date') else None


def now_local():
    return datetime.now(TZ).replace(tzinfo=None)


def atomic_text(path, text, validator=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if validator is not None:
            validator()
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value, validator=None):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if validator is None:
        atomic_text(path, text)
    else:
        atomic_text(path, text, validator=validator)


@contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_digest(record):
    return digest(json.dumps({k: v for k, v in record.items() if k != "checksum"},
                             sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def load_journal(directory=SIGNALS):
    value = read_json(Path(directory) / "daily_state.json")
    if value is not None:
        if (value.get("schema") not in (1, 2) or not isinstance(value.get("text"), str)
                or value.get("signal_id") != digest(value["text"])
                or value.get("checksum") != record_digest(value)):
            raise ValueError("信号记录校验失败，停止使用旧建议")
        if value["schema"] == 2:
            accounts = value.get("account_states")
            updated = value.get('updated_accounts', list(accounts) if isinstance(accounts, dict) else [])
            if (not isinstance(accounts, dict) or not accounts or not isinstance(updated, list) or not updated or
                    set(accounts) - set(ACCOUNT_FILES) or
                    set(updated) - set(accounts) or len(set(updated)) != len(updated) or
                    any(not isinstance(state, dict) or not isinstance(state.get('last_date'), str)
                        or state['last_date'] > value['date'] for state in accounts.values()) or
                    any(accounts[name]['last_date'] != value['date'] for name in updated)):
                raise ValueError("多版本账户提交记录不完整")
        if 'primary_version' in value:
            if (value['schema'] != 2 or value['primary_version'] != PRIMARY_VERSION
                    or value.get('target') != primary_target(value)):
                raise ValueError("主推送目标与当日提交账户不一致")
    return value


def load_saved(directory=SIGNALS):
    journal = load_journal(directory)
    if journal is not None:
        return journal["text"], journal
    path = Path(directory) / "latest.txt"
    return path.read_text(encoding="utf-8"), None


def publish(text, date, target, crash_lock, directory=SIGNALS, account_states=None, accounts_locked=False, validator=None, updated_accounts=None, primary_version=None):
    """One atomic commit; failed compatibility projections cannot orphan a lock."""
    directory = Path(directory)
    if account_states is not None:
        updated_accounts = list(account_states) if updated_accounts is None else list(updated_accounts)
        if (not account_states or set(account_states) - set(ACCOUNT_FILES) or not updated_accounts or
                len(set(updated_accounts)) != len(updated_accounts) or set(updated_accounts) - set(account_states) or
                any(not isinstance(s, dict) or not isinstance(s.get('last_date'), str) or s['last_date'] > date
                    for s in account_states.values()) or
                any(account_states[name]['last_date'] != date for name in updated_accounts)):
            raise ValueError("拒绝提交不完整或跨日的策略账户")
    record = dict(schema=2 if account_states is not None else 1, date=date, signal_id=digest(text), text=text,
                  target=target, crash_lock=crash_lock,
                  committed_at=now_local().isoformat())
    if account_states is not None:
        record["account_states"] = account_states
        record['updated_accounts'] = updated_accounts
    if primary_version is not None:
        record['primary_version'] = primary_version
        if (record['schema'] != 2 or primary_version != PRIMARY_VERSION
                or target != primary_target(record)):
            raise ValueError("主推送仅允许使用该版本当日成功账户的目标")
    record["checksum"] = record_digest(record)
    if validator is None:
        atomic_json(directory / "daily_state.json", record)
    else:
        atomic_json(directory / "daily_state.json", record, validator=validator)
    errors = repair_projections(record, directory, accounts_locked=accounts_locked)
    return record, errors


def repair_projections(record, directory=SIGNALS, accounts_locked=False):
    directory = Path(directory)
    errors = []
    accounts = record.get("account_states", {})
    if set(accounts) - set(ACCOUNT_FILES):
        raise ValueError("未知策略账户投影")
    with ExitStack() as stack:
        if not accounts_locked:
            for name in sorted(accounts):
                stack.enter_context(file_lock((directory / name).with_suffix(".lock")))
        for name, state in accounts.items():
            try:
                path = directory / name
                try:
                    existing = read_json(path)
                except (ValueError, UnicodeError):
                    existing = None  # A verified canonical copy repairs its damaged projection.
                if not isinstance(existing, dict):
                    existing = None
                if existing and (existing.get("last_date") or "") > state["last_date"]:
                    raise ValueError("账户比主提交更新，拒绝倒退覆盖: " + name)
                atomic_json(path, state)
            except OSError as exc:
                errors.append(name + ": " + type(exc).__name__)
            if name == "shadow_v92.json":
                try:
                    output = io.StringIO(newline="")
                    writer = csv.writer(output)
                    writer.writerow(["date", "from", "to", "price", "nav"])
                    for event in state["events"]:
                        if event["trade"]:
                            writer.writerow([event["date"], event["from"] or "", event["to"] or "",
                                             "%.4f" % (event["price"] or 0.), "%.6f" % event["nav"]])
                    atomic_text(directory / "shadow_v92_trades.csv", output.getvalue())
                except OSError as exc:
                    errors.append("shadow_v92_trades.csv: " + type(exc).__name__)
    projections = [(directory / (record["date"] + ".txt"), record["text"]),
                   (directory / "latest.txt", record["text"])]
    if record.get("schema") == 1:
        projections.append((directory / "crash_lock.json", json.dumps(record["crash_lock"])))
    for path, text in projections:
        try:
            atomic_text(path, text)
        except OSError as exc:
            errors.append(path.name + ": " + type(exc).__name__)
    return errors


def execution_window(now=None):
    now = now or now_local()
    return trading_day(now.strftime("%Y-%m-%d")) is not False and START_TIME <= now.strftime("%H:%M:%S") < END_TIME


def trading_day(date):
    """True/False from published calendar; unknown years are NEVER called closed."""
    day = datetime.strptime(date, "%Y-%m-%d")
    if day.weekday() >= 5:
        return False
    calendar = read_json(ROOT / "calendars/sse.json", {})
    year = calendar.get(date[:4])
    if year is None:
        return None
    return not any(start <= date <= end for start, end in year["closed_ranges"])


def signal_info(text, now=None):
    now = now or now_local()
    result = dict(valid=False, actionable=False, date="", generated="", quote_time="",
                  note="信号格式缺失或损坏，仅供检查，不作为操作指令")
    header = re.search(r"^动量轮动信号\s*\|\s*生成 (\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?)\s*\|\s*数据截止 (\d{4}-\d{2}-\d{2})", text, re.M)
    legacy = re.search(r"^★ 建议:\s*\S", text, re.M)
    bundle = re.search(r"^策略版本:\s*(?:V12-R2\s*\|\s*)?V9\.2\s*\|\s*V9\.2\+\s*\|\s*V10-H\s*$", text, re.M)
    if not header or not (legacy or bundle):
        return result
    generated, date = header.groups()
    result.update(generated=generated, date=date)
    quote = re.search(r"^行情时间:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text, re.M)
    if quote:
        result["quote_time"] = quote.group(1)
    try:
        created = datetime.strptime(generated, "%Y-%m-%d %H:%M:%S" if len(generated) == 19 else "%Y-%m-%d %H:%M")
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return result
    if generated[:10] != date:
        result["note"] = "生成日期与行情日期不一致；下列为历史记录，非今日操作指令"
        return result
    result["valid"] = True
    if date != now.strftime("%Y-%m-%d"):
        result["note"] = "当前没有今日有效信号；下列为%s历史记录，非今日操作指令" % date
        return result
    if not quote:
        result["note"] = "缺少行情时间核验；仅作历史记录，不作为操作指令"
        return result
    try:
        stamp = datetime.strptime(quote.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return result
    if stamp.strftime("%Y-%m-%d") != date or not -5 <= (created - stamp).total_seconds() <= 180:
        result["note"] = "生成时行情时间不一致或已过期，暂停操作建议"
        return result
    if not execution_window(now):
        result["note"] = "当前不在14:50–14:55信号执行窗口；仅查看历史记录"
        return result
    if not -5 <= (now - stamp).total_seconds() <= 300 or created > now + timedelta(seconds=5):
        result["note"] = "当前行情已超过有效时限或时间异常，暂停操作建议"
        return result
    result.update(actionable=True, note="当日有效信号；按所示行情时间判断执行")
    return result


def momentum_detail(state, names):
    """Per-asset momentum table lines from a saved decision's diagnostics.

    Read-only display derived from the same committed account state; missing
    or unverifiable diagnostics yield None, never an inferred ranking.
    """
    if not isinstance(state, dict):
        return None
    diag = (state.get("last_decision") or {}).get("diagnostics") or {}
    ranking = diag.get("ranking") or []
    indicators = diag.get("indicators") or {}
    rows = []
    for code in ranking:
        feat = indicators.get(code) or {}
        try:
            score = float(feat.get("score"))
        except (TypeError, ValueError):
            continue
        rows.append((code, score, feat))
    if not rows:
        return None
    holding = state.get("holding")
    date = state.get("last_date") or "未知"
    lines = ["**各 ETF 动量分**（WLS25平滑 · 截至 %s）" % date, "",
             "| # | 标的 | 动量分 | 20日 | 60日 | MA180 |", "|---|---|---|---|---|---|"]
    for idx, (code, score, feat) in enumerate(rows, 1):
        mark = " ◀持有" if code == holding else ""
        lines.append("| %d | %s %s%s | %.1f | %+.1f%% | %+.1f%% | %+.1f%% |" % (
            idx, code, names.get(code, code), mark, score,
            float(feat.get("mom20") or 0.) * 100,
            float(feat.get("mom60") or 0.) * 100,
            float(feat.get("ma180") or 0.) * 100))
    return lines
