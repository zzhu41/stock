"""Small standard-library signal journal shared by generation, query and push.

The journal is the commit point for the signal AND its primary crash lock.
Text/legacy lock files are replaceable projections, never the source of truth
once a journal exists. Compatible with the bot's Python 3.6 interpreter.
"""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parent
SIGNALS = ROOT / "signals"
TZ = timezone(timedelta(hours=8))
START_TIME, END_TIME = "14:50:00", "14:55:00"


def now_local():
    return datetime.now(TZ).replace(tzinfo=None)


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, str(path))
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


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
        if (value.get("schema") != 1 or not isinstance(value.get("text"), str)
                or value.get("signal_id") != digest(value["text"])
                or value.get("checksum") != record_digest(value)):
            raise ValueError("信号记录校验失败，停止使用旧建议")
    return value


def load_saved(directory=SIGNALS):
    journal = load_journal(directory)
    if journal is not None:
        return journal["text"], journal
    path = Path(directory) / "latest.txt"
    return path.read_text(encoding="utf-8"), None


def publish(text, date, target, crash_lock, directory=SIGNALS):
    """One atomic commit; failed compatibility projections cannot orphan a lock."""
    directory = Path(directory)
    record = dict(schema=1, date=date, signal_id=digest(text), text=text,
                  target=target, crash_lock=crash_lock,
                  committed_at=now_local().isoformat())
    record["checksum"] = record_digest(record)
    atomic_json(directory / "daily_state.json", record)
    errors = repair_projections(record, directory)
    return record, errors


def repair_projections(record, directory=SIGNALS):
    directory = Path(directory)
    errors = []
    for path, text in ((directory / (record["date"] + ".txt"), record["text"]),
                       (directory / "latest.txt", record["text"]),
                       (directory / "crash_lock.json", json.dumps(record["crash_lock"]))):
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
    if not header or not re.search(r"^★ 建议:\s*\S", text, re.M):
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
