# -*- coding: utf-8 -*-
"""15:20 watchdog: canonical generation, acknowledged delivery and source health.

Text-file mtimes and a local partial daily bar are not delivery evidence.
Each network probe runs in a bounded child process with exactly one retry.
Historical source comparison excludes today, whose cached price may be a
14:50 observation. Alert tiers: only canonical/delivery/primary failures,
confirmed dual-source divergence and UNKNOWN calendar page DingTalk; degraded
probes, zero-comparison and QVIX publication lag are log-only notes, so a
recurring non-actionable condition cannot train users to ignore a real alarm.
Imports do not send messages; tests inject a temporary root and mock alarms.
"""
from datetime import datetime, timedelta
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

import live_utils
import signal_store as store
from market_data import UNIVERSE, CASH

BASE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = "510300"
_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
DIVERGE_THR = .005
PROBE_TIMEOUT = 6


def em_last_klines(code, n=8, today=None):
    """Read one small qfq page. Errors remain errors, never an empty success."""
    today = today or store.now_local().strftime("%Y-%m-%d")
    prefix = "1" if UNIVERSE[code][1] == "sh" else "0"
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=%s.%s"
           "&fields1=f1,f2,f3&fields2=f51,f53&klt=101&fqt=1&beg=0&lmt=%d&end=%s"
           % (prefix, code, n, today.replace("-", "")))
    with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=4) as response:
        body = response.read(1000001)
    if len(body) > 1000000:
        raise ValueError("Source response too large")
    payload = json.loads(body.decode("utf-8"))
    rows = (payload.get("data") or {}).get("klines") or []
    return [(r.split(",")[0], float(r.split(",")[1])) for r in rows][-n:]


def _probe(code, today):
    """A socket timeout alone does not bound a slowly streaming HTTP response."""
    result = subprocess.run([sys.executable, "-B", os.path.abspath(__file__),
                             "--probe", code, today], cwd=BASE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=PROBE_TIMEOUT)
    if result.returncode:
        raise RuntimeError("Benchmark source probe failed")
    return json.loads(result.stdout)


def _probe_benchmark(today):
    """Benchmark alias kept patchable by delivery tests."""
    return _probe(BENCHMARK, today)


def _probe_with_retry(code, today):
    """Flaky public endpoints get exactly one retry before degrading to a note."""
    probe = (lambda: _probe_benchmark(today)) if code == BENCHMARK else (lambda: _probe(code, today))
    try:
        return probe()
    except Exception:
        return probe()


def tencent_tail(code, n=8):
    with open(os.path.join(BASE, "data", code + ".csv"), newline="", encoding="utf-8") as stream:
        rows = [(r[0], float(r[2])) for r in csv.reader(stream) if r]
    return rows[-n:]


def _closes(rows, today):
    out = {}
    previous = None
    for date, value in rows:
        parsed = datetime.strptime(date, "%Y-%m-%d")
        value = float(value)
        if (parsed.strftime("%Y-%m-%d") != date or date > today
                or (previous is not None and date <= previous)
                or not math.isfinite(value) or value <= 0):
            raise ValueError("Invalid dated close observations")
        out[date] = value
        previous = date
    return out


def _dual_source(alerts, notes, today=None, codes=(BENCHMARK,)):
    """Compare completed common dates of the benchmark AND current holdings.

    Only a confirmed divergence >0.5% pages (data the strategy trades on is
    wrong); unverifiable probes, unreadable caches and zero comparisons are
    log-only notes — they recur under normal network flakiness and must not
    cry wolf. The benchmark stays first: its report keys are load-bearing for
    the UNKNOWN-calendar path and the historical result schema.
    """
    today = today or store.now_local().strftime("%Y-%m-%d")
    report = dict(scope="completed common history only", compared_dates=[],
                  status="DEGRADED", external_has_today=False, assets={})
    for position, code in enumerate(codes):
        is_benchmark = position == 0 and code == BENCHMARK
        label = "基准" if is_benchmark else "持仓%s" % code
        tx, external = {}, {}
        try:
            tx = _closes(tencent_tail(code), today)
        except Exception as exc:
            notes.append("DEGRADED: 腾讯%s历史无法核验（%s）" % (label, type(exc).__name__))
        try:
            external = _closes(_probe_with_retry(code, today) or [], today)
        except Exception as exc:
            notes.append("DEGRADED: 东财%s探测失败/格式无法核验（%s），不能据此判断休市或正常"
                         % (label, type(exc).__name__))
        if is_benchmark:
            report["external_has_today"] = today in external
        common = sorted(d for d in set(tx) & set(external) if d < today)
        asset = dict(compared_dates=common, status="DEGRADED")
        divergences = [(date, abs(external[date] / tx[date] - 1)) for date in common
                       if abs(external[date] / tx[date] - 1) > DIVERGE_THR]
        if not common:
            notes.append("DEGRADED: %s双源已完成共同历史有效对比为0，数据健康尚未验证" % label)
        elif divergences:
            alerts.append("双源%s已完成日收盘偏差>0.5%%: " % label + "; ".join(
                "%s %.2f%%" % (date, deviation * 100) for date, deviation in divergences))
            asset.update(status="FAILED", divergences=divergences)
        else:
            asset["status"] = "OK"
        report["assets"][code] = asset
        if is_benchmark:
            report["compared_dates"] = common
            report["status"] = asset["status"]
            if divergences:
                report["divergences"] = divergences
    return report


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(store.TZ).replace(tzinfo=None)
    return parsed


def _canonical(directory, today, now):
    """Validate the signal at its generation time, not its 15:20 expiry time."""
    try:
        journal = store.load_journal(directory)
        if journal is None or journal.get("date") != today:
            return None, "当日canonical信号未提交；dated.txt/mtime不能替代"
        info = store.signal_info(journal["text"], now)
        if not info["valid"] or info["date"] != today:
            return None, "当日canonical信号正文日期或格式无效"
        generated = datetime.strptime(info["generated"], "%Y-%m-%d %H:%M:%S"
                                      if len(info["generated"]) == 19 else "%Y-%m-%d %H:%M")
        at_generation = store.signal_info(journal["text"], generated)
        committed = _timestamp(journal["committed_at"])
        if (not at_generation["actionable"] or generated > now + timedelta(seconds=5)
                or committed.strftime("%Y-%m-%d") != today
                or committed > now + timedelta(seconds=5) or committed < generated - timedelta(seconds=5)):
            return None, "canonical生成时的行情时间/执行窗口/提交时间未通过核验"
        return journal, None
    except Exception as exc:
        return None, "canonical信号校验失败（%s）" % type(exc).__name__


def _delivery_checks(directory, today, journal, canonical_error):
    failures = []
    if canonical_error:
        failures.append(canonical_error)
    for filename, label in (("generation_status.json", "生成记录"), ("delivery.json", "发送回执")):
        try:
            value = store.read_json(Path(directory) / filename, {})
            if not isinstance(value, dict):
                raise ValueError("Invalid record")
        except Exception as exc:
            failures.append(label + "无法读取（%s）" % type(exc).__name__)
            continue
        expected = "ready" if filename == "generation_status.json" else "sent"
        if value.get("date") != today or value.get("status") != expected:
            failures.append("今日%s未确认%s（记录日期%s，状态%s）" % (
                label, expected, value.get("date", "缺失"), value.get("status", "缺失")))
            continue
        if journal is None or value.get("signal_id") != journal["signal_id"]:
            failures.append(label + "signal_id与当日canonical哈希不匹配")
            continue
        if filename == "delivery.json":
            attempts = value.get("attempts") or []
            if not isinstance(attempts, list) or not any(isinstance(a, dict) and a.get("status") == "sent"
                       and type(a.get("errcode")) is int and a["errcode"] == 0 for a in attempts):
                failures.append("发送回执虽标sent，但没有errcode=0的成功确认记录")
    return failures


def _primary_checks(journal, today):
    """A delivered comparison bundle does not prove the main version worked."""
    if today < store.PRIMARY_START_DATE or journal is None:
        return []
    if journal.get("primary_version") != store.PRIMARY_VERSION:
        return ["DEGRADED: 主推送%s尚未生成；当前仍为旧部署版本记录" % store.PRIMARY_VERSION]
    target = store.primary_target(journal)
    if not target:
        return ["DEGRADED: 主推送未生成（%s）；对照版本已提交或送达不能替代主版本" % store.PRIMARY_VERSION]
    state = journal["account_states"][store.PRIMARY_ACCOUNT]
    try:
        quote = _timestamp(state["quote_timestamp"])
        committed = _timestamp(journal["committed_at"])
        if quote.strftime("%Y-%m-%d") != today or not -5 <= (committed - quote).total_seconds() <= 180:
            raise ValueError("Main quote was stale when committed")
    except (KeyError, TypeError, ValueError):
        return ["DEGRADED: 主推送%s账户行情时间无法核验，不能用对照行情代替" % store.PRIMARY_VERSION]
    body, active = [], False
    headers = ("【影子 %s】" % store.PRIMARY_VERSION, "【主推送 %s】" % store.PRIMARY_VERSION)
    for line in journal["text"].splitlines():
        line = line.strip()
        if line.startswith("【"):
            active = line.startswith(headers)
        if active:
            body.append(line)
    statuses = " ".join(line for line in body if line.startswith("信号状态:"))
    advice = [re.split(r"建议[:：]", line, maxsplit=1)[1].split("|",1)[0].strip()
              for line in body if re.search(r"建议[:：]",line)]
    displayed_target = any(re.search(r"(?:^|(?:买入|继续持有|持有|建仓)\s+)" + re.escape(target) + r"(?=\s|$)",line)
                           for line in advice)
    if (not body or not displayed_target
            or any(word in statuses for word in ("计算失败", "无有效建议", "尚无有效信号", "历史", "尚未启动"))):
        return ["DEGRADED: 主推送%s缺少有效的独立卡片；不得视为主版本正常" % store.PRIMARY_VERSION]
    return []


def _qvix_fresh(notes, today):
    """QVIX is published in the evening: same-day absence at 15:20 is expected
    often enough that paging here would be a standing false alarm. Log only."""
    path = Path(BASE) / "data/qvix50.csv"
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = [(r[0], float(r[1])) for r in csv.reader(stream) if r]
        values = _closes(rows, today)
        if today not in values:
            raise ValueError("No same-day QVIX")
        return True
    except Exception as exc:
        notes.append("DEGRADED: QVIX当日数据未核实（%s）；恐慌通道不可视为正常" % type(exc).__name__)
        return False


def _finish(alerts, status=None, directory=None, notes=()):
    status = status or ("FAILED" if alerts else "OK")
    directory = Path(directory) if directory is not None else Path(BASE) / "signals"
    directory.mkdir(parents=True, exist_ok=True)
    # Record before attempting the alarm, so a network problem cannot erase it.
    logged = alerts + ["[note] " + n for n in notes]
    with (directory / "watchdog.log").open("a", encoding="utf-8") as stream:
        stream.write("%s %s%s\n" % (store.now_local().isoformat(), status,
                                    ": " + " | ".join(logged) if logged else ""))
    if alerts:
        text = "### ⚠️ 动量系统看门狗 %s\n\n" % status + "\n\n".join("- " + a for a in alerts)
        if notes:
            text += "\n\n仅记录不告警：\n" + "\n".join("- " + n for n in notes)
        try:
            alarm_sent = live_utils.send_dingtalk(text)
        except Exception:
            alarm_sent = False
        if not alarm_sent:
            with (directory / "watchdog.log").open("a", encoding="utf-8") as stream:
                stream.write("%s %s: 告警发送未确认；原检查失败状态保留\n" % (store.now_local().isoformat(), status))
    print("watchdog " + status)


def main(now=None, directory=None):
    now = now or store.now_local()
    if now.tzinfo is not None:
        now = now.astimezone(store.TZ).replace(tzinfo=None)
    today = now.strftime("%Y-%m-%d")
    directory = Path(directory) if directory is not None else Path(BASE) / "signals"
    alerts, notes = [], []
    try:
        calendar = store.trading_day(today)
    except Exception:
        calendar = None
    if calendar is False:
        _finish([], "SKIP", directory)
        return dict(status="SKIP", date=today, reason="官方日历休市/周末", alerts=[], notes=[])
    journal, canonical_error = _canonical(directory, today, now)
    # Holdings are the prices the strategy actually trades on; cross-check them
    # alongside the benchmark instead of trusting a single vendor for them.
    holdings = []
    for name in sorted((journal or {}).get("account_states") or {}):
        code = (journal["account_states"][name] or {}).get("holding")
        if code and code != CASH and code not in holdings:
            holdings.append(code)
    dual = _dual_source(alerts, notes, today, (BENCHMARK,) + tuple(holdings))
    assets_failed = any(asset.get("status") == "FAILED" for asset in dual.get("assets", {}).values())
    observed_open = journal is not None or dual.get("external_has_today")
    if calendar is None and not observed_open:
        alerts.insert(0, "UNKNOWN: 官方交易日历未覆盖%s，且没有已核验当日信号/新鲜外部行情；请补充日历，不能按休市跳过" % today)
        status = "UNKNOWN"
    else:
        failures = _delivery_checks(directory, today, journal, canonical_error)
        alerts[:0] = failures
        primary_alerts = _primary_checks(journal, today)
        alerts.extend(primary_alerts)
        if calendar is None:
            alerts.append("DEGRADED: 官方日历未覆盖%s；已凭当日信号/行情确认开市，仍需补充官方日历" % today)
        qvix_ok = _qvix_fresh(notes, today)
        if failures or assets_failed:
            status = "FAILED"
        elif primary_alerts or dual["status"] != "OK" or not qvix_ok or calendar is None:
            status = "DEGRADED"
        else:
            status = "OK"
    _finish(alerts, status, directory, notes)
    return dict(status=status, date=today, calendar=calendar,
                observed_open=observed_open, canonical_verified=journal is not None,
                source_comparison=dual, alerts=alerts, notes=notes)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--probe":
        print(json.dumps(em_last_klines(sys.argv[2], today=sys.argv[3])))
    else:
        raise SystemExit(0 if main()["status"] in ("OK", "SKIP") else 1)
