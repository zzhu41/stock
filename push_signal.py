# -*- coding: utf-8 -*-
"""钉钉推送: 读取 signals/latest.txt, 组装 markdown 推到钉钉群。

配置(均不入库):
  data/dingtalk.webhook  完整 webhook URL(含 access_token), 必填
  data/dingtalk.secret   加签密钥(SEC...), 可选; 无文件则空(仅关键词模式可用)

用法: python3.8 push_signal.py
信号生成后发送；记录送达状态、限次重试、成功后去重，失败返回非零。
"""
import base64
import argparse
import hashlib
import hmac
import json
import importlib.util
import os
import re
import time
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
# The Python3.6 bot imports this file by absolute path from another directory.
_spec = importlib.util.spec_from_file_location("_stock_signal_store", os.path.join(BASE, "signal_store.py"))
store = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(store)
LATEST = os.path.join(BASE, "signals", "latest.txt")
SIGNAL_DIR = os.path.join(BASE, "signals")
SECRET_FILE = os.path.join(BASE, "data", "dingtalk.secret")
WEBHOOK_FILE = os.path.join(BASE, "data", "dingtalk.webhook")
ASSET_PATTERN = r"(\d{6}\s+[^\s|()（）,，;；/]+)"
VERSIONS = ("V9.2", "V9.2+", "V10-H")


def load_file(path):
    """读取单行配置; 无文件返回空串。"""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def signed_url(webhook, secret):
    ts = str(round(time.time() * 1000))
    s = hmac.new(secret.encode("utf-8"),
                 ("%s\n%s" % (ts, secret)).encode("utf-8"),
                 digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(s))
    return "%s&timestamp=%s&sign=%s" % (webhook, ts, sign)


def highlight_advice(advice):
    """只强调操作中的目标标的，保留买入/持有/卖出的原始含义。"""
    highlighted = re.sub(r"((?:买入|继续持有|持有|建仓)\s+)" + ASSET_PATTERN,
                         r"\1**\2**", advice)
    if re.search(r"买入\s+" + ASSET_PATTERN, advice):
        highlighted = "🟧 " + highlighted
    return highlighted


def _shadow_advice(line):
    match = re.search(r"\s*\|\s*建议[:：]\s*(.*)", line)
    if match:
        return line[:match.start()].strip(), match.group(1)
    match = re.match(r"(?:★\s*)?建议[:：]\s*(.*)", line)
    return ("", match.group(1)) if match else (line, None)


def _target_asset(advice):
    target = re.split(r"\s*\|\s*", advice or "", maxsplit=1)[0].strip()
    # A sell leg alone is not an executable target. For a rotation prefer the
    # last explicit buy/hold leg, otherwise require a plain leading asset.
    matches = re.findall(r"(?:买入|继续持有|持有|建仓)\s+" + ASSET_PATTERN, target)
    if matches:
        return matches[-1]
    match = re.match("^" + ASSET_PATTERN, target)
    return match.group(1) if match else None


def _bold_holding(line):
    return re.sub(r"((?:当时)?影子持仓[:：]\s*)(\d{6}(?:\s+[^\s|()（）,，;；/]+)?)",
                  r"\1**\2**", line)


def shadow_markdown(line, historical=False, unavailable=False):
    """影子建议单独展示，不将虚拟目标解释为实盘买入指令。"""
    holding, advice = _shadow_advice(line)
    if historical:
        holding = holding.replace("影子持仓:", "当时影子持仓:")
    result = ["- " + _bold_holding(holding)] if holding else []
    if advice is None or unavailable:
        return result
    target = _target_asset(advice)
    if not target:
        return result
    reason = re.split(r"\s*\|\s*", advice, maxsplit=1)
    reason = reason[1] if len(reason) == 2 else ""
    label = "历史影子目标: " if historical else "影子建议标的: "
    actions = re.findall(r"(买入|继续持有|持有|建仓)\s+" + ASSET_PATTERN,
                         re.split(r"\s*\|\s*", advice, maxsplit=1)[0])
    action = actions[-1][0] + " " if actions and not historical else ""
    result += ["", label + action + "**" + target + "**" + ("（非当前操作指令）" if historical else "")]
    if reason:
        result += ["", "> " + reason]
    return result


def _version(label):
    match = re.fullmatch(r"(?:影子\s*)?(V9\.2\+?|V10-H)", label.strip().replace("＋", "+"), re.I)
    return match.group(1).upper() if match else None


def _version_sections(lines):
    """Allowlisted saved-text sections only; never infer a version from targets."""
    sections, current = {}, None
    for line in lines:
        value = line.strip()
        header = re.match(r"【([^】]+)】", value)
        if header:
            version = _version(header.group(1))
            current = {"lines": [], "failed": False} if version else None
            if version:
                sections[version] = current
            continue
        failure = re.search(r"影子\s*(V9\.2\+?|V10-H)(?![\w.+-]).*计算失败", value, re.I)
        if failure:
            version = failure.group(1).upper()
            current = {"lines": ["信号状态: 本次无有效建议，计算失败", "数据说明: " + value], "failed": True}
            sections[version] = current
            continue
        if re.fullmatch(r"[-=]{3,}", value):
            current = None
            continue
        if current is not None and (value.startswith(("QVIX", "信号状态:", "数据截止:", "行情时间:",
                "策略规则:", "数据说明:", "跟踪说明:", "抄底三口并集:", "建议:", "建议："))
                or "影子持仓" in value or "旧口径净值" in value):
            current["lines"].append(value)
    return sections


def _section_markdown(version, section, info):
    result = ["", "---", "**影子 %s**(虚拟跟踪不下单)" % version, ""]
    if not section or not section["lines"]:
        return result + ["- 信号状态: 尚无有效信号；等待首次生成", "- 不沿用其他版本目标或补记历史交易"]
    lines = section["lines"]
    status = " ".join(line for line in lines if line.startswith("信号状态:"))
    failed = section["failed"] or bool(re.search(r"计算失败|无有效建议|暂停虚拟成交", status))
    pending = bool(re.search(r"无有效信号|等待首次生成", status))
    targets = [_target_asset(_shadow_advice(line)[1]) for line in lines]
    unavailable = failed or pending or not any(targets)
    block_dates = [match.group(1) for line in lines for match in
                   [re.match(r"数据截止[:：]\s*(\d{4}-\d{2}-\d{2})", line)] if match]
    historical = (not info["actionable"] or "历史" in status or "尚未启动" in status
                  or any(day != info["date"] for day in block_dates))
    if unavailable:
        result += ["- 信号状态: 本次无有效建议，计算失败" if failed else "- 信号状态: 尚无有效信号；等待首次生成"]
    elif historical:
        result += ["- 本版本为历史记录；非当前操作指令"]
    if not block_dates and info["date"]:
        result += ["- 数据截止: %s（保存文本日期）" % info["date"]]
    for line in lines:
        if unavailable and line.startswith("信号状态:"):
            # Preserve specific failure details, but do not leave a conflicting
            # stale 'normal/valid' status beside the unavailable label.
            if failed:
                result += ["- " + line]
            continue
        result += shadow_markdown(line, historical=historical, unavailable=unavailable) + [""]
    return result


def build_markdown(text, query=False, now=None, notices=None):
    """Shared three-version view, including old saved cards; no model/account work."""
    lines = text.splitlines()
    title = next((l.strip() for l in lines if "动量轮动信号" in l), "动量轮动信号")
    info = store.signal_info(text, now=now)
    date = info["date"]
    historical = not info["actionable"]
    prem = [l.strip() for l in lines if (re.match(r"\s*\d{6}\s+", l) and "溢价" in l)
            or l.strip().startswith("QDII 溢价:")]
    md = ["### 📊 %s" % title, "",
          ("⚠️ **%s**" if historical else "✅ %s") % info["note"], "",
          "行情时间: %s" % (info["quote_time"] or "未记录")]
    for notice in notices or []:
        md += ["", "⚠️ " + notice]
    for item in prem:
        md += ["", item if item.startswith("QDII ") else "QDII " + item]
    sections = _version_sections(lines)
    for version in VERSIONS:
        md += _section_markdown(version, sections.get(version), info)
    md += ["", "---"]
    versions = "V9.2 / V9.2+ / V10-H（独立虚拟跟踪）"
    if query:
        md += ["最近保存信号 · 14:50开始生成，须同时核对行情时间与信号状态",
               "_行情日期 %s · %s_" % (date or "未知", versions),
               "", "📈 动量跟踪网页: http://120.26.67.168:8081/"]
    else:
        md += ["⏰ 仅在14:50–14:55且对应版本信号有效时参考；过期不补追" if not historical else "历史记录仅供查看，不补追过期信号",
               "_行情日期 %s · %s_" % (date or "未知", versions)]
    return "\n".join(md), date


def is_current_signal(text, today):
    """Legacy date-only helper; actual sending uses store.signal_info freshness."""
    dates = re.search(r"生成\s+(\d{4}-\d{2}-\d{2}).*数据截止\s+(\d{4}-\d{2}-\d{2})", text)
    return bool(dates and dates.group(1) == dates.group(2) == today)


def render_saved_query(path=LATEST, now=None):
    """Read-only query with explicit freshness and delivery/generation status."""
    now = now or store.now_local()
    directory = os.path.dirname(os.path.abspath(path))
    try:
        text, journal = store.load_saved(directory)
        notices = []
        generation = store.read_json(os.path.join(directory, "generation_status.json"), {})
        receipt = store.read_json(os.path.join(directory, "delivery.json"), {})
        today = now.strftime("%Y-%m-%d")
        if generation.get("date") == today and generation.get("status") in ("failed", "running"):
            notices.append("今日信号生成%s；请核对下方数据日期，不沿用旧买入建议" % (
                "失败" if generation["status"] == "failed" else "尚未完成"))
        signal_id = journal["signal_id"] if journal else store.digest(text)
        if receipt.get("signal_id") == signal_id:
            if receipt.get("status") != "sent":
                notices.append("这份信号的自动推送尚未确认送达（%s）" % receipt.get("status", "未知"))
        elif store.signal_info(text, now)["actionable"]:
            notices.append("这份信号尚无自动推送成功记录")
        return build_markdown(text, query=True, now=now, notices=notices)[0]
    except Exception:
        return "⚠️ 信号记录缺失或校验失败，当前没有可用操作建议，请检查生成任务。"


def main(directory=None, now=None, attempts=3, sleeper=None, max_per_run=1):
    directory = directory or SIGNAL_DIR
    fixed_now = now
    current_time = lambda: fixed_now or store.now_local()
    sleeper = sleeper or time.sleep
    try:
        with store.file_lock(os.path.join(directory, "delivery.lock")):
            text, journal = store.load_saved(directory)
            signal_id = journal["signal_id"] if journal else store.digest(text)
            receipt_path = os.path.join(directory, "delivery.json")
            previous = store.read_json(receipt_path, {})
            if previous.get("signal_id") == signal_id and previous.get("status") == "sent":
                print("钉钉推送: 此信号已确认送达，跳过重复发送")
                return 0
            receipt = previous if previous.get("signal_id") == signal_id else dict(
                schema=1, signal_id=signal_id, date=store.signal_info(text, current_time())["date"],
                status="pending", attempts=[])
            if not store.signal_info(text, current_time())["actionable"]:
                receipt.update(status="expired", updated_at=current_time().isoformat())
                store.atomic_json(receipt_path, receipt)
                print("钉钉推送停止: 当前没有有效执行窗口内的新鲜信号")
                return 1
            try:
                webhook = load_file(WEBHOOK_FILE)
                secret = load_file(SECRET_FILE)
                if not webhook:
                    raise ValueError("WebhookNotConfigured")
            except Exception as exc:
                receipt.update(status="failed", error="LocalDeliveryConfigurationUnavailable",
                               updated_at=current_time().isoformat())
                store.atomic_json(receipt_path, receipt)
                print("钉钉推送失败: 本机发送配置缺失或不可读（%s）" % type(exc).__name__)
                return 1
            sent_this_run = 0
            while len(receipt["attempts"]) < attempts and sent_this_run < max_per_run:
                if not store.signal_info(text, current_time())["actionable"]:
                    receipt["status"] = "expired"
                    break
                deadline = current_time().replace(hour=14, minute=55, second=0, microsecond=0)
                remaining = (deadline - current_time()).total_seconds()
                if remaining < 10:
                    receipt["status"] = "expired"
                    break
                md, date = build_markdown(text, now=current_time())
                if receipt["attempts"]:
                    md = "⚠️ 同一信号补发；如已操作，请勿重复下单。\n\n" + md
                attempt = dict(started_at=current_time().isoformat(), status="sending")
                receipt["attempts"].append(attempt)
                sent_this_run += 1
                receipt.update(status="sending", updated_at=attempt["started_at"])
                store.atomic_json(receipt_path, receipt)
                url = signed_url(webhook, secret) if secret else webhook
                body = json.dumps({"msgtype": "markdown", "markdown": {
                    "title": "动量信号 %s" % date, "text": md}}, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=min(8, remaining - 1)) as response:
                        payload = json.loads(response.read(65537).decode("utf-8"))
                    code = payload.get("errcode")
                    if type(code) is int and code == 0:
                        attempt.update(status="sent", errcode=0)
                        receipt.update(status="sent", confirmed_at=current_time().isoformat())
                        store.atomic_json(receipt_path, receipt)
                        print("钉钉推送: 成功（已记录送达状态）")
                        return 0
                    attempt.update(status="rejected", errcode=code if type(code) is int else None)
                    receipt["status"] = "failed"
                except Exception as exc:
                    # A lost HTTP reply cannot prove whether the service sent it.
                    attempt.update(status="uncertain", error=type(exc).__name__)
                    receipt["status"] = "uncertain"
                receipt["updated_at"] = current_time().isoformat()
                store.atomic_json(receipt_path, receipt)
                if len(receipt["attempts"]) < attempts and sent_this_run < max_per_run:
                    sleeper(2 * len(receipt["attempts"]))
            store.atomic_json(receipt_path, receipt)
            print("钉钉推送未确认成功（%s）；已保存失败记录" % receipt["status"])
            return 1
    except BlockingIOError:
        print("钉钉推送: 另一个发送任务正在执行")
        return 1
    except Exception as exc:
        print("钉钉推送失败: %s" % type(exc).__name__)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=SIGNAL_DIR)
    raise SystemExit(main(directory=parser.parse_args().directory))
