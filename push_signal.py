# -*- coding: utf-8 -*-
"""钉钉推送: 读取 signals/latest.txt, 组装 markdown 推到钉钉群。

配置(均不入库):
  data/dingtalk.webhook  完整 webhook URL(含 access_token), 必填
  data/dingtalk.secret   加签密钥(SEC...), 可选; 无文件则空(仅关键词模式可用)

用法: python3.8 push_signal.py
信号生成后由 cron 链式调用; 推送失败只告警不阻断(返回码恒为0)。
"""
import base64
import hashlib
import hmac
import json
import os
import re
import time
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
LATEST = os.path.join(BASE, "signals", "latest.txt")
SECRET_FILE = os.path.join(BASE, "data", "dingtalk.secret")
WEBHOOK_FILE = os.path.join(BASE, "data", "dingtalk.webhook")
ASSET_PATTERN = r"(\d{6}\s+[^\s|()（）,，;；/]+)"


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
    highlighted = re.sub(r"((?:买入|继续持有)\s+)" + ASSET_PATTERN,
                         r"\1**\2**", advice)
    if re.search(r"买入\s+" + ASSET_PATTERN, advice):
        highlighted = "🟧 " + highlighted
    return highlighted


def shadow_markdown(line):
    """影子建议单独展示，不将虚拟目标解释为实盘买入指令。"""
    holding, separator, advice = line.partition(" | 建议: ")
    if not separator:
        return ["- " + line]
    target, _, reason = advice.partition(" | ")
    target = re.sub("^" + ASSET_PATTERN, r"**\1**", target.strip())
    result = ["- " + holding, "", "影子建议标的: " + target]
    if reason:
        result += ["", "> " + reason]
    return result


def build_markdown(text):
    """从信号文本提取关键行, 组装 markdown。"""
    lines = text.splitlines()
    title = next((l.strip() for l in lines if "动量轮动信号" in l), "动量轮动信号")
    date_m = re.search(r"(\d{4}-\d{2}-\d{2})", title)
    date = date_m.group(1) if date_m else ""
    regime = next((l.strip() for l in lines if "牛熊体制" in l), "")
    holding = next((l.strip() for l in lines if "当前持仓" in l), "")
    advice = next((l.strip() for l in lines if "★ 建议" in l), "")
    reason = next((l.strip() for l in lines if "依据" in l), "")
    # 排名前三
    rank = []
    for l in lines:
        m = re.match(r"\s*(\d{6})\s+(\S+)\s+MOM20\s+([+-]\d+\.\d+)%", l)
        if m:
            rank.append("%s %s(MOM20 %s%%)" % (m.group(1), m.group(2), m.group(3)))
        if len(rank) >= 3:
            break
    prem = next((l.strip() for l in lines if "溢价" in l), "")
    md = ["### 📊 %s" % title,
          "",
          "**%s**" % regime,
          "",
          "**%s**" % holding,
          "",
          "## 今日操作",
          "",
          highlight_advice(advice.replace("★ ", "", 1)),
          "",
          "> %s" % reason.strip(),
          "",
          "动量前三: " + " / ".join(rank)]
    if prem:
        md += ["", "QDII " + prem.strip()]
    # 影子区块(0906/v9.2...): 按【影子 标题行分块, 块内收 QVIX/影子持仓行; 影子计算失败行单列
    sections, cur = [], None
    for l in lines:
        s = l.strip()
        if s.startswith("【影子"):
            m = re.match(r"【([^】]+)】", s)
            cur = (m.group(1) if m else s, [])
            sections.append(cur)
        elif "影子" in s and "计算失败" in s:
            sections.append((None, [s]))
            cur = None
        elif cur and (s.startswith("QVIX") or "影子持仓" in s or "旧口径净值" in s):
            cur[1].append(s)
    for label, slines in sections:
        md += ["", "---"]
        if label:
            md.append("**%s**(虚拟跟踪不下单)" % label)
        md.append("")
        for s in slines:
            md += shadow_markdown(s) + [""]
    md += ["",
           "---",
           "⏰ 尾盘 14:30-14:50 限价贴价执行; 操作后回报 代码/价格/金额 记账",
           "_信号 %s · v9.1 (+0906/v9.2影子)_" % date]
    return "\n".join(md), date


def is_current_signal(text, today):
    """生成失败后不得把旧latest.txt重新当作当天信号推送。"""
    dates = re.search(r"生成\s+(\d{4}-\d{2}-\d{2}).*数据截止\s+(\d{4}-\d{2}-\d{2})", text)
    return bool(dates and dates.group(1) == dates.group(2) == today)


def main():
    webhook = load_file(WEBHOOK_FILE)
    if not webhook:
        print("钉钉推送跳过(不阻断): 缺 %s" % WEBHOOK_FILE)
        return
    with open(LATEST, encoding="utf-8") as f:
        text = f.read()
    if not is_current_signal(text, time.strftime("%Y-%m-%d")):
        print("钉钉推送跳过: 信号生成日/行情日期不是今天，请先检查行情与信号生成")
        return
    md, date = build_markdown(text)
    secret = load_file(SECRET_FILE)
    url = signed_url(webhook, secret) if secret else webhook
    body = json.dumps({"msgtype": "markdown",
                       "markdown": {"title": "动量信号 %s" % date, "text": md}},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read().decode("utf-8"))
        print("钉钉推送: %s (errcode=%s %s)" % (
            "成功" if resp.get("errcode") == 0 else "失败",
            resp.get("errcode"), resp.get("errmsg")))
    except Exception as e:
        print("钉钉推送异常(不阻断): %s" % e)


if __name__ == "__main__":
    main()
