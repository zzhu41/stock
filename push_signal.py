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
          "## %s" % advice.replace("★ ", ""),
          "> %s" % reason.strip(),
          "",
          "动量前三: " + " / ".join(rank)]
    if prem:
        md += ["", "QDII " + prem.strip()]
    md += ["",
           "---",
           "⏰ 尾盘 14:30-14:50 限价贴价执行; 操作后回报 代码/价格/金额 记账",
           "_信号 %s · v9.1_" % date]
    return "\n".join(md), date


def main():
    webhook = load_file(WEBHOOK_FILE)
    if not webhook:
        print("钉钉推送跳过(不阻断): 缺 %s" % WEBHOOK_FILE)
        return
    with open(LATEST, encoding="utf-8") as f:
        text = f.read()
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
