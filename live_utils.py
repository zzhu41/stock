# -*- coding: utf-8 -*-
"""实盘运维共享工具: 钉钉告警 + 原子写 + 日历助手。

纯标准库; 钉钉通道复用 push_signal 的 webhook/secret/加签逻辑(只读配置, 不改推送主链路)。
所有发送失败只打印不抛异常 —— 运维工具绝不能把主流程拖死。
"""
import json
import os
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))

import push_signal  # 复用 load_file/signed_url/WEBHOOK_FILE/SECRET_FILE(纯函数, 无副作用)


def send_dingtalk(text, title="动量系统运维"):
    """发送 markdown 告警到钉钉群; 无 webhook 配置或发送失败时打印到 stdout。"""
    webhook = push_signal.load_file(push_signal.WEBHOOK_FILE)
    if not webhook:
        print("[dingtalk 未配置] %s\n%s" % (title, text))
        return False
    secret = push_signal.load_file(push_signal.SECRET_FILE)
    url = push_signal.signed_url(webhook, secret) if secret else webhook
    body = json.dumps({"msgtype": "markdown",
                       "markdown": {"title": title, "text": text}},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            resp = json.loads(r.read().decode("utf-8"))
        ok = resp.get("errcode") == 0
        print("钉钉告警: %s (errcode=%s)" % ("成功" if ok else "失败", resp.get("errcode")))
        return ok
    except Exception as e:
        print("钉钉告警异常(不阻断): %s" % e)
        return False


def atomic_write(path, text):
    """原子写文本: 临时文件 + os.replace(同分区 rename 原子性), 防并发读到半截文件。"""
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def read_last_date(code):
    """读本地缓存最后一根K线日期(不触网络)。"""
    import csv
    path = os.path.join(BASE, "data", "%s.csv" % code)
    last = None
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if r:
                last = r[0]
    return last


def append_log(name, line):
    """追加一行到 signals/<name>(带时间戳)。"""
    path = os.path.join(BASE, "signals", name)
    with open(path, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), line))
