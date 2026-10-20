# -*- coding: utf-8 -*-
"""牢国策略(github.com/marcocrxu/ETFMomentum Top-1 动量轮动)每日 14:49 信号推送。

流程: 指数缓存增量刷新 → ETF qfq 日线(T-1) + 腾讯实时价(T 行) + QQQ-CNY 混合动量
      → 仓库 strategy/signal.py 的 compute_signals 原样计算 → 今日动作
      → 完整版 + 大字版两条消息群发到全部配置群(复用 push_signal 多群通道)。

设计约束:
- 引擎只读加载 /root/external/ETFMomentum/strategy/signal.py, SHA256 校验防漂移
- 幂等: signals/laoguo_state.json 记录当日已推送指纹, 同日同信号不重复推送
- 数据/计算失败推送"计算失败"告知群里, 绝不静默
- 非交易日静默退出; 推送消息固定标记【牢国】
"""
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import laoguo_data  # noqa: E402
import push_signal  # noqa: E402  多群通道: load_targets/signed_url/_ensure_keyword
import signal_store as store  # noqa: E402  trading_day / file_lock / atomic_json

ENGINE_PATH = "/root/external/ETFMomentum/strategy/signal.py"
# 部署基线 commit 的引擎指纹(2026-10-20 下载, 仓库最后更新 2026-08-26); 升级引擎须显式更新此值
ENGINE_SHA256 = ""
STATE_FILE = os.path.join(BASE, "signals", "laoguo_state.json")
LOCK_FILE = os.path.join(BASE, "signals", "laoguo.lock")

CODE_OF = {v: k for k, v in laoguo_data.POOL.items()}  # etf code -> 名称
NAME_TO_TENCENT = {"创业": "159915", "纳指": "513100", "黄金": "518880"}


def engine_sha256(path=ENGINE_PATH):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_engine():
    """只读加载仓库引擎; 记录指纹到状态文件(首次部署固化基线)。"""
    spec = importlib.util.spec_from_file_location("_laoguo_engine", ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_position(actions):
    """从动作序列重放当前持仓: (pos_idx, entry_day_idx, entry_reason)。"""
    pos, entry_day, reason = -1, -1, ""
    for t, _old, new, why in actions:
        if new >= 0:
            entry_day, reason = t, why
        pos = new
    return pos, entry_day, reason


def build_messages(result, names, date_str, quote_note):
    """完整版 + 大字版两条消息。result 见 run_signal 返回。"""
    ranked = sorted(result["momentums"].items(), key=lambda x: -x[1])
    lines = ["### 🏛️ 牢国策略 · ETF 20日动量轮动", "",
             "⏰ %s %s" % (date_str, quote_note), ""]
    for i, (name, mom) in enumerate(ranked):
        icon = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, "")
        arrow = "🟢" if mom >= 0 else "🔴"
        holder = " ← 持有" if name == result["position"] else ""
        lines.append("%s %s **%s**: %+.2f%%%s" % (icon, arrow, name, mom, holder))
    lines.append("    📉 上证指数20日动量: %+.2f%%" % result["sh_momentum"])
    lines.append("")
    if result["position"]:
        lines.append("💼 持仓: **%s** (自%s, %s)" % (
            result["position"], result["entry_date"], result["entry_reason"]))
    else:
        lines.append("💼 持仓: 空仓")
    lines.append("")
    action = result["action"]
    if action["type"] == "买入":
        lines.append("📋 今日动作: **买入 %s** (%s)" % (action["target"], action["reason"]))
    elif action["type"] == "换仓":
        lines.append("📋 今日动作: **卖出 %s → 买入 %s** (%s)"
                     % (action["old"], action["target"], action["reason"]))
    elif action["type"] == "卖出":
        lines.append("📋 今日动作: **卖出 %s → 空仓** (%s)" % (action["old"], action["reason"]))
    else:
        lines.append("📋 今日动作: 无, %s" % ("继续持有" if result["position"] else "继续空仓等待"))
    lines += ["", "---", "_牢国策略(ETFMomentum Top-1动量轮动) · 独立信号, 仅供参考_"]
    full_md = "\n".join(lines)

    # 大字版: 与动作判定同一来源, 一行只回答当前操作(标的带代码)
    def wc(name):
        return "%s %s" % (laoguo_data.POOL.get(name, ""), name) if name else name
    pos = result["position"]
    if action["type"] == "买入":
        big = "# 🟧 买入 %s" % wc(action["target"])
    elif action["type"] == "换仓":
        big = "# 🔄 换仓 → %s" % wc(action["target"])
    elif action["type"] == "卖出":
        big = "# ⏸ 卖出 %s, 空仓等待" % wc(action["old"])
    elif pos:
        big = "# 🟦 持有 %s" % wc(pos)
    else:
        big = "# ⏸ 空仓等待"
    return full_md, big


def send_both(full_md, big_md, date_str, sleeper=time.sleep):
    """完整版 → (5秒) → 大字版, 逐群发送; 返回全部 errcode 列表 [(群, 消息序, code)]。"""
    results = []
    titles = ("【牢国】动量轮动 %s" % date_str, "【牢国】今日操作 %s" % date_str)
    for gi, (webhook, secret) in enumerate(push_signal.load_targets(), 1):
        url = push_signal.signed_url(webhook, secret) if secret else webhook
        for pi, (title, md) in enumerate(zip(titles, (full_md, big_md)), 1):
            if pi > 1:
                sleeper(5)  # 与主推送一致: 固定顺序, 完整版先展示
            text = md if secret else push_signal._ensure_keyword(md)
            body = json.dumps({"msgtype": "markdown",
                               "markdown": {"title": title, "text": text}},
                              ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    code = json.loads(r.read().decode("utf-8")).get("errcode")
            except Exception as exc:
                code = type(exc).__name__
            results.append((gi, pi, code))
    return results


def run_signal(today=None, offline=False, live_quote=None, sender=None, sleeper=time.sleep,
               state_file=None):
    """主流程; 返回 (ok, 摘要)。测试可注入 offline/live_quote/sender/sleeper/state_file。"""
    now = store.now_local()
    today = today or now.strftime("%Y-%m-%d")
    state_file = state_file or STATE_FILE
    if store.trading_day(today) is False:
        return True, "非交易日, 跳过"
    if store.trading_day(today) is None and now.weekday() >= 5:
        return True, "周末且日历未知, 跳过"
    with store.file_lock(LOCK_FILE):
        state = store.read_json(state_file, {})
        engine = load_engine()
        sha = engine_sha256()
        if not state.get("engine_sha256"):
            state["engine_sha256"] = sha  # 首次部署固化基线指纹
        elif state["engine_sha256"] != sha:
            raise ValueError("引擎文件指纹漂移(期望 %s, 实际 %s); 升级须显式确认"
                             % (state["engine_sha256"][:12], sha[:12]))
        data = laoguo_data.build_inputs(today, offline=offline, live_quote=live_quote)
        n = len(data["dates"])
        cfg = engine.SignalConfig()
        actions, _nav, _log = engine.compute_signals(
            cfg, data["etf_names"], np.array(data["price_mat"], dtype=float),
            np.array(data["mom20_mat"], dtype=float), np.array(data["mom10_mat"], dtype=float),
            np.array(data["sh_mom"], dtype=float), np.array(data["cyb_p"], dtype=float),
            np.array(data["sh_p"], dtype=float), n)

        pos, entry_day, entry_reason = replay_position(actions)
        action = {"type": "无"}
        last = actions[-1] if actions else None
        if last and last[0] == n - 1:
            _t, old, new, reason = last
            if old >= 0 and new >= 0:
                action = {"type": "换仓", "old": data["etf_names"][old],
                          "target": data["etf_names"][new], "reason": reason}
            elif new >= 0:
                action = {"type": "买入", "target": data["etf_names"][new], "reason": reason}
            elif old >= 0:
                action = {"type": "卖出", "old": data["etf_names"][old], "reason": reason}

        result = {
            "position": data["etf_names"][pos] if pos >= 0 else "",
            "entry_date": data["dates"][entry_day] if entry_day >= 0 else "",
            "entry_reason": entry_reason,
            "momentums": {nm: round(data["mom20_mat"][i][-1], 2)
                          for i, nm in enumerate(data["etf_names"])
                          if data["mom20_mat"][i][-1] == data["mom20_mat"][i][-1]},
            "sh_momentum": round(data["sh_mom"][-1], 2) if data["sh_mom"][-1] == data["sh_mom"][-1] else 0.0,
            "action": action,
        }
        quote_note = "离线缓存" if offline else "14:49 盘中价"
        full_md, big_md = build_messages(result, data["etf_names"], today, quote_note)
        digest = hashlib.sha256((full_md + big_md).encode("utf-8")).hexdigest()
        if state.get("date") == today and state.get("digest") == digest and state.get("status") == "sent":
            return True, "今日相同信号已推送, 跳过重复发送"
        sender = sender or send_both
        results = sender(full_md, big_md, today, sleeper=sleeper)
        ok = all(code == 0 for _g, _p, code in results)
        state.update(date=today, digest=digest, status="sent" if ok else "failed",
                     sent_at=now.isoformat(), results=[list(r) for r in results])
        store.atomic_json(state_file, state)
        summary = "动作=%s 持仓=%s 推送=%s" % (
            action.get("type"), result["position"] or "空仓", "成功" if ok else results)
        return ok, summary


def main():
    try:
        ok, summary = run_signal()
        print("牢国策略: %s" % summary)
        return 0 if ok else 1
    except Exception as exc:
        # 数据/计算失败也要告知群里, 绝不静默
        try:
            msg = "⚠️ 牢国策略今日信号计算失败: %s" % type(exc).__name__
            send_both(msg, "# ⚠️ 牢国策略今日计算失败", store.now_local().strftime("%Y-%m-%d"))
        except Exception:
            pass
        print("牢国策略失败: %s" % type(exc).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
