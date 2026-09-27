# -*- coding: utf-8 -*-
"""v9.1-0906 影子版本: v9.1 + QVIX 恐慌日放宽深跌抄底口(v10 第七轮 fz25_m4, 首个 6/6 候选)。

只跟踪不下单——每日与主信号一起计算/推送，积累按信号时价格记账的前向样本。
与 v9.1 唯一差异: QVIX(50ETF期权隐波) 相对前250日 z>=2.5 的恐慌日,
抄底触发从 MOM5<=-8% 放宽到 <=-4%(低于年线20%、锁仓5交易日、其余全部同 v9.1)。

数据: data/qvix50.csv(期权论坛 1.optbbs.com 全量CSV, 每日刷新; 拉取失败用缓存并标注陈旧)
状态: signals/shadow_0906.json(v4原始份额/锁仓/虚拟净值/权威事件，单日封存)
流水: signals/shadow_0906_trades.csv(JSON事件的可重建投影；旧流水随旧状态归档)
历史研究见 v10/README.md；六起点重叠，不能替代独立样本外验证。
"""
import csv
import json
import os
import urllib.request
from datetime import datetime

import strategy
from market_data import UNIVERSE
from live_state import signal_calendar, lock_active
from shadow_account import atomic_json, cached_lines, transact

BASE = os.path.dirname(os.path.abspath(__file__))
QVIX_CSV = os.path.join(BASE, "data", "qvix50.csv")
STATE_FILE = os.path.join(BASE, "signals", "shadow_0906.json")
TRADES_FILE = os.path.join(BASE, "signals", "shadow_0906_trades.csv")
PORTFOLIO = os.path.join(BASE, "portfolio.json")

FEAR_Z = 2.5             # QVIX z 阈值(相对前250日, 严格不含当日)
FEAR_M5 = -0.04          # 恐慌日放宽后的抄底触发
CRASH_M5 = -0.08         # v9.1 基线触发(不变)
CRASH_BELOW_MA = 0.20
LOCK_DAYS = 5
FEE = 0.0002             # 换仓双边万二(与回测同口径)


def _fetch_qvix():
    """全量拉取 QVIX 日线写缓存; 失败静默(用旧缓存)。"""
    try:
        req = urllib.request.Request("http://1.optbbs.com/d/csv/d/k.csv",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read().decode("utf-8", "ignore")
        rows = []
        for line in raw.splitlines()[1:]:
            f = line.split(",")
            if len(f) < 5 or not f[0]:
                continue
            try:
                y, m, d = f[0].split("/")
                rows.append(["%s-%02d-%02d" % (y, int(m), int(d)), float(f[4])])
            except ValueError:
                continue
        rows.sort(key=lambda x: x[0])
        if len(rows) > 500:                       # 防接口改版返回残数据覆盖好缓存
            tmp = "%s.tmp.%d" % (QVIX_CSV, os.getpid())
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
            os.replace(tmp, QVIX_CSV)
    except Exception:
        pass


def qvix_state(sig_date):
    """返回 (z, 值, QVIX日期, 恐慌激活, 备注)。
    严格当日口径: QVIX 必须有 sig_date 当日行才参与恐慌判定;
    当日缺失/拉取失败 → 恐慌判定停用并返回醒目报错备注(绝不退回用前几日数据)。"""
    _fetch_qvix()
    if not os.path.exists(QVIX_CSV):
        return None, None, None, False, "⚠️ 无QVIX数据(拉取失败且无缓存), 恐慌判定停用"
    with open(QVIX_CSV, newline="", encoding="utf-8") as f:
        q = [(r[0], float(r[1])) for r in csv.reader(f) if r]
    q = [x for x in q if x[0] <= sig_date]
    if len(q) < 121:
        return None, None, None, False, "⚠️ QVIX样本不足(<121行), 恐慌判定停用"
    d, v = q[-1]
    win = [x for _, x in q[-251:-1]]
    m = sum(win) / len(win)
    s = (sum((x - m) ** 2 for x in win) / len(win)) ** 0.5
    z = (v - m) / s if s > 0 else 0.0
    if d != sig_date:
        return z, v, d, False, "⚠️ 当日(%s)QVIX未取得, 最新仅到%s —— 恐慌判定停用, 请检查数据源" % (
            sig_date, d)
    return z, v, d, z >= FEAR_Z, ""


def _load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            raise ValueError("影子状态文件损坏，停止记账并保留原文件") from exc
    # 首次部署: 影子与实盘 v9.1 持仓对齐起步
    holding = None
    if os.path.exists(PORTFOLIO):
        try:
            with open(PORTFOLIO, encoding="utf-8") as f:
                holding = json.load(f).get("holding") or None
        except Exception:
            pass
    return {"holding": holding, "nav": 1.0, "last_date": None,
            "lock_code": None, "lock_until": None, "last_advice": ""}


def _save_state(st):
    atomic_json(STATE_FILE, st)


def cached_block(signal_date):
    return cached_lines(STATE_FILE, "v9.1-0906", signal_date)


def block(table, histories, live_prices, signal_date=None, *, quotes=None, action_view=None, qvix=None):
    """原决策规则不变；以共享原始报价/公司行动推进独立v4账户。

    live_prices保留旧调用形状；估值只信任quotes/action_view，缺失时明确失败。
    同日读取已封存卡片，不重新获取QVIX或计算目标。
    """
    last_date = signal_date or datetime.now().strftime("%Y-%m-%d")
    context = {}

    def decide(st):
        cal = signal_calendar(histories, last_date)
        context["qvix"] = qvix if qvix is not None else qvix_state(last_date)
        z, v, qd, fear, note = context["qvix"]
        # 锁仓期 > 抄底检测(基线口 ∪ 恐慌放宽口) > v9.1常规信号。
        target, reason = None, ""
        if lock_active(st.get("lock_code"), st.get("lock_trigger_date"), cal, last_date, LOCK_DAYS):
            target = st["lock_code"]
            reason = "抄底锁仓期(触发%s，满5个交易日恢复决策)" % st["lock_trigger_date"]
        else:
            st["lock_code"] = st["lock_trigger_date"] = None
            m5_thr = FEAR_M5 if fear else CRASH_M5
            cand = next((x for x in table if x[0] != st.get("holding") and x[1]["mom5"] <= m5_thr
                         and x[1].get("dist_ma250", 0) < -CRASH_BELOW_MA), None)
            if cand:
                st["lock_code"] = cand[0]
                st["lock_trigger_date"] = last_date
                target = cand[0]
                reason = "%s抄底: %s MOM5 %.1f%% 低于年线 %.1f%%" % (
                    "恐慌放宽(QVIX z=%.1f)" % z if (fear and cand[1]["mom5"] > CRASH_M5) else "深跌",
                    UNIVERSE[cand[0]][0], cand[1]["mom5"] * 100, cand[1]["dist_ma250"] * 100)
            else:
                target, reason = strategy.decide(table, st.get("holding"))
        return target, reason

    def render(st, target, reason):
        z, v, qd, fear, note = context["qvix"]
        name = UNIVERSE[target][0] if target in UNIVERSE else "空仓"
        st["last_advice"] = "%s %s | %s" % (target, name, reason)
        qvix_line = "QVIX %.2f (z=%+.2f, %s) 恐慌加宽: %s%s" % (
            v or 0.0, z or 0.0, qd or "-", "激活" if fear else "未激活",
            " | " + note if note else "")
        lines = ["-" * 56,
                 "【影子 v9.1-0906】QVIX恐慌加宽抄底 · 虚拟跟踪不下单 (v10第七轮候选)",
                 "  " + qvix_line,
                 "  数据截止: %s | 估值时间: %s（当日封存）" % (last_date, st.get("quote_timestamp") or "-"),
                 "  影子持仓: %s | 虚拟净值 %.4f (自%s) | 建议: %s" % (
                     st.get("holding") or "空仓", st["nav"], st["start_date"], st["last_advice"]),
                 "  跟踪说明: 原始份额记账；分红归属旧持仓，按首次有效价格虚拟再投资"]
        if st.get("valuation_note"):
            lines.append("  " + st["valuation_note"])
        return lines

    return transact(STATE_FILE, TRADES_FILE, _load_state, _save_state, "v9.1-0906", last_date,
                    quotes, action_view, FEE, decide, render)


if __name__ == "__main__":
    from market_data import fetch_history, fetch_realtime, prepare_live_histories
    from signal_store import execution_window
    if not execution_window():
        raise SystemExit("正式影子记账仅在14:50–14:55；其它时段请查询保存信号")
    sig_date = datetime.now().strftime("%Y-%m-%d")
    hs = {c: fetch_history(c) for c in UNIVERSE}
    quotes = fetch_realtime(codes=list(UNIVERSE), detailed=True)
    hs = prepare_live_histories(hs, quotes, sig_date)
    lv = {c: q["price"] for c, q in quotes.items()}
    tbl = strategy.rank(hs, on_date=sig_date)
    from v10_live.data import build_live_view
    view = build_live_view(quotes, sig_date)
    qvix = qvix_state(sig_date)
    if not execution_window():
        raise SystemExit("已超过执行窗口，本次不记账")
    print("\n".join(block(tbl, hs, lv, signal_date=sig_date, quotes=quotes,
                           action_view=view, qvix=qvix)))
