# -*- coding: utf-8 -*-
"""v9.2 影子版本: v9.1 + 双通道恐慌抄底并集(v10 第八轮候选 fz25_cv)。

只跟踪不下单——每日与主信号一起计算/推送，积累按信号时价格记账的前向样本。
与 v9.1 唯一差异: 深跌抄底触发口从 1 个变 3 个并集(其余规则全部不变):
  ① 原口径: MOM5<=-8% 且低于年线20%
  ② QVIX恐慌(0906同款): QVIX z>=2.5(相对前250日, 严格当日口径) 当日 MOM5<=-4% 且低于年线20%
  ③ 量能恐慌(本轮新增): 当日量>=2倍前20日均量 且 MOM5<=-4% 且低于年线10%
     量比用报价中的当日累计成交量；回测用全日量，两者的差异尚需前向验证。

数据: QVIX 复用 shadow_0906.qvix_state(严格当日口径, 缺失即停用恐慌判定);
      价格、量比和信号日由 market_data 的带日期报价统一对齐。
状态: signals/shadow_v92.json(v4原始份额/锁仓/权威事件与当天封存卡片)
流水: signals/shadow_v92_trades.csv(JSON事件的可重建投影；旧流水随旧状态归档)
旧研究结果归档于 v10/README.md；修复后的评估见 reports/correctness_review.md。
历史扫描结果不代表独立样本外验证，也不是修复后前向业绩。
"""
import json
import os
from datetime import datetime

import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD
from live_state import signal_calendar, lock_active
from shadow_account import atomic_json, cached_lines, transact

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE, "signals", "shadow_v92.json")
TRADES_FILE = os.path.join(BASE, "signals", "shadow_v92_trades.csv")
PORTFOLIO = os.path.join(BASE, "portfolio.json")

FEAR_Z = 2.5             # QVIX z 阈值(同 0906)
FEAR_M5 = -0.04          # ② QVIX恐慌日放宽后的 MOM5 口
FEAR_DEP = 0.20          # ② 低于年线深度(同基线)
CV_VOL = 2.0             # ③ 量比阈值(当日量/前20日均量)
CV_M5 = -0.04            # ③ 量能恐慌放宽后的 MOM5 口
CV_DEP = 0.10            # ③ 低于年线深度(放宽)
CRASH_M5 = -0.08         # ① v9.1 基线触发(不变)
CRASH_BELOW_MA = 0.20
LOCK_DAYS = 5
FEE = 0.0002             # 换仓双边万二(与回测同口径)
CRASH_POOL = set(STOCK_POOL) | set(GLOBAL_POOL) | {GOLD}


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
    return cached_lines(STATE_FILE, "v9.2", signal_date)


def _vol_ratio20(histories, code, last_date):
    """当日(盘中)量 / 前20日均量。数据不足返回 0(不触发)。"""
    rows = [r for r in histories.get(code, []) if r[0] <= last_date and len(r) > 3]
    if not rows or rows[-1][0] != last_date:
        raise ValueError("%s 缺少 %s 当日成交量" % (code, last_date))
    if len(rows) < 21:
        return 0.0
    avg20 = sum(r[3] for r in rows[-21:-1]) / 20.0
    return rows[-1][3] / avg20 if avg20 > 0 else 0.0


def _trigger_channels(ind, vr, fear):
    """返回触发的通道列表(可多个): '①深跌' / '②QVIX恐慌' / '③量能恐慌'。"""
    ch = []
    if ind["mom5"] <= CRASH_M5 and ind.get("dist_ma250", 0) < -CRASH_BELOW_MA:
        ch.append("①深跌")
    if fear and ind["mom5"] <= FEAR_M5 and ind.get("dist_ma250", 0) < -FEAR_DEP:
        ch.append("②QVIX恐慌")
    if vr >= CV_VOL and ind["mom5"] <= CV_M5 and ind.get("dist_ma250", 0) < -CV_DEP:
        ch.append("③量能恐慌")
    return ch


def block(table, histories, live_prices, signal_date=None, *, quotes=None, action_view=None, qvix=None):
    """原三通道规则不变；共享原始行情记账，锁内封存同日目标和卡片。"""
    import shadow_0906                       # 复用 QVIX 严格当日口径(同源同窗同阈值)
    last_date = signal_date or datetime.now().strftime("%Y-%m-%d")
    context = {}

    def decide(st):
        cal = signal_calendar(histories, last_date)
        context["qvix"] = qvix if qvix is not None else shadow_0906.qvix_state(last_date)
        z, v, qd, fear, note = context["qvix"]
        # 2) 决策: 锁仓期 > 抄底检测(① ∪ ② ∪ ③, score 最强者, 排除当前持仓——同 lab 口径) > v9.1 常规信号
        target, reason = None, ""
        if lock_active(st.get("lock_code"), st.get("lock_trigger_date"), cal, last_date, LOCK_DAYS):
            target = st["lock_code"]
            reason = "抄底锁仓期(触发%s，满5个交易日恢复决策)" % st["lock_trigger_date"]
        else:
            st["lock_code"] = st["lock_trigger_date"] = None
            cand, cand_ch, cand_vr = None, [], 0.0
            for x in table:
                if x[0] not in CRASH_POOL or x[0] == st.get("holding"):
                    continue
                vr = _vol_ratio20(histories, x[0], last_date)
                ch = _trigger_channels(x[1], vr, fear)
                if ch:
                    cand, cand_ch, cand_vr = x, ch, vr
                    break
            if cand:
                st["lock_code"] = cand[0]
                st["lock_trigger_date"] = last_date
                target = cand[0]
                reason = "抄底(%s): %s MOM5 %.1f%% 低于年线 %.1f%% 量比 %.1f" % (
                    "∪".join(cand_ch), UNIVERSE[cand[0]][0], cand[1]["mom5"] * 100,
                    cand[1]["dist_ma250"] * 100, cand_vr)
            else:
                target, reason = strategy.decide(table, st.get("holding"))
        return target, reason

    def render(st, target, reason):
        z, v, qd, fear, note = context["qvix"]
        name = UNIVERSE[target][0] if target in UNIVERSE else "空仓"
        st["last_advice"] = "%s %s | %s" % (target, name, reason)
        qvix_line = "QVIX %.2f (z=%+.2f, %s) 恐慌: %s%s" % (
            v or 0.0, z or 0.0, qd or "-", "激活" if fear else "未激活",
            " | " + note if note else "")
        lines = ["-" * 56,
                 "【影子 v9.2】QVIX∪量能恐慌抄底 · 虚拟跟踪不下单 (v10第八轮候选 fz25_cv)",
                 "  " + qvix_line,
                 "  抄底三口并集: ①MOM5≤-8%∧低年线20% ②恐慌日MOM5≤-4%∧低年线20% ③量比≥2∧低年线10%且MOM5≤-4%",
                 "  数据截止: %s | 估值时间: %s（当日封存）" % (last_date, st.get("quote_timestamp") or "-"),
                 "  影子持仓: %s | 虚拟净值 %.4f (自%s) | 建议: %s" % (
                     st.get("holding") or "空仓", st["nav"], st["start_date"], st["last_advice"]),
                 "  跟踪说明: 原始份额记账；分红归属旧持仓，按首次有效价格虚拟再投资"]
        if st.get("valuation_note"):
            lines.append("  " + st["valuation_note"])
        return lines

    return transact(STATE_FILE, TRADES_FILE, _load_state, _save_state, "v9.2", last_date,
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
    import shadow_0906
    view = build_live_view(quotes, sig_date)
    qvix = shadow_0906.qvix_state(sig_date)
    if not execution_window():
        raise SystemExit("已超过执行窗口，本次不记账")
    print("\n".join(block(tbl, hs, lv, signal_date=sig_date, quotes=quotes,
                           action_view=view, qvix=qvix)))
