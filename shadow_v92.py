# -*- coding: utf-8 -*-
"""v9.2 影子版本: v9.1 + 双通道恐慌抄底并集(v10 第八轮候选 fz25_cv, 六起点全赢且支配双亲本)。

只跟踪不下单——每日与主信号一起计算/推送, 攒前向样本, 复现 ~69% 5日胜率画像再议转正。
与 v9.1 唯一差异: 深跌抄底触发口从 1 个变 3 个并集(其余规则全部不变):
  ① 原口径: MOM5<=-8% 且低于年线20%
  ② QVIX恐慌(0906同款): QVIX z>=2.5(相对前250日, 严格当日口径) 当日 MOM5<=-4% 且低于年线20%
  ③ 量能恐慌(本轮新增): 当日量>=2倍前20日均量 且 MOM5<=-4% 且低于年线10%
     机制: 深跌+巨量=被迫卖盘(赎回/强平)出清的可观测签名。量比用 14:50 盘中量(约95%全日),
     回测用全日量——口径差异正是前向验证的核心项之一。

数据: QVIX 复用 shadow_0906.qvix_state(严格当日口径, 缺失即停用恐慌判定);
      量比用 market_data 增量缓存的当日(盘中)量, 无前视。
状态: signals/shadow_v92.json(幂等: 同一交易日重跑不重复记账)
流水: signals/shadow_v92_trades.csv(date,from,to,price,nav)
回测: 2014起终值 1.286x v9.1, 六起点 1.286/1.286/1.388/1.258/1.258/1.124 全赢,
      回撤不变(-26.99%); 证据链 v10/README.md 第八轮
"""
import csv
import json
import os

import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD

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
        except Exception:
            pass
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
    tmp = "%s.tmp.%d" % (STATE_FILE, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)


def _log_trade(date, frm, to, price, nav):
    new = not os.path.exists(TRADES_FILE)
    with open(TRADES_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "from", "to", "price", "nav"])
        w.writerow([date, frm or "", to, "%.4f" % price, "%.6f" % nav])


def _vol_ratio20(histories, code, last_date):
    """当日(盘中)量 / 前20日均量。数据不足返回 0(不触发)。"""
    rows = [r for r in histories.get(code, []) if r[0] <= last_date and len(r) > 3]
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


def block(table, histories, live_prices):
    """计算影子信号并推进状态(同日幂等), 返回信号文本行列表。异常由调用方兜底。"""
    import shadow_0906                       # 复用 QVIX 严格当日口径(同源同窗同阈值)
    cal = [r[0] for r in histories["510300"]]
    last_date = cal[-1]
    close_of = {c: {r[0]: r[2] for r in rows} for c, rows in histories.items()}
    st = _load_state()
    z, v, qd, fear, note = shadow_0906.qvix_state(last_date)
    info = {c: ind for c, ind in table}

    rerun = st.get("last_date") == last_date     # 同日重跑: 只展示不推进
    if not rerun:
        # 1) 净值推进: 昨日持仓吃 prev->last 收益(与回测 T 日收盘口径一致)
        prev_date = st.get("last_date")
        h = st.get("holding")
        if prev_date and h and prev_date in close_of.get(h, {}) and last_date in close_of.get(h, {}):
            st["nav"] *= close_of[h][last_date] / close_of[h][prev_date]

        # 2) 决策: 锁仓期 > 抄底检测(① ∪ ② ∪ ③, score 最强者, 排除当前持仓——同 lab 口径) > v9.1 常规信号
        target, reason = None, ""
        if st.get("lock_until") and st["lock_until"] >= last_date and st.get("lock_code") in UNIVERSE:
            target = st["lock_code"]
            reason = "抄底锁仓期(至%s)" % st["lock_until"]
        else:
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
                idx = cal.index(last_date)
                st["lock_code"] = cand[0]
                st["lock_until"] = cal[min(idx + LOCK_DAYS, len(cal) - 1)]
                target = cand[0]
                reason = "抄底(%s): %s MOM5 %.1f%% 低于年线 %.1f%% 量比 %.1f" % (
                    "∪".join(cand_ch), UNIVERSE[cand[0]][0], cand[1]["mom5"] * 100,
                    cand[1]["dist_ma250"] * 100, cand_vr)
            else:
                target, reason = strategy.decide(table, st.get("holding"))
        # 3) 换仓记账(虚拟); 首次部署仅对齐起步, 不计费不记流水
        if target != st.get("holding"):
            if prev_date:
                st["nav"] *= 1 - FEE
                price = live_prices.get(target) or close_of.get(target, {}).get(last_date, 0.0)
                _log_trade(last_date, st.get("holding"), target, price, st["nav"])
            st["holding"] = target
        if not st.get("start_date"):
            st["start_date"] = last_date
        st["last_date"] = last_date
        name = UNIVERSE[target][0] if target in UNIVERSE else "空仓"
        st["last_advice"] = "%s %s | %s" % (target, name, reason)
        _save_state(st)

    qvix_line = "QVIX %.2f (z=%+.2f, %s) 恐慌: %s%s" % (
        v or 0.0, z or 0.0, qd or "-", "激活" if fear else "未激活",
        " | " + note if note else "")
    return [
        "-" * 56,
        "【影子 v9.2】QVIX∪量能恐慌抄底 · 虚拟跟踪不下单 (v10第八轮候选 fz25_cv)",
        "  " + qvix_line,
        "  抄底三口并集: ①MOM5≤-8%∧低年线20% ②恐慌日MOM5≤-4%∧低年线20% ③量比≥2∧MOM5≤-4%∧低年线10%",
        "  影子持仓: %s | 虚拟净值 %.4f (自%s) | 建议: %s" % (
            st.get("holding") or "空仓", st.get("nav", 1.0),
            st.get("start_date") or st.get("last_date"), st.get("last_advice", "")),
    ]


if __name__ == "__main__":
    from market_data import fetch_history, fetch_realtime
    hs = {c: fetch_history(c) for c in UNIVERSE}
    lv = {c: p for c, (n, p) in fetch_realtime().items()}
    tbl = strategy.rank(hs, live_prices=lv)
    print("\n".join(block(tbl, hs, lv)))
