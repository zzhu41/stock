# -*- coding: utf-8 -*-
"""v9.1-0906 影子版本: v9.1 + QVIX 恐慌日放宽深跌抄底口(v10 第七轮 fz25_m4, 首个 6/6 候选)。

只跟踪不下单——每日与主信号一起计算/推送, 攒前向样本, 前向复现 3/4 胜率画像再议转正。
与 v9.1 唯一差异: QVIX(50ETF期权隐波) 相对前250日 z>=2.5 的恐慌日,
抄底触发从 MOM5<=-8% 放宽到 <=-4%(低于年线20%、锁仓5交易日、其余全部同 v9.1)。

数据: data/qvix50.csv(期权论坛 1.optbbs.com 全量CSV, 每日刷新; 拉取失败用缓存并标注陈旧)
状态: signals/shadow_0906.json(持仓/锁仓/虚拟净值, 幂等: 同一交易日重跑不重复记账)
流水: signals/shadow_0906_trades.csv(date,from,to,price,nav)
回测: 2014起终值 1.132x v9.1, 六起点全赢, 回撤不变; 证据链 v10/README.md 第七轮
"""
import csv
import json
import os
import urllib.request

import strategy
from market_data import UNIVERSE

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


def block(table, histories, live_prices):
    """计算影子信号并推进状态(同日幂等), 返回信号文本行列表。异常由调用方兜底。"""
    cal = [r[0] for r in histories["510300"]]
    last_date = cal[-1]
    close_of = {c: {r[0]: r[2] for r in rows} for c, rows in histories.items()}
    st = _load_state()
    z, v, qd, fear, note = qvix_state(last_date)
    info = {c: ind for c, ind in table}

    rerun = st.get("last_date") == last_date     # 同日重跑: 只展示不推进
    if not rerun:
        # 1) 净值推进: 昨日持仓吃 prev->last 收益(与回测 T 日收盘口径一致)
        prev_date = st.get("last_date")
        h = st.get("holding")
        if prev_date and h and prev_date in close_of.get(h, {}) and last_date in close_of.get(h, {}):
            st["nav"] *= close_of[h][last_date] / close_of[h][prev_date]

        # 2) 决策: 锁仓期 > 抄底检测(基线口 ∪ 恐慌放宽口) > v9.1 常规信号
        target, reason = None, ""
        if st.get("lock_until") and st["lock_until"] >= last_date and st.get("lock_code") in UNIVERSE:
            target = st["lock_code"]
            reason = "抄底锁仓期(至%s)" % st["lock_until"]
        else:
            m5_thr = FEAR_M5 if fear else CRASH_M5
            cand = next((x for x in table if x[1]["mom5"] <= m5_thr
                         and x[1].get("dist_ma250", 0) < -CRASH_BELOW_MA), None)
            if cand and cand[0] != st.get("holding"):
                idx = cal.index(last_date)
                st["lock_code"] = cand[0]
                st["lock_until"] = cal[min(idx + LOCK_DAYS, len(cal) - 1)]
                target = cand[0]
                reason = "%s抄底: %s MOM5 %.1f%% 低于年线 %.1f%%" % (
                    "恐慌放宽(QVIX z=%.1f)" % z if (fear and cand[1]["mom5"] > CRASH_M5) else "深跌",
                    UNIVERSE[cand[0]][0], cand[1]["mom5"] * 100, cand[1]["dist_ma250"] * 100)
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

    qvix_line = "QVIX %.2f (z=%+.2f, %s) 恐慌加宽: %s%s" % (
        v or 0.0, z or 0.0, qd or "-", "激活" if fear else "未激活",
        " | " + note if note else "")
    return [
        "-" * 56,
        "【影子 v9.1-0906】QVIX恐慌加宽抄底 · 虚拟跟踪不下单 (v10第七轮候选)",
        "  " + qvix_line,
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
