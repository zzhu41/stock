# -*- coding: utf-8 -*-
"""每日信号脚本：拉行情 -> 算信号 -> 结合账本持仓给建议 -> 写 signals/ 文件。

盘中(9:30-15:00)用实时价计算, 盘后自动用收盘价。用法:
  python3.8 signal_daily.py          # 打印信号并写入 signals/

v9: 深跌恐慌抄底(危机Alpha)——任何标的 MOM5<=-8% 且低于年线20% 时, 建议抄底买入并锁仓5个交易日
(锁仓状态存 signals/crash_lock.json)。
"""
import json
import os
import time
from datetime import datetime

import strategy
from market_data import UNIVERSE, CASH, fetch_history, fetch_realtime, prepare_live_histories
from live_state import lock_active

BASE = os.path.dirname(os.path.abspath(__file__))
SIGNAL_DIR = os.path.join(BASE, "signals")
PORTFOLIO = os.path.join(BASE, "portfolio.json")
CRASH_LOCK = os.path.join(SIGNAL_DIR, "crash_lock.json")
CRASH_MOM5 = -0.08        # v9 抄底触发: MOM5<=-8%
CRASH_BELOW_MA = 0.20     # 且低于年线20%
CRASH_LOCK_DAYS = 5       # 锁仓5个交易日


def _atomic_write(path, text):
    """原子写(临时文件+rename): 防钉钉机器人并发读到半截文件。"""
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load_portfolio():
    if os.path.exists(PORTFOLIO):
        with open(PORTFOLIO, encoding="utf-8") as f:
            return json.load(f)
    return {"holding": None, "shares": 0, "cost": 0.0, "entry_date": None}


def main():
    now = datetime.now()
    histories = {}
    for code in UNIVERSE:
        histories[code] = fetch_history(code)
        time.sleep(1.0)
    signal_date = now.strftime("%Y-%m-%d")
    quotes = fetch_realtime(codes=list(UNIVERSE), detailed=True)
    histories = prepare_live_histories(histories, quotes, signal_date)
    live_prices = {c: q["price"] for c, q in quotes.items()}

    pf = load_portfolio()
    holding = pf.get("holding") or None

    table = strategy.rank(histories, on_date=signal_date)
    target, reason, act, tbl = strategy.advice(table, holding)

    # --- v9 深跌恐慌抄底: 锁仓期优先, 其次检测新触发 ---
    last_date = signal_date
    cal = [r[0] for r in histories["510300"]]
    lock = None
    if os.path.exists(CRASH_LOCK):
        try:
            with open(CRASH_LOCK, encoding="utf-8") as f:
                lock = json.load(f)
        except Exception as exc:
            raise ValueError("抄底锁仓文件无法读取，停止信号以免丢失锁仓") from exc
    lock_note = ""
    if lock and "lock_until" in lock:
        # 旧实现的截止日等于触发日；只采用已保存的真实触发记录，绝不猜未来日历。
        lock_active(lock.get("code"), lock.get("trigger_date"), cal, last_date, CRASH_LOCK_DAYS)
        lock_note = "旧锁仓已按记录的触发日 %s 迁移为交易日计数" % lock["trigger_date"]
        lock = {"code": lock["code"], "trigger_date": lock["trigger_date"],
                "lock_days": CRASH_LOCK_DAYS}
        _atomic_write(CRASH_LOCK, json.dumps(lock, ensure_ascii=False))
    if lock and lock_active(lock.get("code"), lock.get("trigger_date"), cal, last_date,
                            CRASH_LOCK_DAYS):
        # 锁仓期: 强制继续持有抄底标的
        code = lock["code"]
        target, reason = code, "恐慌抄底锁仓期(触发%s，满5个交易日恢复决策)" % lock["trigger_date"]
        act = "%s %s %s (抄底锁仓，触发%s)" % (
            "继续持有" if holding == code else "买入", code, UNIVERSE[code][0], lock["trigger_date"])
    else:
        # 检测新抄底触发(全池, 含持仓外的)
        cand = next((x for x in table if x[0] != holding and x[1]["mom5"] <= CRASH_MOM5
                     and x[1].get("dist_ma250", 0) < -CRASH_BELOW_MA), None)
        if cand:
            target = cand[0]
            reason = "恐慌抄底: %s MOM5 %.1f%% 且低于年线 %.1f%%" % (
                UNIVERSE[cand[0]][0], cand[1]["mom5"] * 100, cand[1]["dist_ma250"] * 100)
            act = "买入 %s %s (抄底! 锁仓5个交易日，触发%s)" % (cand[0], UNIVERSE[cand[0]][0], last_date)
            if not os.path.isdir(SIGNAL_DIR):
                os.makedirs(SIGNAL_DIR)
            _atomic_write(CRASH_LOCK, json.dumps(
                {"code": cand[0], "trigger_date": last_date, "lock_days": CRASH_LOCK_DAYS}))

    lines = [
        "=" * 56,
        "动量轮动信号 | 生成 %s | 数据截止 %s" % (now.strftime("%Y-%m-%d %H:%M"), last_date),
        "=" * 56,
        tbl,
        "-" * 56,
    ]
    if holding:
        cur = live_prices[holding]
        pnl = (cur / pf["cost"] - 1) * 100 if pf.get("cost") else 0.0
        lines.append("当前持仓: %s %s | 成本 %.3f (%s) | 现价 %.3f | 浮动盈亏 %+.2f%%"
                     % (holding, UNIVERSE[holding][0], pf["cost"],
                        pf.get("entry_date") or "-", cur, pnl))
    else:
        lines.append("当前持仓: 空仓")
    lines += ["★ 建议: %s" % act, "  依据: %s" % reason, "-" * 56]
    if lock_note:
        lines.append("  " + lock_note)
    try:                                # QDII 溢价提示(>2%勿追); 失败不影响信号主流程
        import premium
        lines += premium.signal_block()
    except Exception:
        pass
    try:                                # v9.1-0906 影子版本(QVIX恐慌加宽抄底, 虚拟跟踪不下单)
        import shadow_0906
        lines += shadow_0906.block(table, histories, live_prices, signal_date=signal_date)
    except Exception as e:              # 及时报错进信号卡(钉钉可见), 不阻断主信号
        lines += ["-" * 56, "⚠️ 影子 v9.1-0906 计算失败: %r" % e]
    try:                                # v9.2 影子版本(QVIX∪量能恐慌抄底, 虚拟跟踪不下单)
        import shadow_v92
        lines += shadow_v92.block(table, histories, live_prices, signal_date=signal_date)
    except Exception as e:              # 不阻断主信号
        lines += ["-" * 56, "⚠️ 影子 v9.2 计算失败: %r" % e]
    try:                                # V10-H: 独立数据、配置和虚拟账户，最多等待60秒
        import shadow_v10
        lines += shadow_v10.block(quotes, signal_date=signal_date)
    except Exception as e:
        lines += ["-" * 56, "⚠️ 影子 V10-H 计算失败: %r" % e]
    lines.append("操作后请记账: python3.8 record.py buy|sell <代码> <价格> <金额元>")

    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(SIGNAL_DIR):
        os.makedirs(SIGNAL_DIR)
    for path in (os.path.join(SIGNAL_DIR, "%s.txt" % last_date),
                 os.path.join(SIGNAL_DIR, "latest.txt")):
        _atomic_write(path, text + "\n")
    return target


if __name__ == "__main__":
    main()
