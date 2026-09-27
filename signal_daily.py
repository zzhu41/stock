# -*- coding: utf-8 -*-
"""每日信号脚本：拉行情 -> 算信号 -> 结合账本持仓给建议 -> 写 signals/ 文件。

正式信号仅在14:50–14:55窗口生成；定时任务通过daily_job施加整体时限。
  python3.8 daily_job.py            # 生成并推送，记录失败/送达
  python3.8 daily_job.py --warmup   # 提前只更新历史缓存

v9: 深跌恐慌抄底(危机Alpha)——任何标的 MOM5<=-8% 且低于年线20% 时, 建议抄底买入并锁仓5个交易日
(锁仓状态存 signals/crash_lock.json)。
"""
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import os
import time
from datetime import datetime

import strategy
from market_data import UNIVERSE, CASH, fetch_history, fetch_realtime, prepare_live_histories
from live_state import lock_active
import signal_store

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


def fetch_histories():
    # Independent assets can refresh together; daily_job bounds the whole process.
    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(zip(UNIVERSE, pool.map(fetch_history, UNIVERSE)))


def optional_blocks(table, histories, quotes, signal_date):
    payload = dict(table=table, histories=histories, quotes=quotes, date=signal_date)
    try:
        result = subprocess.run([sys.executable, "-B", "-m", "daily_extras"], cwd=BASE,
            input=json.dumps(payload, allow_nan=False), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        value = json.loads(result.stdout)
        if result.returncode or not isinstance(value.get("lines"), list):
            raise ValueError("optional worker failed")
        return value["lines"]
    except Exception:
        lines = ["-" * 56, "  QDII 溢价: 本次查询未完成，请另行核验"]
        # Optional workers may have committed before losing their stdout reply.
        for name, filename in (("v9.1-0906", "shadow_0906.json"), ("v9.2", "shadow_v92.json"),
                               ("V10-H", "shadow_v10.json")):
            try:
                if name == "V10-H":
                    import shadow_v10
                    saved = shadow_v10.saved_block(signal_date)
                else:
                    import importlib
                    module = importlib.import_module("shadow_0906" if name == "v9.1-0906" else "shadow_v92")
                    saved = module.cached_block(signal_date)
                if saved:
                    lines.extend(saved)
                    continue
            except Exception:
                pass
            lines.extend(["-" * 56, "⚠️ 影子 %s 计算失败: 辅助计算超时或未完成，本次无新建议" % name])
        return lines


def main(now=None):
    with signal_store.file_lock(os.path.join(SIGNAL_DIR, "generation.lock")):
        return generate(now=now)


def generate(now=None):
    fixed_now = now
    now = now or signal_store.now_local()
    if not signal_store.execution_window(now):
        raise ValueError("正式信号仅在14:50–14:55生成；其它时段请查看历史信号")
    signal_date = now.strftime("%Y-%m-%d")
    journal = signal_store.load_journal(SIGNAL_DIR)
    if journal and journal["date"] > signal_date:
        raise ValueError("已有信号日期晚于当前时钟，禁止倒退覆盖")
    if journal and journal["date"] == signal_date:
        signal_store.repair_projections(journal, SIGNAL_DIR)
        print(journal["text"])
        return journal["target"]
    histories = fetch_histories()
    quotes = fetch_realtime(codes=list(UNIVERSE), detailed=True)
    histories = prepare_live_histories(histories, quotes, signal_date, now=fixed_now or signal_store.now_local())
    live_prices = {c: q["price"] for c, q in quotes.items()}

    pf = load_portfolio()
    holding = pf.get("holding") or None

    table = strategy.rank(histories, on_date=signal_date)
    target, reason, act, tbl = strategy.advice(table, holding)

    # --- v9 深跌恐慌抄底: 锁仓期优先, 其次检测新触发 ---
    last_date = signal_date
    cal = [r[0] for r in histories["510300"]]
    lock = None
    if journal is not None:
        lock = journal.get("crash_lock")
    elif os.path.exists(CRASH_LOCK):
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
    if lock and lock_active(lock.get("code"), lock.get("trigger_date"), cal, last_date,
                            CRASH_LOCK_DAYS):
        # 锁仓期: 强制继续持有抄底标的
        code = lock["code"]
        target, reason = code, "恐慌抄底锁仓期(触发%s，满5个交易日恢复决策)" % lock["trigger_date"]
        act = "%s %s %s (抄底锁仓，触发%s)" % (
            "继续持有" if holding == code else "买入", code, UNIVERSE[code][0], lock["trigger_date"])
        if holding and holding != code:
            act = "卖出 %s %s -> " % (holding, UNIVERSE[holding][0]) + act
    else:
        lock = None
        # 检测新抄底触发(全池, 含持仓外的)
        cand = next((x for x in table if x[0] != holding and x[1]["mom5"] <= CRASH_MOM5
                     and x[1].get("dist_ma250", 0) < -CRASH_BELOW_MA), None)
        if cand:
            target = cand[0]
            reason = "恐慌抄底: %s MOM5 %.1f%% 且低于年线 %.1f%%" % (
                UNIVERSE[cand[0]][0], cand[1]["mom5"] * 100, cand[1]["dist_ma250"] * 100)
            act = "买入 %s %s (抄底! 锁仓5个交易日，触发%s)" % (cand[0], UNIVERSE[cand[0]][0], last_date)
            if holding:
                act = "卖出 %s %s -> " % (holding, UNIVERSE[holding][0]) + act
            lock = {"code": cand[0], "trigger_date": last_date, "lock_days": CRASH_LOCK_DAYS}

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
    lines += optional_blocks(table, histories, quotes, signal_date)
    lines.append("操作后请记账: python3.8 record.py buy|sell <代码> <价格> <金额元>")

    finished = fixed_now or signal_store.now_local()
    if not signal_store.execution_window(finished):
        raise ValueError("生成已超过14:55截止时间，不发布迟到买入指令")
    # Verify again after optional work; a stale snapshot cannot become a new card.
    prepare_live_histories(histories, quotes, signal_date, now=finished)
    lines[1] = "动量轮动信号 | 生成 %s | 数据截止 %s" % (finished.strftime("%Y-%m-%d %H:%M:%S"), last_date)
    lines.insert(2, "行情时间: %s" % min(q["timestamp"] for q in quotes.values()))
    text = "\n".join(lines) + "\n"
    signal_store.publish(text, last_date, target, lock, SIGNAL_DIR)
    print(text)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", action="store_true", help="Only refresh history caches; never create signals/accounts")
    args = parser.parse_args()
    if args.warmup:
        fetch_histories()
    else:
        main()
