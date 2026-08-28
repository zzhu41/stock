# -*- coding: utf-8 -*-
"""账本：记录实际成交，维护持仓状态。用法:
  python3.8 record.py buy  <代码> <价格> <金额元>     # 买入(全仓语义: 用全部可用资金)
  python3.8 record.py sell <代码> <价格>              # 卖出全部持仓
  python3.8 record.py status                          # 查看当前状态
  python3.8 record.py history                         # 查看操作历史
"""
import csv
import json
import os
import sys
from datetime import datetime

from market_data import UNIVERSE, fetch_realtime, fetch_history

BASE = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO = os.path.join(BASE, "portfolio.json")
TRADES = os.path.join(BASE, "trades.csv")


def load():
    if os.path.exists(PORTFOLIO):
        with open(PORTFOLIO, encoding="utf-8") as f:
            return json.load(f)
    return {"holding": None, "shares": 0, "cost": 0.0, "entry_date": None,
            "cash": 0.0, "realized": 0.0}


def save(pf):
    """原子写(临时文件+rename): 防与信号/跟踪任务并发读到半截 JSON。"""
    tmp = "%s.tmp.%d" % (PORTFOLIO, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PORTFOLIO)


def log_trade(action, code, price, shares, amount):
    new = not os.path.exists(TRADES)
    with open(TRADES, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "action", "code", "name", "price", "shares", "amount"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), action, code,
                    UNIVERSE.get(code, ("?",))[0], price, shares, amount])


def buy(pf, code, price, amount):
    if code not in UNIVERSE:
        sys.exit("未知代码 %s，可选: %s" % (code, ",".join(UNIVERSE)))
    if pf["holding"]:
        sys.exit("当前持有 %s，请先 sell" % pf["holding"])
    if pf["cash"] and abs(amount - pf["cash"]) / max(pf["cash"], 1) > 0.05:
        print("提示: 买入金额 %.0f 与可用资金 %.0f 不一致，按实际买入额记账" % (amount, pf["cash"]))
    shares = int(amount / price / 100) * 100   # ETF 100份整数倍
    if shares <= 0:
        sys.exit("金额不足买100份")
    real_amount = shares * price
    pf.update(holding=code, shares=shares, cost=price,
              entry_date=datetime.now().strftime("%Y-%m-%d"),
              cash=round(pf["cash"] - real_amount, 2) if pf["cash"] else 0.0)
    log_trade("buy", code, price, shares, real_amount)
    save(pf)
    print("已记录买入: %s %s %.3f元 x %d份 = %.0f元" % (code, UNIVERSE[code][0], price, shares, real_amount))


def sell(pf, code, price):
    if pf["holding"] != code:
        sys.exit("当前未持有 %s (持仓: %s)" % (code, pf["holding"]))
    amount = pf["shares"] * price
    pnl = (price / pf["cost"] - 1) * 100
    pf["realized"] += pf["shares"] * (price - pf["cost"])
    pf["cash"] = round(pf["cash"] + amount, 2)
    log_trade("sell", code, price, pf["shares"], amount)
    print("已记录卖出: %s %s %.3f元 x %d份 = %.0f元 | 本笔盈亏 %+.2f%% | 累计已实现 %.0f元"
          % (code, UNIVERSE[code][0], price, pf["shares"], amount, pnl, pf["realized"]))
    pf.update(holding=None, shares=0, cost=0.0, entry_date=None)
    save(pf)


def status(pf):
    print("== 账本状态 ==")
    if pf["holding"]:
        code = pf["holding"]
        try:
            cur = fetch_realtime([code]).get(code, (None, None))[1]
        except Exception:
            cur = fetch_history(code)[-1][2]
        mv = pf["shares"] * cur
        pnl = (cur / pf["cost"] - 1) * 100
        print("持仓: %s %s | %d份 | 成本 %.3f (%s) | 现价 %.3f | 市值 %.0f | 浮盈 %+.2f%%"
              % (code, UNIVERSE[code][0], pf["shares"], pf["cost"],
                 pf["entry_date"], cur, mv, pnl))
        print("市值+现金合计: %.0f | 累计已实现盈亏: %.0f" % (mv + pf["cash"], pf["realized"]))
    else:
        print("空仓 | 现金 %.0f | 累计已实现盈亏 %.0f" % (pf["cash"], pf["realized"]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pf = load()
    cmd = sys.argv[1]
    if cmd == "buy" and len(sys.argv) == 5:
        buy(pf, sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    elif cmd == "sell" and len(sys.argv) == 4:
        sell(pf, sys.argv[2], float(sys.argv[3]))
    elif cmd == "status":
        status(pf)
    elif cmd == "history":
        with open(TRADES, encoding="utf-8") as f:
            print(f.read())
    else:
        sys.exit(__doc__)
