# -*- coding: utf-8 -*-
"""第十轮: 激进组合矩阵 (v7基线)。含次方量化机制(trailing止盈+冷却/加权斜率)与社区思路。"""
import backtest as bt
import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
S = "2014-01-01"


def show(name, r):
    yd = dict(bt.yearly(r["daily"]))
    print("%s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%% | 换手 %d || 18:%+.0f 20:%+.0f 22:%+.0f 25:%+.0f 26:%+.0f"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100,
             r["switches"], yd.get("2018", 0) * 100, yd.get("2020", 0) * 100,
             yd.get("2022", 0) * 100, yd.get("2025", 0) * 100, yd.get("2026", 0) * 100), flush=True)


def backtest_weekly(start):
    """周频信号: 每周最后一个交易日才决策, 其余天维持。"""
    days = [d for d in calendar if start <= d]
    close_of = {c: {r[0]: r[2] for r in rows} for c, rows in histories.items()}
    nav, peak, max_dd = 1.0, 1.0, 0.0
    holding, switches, daily = None, 0, []
    for i, d in enumerate(days):
        if i > 0 and holding is not None:
            prev, cur = close_of[holding].get(days[i - 1]), close_of[holding].get(d)
            if prev and cur:
                nav *= cur / prev
        # 周五(或本周最后交易日)才决策: 下周是否同周判断用 iso 周
        decide_today = (i == len(days) - 1) or (days[i + 1][:4] + days[i + 1][5:7] + "%02d" % 0 or True)
        # 简化: 下周第一天不同(iso年周不同)则今天是本周最后一天
        import datetime as _dt
        cur_w = _dt.date(*map(int, d.split("-"))).isocalendar()[:2]
        nxt_w = _dt.date(*map(int, days[i + 1].split("-"))).isocalendar()[:2] if i + 1 < len(days) else (0, 0)
        if cur_w != nxt_w:
            table = strategy.rank(histories, on_date=d)
            target, _ = strategy.decide(table, holding, 99)
        else:
            target = holding
        if target != holding:
            if i > 0:
                switches += 1
                nav *= (1 - bt.FEE * 2)
            holding = target
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1)
        daily.append((d, nav, holding))
    n = len(daily)
    rets = [daily[i][1] / daily[i - 1][1] - 1 for i in range(1, n)]
    mean = sum(rets) / len(rets)
    std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
    years = n / 244
    return {"nav": nav, "ann": nav ** (1 / years) - 1, "max_dd": max_dd,
            "sharpe": mean / std * 244 ** 0.5 if std else 0, "switches": switches, "daily": daily}


print("== 基线 ==")
show("v7 基线             ", bt.backtest(histories, calendar, start=S))

print("\n== A. 进场动量阈值(强趋势才进) ==")
for x in (0.02, 0.03, 0.05):
    show("MOM20>+%.0f%%才进场  " % (x * 100), bt.backtest(histories, calendar, start=S, enter_mom_min=x))

print("\n== B. 动量平方score(激进追强) ==")
show("mom^2/vol排名       ", bt.backtest(histories, calendar, start=S, score_square=True))

print("\n== C. 更敏感缓冲 ==")
show("BUFFER 0.5%%         ", bt.backtest(histories, calendar, start=S, buffer=0.005))

print("\n== D. 次方量化机制 ==")
show("trail5%%止盈+冷却5天  ", bt.backtest(histories, calendar, start=S, trail_stop=0.05, trail_cool=5))
show("trail8%%+冷却3天      ", bt.backtest(histories, calendar, start=S, trail_stop=0.08, trail_cool=3))
show("25日加权斜率score   ", bt.backtest(histories, calendar, start=S, score_wls=True))

print("\n== E. 过热关闭(让利润奔跑) ==")
show("OH关闭              ", bt.backtest(histories, calendar, start=S, overheat=9.9))

print("\n== F. 周频信号 ==")
show("周频决策            ", backtest_weekly(S))

print("\n== G. 组合抽查 ==")
show("平方score+进场2%%    ", bt.backtest(histories, calendar, start=S, score_square=True, enter_mom_min=0.02))
show("trail8%%+OH关闭      ", bt.backtest(histories, calendar, start=S, trail_stop=0.08, trail_cool=3, overheat=9.9))
print("DONE")
