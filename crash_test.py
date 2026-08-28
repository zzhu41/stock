# -*- coding: utf-8 -*-
"""平台归因: 1)价格口径核对 2)恐慌抄底模块在v7框架的实测。

抄底规则(仿平台): 竞赛池标的出现 5日暴跌(MOM5<=-X%) 且未持仓 -> 即使MOM20<=0也买入,
锁仓 LOCK 天(期内不操作), 解锁后回到正常信号。测试 X=6/8/10%。"""
import backtest as bt
import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

print("== 1. 纳指513100 价格口径核对 (平台@2020-02-13 = 0.839) ==")
close513 = {r[0]: r[2] for r in histories["513100"]}
for d in ("2020-02-13", "2020-02-19", "2020-04-17", "2022-03-24"):
    print("  我们数据 %s: %.3f" % (d, close513.get(d, -1)))

FEE = 0.0001
TD = 244


def backtest_crash(start, crash_mom5=-0.08, lock=5, end="9999"):
    """v7 + 恐慌抄底: 竞赛池标的 MOM5<=crash_mom5 触发抄底买入, 锁仓 lock 天。"""
    days = [d for d in calendar if start <= d <= end]
    close_of = {c: {r[0]: r[2] for r in rows} for c, rows in histories.items()}
    nav, peak, max_dd = 1.0, 1.0, 0.0
    holding, switches, daily = None, 0, []
    lock_until = -1
    crash_buys = []
    for i, d in enumerate(days):
        if i > 0:
            prev, cur = close_of[holding].get(days[i - 1]), close_of[holding].get(d)
            if prev and cur:
                nav *= cur / prev
        table = strategy.rank(histories, on_date=d)
        info = {c: ind for c, ind in table}
        bull = strategy._is_bull(info)
        comp_pool = STOCK_POOL + GLOBAL_POOL if bull else GLOBAL_POOL + [GOLD]
        if i < lock_until:
            target = holding                     # 锁仓期不操作
        else:
            target, _ = strategy.decide(table, holding, 99)
            # 恐慌抄底: 竞赛池中MOM5最深跌者触发(即便动量未转正)
            cand = None
            for c, ind in table:
                if c in comp_pool and ind["mom5"] <= crash_mom5:
                    cand = c
                    break
            if cand and cand != holding:
                target = cand
                lock_until = i + lock
                crash_buys.append((d, cand))
        if target != holding:
            if i > 0:
                switches += 1
                nav *= (1 - FEE * 2)
            holding = target
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1)
        daily.append((d, nav, holding))
    n = len(daily)
    rets = [daily[i][1] / daily[i - 1][1] - 1 for i in range(1, n)]
    mean = sum(rets) / len(rets)
    std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
    years = n / TD
    return {"nav": nav, "ann": nav ** (1 / years) - 1, "max_dd": max_dd,
            "sharpe": mean / std * TD ** 0.5 if std else 0,
            "switches": switches, "daily": daily, "crash_buys": crash_buys}


print("\n== 2. v7 + 恐慌抄底 实测 (2014起) ==")
r0 = bt.backtest(histories, calendar, start="2014-01-01")
print("  v7 基线: 年化 %+5.1f%% | 回撤 %6.1f%% | 终值 %+.0f%% | 换手 %d"
      % (r0["ann"] * 100, r0["max_dd"] * 100, (r0["nav"] - 1) * 100, r0["switches"]))
for x, lock in ((-0.06, 5), (-0.08, 5), (-0.10, 5), (-0.08, 7), (-0.10, 7)):
    r = backtest_crash("2014-01-01", x, lock)
    yrs = " ".join("%s:%+.0f" % (y[2:], x2 * 100) for y, x2 in bt.yearly(r["daily"]))
    print("  抄底 MOM5<=%+4.0f%% 锁%d天: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 终值 %+.0f%% | 换手 %d | 抄底 %d 次 || %s"
          % (x * 100, lock, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], (r["nav"] - 1) * 100,
             r["switches"], len(r["crash_buys"]), yrs))
    if r["crash_buys"]:
        print("    抄底记录: " + " ".join("%s(%s)" % (d[5:], UNIVERSE[c][0][:4]) for d, c in r["crash_buys"]))
print("DONE")
