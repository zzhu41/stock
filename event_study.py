# -*- coding: utf-8 -*-
"""事件研究 + 熊市跨境过滤变体。

Part1 事件研究: 全部标的2013起, 单日跌<=-X%且MOM20>0(动量持仓态)后 5/10/20 日收益,
               对比无条件分布 —— 检验"急跌后继续跌"是否稳定统计现象。
Part2 熊市变体: 仅熊市时跨境池须站年线 / 熊市只黄金货币。
"""
import backtest
import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

print("== Part1 事件研究: 单日急跌后的前向收益 (2013起, 全部10只风险标的) ==")
for thresh in (0.03, 0.04, 0.05):
    for cond_mom in (False, True):
        ev5, ev10, ev20 = [], [], []
        for c in UNIVERSE:
            if c == CASH:
                continue
            rows = histories[c]
            closes = [r[2] for r in rows]
            for i in range(61, len(rows) - 20):
                ret1 = closes[i] / closes[i - 1] - 1
                if ret1 > -thresh:
                    continue
                mom20 = closes[i] / closes[i - 20] - 1
                if cond_mom and mom20 <= 0:
                    continue
                ev5.append(closes[i + 5] / closes[i] - 1)
                ev10.append(closes[i + 10] / closes[i] - 1)
                ev20.append(closes[i + 20] / closes[i] - 1)
        if not ev5:
            continue
        n = len(ev5)
        neg5 = sum(1 for x in ev5 if x < 0) / n
        tag = "MOM20>0" if cond_mom else "无条件 "
        print("  单日<=-%3.1f%% %s: %4d 次 | 后5日均值 %+5.2f%% (跌占比%.0f%%) | 后10日 %+5.2f%% | 后20日 %+5.2f%%"
              % (thresh * 100, tag, n, 100 * sum(ev5) / n, 100 * neg5,
                 100 * sum(ev10) / n, 100 * sum(ev20) / n))
# 无条件基准: 全样本任意日
all5, all10, all20 = [], [], []
for c in UNIVERSE:
    if c == CASH:
        continue
    closes = [r[2] for r in histories[c]]
    for i in range(61, len(closes) - 20):
        all5.append(closes[i + 5] / closes[i] - 1)
        all10.append(closes[i + 10] / closes[i] - 1)
        all20.append(closes[i + 20] / closes[i] - 1)
print("  基准(任意日)      : %4d 天 | 后5日均值 %+5.2f%% (跌占比%.0f%%) | 后10日 %+5.2f%% | 后20日 %+5.2f%%"
      % (len(all5), 100 * sum(all5) / len(all5),
         100 * sum(1 for x in all5 if x < 0) / len(all5),
         100 * sum(all10) / len(all10), 100 * sum(all20) / len(all20)))

print("\n== Part2 熊市跨境约束变体 (2018起) ==")
# 临时 monkey-patch decide 实现两个变体, 不动 strategy.py 主逻辑
orig_decide = strategy.decide


def decide_bear_filter(table, holding, holding_days=99, mode=None):
    info = {c: ind for c, ind in table}
    bull = strategy._is_bull(info)
    if bull or mode is None:
        return orig_decide(table, holding, holding_days)
    # 熊市: 按模式调整竞赛池
    if mode == "ma":       # 熊市跨境须站年线
        pool = [c for c in GLOBAL_POOL if info.get(c, {}).get("above_ma")] + [GOLD]
    elif mode == "noglobal":  # 熊市只要黄金
        pool = [GOLD]
    best = next((t for t in table if t[0] in pool and strategy._enter_ok(t[1], t[0])), None)
    if best:
        target, reason = best[0], "熊市约束池动量第一"
    elif GOLD in info and info[GOLD]["mom20"] > 0:
        target, reason = GOLD, "黄金备胎"
    else:
        target, reason = CASH, "熊市空仓"
    if holding and holding != target and holding in info:
        h = info[holding]
        if holding in STOCK_POOL:
            return target, "熊市体制, A股无条件离场"
        exit_why = strategy._exit_hit(h)
        if exit_why:
            return target, exit_why
        c_mom = info[target]["mom20"] if target in info else 0.0
        if c_mom - h["mom20"] < strategy.BUFFER:
            return holding, "滞回缓冲"
    return target, reason


for mode, name in ((None, "基线(熊市跨境自由)"), ("ma", "熊市跨境须站年线"), ("noglobal", "熊市仅黄金货币")):
    strategy.decide = (lambda t, h, hd=99, m=mode: orig_decide(t, h, hd)) if mode is None \
        else (lambda t, h, hd=99, m=mode: decide_bear_filter(t, h, hd, m))
    r = backtest.backtest(histories, calendar, start="2018-01-01")
    yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in backtest.yearly(r["daily"]))
    print("  %-14s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 卡玛 %4.2f || %s"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], r["calmar"], yrs))
strategy.decide = orig_decide

print("\n== Part3 组合: panic4%% + 熊市约束 ==")
for mode, name in ((None, "panic4%          "), ("ma", "panic4%+熊跨年线 "), ("noglobal", "panic4%+熊仅黄金 ")):
    strategy.decide = (lambda t, h, hd=99, m=mode: orig_decide(t, h, hd)) if mode is None \
        else (lambda t, h, hd=99, m=mode: decide_bear_filter(t, h, hd, m))
    r = backtest.backtest(histories, calendar, start="2018-01-01", panic_drop=0.04)
    yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in backtest.yearly(r["daily"]))
    print("  %-14s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 卡玛 %4.2f || %s"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], r["calmar"], yrs))
strategy.decide = orig_decide
