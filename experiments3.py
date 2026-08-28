# -*- coding: utf-8 -*-
"""第三轮结构变体: 轻量参数(dual_mom/score_mom/ma_slope/rank_exit) + Top2等权持仓引擎。"""
import backtest
import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
FEE = 0.0001
TD = 244


def backtest_top2(start, end="9999"):
    """Top2 等权(各50%)持仓: 不足2只时余仓货币; 替换需挑战者MOM超最弱持仓BUFFER; 不再平衡。"""
    strategy.BUFFER, strategy.OVERHEAT, strategy.PANIC_DROP = 0.02, 0.40, 0.04
    strategy.MOM_MAIN, strategy.MA_BULL = 20, 250
    strategy.BULL_VOTE = False
    strategy.MIN_ROWS = max(strategy.MA_BULL, max(strategy.MOM_WINDOWS) + 1) + strategy.VOL_DAYS
    days = [d for d in calendar if start <= d <= end]
    close_of = {c: {r[0]: r[2] for r in rows} for c, rows in histories.items()}
    nav, peak, max_dd = 1.0, 1.0, 0.0
    holdings = {}          # code -> 目标权重(风险), 余仓货币
    daily, switches = [], 0
    for i, d in enumerate(days):
        if i > 0:
            r_day, wsum = 0.0, 0.0
            for c, w in holdings.items():
                p0, p1 = close_of[c].get(days[i - 1]), close_of[c].get(d)
                if p0 and p1:
                    r_day += w * (p1 / p0 - 1)
                wsum += w
            c0, c1 = close_of[CASH].get(days[i - 1]), close_of[CASH].get(d)
            r_cash = (c1 / c0 - 1) if (c0 and c1) else 0.0
            nav *= 1 + r_day + (1 - wsum) * r_cash
        table = strategy.rank(histories, on_date=d)
        info = {c: ind for c, ind in table}
        bull = strategy._is_bull(info)
        comp_pool = STOCK_POOL + GLOBAL_POOL if bull else GLOBAL_POOL + [GOLD]
        comp = [c for c, ind in table if c in comp_pool and strategy._enter_ok(ind, c)]
        if not comp and bull and GOLD in info and info[GOLD]["mom20"] > 0:
            comp = [GOLD]          # 牛市备胎
        # 1) 硬移除: 熊市A股 / 触离场线 / 跌出候选
        keep = {}
        for c, w in holdings.items():
            if (not bull and c in STOCK_POOL) or c not in comp:
                continue
            if c in info and strategy._exit_hit(info[c]):
                continue
            keep[c] = w
        # 2) 补齐/替换到2只
        for c in comp:
            if len(keep) >= 2:
                break
            if c not in keep:
                keep[c] = 0.0
        if len(keep) > 2:
            keep = dict(sorted(keep.items(), key=lambda kv: comp.index(kv[0]))[:2])
        # 滞回: 新面孔替换老持仓需 MOM 差 > BUFFER (仅当集合变化由排名引起时)
        old_codes, new_codes = set(holdings), set(keep)
        if old_codes != new_codes and old_codes - new_codes and not (
                any((not bull and c in STOCK_POOL) or c not in comp or
                    (c in info and strategy._exit_hit(info[c])) for c in old_codes - new_codes)):
            # 被换掉的并非硬移除 -> 检查滞回
            removed = old_codes - new_codes
            added = new_codes - old_codes
            for rc in list(removed):
                for ac in list(added):
                    if rc in info and info[ac]["mom20"] - info[rc]["mom20"] < strategy.BUFFER:
                        keep.pop(ac)
                        keep[rc] = holdings[rc]
                        added.discard(ac)
        # 权重标准化: 等权
        n = len(keep)
        keep = {c: 0.5 if n == 2 else (0.5 if n == 1 else 0.0) for c in keep}
        turn = sum(abs(keep.get(c, 0) - holdings.get(c, 0)) for c in set(keep) | set(holdings))
        if i > 0 and turn > 1e-9:
            switches += 1
            nav *= 1 - turn * FEE * 2
        holdings = keep
        peak = max(peak, nav)
        max_dd = min(max_dd, nav / peak - 1)
        daily.append((d, nav, ",".join(holdings) or "CASH"))
    n = len(daily)
    rets = [daily[i][1] / daily[i - 1][1] - 1 for i in range(1, n)]
    mean = sum(rets) / len(rets)
    std = (sum((r - mean) ** 2 for r in rets) / len(rets)) ** 0.5
    years = n / TD
    return {"nav": nav, "ann": nav ** (1 / years) - 1, "max_dd": max_dd,
            "sharpe": mean / std * TD ** 0.5 if std else 0,
            "calmar": (nav ** (1 / years) - 1) / abs(max_dd) if max_dd else 0,
            "daily": daily, "switches": switches}


print("== 1. 轻量结构变体 (2014起, 基线=v6) ==")
variants = [
    ("基线 v6          ", {}),
    ("双动量确认 MOM60>0", {"dual_mom": True}),
    ("MOM60慢动量排名   ", {"score_mom": 60}),
    ("牛熊加斜率        ", {"ma_slope": True}),
    ("排名缓冲 top2     ", {"rank_exit_n": 2}),
    ("排名缓冲 top3     ", {"rank_exit_n": 3}),
    ("排名缓冲 top4     ", {"rank_exit_n": 4}),
]
for name, kw in variants:
    r = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
    yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in backtest.yearly(r["daily"]))
    print("%s: %+5.1f%% | %6.1f%% | %4.2f | %4.2f || %s"
          % (name, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], r["calmar"], yrs))

print("\n== 2. Top2 等权持仓 (2014起) ==")
r1 = backtest.backtest(histories, calendar, start="2014-01-01")
r2 = backtest_top2("2014-01-01")
for tag, r in (("Top1(v6)", r1), ("Top2等权", r2)):
    yrs = " ".join("%s:%+.0f" % (y[2:], x * 100) for y, x in backtest.yearly(r["daily"]))
    print("%s: 年化 %+5.1f%% | 回撤 %6.1f%% | 夏普 %4.2f | 卡玛 %4.2f | 总收益 %+.0f%% || %s"
          % (tag, r["ann"] * 100, r["max_dd"] * 100, r["sharpe"], r["calmar"],
             (r["nav"] - 1) * 100, yrs))
print("DONE")
