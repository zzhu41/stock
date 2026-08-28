# -*- coding: utf-8 -*-
"""回撤区间归因：找出 Top 回撤 (peak->trough->recovery)，映射区间内持仓与市场状态。"""
import backtest
import strategy
from market_data import UNIVERSE, STOCK_POOL, GLOBAL_POOL, GOLD, CASH, fetch_history

START = "2018-01-01"
histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
r = backtest.backtest(histories, calendar, start=START)
daily, trades = r["daily"], r["trades"]

# ---- 1. 提取所有回撤区间 (peak -> trough -> recovery) ----
peak_nav, peak_date = daily[0][1], daily[0][0]
trough_nav, trough_date = daily[0][1], daily[0][0]
episodes = []
in_dd = False
for d, nav, h in daily:
    if nav >= peak_nav:
        if in_dd:
            episodes.append({"peak": peak_date, "trough": trough_date, "recovery": d,
                             "depth": trough_nav / peak_nav - 1, "peak_nav": peak_nav,
                             "trough_nav": trough_nav})
            in_dd = False
        peak_nav, peak_date = nav, d
        trough_nav, trough_date = nav, d
    else:
        in_dd = True
        if nav < trough_nav:
            trough_nav, trough_date = nav, d
if in_dd:  # 未修复的回撤
    episodes.append({"peak": peak_date, "trough": trough_date, "recovery": None,
                     "depth": trough_nav / peak_nav - 1, "peak_nav": peak_nav,
                     "trough_nav": trough_nav})

episodes.sort(key=lambda e: e["depth"])
idx = {d: i for i, (d, _, _) in enumerate(daily)}
close_of = {c: {row[0]: row[2] for row in rows} for c, rows in histories.items()}


def legs_in(d0, d1):
    """区间内的持仓段: (enter, exit, code, ret)。"""
    segs, cur_h, enter, nav0 = [], None, None, None
    for d, nav, h in daily:
        if d < d0 or d > d1:
            continue
        if cur_h is None:
            cur_h, enter, nav0 = h, d, nav
        elif h != cur_h:
            segs.append((enter, d, cur_h, nav / nav0 - 1))
            cur_h, enter, nav0 = h, d, nav
    if cur_h is not None:
        segs.append((enter, d1, cur_h, daily[idx[d1]][1] / nav0 - 1))
    return segs


print("== 基准 v5: 年化 %+.1f%% 最大回撤 %.1f%% 夏普 %.2f ==\n" %
      (r["ann"] * 100, r["max_dd"] * 100, r["sharpe"]))
print("== Top 8 回撤区间 ==")
for e in episodes[:8]:
    i0, i1 = idx[e["peak"]], idx[e["trough"]]
    dur = i1 - i0
    rec = "未修复(至今 %d 天)" % (len(daily) - 1 - i1) if not e["recovery"] \
        else "%s (修复用 %d 天)" % (e["recovery"], idx[e["recovery"]] - i1)
    print("\n--- 回撤 %.1f%% | %s -> %s (%d个交易日) | %s" %
          (e["depth"] * 100, e["peak"], e["trough"], dur, rec))
    # 区间市场状态
    bull0 = strategy.rank(histories, on_date=e["peak"])
    info = {c: x for c, x in bull0}
    bull = info.get("510300", {}).get("above_ma", True)
    print("    峰值日体制: %s" % ("牛市" if bull else "熊市"))
    # 持仓段
    for enter, exit_, code, ret in legs_in(e["peak"], e["trough"]):
        name = UNIVERSE[code][0] if code else "?"
        role = dict((c, v[2]) for c, v in UNIVERSE.items()).get(code, "?")
        print("    %s~%s %-7s %-6s [%s] %+.1f%%" % (enter, exit_, code, name, role, ret * 100))
    # 同期各标的涨跌幅 (peak->trough)
    movers = []
    for c in UNIVERSE:
        p0, p1 = close_of[c].get(e["peak"]), close_of[c].get(e["trough"])
        if p0 and p1:
            movers.append((c, p1 / p0 - 1))
    movers.sort(key=lambda t: t[1])
    s = "  ".join("%s%+.0f%%" % (UNIVERSE[c][0][:4], x * 100) for c, x in movers[:4])
    print("    区间最弱: " + s)
    s = "  ".join("%s%+.0f%%" % (UNIVERSE[c][0][:4], x * 100) for c, x in movers[-3:])
    print("    区间最强: " + s)

# ---- 2. 汇总：回撤期间按持仓角色统计损失 ----
print("\n\n== 各回撤区间损失按持仓角色汇总 ==")
role_loss = {}
for e in episodes[:8]:
    for enter, exit_, code, ret in legs_in(e["peak"], e["trough"]):
        if ret < 0:
            role = dict((c, v[2]) for c, v in UNIVERSE.items()).get(code, "?")
            role_loss.setdefault(role, [0, 0.0])
            role_loss[role][0] += 1
            role_loss[role][1] += ret
for role, (n, tot) in sorted(role_loss.items(), key=lambda kv: kv[1][1]):
    print("  %-6s: %d 段亏损, 累计 %+.1f%%" % (role, n, tot * 100))
