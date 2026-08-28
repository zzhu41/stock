# -*- coding: utf-8 -*-
"""亏损来源诊断：拆解回测交易的胜率、盈亏结构、whipsaw 分布。"""
from collections import Counter
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]
r = backtest.backtest(histories, calendar, start="2014-01-01")
trades, daily = r["trades"], r["daily"]
nav_of = {d: nav for d, nav, _ in daily}

# 每笔持仓的盈亏：相邻换仓之间的净值变化
legs = []
for i in range(1, len(trades)):
    d0, frm, to, nav0 = trades[i - 1]
    d1 = trades[i][0]
    legs.append({"code": to, "entry": d0, "exit": d1,
                 "ret": trades[i][3] / nav0 - 1,
                 "days": len([x for x in nav_of if d0 < x <= d1])})
# 最后一笔到现在
if trades:
    d0, frm, to, nav0 = trades[-1]
    legs.append({"code": to, "entry": d0, "exit": daily[-1][0],
                 "ret": daily[-1][1] / nav0 - 1,
                 "days": len([x for x in nav_of if d0 < x <= daily[-1][0]])})

wins = [x for x in legs if x["ret"] > 0]
losses = [x for x in legs if x["ret"] <= 0]
print("== 交易结构 (2014 起, %d 笔持仓) ==" % len(legs))
print("胜率 %d/%d = %.0f%% | 平均盈利 %+.1f%% | 平均亏损 %.1f%% | 盈亏比 %.1f"
      % (len(wins), len(legs), 100.0 * len(wins) / len(legs),
         100 * sum(x["ret"] for x in wins) / len(wins),
         100 * sum(x["ret"] for x in losses) / len(losses),
         -sum(x["ret"] for x in wins) / len(wins) / (sum(x["ret"] for x in losses) / len(losses))))

quick = [x for x in losses if x["days"] <= 10]
print("\n== 快速打脸 (持仓<=10天的亏损, whipsaw 特征) ==")
print("%d 笔, 占亏损 %.0f%%, 合计亏损 %.1f%%, 平均每笔 %.1f%%"
      % (len(quick), 100.0 * len(quick) / len(losses),
         100 * sum(x["ret"] for x in quick),
         100 * sum(x["ret"] for x in quick) / len(quick)))

print("\n== 亏损按标的分布 ==")
cnt = Counter(x["code"] for x in losses)
amt = Counter()
for x in losses:
    amt[x["code"]] += x["ret"]
for c, n in cnt.most_common():
    print("  %-7s %s: %d笔, 累计 %.1f%%" % (c, UNIVERSE[c][0], n, 100 * amt[c]))

print("\n== 亏损按年份分布 ==")
ycnt = Counter(x["entry"][:4] for x in losses)
yamt = Counter()
for x in losses:
    yamt[x["entry"][:4]] += x["ret"]
for y in sorted(ycnt):
    print("  %s: %d笔, 累计 %+.1f%%" % (y, ycnt[y], 100 * yamt[y]))

print("\n== 最大5笔亏损 ==")
for x in sorted(legs, key=lambda z: z["ret"])[:5]:
    print("  %s ~ %s %s %s: %.1f%% (%d天)"
          % (x["entry"], x["exit"], x["code"], UNIVERSE[x["code"]][0], 100 * x["ret"], x["days"]))
