# -*- coding: utf-8 -*-
"""不空仓兜底态解剖: 兜底天数占比/逐年分布/兜底日与全日收益对比。
回答: 兜底侧改进的理论天花板有多大? 降仓/门控的代价来自哪里?
归因口径: T 日信号定的持仓赚 T+1 日收益 → 用前一日的兜底标记归因当日 logR。"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lab


def main():
    lab.init_data()
    lab.CFG.clear()
    lab.CFG.update({"use_copies": True, "trace": True})
    lab.STATE.update(last_date=None, holding=None, pending=None, count=0,
                     ne_fb=False, ne_fb_bull=True)
    del lab.TRACE[:]
    r = lab.backtest_v10(lab.H, lab.CAL, start="2014-01-01")
    nav_of = {d: nav for d, nav, _ in r["daily"]}
    days = [t[0] for t in lab.TRACE]

    n_fb = sum(1 for t in lab.TRACE if t[3])
    print("总交易日 %d, 兜底态 %d 天 (%.1f%%)" % (len(days), n_fb, 100.0 * n_fb / len(days)))

    # 逐年: 兜底天数 + 兜底持仓赚到的 logR(前日兜底 → 当日收益归兜底)
    by_year = {}
    fb_lr = nf_lr = 0.0
    n_fb_earn = 0
    for i in range(1, len(lab.TRACE)):
        prev, cur = lab.TRACE[i - 1], lab.TRACE[i]
        d = cur[0]
        lr = math.log(nav_of[d] / nav_of[days[i - 1]])
        y = d[:4]
        a = by_year.setdefault(y, {"days": 0, "fb": 0, "lr_fb": 0.0, "lr": 0.0})
        a["days"] += 1
        a["lr"] += lr
        if prev[3]:                      # 前日处于兜底态 → 当日收益由兜底持仓贡献
            a["fb"] += 1
            a["lr_fb"] += lr
            fb_lr += lr
            n_fb_earn += 1
        else:
            nf_lr += lr
    print("\n年份  交易日  兜底赚资天  占比   全年logR  兜底logR  兜底持仓")
    for y in sorted(by_year):
        a = by_year[y]
        holds = sorted({t[1] for t in lab.TRACE if t[3] and t[0][:4] == y})
        print("%s %6d %8d %6.1f%% %+9.3f %+9.3f  %s"
              % (y, a["days"], a["fb"], 100.0 * a["fb"] / max(a["days"], 1),
                 a["lr"], a["lr_fb"], "+".join(h[-4:] for h in holds)))

    print("\n兜底持仓日 logR 合计 %+.3f (%d天, 日均%+.4f%%)" % (fb_lr, n_fb_earn, 100 * fb_lr / max(n_fb_earn, 1)))
    print("其余持仓日 logR 合计 %+.3f (%d天, 日均%+.4f%%)" % (nf_lr, len(days) - 1 - n_fb_earn, 100 * nf_lr / max(len(days) - 1 - n_fb_earn, 1)))
    print("全程 logR %+.3f → 兜底贡献 %.1f%%" % (fb_lr + nf_lr, 100.0 * fb_lr / (fb_lr + nf_lr)))


if __name__ == "__main__":
    main()
