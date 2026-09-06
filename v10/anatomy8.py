# -*- coding: utf-8 -*-
"""第八轮事件解剖: 量能恐慌抄底(cv20_m5d15) vs 基线 的抄底事件对比(2014 起)。
列出新事件(量比并集引入)的入场日/标的/锁仓5日收益/20日收益, 以及当日基线持仓的同期收益(机会差)。
只读分析, 不改任何状态。用法: python3.8 anatomy8.py [variant]"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lab  # noqa: E402


def run(cfg):
    lab.CFG.clear()
    lab.CFG.update(cfg)
    lab.STATE.update(last_date=None, holding=None, pending=None, count=0,
                     ne_fb=False, ne_fb_bull=True, ne_fb_raw=False, ne_fb_cand=None)
    return lab.backtest_v10(lab.H, lab.CAL, start="2014-01-01")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "cv20_m5d15"
    cfg = dict(lab.VARIANTS[name][0])
    lab.init_data()
    base = run({})
    vari = run(cfg)
    base_buys = set(d for d, _ in base["crash_buys"])
    print("基线抄底 %d 次, %s %d 次" % (len(base["crash_buys"]), name, len(vari["crash_buys"])))
    print("基线事件: %s" % ", ".join(sorted(base_buys)))
    close_of = {c: {r[0]: r[2] for r in rows} for c, rows in lab.H.items()}
    daily_base = {d: (nav, h) for d, nav, h in base["daily"]}
    days = [d for d in lab.CAL if d >= "2014-01-01"]
    print("\n=== %s 新增事件解剖(入场日, 标的, 5日/20日收益, 当日量比, 基线同期收益) ===" % name)
    for d, code in vari["crash_buys"]:
        tag = "基线同有" if d in base_buys else "★新增"
        i = days.index(d)
        p0 = close_of[code].get(d)
        p5 = close_of[code].get(days[i + 5]) if i + 5 < len(days) else None
        p20 = close_of[code].get(days[i + 20]) if i + 20 < len(days) else None
        r5 = (p5 / p0 - 1) * 100 if p0 and p5 else float("nan")
        r20 = (p20 / p0 - 1) * 100 if p0 and p20 else float("nan")
        # 当日量比(复算, 与指标口径一致: 当日量/前20日均量)
        rows = lab.H[code]
        vols = {r[0]: r[3] for r in rows}
        prev20 = [vols.get(days[j], 0.0) for j in range(i - 20, i)]
        avg20 = sum(prev20) / 20.0
        vr = vols.get(d, 0.0) / avg20 if avg20 > 0 else 0.0
        # 基线当日持仓的同期5日收益
        bh = daily_base.get(d, (None, None))[1]
        br5 = float("nan")
        if bh and i + 5 < len(days):
            b0, b5 = close_of[bh].get(d), close_of[bh].get(days[i + 5])
            if b0 and b5:
                br5 = (b5 / b0 - 1) * 100
        print("%s %-7s %s | 5日 %+6.1f%% | 20日 %+6.1f%% | 量比 %.1f | 基线持 %s 5日 %+6.1f%%"
              % (d, code, tag, r5, r20, vr, bh, br5))


if __name__ == "__main__":
    main()
