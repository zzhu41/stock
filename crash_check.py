# -*- coding: utf-8 -*-
"""深跌抄底的样本内敏感性：显式比较 v8.1（无抄底）与 v9。"""
import backtest
from strategy_versions import backtest_kwargs, load_local_histories


def run(histories, calendar, start="2014-01-01", include_crash=True, **overrides):
    params = backtest_kwargs("v9", **overrides)
    if not include_crash:
        params["crash_mom5"] = 0.0
    return backtest.backtest(histories, calendar, start=start, **params)


def main():
    histories = load_local_histories()
    calendar = [r[0] for r in histories["510300"]]
    print("== 1. 参数网格 (mom5阈值 x 深跌阈值, v9, 2014起) ==")
    print("       mom5:  -6%            -8%            -10%")
    for bm in (0.12, 0.15, 0.18, 0.20, 0.25, 0.30):
        row = "  深跌%4.0f%%: " % (bm * 100)
        for cm in (-0.06, -0.08, -0.10):
            r = run(histories, calendar, crash_mom5=cm, crash_below_ma=bm)
            row += "%+5.1f%%/%+.0f%%/dd%4.1f  " % (
                r["ann"] * 100, (r["nav"] - 1) * 100, r["max_dd"] * 100)
        print(row, flush=True)

    print("\n== 2. 锁仓邻域 (mom5-8%, 深跌20%) ==")
    for lk in (3, 5, 7, 10):
        r = run(histories, calendar, crash_lock=lk)
        print("  锁%d天: 年化 %+5.1f%% | 回撤 %6.1f%% | 终值 %+.0f%%" % (
            lk, r["ann"] * 100, r["max_dd"] * 100, (r["nav"] - 1) * 100), flush=True)

    print("\n== 3. 嵌套起点敏感性 (固定v9配置，不是独立样本外验证) ==")
    for start in ("2014-01-01", "2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"):
        r0 = run(histories, calendar, start=start, include_crash=False)
        r1 = run(histories, calendar, start=start)
        print("  %s起: 无抄底[%+5.1f%% %+.0f%% dd%5.1f] v9[%+5.1f%% %+.0f%% dd%5.1f] 终值比%.4fx" % (
            start[:4], r0["ann"] * 100, (r0["nav"] - 1) * 100, r0["max_dd"] * 100,
            r1["ann"] * 100, (r1["nav"] - 1) * 100, r1["max_dd"] * 100,
            r1["nav"] / r0["nav"]), flush=True)

    print("\n== 4. 逐年 (2014起) ==")
    r0 = run(histories, calendar, include_crash=False)
    r1 = run(histories, calendar)
    y0, y1 = dict(backtest.yearly(r0["daily"])), dict(backtest.yearly(r1["daily"]))
    for year in sorted(y0):
        print("  %s: 无抄底 %+6.1f%% | v9 %+6.1f%% | 差 %+5.1fpp" % (
            year, y0[year] * 100, y1[year] * 100, (y1[year] - y0[year]) * 100))
    print("DONE")


if __name__ == "__main__":
    main()
