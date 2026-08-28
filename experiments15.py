# -*- coding: utf-8 -*-
"""借鉴平台机制提升非大牛市收益 (v9基线)。重点看平台优势年份: 17/19/20/21/22/23。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]


def show(name, **kw):
    r14 = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
    r16 = backtest.backtest(histories, calendar, start="2016-01-01", **kw)
    y14 = dict(backtest.yearly(r14["daily"]))
    print("%-24s: 14起[%+5.1f%% %6.1f%% %+.0f%%] 16起[%+5.1f%% %6.1f%% %+.0f%%] || 17:%+.0f 19:%+.0f 20:%+.0f 21:%+.0f 22:%+.0f 23:%+.0f"
          % (name, r14["ann"] * 100, r14["max_dd"] * 100, (r14["nav"] - 1) * 100,
             r16["ann"] * 100, r16["max_dd"] * 100, (r16["nav"] - 1) * 100,
             y14.get("2017", 0) * 100, y14.get("2019", 0) * 100, y14.get("2020", 0) * 100,
             y14.get("2021", 0) * 100, y14.get("2022", 0) * 100, y14.get("2023", 0) * 100), flush=True)


print("== 基线 ==")
show("v9 基线               ")

print("\n== A. min_hold 重测(v9基线) ==")
show("最短持有 5 天         ", min_hold=5)
show("最短持有 10 天        ", min_hold=10)

print("\n== B. 趋势强度自适应缓冲(MOM20>10%→缓冲5%) ==")
show("强趋势缓冲5%%          ", trend_buf_on=True)

print("\n== C. 分池缓冲 ==")
show("A股2%%/跨境3%%/黄金3%%    ", pool_buffer={"stock": 0.02, "global": 0.03, "gold": 0.03})
show("A股2%%/跨境4%%/黄金3%%    ", pool_buffer={"stock": 0.02, "global": 0.04, "gold": 0.03})

print("\n== D. 斜率连续为正确认(缓冲x2) ==")
show("斜率连续8日正→缓冲x2  ", slope_days_buf=8)
show("斜率连续10日正→缓冲x2 ", slope_days_buf=10)

print("\n== E. 组合 ==")
show("强趋势缓冲+分池       ", trend_buf_on=True, pool_buffer={"stock": 0.02, "global": 0.03, "gold": 0.03})
show("强趋势缓冲+min_hold5  ", trend_buf_on=True, min_hold=5)
print("DONE")
