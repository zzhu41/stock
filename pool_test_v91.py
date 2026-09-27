# -*- coding: utf-8 -*-
"""v9.1 池子组合扫描(带基线保护): LOO + 窄池 + 精选 + 去弱留强。
临时修改竞赛池，断言普通轮动和抄底都未交易池外标的；异常时也恢复池。"""
from functools import partial
import backtest
import strategy
from market_data import UNIVERSE, GOLD, CASH
from strategy_versions import (backtest_kwargs, load_local_histories,
                               STOCK_CODES, GLOBAL_CODES)


def run_pool(histories, calendar, stock, glob, start="2014-01-01"):
    orig_s, orig_g = strategy.STOCK_POOL, strategy.GLOBAL_POOL
    strategy.STOCK_POOL, strategy.GLOBAL_POOL = list(stock), list(glob)
    try:
        result = backtest.backtest(histories, calendar, start=start, **backtest_kwargs("v9.1"))
        allowed = set(stock) | set(glob) | {GOLD, CASH}
        outside = {holding for _, _, holding in result["daily"]} - allowed
        if outside:
            raise AssertionError("Pool test traded excluded assets: %s" % sorted(outside))
        return result
    finally:
        strategy.STOCK_POOL, strategy.GLOBAL_POOL = orig_s, orig_g


def show(name, stock, glob, histories, calendar):
    r14 = run_pool(histories, calendar, stock, glob, start="2014-01-01")
    r16 = run_pool(histories, calendar, stock, glob, start="2016-01-01")
    y14 = dict(backtest.yearly(r14["daily"]))
    print("%-20s: 14起[%+5.1f%% %6.1f%% %+.0f%%] 16起[%+5.1f%% %6.1f%% %+.0f%%] || 17:%+.0f 19:%+.0f 21:%+.0f 22:%+.0f 25:%+.0f"
          % (name, r14["ann"] * 100, r14["max_dd"] * 100, (r14["nav"] - 1) * 100,
             r16["ann"] * 100, r16["max_dd"] * 100, (r16["nav"] - 1) * 100,
             y14.get("2017", 0) * 100, y14.get("2019", 0) * 100, y14.get("2021", 0) * 100,
             y14.get("2022", 0) * 100, y14.get("2025", 0) * 100), flush=True)


A7 = list(STOCK_CODES)
G2 = list(GLOBAL_CODES)


def main():
    histories = load_local_histories()
    calendar = [r[0] for r in histories["510300"]]
    display = partial(show, histories=histories, calendar=calendar)
    print("== v9.1 固定基线（池敏感性不是独立样本外验证） ==")
    display("现池(A7+跨2)      ", A7, G2)

    print("\n== 1. leave-one-out ==")
    for c in A7 + G2:
        display("剔%s" % UNIVERSE[c][0][:5], [x for x in A7 if x != c], [x for x in G2 if x != c])

    print("\n== 2. 窄池与精选 ==")
    display("平台3只(纳指黄金创业)", ["159915"], ["513100"])
    display("窄池+有色(4只)     ", ["159915", "512400"], ["513100"])
    display("窄池+有色+创新药(5)", ["159915", "512400"], ["513100", "513120"])
    display("进攻全明星6只      ", ["159915", "588080", "512400"], G2)
    display("防守精选6只        ", ["510300", "510500", "512890"], G2)
    display("去三宽基留4A股     ", ["159915", "588080", "512400", "512890"], G2)
    display("仅跨境+黄金(纯防御)", [], G2)
    print("DONE")


if __name__ == "__main__":
    main()
