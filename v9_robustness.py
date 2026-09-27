# -*- coding: utf-8 -*-
"""v9 五参数联合邻域敏感性：同一历史的扰动结果，不是样本外或收敛证明。"""
from contextlib import contextmanager
import math
import backtest
import strategy
from strategy_versions import backtest_kwargs, load_local_histories

# 固定种子保证可复现(不用Date.now): 手工指定20组扰动
PERTURBS = [
    (0.9, 1.1, 1.0, 1.0, 1.0), (1.1, 0.9, 1.0, 1.1, 0.9), (1.0, 1.0, 1.1, 0.9, 1.1),
    (0.8, 1.0, 0.9, 1.2, 1.0), (1.2, 1.0, 1.0, 0.8, 0.9), (1.0, 0.8, 1.0, 1.0, 1.2),
    (0.9, 0.9, 1.1, 1.1, 1.0), (1.1, 1.1, 0.9, 0.9, 1.1), (1.0, 1.2, 1.0, 1.0, 0.8),
    (0.85, 1.05, 1.15, 0.95, 1.1), (1.15, 0.95, 0.85, 1.05, 0.9),
    (0.95, 1.15, 0.95, 1.15, 1.05), (1.05, 0.85, 1.05, 0.85, 0.95),
    (0.8, 0.9, 1.2, 1.0, 1.1), (1.2, 1.1, 0.8, 1.1, 0.85),
    (0.9, 1.2, 0.9, 0.85, 1.15), (1.1, 0.8, 1.1, 1.2, 0.8),
    (0.85, 1.0, 0.85, 1.0, 1.2), (1.2, 0.85, 1.15, 0.9, 1.0),
    (1.0, 1.1, 1.0, 1.15, 0.85),
]
PARAMS = ("panic_drop", "wls_window", "bear_enter_mom", "buffer", "crash_below_ma")
BASE = (0.04, 25, 0.07, 0.02, 0.20)


def perturbation_parameters(multipliers):
    p, w, b, buf, cbm = multipliers
    # v9 必须关闭分池缓冲，否则传入 buffer 会被 v9.1 默认配置覆盖。
    kw = backtest_kwargs("v9", panic_drop=BASE[0] * p,
                         bear_enter_mom=BASE[2] * b, buffer=BASE[3] * buf,
                         crash_below_ma=BASE[4] * cbm)
    target_window = max(10, int(round(BASE[1] * w)))
    return kw, target_window


@contextmanager
def wls_window(target_window):
    """固定指定 WLS 窗口；异常时也恢复函数，不借用自适应开关。"""
    orig_indicators = strategy.indicators

    def patched(closes, volumes=None, _orig=orig_indicators, _tw=target_window):
        out = _orig(closes, volumes)
        if strategy.SCORE_WLS and out is not None:
            # 重算目标窗口WLS斜率
            seg = closes[-_tw:]
            n = len(seg)
            xs = list(range(n))
            wts = [i + 1 for i in xs]
            wsum = sum(wts)
            wmx = sum(wts[i2] * xs[i2] for i2 in xs) / wsum
            wmy = sum(wts[i2] * seg[i2] for i2 in xs) / wsum
            wsxy = sum(wts[i2] * (xs[i2] - wmx) * (seg[i2] - wmy) for i2 in xs)
            wsxx = sum(wts[i2] * (xs[i2] - wmx) ** 2 for i2 in xs)
            slope = (wsxy / wsxx) / wmy * 250 if wsxx > 0 and wmy > 0 else 0.0
            out["score"] = slope / out["vol"] if out["vol"] > 0 else 0.0
        return out

    strategy.indicators = patched
    try:
        yield
    finally:
        strategy.indicators = orig_indicators


def main():
    histories = load_local_histories()
    calendar = [r[0] for r in histories["510300"]]
    r0 = backtest.backtest(histories, calendar, start="2014-01-01", **backtest_kwargs("v9"))
    print("基线 v9: 年化 %+5.1f%% | 终值 %+.0f%%\n" %
          (r0["ann"] * 100, (r0["nav"] - 1) * 100))
    print("扰动组合(panic/WLS窗/熊门/缓冲/深跌) -> 年化 | 终值比基线 | 判定")
    counts = {"劣于基线": 0, "优于基线": 0, "持平": 0}
    for i, multipliers in enumerate(PERTURBS):
        kw, target_window = perturbation_parameters(multipliers)
        with wls_window(target_window):
            r = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
        if math.isclose(r["nav"], r0["nav"], rel_tol=1e-10):
            verdict = "持平"
        else:
            verdict = "劣于基线" if r["nav"] < r0["nav"] else "优于基线"
        counts[verdict] += 1
        print("  #%02d (%.3f/%-2d/%.3f/%.3f/%.2f): %+5.1f%% | %.4fx | %s" %
              (i, kw["panic_drop"], target_window, kw["bear_enter_mom"], kw["buffer"],
               kw["crash_below_ma"], r["ann"] * 100, r["nav"] / r0["nav"], verdict), flush=True)
    print("\n邻域结果: %s" % counts)
    print("邻域输赢仅反映本段历史敏感性；局部峰值不证明稳健，也不能排除过拟合。")


if __name__ == "__main__":
    main()
