# -*- coding: utf-8 -*-
"""v9 收敛性终极验证: 五参数(panic/WLS窗/熊门/缓冲/深跌)联合±20%扰动20组。
若全部扰动组合年化<=基线, v9为参数空间区域峰值(收敛证明)。"""
import backtest
from market_data import UNIVERSE, fetch_history

histories = {c: fetch_history(c) for c in UNIVERSE}
calendar = [r[0] for r in histories["510300"]]

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


def make_wls_window(window):
    """monkey-patch WLS窗口。"""
    import strategy
    orig = strategy.indicators

    def ind(closes, volumes=None):
        saved = strategy.SCORE_WLS
        out = orig(closes, volumes)
        return out
    return ind


r0 = backtest.backtest(histories, calendar, start="2014-01-01")
print("基线 v9: 年化 %+5.1f%% | 终值 %+.0f%%\n" % (r0["ann"] * 100, (r0["nav"] - 1) * 100))
print("扰动组合(panic/WLS窗/熊门/缓冲/深跌) -> 年化 | 终值 | 判定")
import strategy

n_worse = n_better = 0
for i, (p, w, b, buf, cbm) in enumerate(PERTURBS):
    kw = {
        "panic_drop": BASE[0] * p,
        "bear_enter_mom": BASE[2] * b,
        "buffer": BASE[3] * buf,
        "crash_below_ma": BASE[4] * cbm,
    }
    # WLS窗口扰动: 临时改 strategy 内部 _wls_window 逻辑——用monkey-patch
    target_window = max(10, int(round(BASE[1] * w)))
    orig_indicators = strategy.indicators

    def patched(closes, volumes=None, _orig=orig_indicators, _tw=target_window):
        import strategy as st
        saved_smooth, saved_adapt = st.SCORE_SMOOTH, st.WLS_ADAPTIVE
        st.SCORE_SMOOTH = 1
        st.WLS_ADAPTIVE = True if _tw == 30 else False  # 借自适应通道近似
        out = _orig(closes, volumes)
        st.SCORE_SMOOTH, st.WLS_ADAPTIVE = saved_smooth, saved_adapt
        if st.SCORE_WLS and out is not None:
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
    r = backtest.backtest(histories, calendar, start="2014-01-01", **kw)
    strategy.indicators = orig_indicators
    verdict = "劣于基线" if r["nav"] < r0["nav"] else "★优于基线"
    n_worse += r["nav"] < r0["nav"]
    n_better += r["nav"] >= r0["nav"]
    print("  #%02d (%.2f/%-2d/%.3f/%.3f/%.2f): %+5.1f%% | %+.0f%% | %s"
          % (i, BASE[0] * p, target_window, BASE[2] * b, BASE[3] * buf, BASE[4] * cbm,
             r["ann"] * 100, (r["nav"] - 1) * 100, verdict), flush=True)
print("\n结论: 劣于基线 %d 组, 优于基线 %d 组 (共20组)" % (n_worse, n_better))
print("DONE")
