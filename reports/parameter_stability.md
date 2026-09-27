# v9 / v9.1 参数邻域稳定性

固定区间：2014-01-01 至 2026-09-24；次日开盘重放，单边佣金 1bp + 滑点 10bp。
预先限定单因素邻域，不据收益挑选新参数。全部历史均已参与过研究，以下是样本内敏感性，不是样本外验证。

| 版本/场景 | 年化 | 最大回撤 | 成交切换次数 | 2014–17年化 | 2018–21年化 | 2022–25年化 | 2026年化至截止日 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v9:baseline | 36.24% | -35.49% | 273 | 34.06% | 38.97% | 42.64% | 4.03% |
| v9:panic_drop=0.03 | 35.34% | -36.61% | 315 | 34.16% | 34.89% | 46.24% | -5.14% |
| v9:panic_drop=0.05 | 35.39% | -34.37% | 248 | 34.34% | 35.95% | 45.36% | -6.16% |
| v9:wls_window=20 | 31.39% | -35.94% | 285 | 36.10% | 34.38% | 33.09% | -10.74% |
| v9:wls_window=30 | 34.18% | -28.65% | 268 | 34.23% | 38.43% | 34.67% | 10.72% |
| v9:crash_mom5=-0.06 | 36.27% | -35.49% | 281 | 33.99% | 41.58% | 40.55% | 2.55% |
| v9:crash_mom5=-0.1 | 32.35% | -35.49% | 261 | 37.43% | 32.51% | 33.01% | 4.03% |
| v9:crash_below_ma=0.15 | 32.86% | -35.49% | 285 | 35.37% | 38.95% | 30.36% | 4.03% |
| v9:crash_below_ma=0.25 | 33.10% | -40.88% | 260 | 30.41% | 35.25% | 40.35% | 2.26% |
| v9:crash_lock=3 | 35.44% | -35.49% | 279 | 34.60% | 35.44% | 43.10% | 3.78% |
| v9:crash_lock=7 | 36.16% | -35.49% | 271 | 36.03% | 39.73% | 42.09% | -5.80% |
| v9:buffer=0.01 | 36.78% | -35.49% | 293 | 40.93% | 37.54% | 38.62% | 4.61% |
| v9:buffer=0.03 | 35.01% | -35.49% | 263 | 35.36% | 37.35% | 39.19% | 2.64% |
| v9.1:baseline | 37.13% | -35.49% | 268 | 33.28% | 38.83% | 44.46% | 12.88% |
| v9.1:panic_drop=0.03 | 35.39% | -36.61% | 310 | 33.38% | 33.94% | 46.14% | 2.92% |
| v9.1:panic_drop=0.05 | 36.27% | -34.37% | 243 | 33.55% | 35.82% | 47.22% | 1.81% |
| v9.1:wls_window=20 | 33.81% | -34.74% | 277 | 35.86% | 34.25% | 39.39% | -3.16% |
| v9.1:wls_window=30 | 35.45% | -26.04% | 265 | 33.99% | 37.79% | 37.61% | 20.13% |
| v9.1:crash_mom5=-0.06 | 37.06% | -35.49% | 275 | 33.20% | 41.44% | 42.02% | 11.26% |
| v9.1:crash_mom5=-0.1 | 33.21% | -35.49% | 256 | 36.62% | 32.38% | 34.71% | 12.88% |
| v9.1:crash_below_ma=0.15 | 33.72% | -35.49% | 280 | 34.58% | 38.81% | 32.03% | 12.88% |
| v9.1:crash_below_ma=0.25 | 33.97% | -41.09% | 255 | 29.64% | 35.11% | 42.15% | 10.95% |
| v9.1:crash_lock=3 | 36.32% | -35.49% | 274 | 33.81% | 35.31% | 44.93% | 12.60% |
| v9.1:crash_lock=7 | 37.13% | -35.49% | 264 | 35.23% | 39.59% | 44.19% | 2.21% |
| v9.1:global_buffer=0.02 | 37.11% | -35.49% | 270 | 34.06% | 38.97% | 43.39% | 12.88% |
| v9.1:global_buffer=0.04 | 36.04% | -35.49% | 266 | 32.40% | 36.57% | 44.15% | 12.88% |
| v9.1:gold_buffer=0.02 | 36.26% | -35.49% | 271 | 33.28% | 38.83% | 43.70% | 4.03% |
| v9.1:gold_buffer=0.04 | 37.13% | -35.49% | 268 | 33.28% | 38.83% | 44.46% | 12.88% |

## 数值摘要

- v9 基线年化 36.24%、回撤 -35.49%；12 个邻居年化范围 31.39%～36.78%，终值为基线 0.631x～1.051x；2 个高于基线、0 个持平。
- v9.1 基线年化 37.13%、回撤 -35.49%；14 个邻居年化范围 33.21%～37.13%，终值为基线 0.692x～1.000x；1 个高于基线、1 个持平。

## 分段与缓存核验

分段保留连续净值与跨年持仓。每段收益乘积已断言等于全段终值；分段回撤从该段期初净值重新计算，不代表全程峰值下的回撤。JSON 同时保存分段累计收益、回撤和成交次数。
WLS 20/25/30 独立缓存，其他影响指标的配置纳入缓存键；缓存只保存指标，不缓存仓位、锁仓或决策状态。三个预先指定的短区间情景与未缓存引擎逐日净值、交易、抄底及开盘重放结果完全一致。

## 方法限制

- All periods were already used for strategy research; this is in-sample sensitivity, not OOS validation.
- Neighbors are deliberately limited, not an exhaustive or probabilistic overfitting test.
- All performance segments inherit positions and wealth from one 2014-start path; no segment restarts.
- Each scenario's fixed closing target path is repriced at next open with 1bp commission plus 10bp adverse slippage per side.
- Missing quotes defer fills without re-solving targets from actual deferred holdings; limit-lock and intraday liquidity are not modeled.
- WLS window changes the score only; other window-sensitive feature switches are disabled in the frozen configurations.
- Annualization uses 244 sessions per year, including the initial uninvested signal day; trade counts include first entry.
- Point-in-time asset-selection and adjusted-history biases remain; no future return guarantee follows from a smooth neighborhood.

复现：`python3 -B research_parameter_stability.py`。仅写本报告及配套 JSON，不修改生产策略或实盘状态。
