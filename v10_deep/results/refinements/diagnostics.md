# v10 深入研究：完整候选族条件诊断

阶段 `refinements`，全部 9558 个候选；基准 `vd_368146734e874b98662b`。

**整个候选族由已经看过的历史自适应构造；这些统计不是干净样本外检验，也没有校正所有旧试验。**

|区块长度|候选数|重采样次数|White-style p|Monte Carlo SE|
|---|---:|---:|---:|---:|
|20|9558|1000|0.9690|0.0055|
|60|9558|1000|0.9610|0.0061|

固定四段历史重选（2018–2025 拼接）的年化 34.03%，基准 44.34%；没有重建策略切换时的持仓和成本，不能当可交易策略。

|随后历史区间|训练时所选|重选流年化|基准年化|
|---|---|---:|---:|
|2018-01-01–2019-12-31|vd_ea037f3b092485984557|39.70%|35.11%|
|2020-01-01–2021-12-31|vd_b9bad8d94c0a596bd3c9|26.18%|38.69%|
|2022-01-01–2023-12-31|vd_1f0f12b03c7ee3bee2d6|16.40%|25.00%|
|2024-01-01–2025-12-31|vd_0fb9d84e8e1b5c3fb28e|57.20%|85.32%|

冻结候选 `vd_7d52c290adc6c4503f7d`（primary/guarded）：

|连续路径分段|候选年化|基准年化|候选累计|基准累计|
|---|---:|---:|---:|---:|
|full|53.31%|46.32%|22566.90%|12432.65%|
|selection_2014_2025|52.50%|44.18%|15481.78%|7864.15%|
|early_2014_2017|48.30%|43.88%|384.52%|329.16%|
|middle_2018_2021|46.26%|36.88%|355.52%|249.73%|
|recent_2022_2025|63.58%|52.23%|605.98%|430.63%|
|report_only_2026|67.16%|86.17%|45.47%|57.36%|

2026 为不足一年的已知历史，表内年化仅机械换算，应同时看累计收益。年度、前 20 个优势/劣势日、收益贡献、费用和固定路径排除早期年份诊断详见 JSON。

方法限制：

- The entire candidate family was adaptively designed using already known history, including knowledge of later outcomes; this is not clean out-of-sample evidence.
- The max test is conditional on this supplied complete stage; it does not correct all previous, discarded, or unrecorded searches, nor the adaptive construction of this family.
- White-style means paired, unstudentized, zero-centered circular-block maximum mean log-return differences; it is not Hansen SPA, PBO, or a probability of future profit.
- The four historical re-selection folds reuse a family designed with known future history; stitched model return streams omit transfers and model-switch transaction costs and are not a tradable strategy.
- The 2026 tail was already inspected. It is displayed separately and does not enter the bootstrap or historical re-selection calculations.
- Same-close decisions/fills, fee 1 bp per side with first evaluation day free, and reconstructed close total-return prices retain the original execution and dividend assumptions.
