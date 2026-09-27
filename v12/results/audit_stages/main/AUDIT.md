# V12 main 冻结证据审计

全部 439 行 × 4 情景的全期、分块、逐年、尾段指标及十项门槛复算一致；合格 0 项，primary=None。

6 条独立 Python reference 全路径与存档 returns/holdings/summary 逐值一致。

|标识|年化|最大回撤|尾段收益|失败门槛|
|---|---:|---:|---:|---|
|v12_10b7b1a6687fd049aebe|43.2398%|-24.1648%|41.6882%|main_cagr, main_drawdown, tail_return, close_11bp_cagr, close_11bp_drawdown, lag1_11bp_cagr, lag1_11bp_drawdown, middle_cagr, recent_cagr|
|v12_24552ec3549d227cc2e1|46.3175%|-24.1648%|57.3632%|main_cagr, main_drawdown, close_11bp_cagr, close_11bp_drawdown, lag1_11bp_cagr, lag1_11bp_drawdown, middle_cagr, recent_cagr|
|v12_47b31cea55c688ab7a2a|49.9525%|-23.9923%|91.1894%|main_cagr, main_drawdown, close_11bp_cagr, close_11bp_drawdown, lag1_11bp_cagr, middle_cagr|
|v12_55ec5c56a477b7158b7b|50.1327%|-21.6824%|75.2675%|main_cagr, main_drawdown, close_11bp_cagr, lag1_11bp_cagr, lag1_11bp_drawdown, middle_cagr, recent_cagr|
|v12_6ace1b82572e526d2b75|54.5133%|-20.6111%|52.4147%|tail_return|
|v12_cca586e367aa3071f32a|53.3104%|-23.8833%|45.4706%|main_cagr, main_drawdown, tail_return, close_11bp_drawdown|
|v12_d1d9ea928d1692b16fb2|43.8058%|-24.1648%|50.7414%|main_cagr, main_drawdown, tail_return, close_11bp_cagr, close_11bp_drawdown, lag1_11bp_cagr, lag1_11bp_drawdown, middle_cagr, recent_cagr|
|v12_d961c9516f25610bda74|55.1356%|-27.2753%|44.2679%|main_drawdown, tail_return, close_11bp_drawdown, lag1_11bp_drawdown|

这是已知历史上的实现与存档审计，不能证明样本外优势或消除此前自适应搜索偏差。
