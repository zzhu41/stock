# V12 consensus 冻结证据审计

全部 293 行 × 4 情景的全期、分块、逐年、尾段指标及十项门槛复算一致；合格 0 项，primary=None。

4 条独立 Python reference 全路径与存档 returns/holdings/summary 逐值一致。

|标识|年化|最大回撤|尾段收益|失败门槛|
|---|---:|---:|---:|---|
|v12_10b7b1a6687fd049aebe|43.2398%|-24.1648%|41.6882%|main_cagr, main_drawdown, tail_return, close_11bp_cagr, close_11bp_drawdown, lag1_11bp_cagr, lag1_11bp_drawdown, middle_cagr, recent_cagr|
|v12_24552ec3549d227cc2e1|46.3175%|-24.1648%|57.3632%|main_cagr, main_drawdown, close_11bp_cagr, close_11bp_drawdown, lag1_11bp_cagr, lag1_11bp_drawdown, middle_cagr, recent_cagr|
|v12_55ec5c56a477b7158b7b|50.1327%|-21.6824%|75.2675%|main_cagr, main_drawdown, close_11bp_cagr, lag1_11bp_cagr, lag1_11bp_drawdown, middle_cagr, recent_cagr|
|v12_9804bce55872af4a0911|49.7666%|-30.2424%|83.6263%|main_cagr, main_drawdown, close_11bp_cagr, close_11bp_drawdown, lag1_11bp_cagr, early_cagr, middle_cagr, recent_cagr|
|v12_bb36c1de9e534b1cf332|53.6379%|-20.1645%|59.0291%|main_cagr|
|v12_cca586e367aa3071f32a|53.3104%|-23.8833%|45.4706%|main_cagr, main_drawdown, tail_return, close_11bp_drawdown|
|v12_d1d9ea928d1692b16fb2|43.8058%|-24.1648%|50.7414%|main_cagr, main_drawdown, tail_return, close_11bp_cagr, close_11bp_drawdown, lag1_11bp_cagr, lag1_11bp_drawdown, middle_cagr, recent_cagr|

这是已知历史上的实现与存档审计，不能证明样本外优势或消除此前自适应搜索偏差。
