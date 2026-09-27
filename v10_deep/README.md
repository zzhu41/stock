# v10 深入迭代：H研究候选

主口径：2014-01-02—2026-09-24，统一修正行情、原同收盘成交、单边万一、首日免费。
目前冻结的新H年化 **53.31%**、最大回撤 **23.88%**；同数据v9.2为 **46.32% / 24.16%**。历史收益目标达成，未来稳定超额尚未证明。

- [完整报告、逐年及风险说明](REPORT.md)
- [冻结配置及文件哈希](profiles.json)
- [原始研究协议](PROTOCOL.md)、[最后一轮组合协议](REFINEMENT_PROTOCOL.md)
- [数据审查](DATA_REVIEW.md)、[原始论文及适用边界](LITERATURE.md)
- [全部9558个配置的最终结果](results/refinements/evaluation.json)
- [多重筛选与历史重选诊断](results/refinements/diagnostics.md)
- [费用、滞后一日及分红精度压力](results/refinements/stress/evaluation.json)
- [局部敏感性与消融](results/refinements/neighborhood/evaluation.json)
- [独立多持仓小族](results/portfolios/REPORT.md)：未通过S门槛，未替换此前S版本。

## 新H做了什么

保持原v9.1/v9.2的11只ETF及单一标的轮动、熊市门槛、缓冲、过热处理和危机框架，改变三点：

1. 评分改成 **20日WLS相对斜率/20日日收益波动率，取最近3个有效交易日评分均值**（回归斜率除以加权均价，再乘250年化）。
2. 沪深300牛熊判断的均线由250日改为 **180日**。
3. 单日急跌退出线改为 **1.5×20日日收益标准差，并限制在2%至10%之间**，替代固定4%。

这不是生产根脚本现有参数的简单粘贴：新评分平滑的是完整风险调整评分，精确实现与固定配置在本目录。

## 运行

依赖Python3.8+、NumPy、支持OpenMP的g++；作图另需Matplotlib。当前环境NumPy1.24.4。

```bash
OPENBLAS_NUM_THREADS=1 python3.8 -B -m v10_deep.cli backtest growth
python3.8 -B -m v10_deep.cli signal growth --end 2026-09-24
python3.8 -B -m v10_deep.cli backtest v92
python3.8 -B -m v10_deep.cli backtest simple
```

首次自动构建本目录缓存及本地C++库；只读输入快照，不拉实时行情、不写持仓、不发消息。
`growth`是本轮2014–2025选择后冻结的H；`simple`是已登记的单项改动对照（原v9.2仅给普通轮动增加最短2日持有），不是通过独立验证的新S。`v92`是同实现、同数据的旧基准。
`signal`只显示指定历史收盘的模型目标，不是当前14:50可执行指令。

## 必须一起看的结果

H在已登记11种压力中均保持相对v9.2的全程收益优势，但滞后一日信号时年化降至38.18%、回撤升至35.13%。
2026截至09-24，H累计45.47%，低于v9.2的57.36%。全9558族的White-style检验未显著，四段历史重选仅一段胜基准。
这些是已研究历史中的条件性结果，不能把53.31%当作实盘承诺，不能说过拟合已经消除。

## 复核

```bash
python3.8 -B -m unittest discover -s v10_deep/tests -v
python3.8 -B -m v10_deep.archives restore --stage refinements
python3.8 -B -m v10_deep.reference_fidelity --stage refinements
```

完整扫描各阶段已冻结，不能在看结果后覆盖选择文件。收益矩阵有无损归档和哈希收据；需要重跑统计时先按归档工具恢复对应阶段的`.npy`。
研究启动时的535个旧文件记录在`protected_manifest.json`；它用于证明本轮未改旧策略，不要求未来正常维护永远不能改生产文档/展示代码。
