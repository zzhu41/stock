# 本轮参考论文及采用边界

论文用于提出可解释机制，不是为具体 ETF 池、参数或回测收益背书。以下方向在 A/B/C 中均保留失败结果。

| 原始来源 | 对本轮实验的启发 | 不能直接移植的部分 |
| --- | --- | --- |
| Gârleanu & Pedersen，*Dynamic Trading with Predictable Returns and Transaction Costs*，2013，[NBER 作者摘要](https://www.nber.org/papers/w15205)、[作者讲义](https://pages.stern.nyu.edu/~lpederse/papers/DynamicTrading_Slides.pdf) | 考虑信号持久性和交易成本，测试向目标逐步调仓 | 论文有明确动态最优化模型；本轮固定1/2、1/3调整速度是简化实验，未声称复现其最优解。NBER全文直连403，使用可读摘要及作者讲义。 |
| Moreira & Muir，*Volatility Managed Portfolios*，2017，[NBER 原文](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf) | 高波动时少承担风险；测试既有H/simple信号上面的无杠杆风险权重 | 论文主要是按先前月度方差缩放因子。本轮按先前20/60报价波动、每日封顶100%仓位，频率、资产、约束均不同，不称论文策略复现。 |
| Cederburg et al.，*On the performance of volatility-managed portfolios*，2020，[作者全文](https://www.lehigh.edu/~xuy219/research/COWY.pdf) | 把风险缩放作为需要验证的假说，加入费用、历史重选及参数敏感性 | 论文研究103个股票策略，未发现风险缩放普遍优于原策略；事后最优组合也不等于当时可实施组合。本轮不能因“加了风控”就宣称更可靠。 |
| Goulding, Harvey & Mazzoleni，*Momentum turning points*，2023，[作者全文](https://people.duke.edu/~charvey/Research/Published_Papers/P158_Momentum_turning_points.pdf) | 多期限信号在趋势转折处可能有价值，C保留已研究的10/20/40期限组合 | 论文的月度、多空与较长期限环境不同于本项目的日频、仅做多ETF。它不支持恰好10/20/40或提高2026收益的保证。 |

截至2026-09-28重新核对上述原始来源。本文没有从论文取收益数字代替本项目实测，没有声称它们支持 54% 年化或保证降低过拟合。
