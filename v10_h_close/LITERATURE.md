# 原收盘口径H研究：文献依据与不能外推的部分

阅读日期：2026-09-27。本轮由用户明确指定，把原同日收盘、原手续费口径作为主要比较目标。这是新的选型批次，不把此前次日开盘搜索的名次直接移来，也不将更漂亮的收盘曲线解释成已经可实盘取得的收益。具体口径和选择门槛见 [PROTOCOL.md](PROTOCOL.md)。

本轮并列交付“2014–2025原收盘收益最高者”和“预先固定风险/后段/费用门槛下的最高者”，两者可以相同，也允许没有风险约束合格者。此划分是用户目标下的研究约定，不是论文给出的最优投资组合或安全保证。

## 大量技术规则可以认真比较，但要报告整个搜索过程

Sullivan、Timmermann、White，**Data-Snooping, Technical Trading Rule Performance, and the Bootstrap**，*Journal of Finance*（1999）。论文扩展技术规则集合，并用完整规则集合的Reality Check方法讨论数据窥探；不同历史段的结果并不一致。这里链接的是1998年10月作者讨论稿。[LSE原文](https://eprints.lse.ac.uk/119144/1/dp303.pdf)

**本项目推论：** 可以在用户允许的已研究历史中认真寻找更高的原收盘收益，但必须登记候选、实际使用的历史区间、淘汰理由、全部结果和撮合假设。只留下最高曲线，或把选型区间改名为验证期，并不能得到可靠的泛化证据。

## 自身趋势、多周期与单一牛熊门是不同假设

Moskowitz、Ooi、Pedersen，**Time Series Momentum**，*Journal of Financial Economics*（2012），研究各交易品种自身过去收益与后续趋势；不是用一只沪深300ETF的均线决定所有A股能否持有。[原文](https://fairmodel.econ.yale.edu/ec439/mosk.pdf)

Hurst、Ooi、Pedersen，**A Century of Evidence on Trend-Following Investing**，*Journal of Portfolio Management*（2017），研究固定多周期组合及跨资产趋势的长历史证据。[原文](https://images.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/AQR-JPM-Fall-2017.pdf)

**本项目推论：** 原收盘模式下可以比较单窗口、固定窗口平均、体制消融和资产池结构；但WLS25/30、沪深300MA250门控、黄金备用、五日危机抄底都是本项目自己的具体设计，不是论文已经证明有效的中国ETF策略。更多高度相关窗口也不是更多独立确认。

## 固定手续费与真实执行成本不能等同

Novy-Marx、Velikov，**A Taxonomy of Anomalies and Their Trading Costs**，*Review of Financial Studies*（2016），研究扣交易成本后的异常策略及买入/持有门槛等成本缓解方法，指出成本会改变利润和统计评价。[作者工作论文及期刊信息](https://www.nber.org/papers/w20721)

Frazzini、Israel、Moskowitz，**Trading Costs**（2018），使用大规模真实执行数据研究成本随订单、资产、规模和市场条件变化的情形。[作者机构原始研究页面](https://www.aqr.com/insights/research/working-paper/trading-costs)

**本项目推论：** 主表可忠实复现根回测的换仓扣费公式及首日免费约定，但这只是指定的研究成本口径。增加FEE的同价压力测试检查费用敏感性；另报下一开盘执行可以检查时点敏感性。二者都不能替代真实14:50报价、成交回执、盘口和资金规模数据，也不能用论文反推“万二足够覆盖全部成本”。

## 多重选择检验不能跨口径直接借用

White，**A Reality Check for Data Snooping**（2000），针对整个已搜索模型集合最优表现相对基准进行检验。[原文](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf)

Bailey、López de Prado，**The Deflated Sharpe Ratio**（2014），讨论选择偏差、未入选试验及收益非正态性。[原文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

**本项目推论：** 此前对次日开盘收益、2014–2021候选矩阵做过的统计结果，不是本轮原收盘、2014–2025选择过程的检验。若本轮报告校正统计，必须使用相同主口径与实际选型区间的完整候选矩阵；只对本轮赢家计算普通bootstrap，不能说已校正大量筛选。没有实施相应校正时就明确未实施，不借用旧p值证明成功或失败。

## 近期因子复现研究也不提供本地ETF赢家的背书

Jensen、Kelly、Pedersen，**Is There a Replication Crisis in Finance?**，*Journal of Finance*（2023），使用主题相关结构、贝叶斯框架和跨国数据讨论因子复现。[出版社原文](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249)

**本项目推论：** 不能机械地把每个相关参数视为独立彩票，也不能因为都属于趋势主题就免除选择偏差。本轮的既有规则、ETF选择及2026历史都已看过。2014–2021辅助审计仅意味着它不决定本轮主排名，不意味着它是统计独立样本。最终结论只能是某个冻结配置在规定历史、手续费、风险门槛和诊断下的表现；不承诺消除过拟合或未来仍达到50%以上。
