# V11：文献依据、实现对应与证据边界

查阅日期2026-09-27。只采用作者论文、作者机构或正式出版机构来源。论文用于定义可检验问题，不能证明本项目11只中国ETF、短窗口或某个阈值有稳定超额。2014—2026全部历史在此前研究中已被查看；本轮开发、确认和尾段都不能重新称为干净样本外。

## 多期限趋势与转折

**Moskowitz、Ooi、Pedersen（2012），Time Series Momentum。** 研究覆盖股指、债券、货币和商品期货，区分各资产自身趋势与截面排名。本文献支持检验多个期限的共同趋势，但不是WLS、Huber或百分位排名的特定最优参数证明。[原论文](https://fairmodel.econ.yale.edu/ec439/mosk.pdf)

**Hurst、Ooi、Pedersen（2017），A Century of Evidence on Trend-Following Investing。** 已查阅作者原PDF，重点包括构造方法和信号滞后诊断：文章使用67个市场、月度调整、1/3/12个月趋势等权组合，并进行风险缩放；文中也讨论了迟延信号及快速反转的影响。V11的固定多期限共识、长期简化控制和滞后一日收盘检查受这些研究问题启发，但我们的长仓ETF、无杠杆、日线规则和费用模型并不复现其期货多空组合。[作者原PDF](https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/AQR-JPM-Fall-2017.pdf)

**Goulding、Harvey、Mazzoleni（2023），Momentum Turning Points。** 已核对作者机构页面和作者SSRN摘要；本次Duke PDF链接超时，未声称完整阅读全文。论文讨论快慢趋势信号在转折中的不同反应，以及两者组合的信息。它支持同时看慢信号迟钝与快信号误报的成本，不支持事后挑最快或最慢窗口，也不证明V11的固定等权组合最优。[Duke作者机构页面](https://scholars.duke.edu/publication/1586630)、[作者摘要](https://doi.org/10.2139/ssrn.3489539)、[作者PDF入口](https://people.duke.edu/~charvey/Research/Published_Papers/P158_Momentum_turning_points.pdf)

Huber/Theil–Sen等稳健回归是本轮的工程假说：降低个别价格点对斜率的影响。真实暴跌也可能携带有用信息，不能因为回归更稳健就推导出金融收益更稳健，更不能把真实极端行情裁成错误值后报告获利。

## 风险退出、危机与换仓成本

**Daniel、Moskowitz（2016；NBER稿2014），Momentum Crashes。** 作者研究动量在恐慌及反弹中的罕见严重损失，并讨论动态风险调整。其多空动量的输家反弹暴露不能直接变成单边ETF“跌得多就全仓抄底”的依据。V11保留关闭抄底、仅价格深跌和原通道对照，检查事件簇而非把相邻反弹日当独立成功。[NBER原论文页面](https://www.nber.org/papers/w20439)

**Gârleanu、Pedersen（2013），Dynamic Trading with Predictable Returns and Transaction Costs。** 作者把可预测收益、信号衰减和交易成本放入同一动态交易问题。其启发是评估延后/保留持仓的净收益及错失机会，不能只统计少了几次交易。文献没有规定“持有2日”“确认3次”或本ETF池的最优缓冲。[NBER作者论文](https://www.nber.org/papers/w15205)

**Barroso、Santa-Clara，Momentum Has Its Moments。** 原作者稿SSRN编号2041429在本轮访问中未取得全文，因此这里仅保留查阅线索，不将其公式、最优参数或结果作为已核验实现要求。本轮的波动率退出门不是波动率目标仓位策略。[作者稿入口](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2041429)

## 搜索校正和时间依赖

**White（2000），A Reality Check for Data Snooping。** 关注整个规格搜索中最好模型相对基准的证据，不能把搜索后的赢家视作唯一事前假说。[原出版论文](https://doi.org/10.1111/1468-0262.00152)

本目录实现的是明确限定的 **White-style** 检验。对每个已登记候选计算净简单收益的 `log(1+r_candidate)-log(1+r_H)`，按候选去均值，在同一次时间重采样中同步抽取全族，再计算 `sqrt(T)*max(0,各候选平均差)`。保留差策略、首个日收益和样本余数；不按结果裁TopK。它检验给定费用/时钟的一族平均log超额，不能自动验证所有回撤或多重筛选条件，也没有追溯校正整个项目先前的自适应研究。

**Politis、Romano（1994），The Stationary Bootstrap。** 平稳区块重采样用随机块长处理弱依赖观测，本轮实现几何长度、首点均匀抽取、循环接续和末块截断；另提供固定长度循环区块作清楚命名的替代。20/60是登记的区块长度或平均长度，两项应完整报告，不能挑较小p值。样本非平稳、少数危机和结构变化仍限制解释。[原出版论文](https://doi.org/10.1080/01621459.1994.10476870)

**Hansen（2005），A Test for Superior Predictive Ability。** SPA通过学生化和不同的零假设处理减少差备选项带来的影响。本轮未实现这些必要步骤，故不能把当前未学生化的最不利零均值检验改名为SPA。[原出版论文](https://doi.org/10.1198/073500105000000063)

**Bailey、López de Prado（2014），The Deflated Sharpe Ratio。** 作者强调多重选择、未报告尝试及非正态分布对Sharpe解释的影响。已查阅作者PDF摘要与相关讨论。本轮不猜测整个旧项目的有效独立尝试数，因而不输出貌似精确的全项目DSR/PBO；White-style p也不是未来失败概率。[作者原PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

## 诊断输出应如何解读

- 1000次重采样的p值最小报告粒度为`1/1001`；不能打印成绝对0。Monte Carlo误差只描述有限模拟，边界标准误为0不代表尾概率完全确定，所以同时报告二项Wilson区间。
- 前10个正优势日分别除以**净log超额**和**所有正log超额**。前者会因其它日期抵消而超过100%；净超额非正时不给“占净优势比例”。两者都不是总策略利润占比。
- 事件前后区间只描述已实现的连续路径。跨簇周边窗口可能重叠，不可相加或当作独立样本；删除优势日不是重跑了一个可交易策略。
- 一族通过统计诊断，也不自动证明最终复杂选择器、某个具体赢家或未来执行收益有效。真正新增的未见证据来自冻结之后的前向观察。
