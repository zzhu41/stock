# 大规模搜索的文献依据与验证边界

阅读日期：2026-09-27。本轮允许广泛枚举规则和资产池；文献用于提出可检查的机制、交易成本和统计流程，不预言本ETF池中的赢家。只采用原始论文、作者页面、作者提交稿或出版社信息。具体登记与选择规则见 [PROTOCOL.md](PROTOCOL.md)。

## 1. 动量：可以搜索，不能把所有趋势形式混为一个已证实事实

Moskowitz、Ooi、Pedersen，**Time Series Momentum**，*Journal of Financial Economics* 104（2012），228–250。论文研究58个期货/远期工具，报告1–12个月自身历史收益与后续趋势的关系；自身收益趋势不等于横截面追逐最强者。[原文](https://fairmodel.econ.yale.edu/ec439/mosk.pdf)

**本项目用法：** 将自身趋势过滤、相对排名、回归斜率分别登记为机制族，而不是把每种打分函数都说成论文已验证。长期期货多空结果不能直接外推成中国ETF短周期多头轮动有效。

### 单一沪深300牛熊门不是TSM论文的必要条件

时间序列动量按**每个交易品种自身**的过去收益判断方向。它不等于“沪深300跌破MA250，因此所有A股ETF都不能买”；上述单一基准门控是本项目额外的体制假设，不能借TSM论文直接证明。[原文中的逐品种时间序列动量构造](https://fairmodel.econ.yale.edu/ec439/mosk.pdf)

**本轮消融推论：** 用户授权的 `open_stock`、`always_bull`、`all_assets` 分别检验放开熊市A股、固定原牛市路径、全风险资产统一竞争。这三者不是同一个开关：第一种仍有熊市7%进场门，第二种保留黄金备胎结构，第三种让黄金与其他风险资产共同排名。去掉单一指数门控是合理的机制消融，但论文没有保证收益一定提高，也没有证明本ETF池中强趋势能在系统性下跌时继续有效。它们需要承担同样的成本、跨期及全候选搜索校正检查，不因有文献动机而得到统计豁免。

## 2. 固定集成：分散参数选择，不学习看起来最好的权重

Hurst、Ooi、Pedersen，**A Century of Evidence on Trend-Following Investing**，*Journal of Portfolio Management* 44(1)（2017），15–29。论文把1、3、12个月时间序列动量等权组合，在多资产历史中研究趋势，并明确采用风险尺度控制和交易成本假设。[作者机构提供的原文](https://images.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/AQR-JPM-Fall-2017.pdf)

**本项目用法：** 可以搜索预先列出的单窗口、固定多窗口均值或独立账户组合。必须区分平均信号与平均账户资金路径，并把窗口/权重尝试计入候选登记；集成后不能自动声称消除了过拟合。相邻窗口高度相关，并不是多个独立证据。

## 3. 低换手：交易规则中的买入/持有差别可以有经济意义

Novy-Marx、Velikov，**A Taxonomy of Anomalies and Their Trading Costs**，*Review of Financial Studies* 29(1)（2016），104–147。研究异常策略扣费表现和成本缓解方法，发现买入与继续持有采用不同门槛是有效的简单办法；交易成本降低利润及统计显著性。[作者NBER工作论文及期刊版本信息](https://www.nber.org/papers/w20721)，[作者稿PDF](https://www.nber.org/system/files/working_papers/w20721/w20721.pdf)

**本项目用法：** 缓冲、调仓频率、持有条件可作为明确的成本机制枚举，并同时报告成交金额换手。不能把某个缓冲档事后最优当成普适参数，也不能借论文给本ETF执行假设背书。须按每条真实成交腿扣费，不能通过免费再平衡或订单抵消制造低成本。

## 4. White Reality Check：应该检验搜索中的最大者

White，**A Reality Check for Data Snooping**，*Econometrica* 68(5)（2000），1097–1126。论文研究同一数据反复用于模型选择时的检验问题，其零假设针对整个已检查模型集合中最优者相对基准没有预测优势；并不是只对最后选中的一个赢家重新做普通检验。[论文原文](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf)，[出版社页面](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00152)

**本项目用法：** 保存全部候选同步逐日净收益矩阵；若实现重采样最大统计量，需要对整个候选集合重定心、同步抽时间区块，再在每次抽样里取最大者。只给冠军配对bootstrap区间，不能冒称已校正数千次筛选。只保留TopK再校正也不能代表原完整搜索。

## 5. Hansen SPA：不要用大量无关劣模型掩盖检验，也不要随意删除它们

Hansen，**A Test for Superior Predictive Ability**，*Journal of Business & Economic Statistics* 23(4)（2005），365–380。SPA对Reality Check采用学生化统计量及依样本构造的零假设分布，目标是提高功效、降低差模型和无关备选的影响。[作者研究页的2005年条目](https://reinhardhansen.github.io/research.html)，[作者提交稿](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID264569_code244328.pdf?abstractid=264569)

**本项目用法：** 若没有完整实现方差估计、学生化、重定心和退化列处理，就不能把简单最大值bootstrap命名为Hansen SPA。宁可准确标记“固定候选集合的White-style区块最大均值诊断”。SPA/Reality Check都依赖数据与重采样假设，也不会自动处理所有历史选池、公开历史知识或漏记的旧研究。

## 6. 大量相关因子：不能机械计独立次数，也不能因此免除多重选择

Jensen、Kelly、Pedersen，**Is There a Replication Crisis in Finance?**，*Journal of Finance* 78(5)（2023），2465–2518。论文使用贝叶斯复现框架、因子主题分组及93个国家的数据，对笼统的因子不可复现结论提出不同证据；相关因子可归入较少经济主题。[出版社原文](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249)，[作者机构公开终稿](https://research-api.cbs.dk/ws/portalfiles/portal/95651880/theis_ingerslev_jensen_et_al_is_there_a_replication_crisis_in_finance_publishersversion.pdf)

**本项目用法：** 报告配置数、不同收益路径数和机制族，而不把它们都当独立试验；同步重采样应保留候选之间的相关性。论文的全球因子证据不意味着本地ETF搜索赢家已经复现，也不能据此忽略只发表最好结果的偏差。

## 7. DSR与PBO：需要完整选择信息，不是报告冠军好看的装饰

Bailey、López de Prado，**The Deflated Sharpe Ratio**（2014），把多重选择及收益非正态性纳入夏普评价，需要未入选试验和收益分布的信息。[作者原文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

Bailey、Borwein、López de Prado、Zhu，**The Probability of Backtest Overfitting**，作者2015年修订稿，提出通过候选集合的组合对称交叉验证研究选择退化，并讨论已熟悉公开历史对留出检验的影响。[作者原文](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

**本项目用法：** 本轮完整矩阵有助于诊断“本轮这个候选集合”的选择风险，但此前研究链并不因新目录而消失。任何DSR、PBO、最大统计量或滚动重选结果都必须说明覆盖哪一批、哪段数据、哪些假设；不能把一个小p值说成未来盈利概率，也不能声称已经恢复真正未见OOS。

## 本轮研究行动

允许认真搜寻更多动量形式、买入/持有缓冲、调仓频率和资产池组合；用户扩大了试探范围，不沿用此前7个候选的上限。但每批须在运行前固化生成规则、列表及总数，保存失败和中途停止的记录。开发冻结赢家、开发TopK、全历史回看最高者必须分别展示。论文检验方法用于暴露搜索中的不确定性，不用于给全历史最高曲线制造“通过验证”的标签。
