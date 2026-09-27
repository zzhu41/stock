# ETF 动量轮动策略 · 版本复算

当前引擎可按显式 v9、v9.1 参数复算。旧论文和历史实验数字是当时代码及数据口径的存档；修复逻辑、修正前复权历史后，不承诺重现旧数字。

## 环境

- Python 3.8+；策略复算使用标准库，追加的区块重采样统计脚本 `research_overfitting.py` 使用 NumPy（本次环境为 1.24.4）。
- 本地 `data/` 中已有十一只基线 ETF 的 CSV。下述版本复算和四个敏感性脚本只读本地文件，不拉行情、不写实盘状态。

## 文件

| 文件 | 作用 |
|---|---|
| `market_data.py` | 行情管道：腾讯前复权日 K（含成交量），本地 CSV 缓存 + 增量更新 |
| `strategy.py` | 策略核心：牛熊开关 / WLS25 排名 / 缓冲 / 三重离场 / 熊市门槛 / 永不空仓兜底 |
| `backtest.py` | 当前回测引擎；直接运行的默认版本是 v9.1，会调用行情更新 |
| `strategy_versions.py` | 显式固定全部行为参数；v9 统一 2% 缓冲，v9.1 分池 2%/3%/3%；本地 CSV 复算入口 |
| `v10/lab.py` | v9.2 对应 `fz25_cv`，在 v9.1 参数上叠加 QVIX 与量能恐慌钩子 |
| `crash_check.py` | v8.1 无抄底对照与 v9 抄底参数敏感性 |
| `experiments15.py` | v9 基线与分池缓冲等候选对照 |
| `v9_robustness.py` | v9 五轴邻域敏感性，包含实际生效的统一缓冲变化 |
| `pool_test_v91.py` | v9.1 剔除/缩小池测试，断言未交易被剔除标的 |
| `research_parameter_stability.py` | 固定成本下的参数邻域，记录连续路径的分段结果，不选新参数 |
| `research_crash_ablation.py` | 抄底通道消融、事件归因及成本后逐日路径 |
| `research_overfitting.py` | 读取消融路径，计算收益集中度和配对区块重采样区间 |

## 复现步骤

```bash
python3.8 strategy_versions.py v9 --start 2014-01-01 --end 2026-09-24
python3.8 strategy_versions.py v9.1 --start 2014-01-01 --end 2026-09-24
python3.8 evaluate_versions.py --end 2026-09-24
python3.8 -B -m unittest discover -s tests -v
python3.8 v10/freeze_v91.py
```

按需要修改起止日；应保证所有版本使用同一份数据、同一交易区间和费用假设。CLI 显示实际回测区间和引擎费用。`evaluate_versions.py` 为三个版本统一记录价格与QVIX数据哈希、截止日、费用和成交假设，输出 `reports/correctness_review.json` 与可读报告；v9.2 使用 `fz25_cv` 钩子。

`freeze_v91.py` 使用固定截止日及数据哈希，无网络请求。修复前 `v91_baseline.json` 保留不改，新回归锚为 `v91_corrected_baseline.json`；追加行情不影响旧截止日的校验，历史复权修订则明确报告数据变化。

追加过拟合诊断（本地离线，固定截止日为 2026-09-24）：

```bash
python3.8 -B research_parameter_stability.py
python3.8 -B research_crash_ablation.py
python3.8 -B research_overfitting.py
```

最后一步依赖前一步输出的 `reports/crash_ablation.json`，结果写入 `reports/`。区块重采样区间只描述已选中历史路径的条件不确定性，未校正历次规则和ETF池筛选，不是过拟合概率或未来获利概率。研究选择历史另见 `reports/research_selection_audit.md`。

这些配置固定的是参数。保留代码版本、CSV 内容哈希、截止日、费用和撮合假设，才能审计一次具体运行；当前代码修正后的结果与历史输出不同，应保留并解释差异，不能覆盖旧档案后声称逐位重现。

## 注意

- 六起点共享后段交易，是嵌套的样本内敏感性检查，不能解释为六份独立样本外证据。
- 参数邻域的平台或局部峰值不能排除反复筛选历史导致的过拟合。
- 本库含实盘信号、推送和记账组件；研究复算无需运行 `signal_daily.py`。
- 前复权历史修正可能改变历史信号，差异并非只有新增交易日的尾差。
