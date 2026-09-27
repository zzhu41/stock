# v10-H：继续迭代与复权核查

本轮在独立目录内研究，原v9/v9.1/v9.2、旧H/S和生产数据均不修改。

**结论：本轮未取得实质收益升级。** 同一修正数据下，2014-01-02—2026-09-24新H候选年化45.62%，v9.2为46.32%。全6544项的全程最高者就是原v9.2规则。

最初按原收盘口径搜索6544个配置，得到年化54.75%的候选。但归因核查发现，加减式前复权价格会放大分红ETF的历史百分比收益，这个结果已撤回为有效收益改进的证据。原v9.2的50.42%也只保留为旧输入的可复现数字。

修正批次仍采用原同收盘撮合、首日免费、后续换仓净值乘0.9998的约定；所有24只ETF统一用原始价格、公告折算比例与重建分红计量。原始与修正批次分开保存。

- [完整比较、逐年收益、消融及风险](corrected_results/REPORT.md)
- [净值与回撤图](corrected_results/comparison.png)、[PDF版](corrected_results/comparison.pdf)
- [冻结候选及代码/数据哈希](profiles.json)
- [修正协议及限制](CORRECTION_PROTOCOL.md)
- [原批次协议](PROTOCOL.md)、[文献依据](LITERATURE.md)
- [复权错误归因](results/attribution_audit.json)、[原结果状态](results/STATUS.json)
- [份额折算公告核查](results/corporate_actions_verified.json)
- [修正数据清单及每只资产的重建诊断](corrected_manifest.json)

## 使用冻结候选

在项目根目录运行（本机使用Python 3.8）：

```bash
python3.8 -B -m v10_h_close.cli backtest growth
python3.8 -B -m v10_h_close.cli backtest guarded
python3.8 -B -m v10_h_close.cli signal growth --end 2026-09-24
```

`growth`是2014–2025收益最高的配置，`guarded`是满足预定风险/近期表现/手续费门槛的配置。两者可能同一配置；若没有合格者，则不提供`guarded`。
`exploratory`只展示全历史最高者，明确带有后见选择性质。
本批它就是已有v9.2控制，不代表另一套新H；`growth`和`guarded`也选中同一个配置。

这些入口只读取历史数据、打印模型结果，不发送订单、消息或改账户持仓。`signal`是指定历史收盘的目标，不是14:50可执行指令。

## 数据和执行边界

修正序列假设现金分红在除息日收盘立即、免费再投；尚未建模支付日等待、份额取整与实盘成本。部分现金事件仍由同源价格反推，不等于全部独立核验。
`corrected_snapshots`的open字段是close占位，禁止用于次日开盘或盘中执行。修正批次未提供下一开盘收益，原批次的该项压力也不能替代修正后的真实执行验证。
历史已经多轮研究；本轮不用2026选型也不会使它重新成为干净样本外。

## 复核与复现

```bash
python3.8 -B -m unittest discover -s v10_h_close/tests -v
python3.8 -B -m v10_h_close.matrix_archive verify
```

每批的四个完整日收益矩阵均保留无损gzip与SHA256清单。重跑统计之前恢复对应矩阵：

```python
from pathlib import Path
from v10_h_close.matrix_archive import restore, verify
restore(Path("v10_h_close/corrected_results"))
verify(Path("v10_h_close/corrected_results"))
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.8 -B -m v10_h_close.corrected_scan statistics
python3.8 -B -m v10_h_close.report
```

选型和行情清单有防覆盖检查。若要从头重建，使用新的研究副本，不删除或覆写本目录冻结的选择记录。`results/`是保留的失效原输入批次；`corrected_results/`是修正批次，不能混用它们的矩阵、ID顺序或统计值。
