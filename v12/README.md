# V12 研究档案

本轮完成三阶段搜索，合并去重后1,807个配置。**正式合格策略为空；保留三个研究对照，未接入每日推送或动量查询。** 生产继续只有V9.2、V9.2+、V10-H。

- [研究判断和主要问题](REPORT.md)
- [全期、逐年、费用及现金精度数值](RESULTS.md)
- [原协议](PROTOCOL.md)、[B扩展登记](CONSENSUS_PROTOCOL.md)、[C扩展及停止边界](EXANTE_PROTOCOL.md)
- [诊断冻结计划](DIAGNOSTIC_PROTOCOL.md)、[历史重选计划](RESELECTION_PROTOCOL.md)
- [论文及实际采用边界](LITERATURE.md)
- [冻结配置](profiles.json)、[仅模型运行依赖](runtime_manifest.json)

数据固定至2026-09-24；2026已经参与本轮已知历史优化，不是样本外验证。主口径是修正总回报、理想同收盘、单边1bp、首日免费、244交易日年化。没有14:50成交价、下一开盘、真实整手、分红到账延迟或未来收益保证。

## 只读复算

Python3.8+、NumPy。以下入口用独立Python状态机，不写模型缓存、前向账户或推送，不拉实时行情。`balanced_a/b/c`都是未通过全部资格的对照；默认`balanced_c`只为研究入口方便，不代表生产推荐。

```bash
python3.8 -B -m v12.cli backtest balanced_c
python3.8 -B -m v12.cli backtest balanced_b --lag 1 --fee-bp 11
python3.8 -B -m v12.cli backtest h
python3.8 -B -m v12.cli signal balanced_c --end 2026-09-24
```

还可指定`balanced_a`、`v9`、`v91`、`v92`、`simple`；simple即V9.2+。`signal`输出指定历史收盘的模型目标，不是当前交易指令。历史截止必须是冻结样本内实际交易日，波动和评分在物理截断的前缀上构建。

## 复现证据

原始矩阵与大JSON已无损压缩，完整原始/压缩字节SHA及大小在`path_archives.json`。Git内保留压缩件，以下恢复只补缺失文件，已有内容不一致时拒绝覆盖。

```bash
python3.8 -B -m v12.archives verify
python3.8 -B -m v12.archives restore
python3.8 -B -m v12.verify
OPENBLAS_NUM_THREADS=1 python3.8 -B -m unittest discover -s v12/tests -v
```

完整扫描使用g++/OpenMP，绘图使用Matplotlib。A/B/C选择与源收据已经冻结，扫描工具会拒绝覆盖已完成的选择；不要删除选择文件后悄悄改参数重跑。复现实测数值可用只读CLI和归档中的逐日路径。

`protected_manifest.json`证明本轮研究未改动895个旧代码/研究文件。它是研究实施期的保护检查，不把以后正常维护推送页面当作模型失效；只读CLI校验的是`runtime_manifest.json`内模型和输入依赖。
