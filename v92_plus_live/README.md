# V9.2+ 独立前向适配

对应 `v10_deep/profiles.json` 的冻结 `simple` 配置（`vd_d2a02ab14be481f562fd`）：V9.2只增加普通健康轮动最短持有2日。WLS25、MA250、固定4%急跌触发及原三通道抄底/5日锁仓不变，2日限制不覆盖原有风险和抄底例外。

`policy.py`使用明确的原WLS25算术及冻结参考决策器，按实际入场日期计算持有年龄；`ledger.py`维护独立原始份额、分红权益和净值。新账户首次有效信号从1开始，位于`signals/shadow_v92_plus.json`，不复制V9.2或H的持仓、净值、事件。

生产入口是`daily_job.py`→`signal_daily.py`→`daily_extras.py`：三个版本共享同一批校验TR/原始行情和封存QVIX快照，各自在隔离stage目录准备，主journal统一提交。`shadow_v92_plus.saved_block`仅作只读查询/恢复；独立测试run须显式传临时state_path，不运行生产账户作为测试。

当前为虚拟前向观察，不向券商下单。14:50价格与累计量不等于历史最终收盘；现金分红使用虚拟即时再投资约定，不模拟到账延迟、整手与真实成交限制。代码不把历史50.13%年化或2026的75.27%累积收益写入前向账户。

验证：

```bash
python3.8 -B -m unittest tests.test_v92_plus_live -q
python3.8 -B -m unittest discover -s tests -q
```

测试覆盖3097交易日评分/目标与冻结simple、同状态min_hold0基准的逐值一致，以及两日持有、风险例外、锁仓、公司行动、同日幂等和提交前时效。
