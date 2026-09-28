# V12-R2 用户指定主观察适配

绑定唯一候选 `v12r2_08b522c2e1aa291f054d`（完整hash见constants.py）。
用户指定它为主推送/观察模型；旧研究的 `primary=None` 与未合格结论没有被重写。
本模块不提交订单。

- WLS25原list/sum算术，四个自身有效报价的评分均值；MA180。
- 急跌尺度为**上一实际报价**的VOL20/VOL60较大值×1.4，裁剪2%—10%；不再额外滞后信号。
- 普通健康轮动最短持有2个后续基准观察日；年龄从实际虚拟入场日计算。
- 仅深跌/量能抄底，完全不读QVIX；已有5日抄底锁优先于急跌退出，锁内风险退出开关保持0。

独立账户为 `signals/shadow_v12_r2.json`，使用strategy_id、candidate_id和完整hash三重身份。
首次有效窗口NAV从1开始，不继承V9.2、V9.2+、H或个人账户。最早允许2026-09-29的新窗口，
不补记9/28。只接受当前交易日14:50—14:55内新鲜且同步的11只报价。

账本使用原始价格与份额，现金分红归原持有人、拆分按原份额记账；虚拟现金即时再投资。
未知行动/无依据缺报价拒绝推进，显式验证的零行动停牌区间只延续份额。已经封存的同日卡片只读返回。
最后提交前（包括fsync后）再次校验时效；state_path可指主生成器的隔离stage目录。

```python
from v12_live.runtime import run
run(quotes, signal_date, state_path=None, now=None, build_view=None, decide=None)
```

`now`、view和decide注入用于离线测试。查询仅使用 `shadow_v12_r2.saved_block(date)` 或canonical账本，
不能为了展示而调用run。网页使用web_id `v12-r2`，显示“V12-R2”。

离线验证：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3.8 -B -m unittest discover -s tests -p 'test_v12_live*.py' -v
```

包括3097日逐日score、decision risk feature、target/panic/crash及费用后P&L精确保真；
另以真实冻结seed＋人工9/28完成价和9/29—30报价测试实际data→policy→ledger→runtime。
测试只写临时目录、禁止网络；它们不意味着已产生实盘或前向收益。
