# V12-R2研究档案

[报告](REPORT.md) · [事前协议](PROTOCOL.md) · [2021诊断](diagnosis/REPORT.md)

严格与公开折衷均无合格升级；保留完整866条已知历史路径与解释项，生产版本不变。

```bash
python3.8 -B -m v12_r2.cli backtest c4
python3.8 -B -m v12_r2.cli backtest c --lag 1 --fee-bp 11
python3.8 -B -m v12_r2.cli signal h --end 2026-09-24
```

只读历史查询，不拉行情、不写账户、不下单。2026是部分年度，且已参与优化。
