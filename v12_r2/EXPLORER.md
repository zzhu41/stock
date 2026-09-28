# 只读查看未合格的R2解释项

必须显式指定角色，不默认选择任何新版本；不重新回测或选型、不联网、不写账户、不下单。
原R2正式主候选和折衷候选继续为None，原53文件封存记录保持不变。

```bash
python3.8 -B -m v12_r2.explore recovery_2021
python3.8 -B -m v12_r2.explore recovery_2021 --end 2021-12-31
python3.8 -B -m v12_r2.explore recovery_2021 --lag 1 --fee-bp 11
python3.8 -B -m v12_r2.explore highest_cagr
python3.8 -B -m v12_r2.explore highest_2021
```

| 角色 | 固定ID | 主口径全期年化 / 2021 / 2026截至9-24 |
|---|---|---|
| `recovery_2021`（别名`main_three_gates_1`） | `v12r2_9a8799a5f7dd7581608e` | 55.04% / 18.70% / 46.82% |
| `highest_cagr` | `v12r2_08b522c2e1aa291f054d` | 55.27% / 7.08% / 74.91% |
| `highest_2021` | `v12r2_7743224871bb70b78781` | 50.21% / 22.60% / 37.03% |

`recovery_2021`仅指同时满足三个主口径指标的唯一解释项，不代表通过压力或整体资格。
它在延迟一天、单边11bp时全期最大回撤38.86%、2026累计−8.13%，失败原因不会被隐藏。

只支持原四情景：lag0/1 × 单边1/11bp。`--end`须为冻结路径的实际观察日。
输出包含effective config、实际评分窗口/平滑、全期及逐年统计、原完整期间失败门槛和来源哈希。
逐年`total_return`为累计收益；2026不是完整年度。截短历史不会重评原资格或免费重启账户。
`historical_close_target`只是历史模型收盘持仓，不是今天的交易建议；不接受任意ID或“救回primary”的新角色。

新入口及测试单独记录在[explore_receipt.json](explore_receipt.json)，不覆盖原release receipt。
