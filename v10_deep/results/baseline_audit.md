# 修正后 v9.2 基线诊断

只读已有路径；未重新选参或改变持仓。固定同日收盘、单边万一、无滑点，首次评估日免费。

| 区间 | 累计收益 | 年化 | 最大回撤 | 换仓 | <=3日已平仓段 | 普通五日内往返相对原仓胜/总 |
|---|---:|---:|---:|---:|---:|---:|
| full | 12432.65% | 46.32% | -24.16% | 272 | 77 | 15/49 |
| selection_2014_2025 | 7864.15% | 44.18% | -24.16% | 243 | 64 | 9/41 |
| recent_2022_2025 | 430.63% | 52.23% | -18.81% | 86 | 22 | 1/14 |
| 2026_ytd | 57.36% | 86.17% | -14.69% | 29 | 13 | 6/8 |

## 主要发现

- 全程佣金对固定持仓路径的年化拖累约 0.63 个百分点；主要改进空间若存在，应来自持仓选择和错误切换，而非把万一费用降到零。
- 2022–2025 普通五日内往返相对原仓 1/14 胜，平均 -2.43%；2026 则 6/8 胜，平均 1.75%。这些往返可能重叠，不能相加当成可赚取收益。
- 26 次已记录危机入场中，触发组合为 {'deep': 6, 'volume': 17, 'deep+volume': 1, 'deep+qvix': 2}；没有 QVIX 单独触发的记录。先验证删除冗余 QVIX 通道是否逐日同路径，不能把这一简化冒称收益提升。
- 抄底标的五日正收益 20/26，但扣两次原费率后优于原持仓固定持有的只有 15/26；绝对反弹并不自动等于更好的资产配置。

## 逐年与额外危机通道的增量

v9.1 对照来自同一修正快照和已保存回测，不是重新拟合。

| 年份 | v9.2收益 | v9.1收益 | v9.2/v9.1财富比 | 换仓 | 抄底 |
|---|---:|---:|---:|---:|---:|
| 2014 | 65.58% | 65.58% | 1.0000x | 7 | 0 |
| 2015 | 90.55% | 90.55% | 1.0000x | 19 | 1 |
| 2016 | 13.21% | 30.33% | 0.8686x | 18 | 4 |
| 2017 | 20.15% | 10.09% | 1.0913x | 22 | 3 |
| 2018 | 35.83% | 17.57% | 1.1553x | 21 | 5 |
| 2019 | 34.22% | 34.22% | 1.0000x | 17 | 0 |
| 2020 | 63.95% | 63.95% | 1.0000x | 31 | 0 |
| 2021 | 17.00% | 17.00% | 1.0000x | 22 | 0 |
| 2022 | 23.93% | 13.32% | 1.0937x | 18 | 6 |
| 2023 | 25.62% | 25.93% | 0.9975x | 19 | 1 |
| 2024 | 74.43% | 74.43% | 1.0000x | 24 | 3 |
| 2025 | 95.41% | 95.65% | 0.9987x | 25 | 1 |
| 2026 | 57.36% | 50.74% | 1.0439x | 29 | 2 |

## 待检验的经济机制

- **rotation_quality**：Add a fixed confidence/confirmation requirement only to healthy ordinary rank rotations, retaining panic and crash exceptions. Generic minimum holding/cooldowns can suppress valuable fast switches; 2026 has both helpful and harmful quick reversals.
- **crash_displacement**：Make event override conditional on whether the existing asset already offers comparable recovery/trend quality. Five-day stay-put returns are hindsight diagnostics, not inputs; event sample is small and clustered.
- **exit_symmetry**：Test a small fixed volatility-normalized emergency threshold against the existing universal 4% exit, separately from ordinary rotation confirmation. Do not improve mean return simply by removing tail protection or by tuning to isolated crisis dates.

完整 JSON 保存逐笔持仓段、进出原因代理、危机事件条件、原持仓参照、资产/角色贡献和数据指纹。

## 限制

- This is attribution of existing paths, not a new parameter scan, a validated strategy improvement or a clean OOS test.
- Corrected dividends include inferred cash flows and immediate free ex-date reinvestment; the historical same-close fill remains idealized.
- Reason proxies are reconstructed from available indicators and known rule precedence, not original execution-reason logs.
- Episode net return assigns the next switch's whole legacy 2bp fee to the exiting episode; episodes are grouped by exit date and can cross calendar boundaries.
- Roundtrips can overlap. Their stay-put comparisons must not be added as attainable incremental profits.
- Crash five-session comparisons hold the displaced asset fixed and charge two switch factors; they do not rerun the strategy without the crash.
- 2026 is partial and already observed; per-year CAGR annualizes this partial path, while total_return reports the actual observed gain.
- New mechanism tests require dated inputs, fixed budgets and explicit comparison with the same baseline; this audit does not authorize changing production.
