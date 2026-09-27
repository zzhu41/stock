# Fixed portfolio diversification experiment

36 configurations were registered before their simulations. The S choice uses 2014–2025 only; the already known 2026 tail is report-only and is not a clean holdout.

S: `None`; eligible candidates: 0. Portfolio return champion: `top1_wls25_v20_original_legacy_gate`. This champion does not replace the single-ETF H choice.

Close-only corrected TR indices; immediate free ex-date dividend reinvestment. Open columns are placeholders and are never executed. First evaluation day is free; thereafter cost is pre-rebalance NAV × fee × L1 security-weight turnover. Both 1 bp and 5 bp paths are freshly simulated. Units drift until membership changes. A missing necessary quote discards the entire rebalance without pending orders.

Top1 here is the no-crash, no momentum-gap-buffer control for this portfolio family; it is not v9.2. All k values can fall back to one asset. Selection requires configured k ≥ 2, not two actual members on every date.

| Candidate | Train CAGR | Train DD | Full CAGR | Full DD | Full 5 bp CAGR | 2022–25 CAGR |
|---|---:|---:|---:|---:|---:|---:|
| v9 | 42.2101% | -24.1648% | 43.2398% | -24.1648% | 40.9636% | 48.2689% |
| v9.1 | 42.0690% | -24.1648% | 43.8058% | -24.1648% | 41.5652% | 48.9771% |
| v9.2 | 44.1839% | -24.1648% | 46.3175% | -24.1648% | 43.8289% | 52.2325% |
| top1_wls25_v20_original_legacy_gate | 29.9967% | -25.1864% | 31.0114% | -25.1864% | 27.9493% | 21.9614% |

The 36-row development/evaluation files retain every candidate, yearly returns, and period statistics. Each path archive contains both fee runs with daily NAV/weights, returns, signals, actual model fills, and missing-price diagnostics. Full runs preserve both training return prefixes exactly. The three old controls are copied from the prior frozen mechanisms artifacts, not resimulated.

This is adaptive research on previously inspected history. Registration and a frozen selection prevent changing this 36-member set after its results; they do not erase prior selection bias or prove expected live performance.
