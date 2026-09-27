"""Render the completed bounded study and freeze its two local research profiles."""
import csv
from datetime import datetime
import hashlib
import json
import os

from .candidates import get_candidate
from .data import BASE, verify_protected
from .research import OUT, registered, sha, dump


def profiles(report):
    if report["registered"] != registered():
        raise ValueError("Evaluation fingerprints do not match current implementation")
    variants = {}
    for route, variant, name in (("high_return", "growth", "v10-H"), ("robust", "robust", "v10-S")):
        selected = report["selection"][route]
        if selected is None:
            continue
        config = json.loads(json.dumps(get_candidate(selected)))
        raw = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        variants[variant] = dict(name=name, status="frozen_research_candidate_not_deployed",
                                 selected_candidate=selected, config=config,
                                 config_sha256=hashlib.sha256(raw).hexdigest(),
                                 selection_basis="2014-2021 development only; later known history reported without re-selection",
                                 new_signal_alpha_claim=False if variant == "growth" and selected == "control_v91" else None)
    implementation = dict(report["registered"]["source_sha256"])
    implementation.update({name: sha(BASE / name) for name in ("strategy.py", "cli.py", "report.py")})
    return dict(variants=variants, evaluation_sha256=sha(OUT / "evaluation.json"),
                selection_sha256=sha(OUT / "selection.json"),
                implementation_sha256=implementation,
                data_manifest_sha256=sha(BASE / "data_manifest.json"),
                user_scope="Isolated research; existing production versions and jobs untouched")


def render(r):
    high, robust = r["selection"]["high_return"], r["selection"]["robust"]
    selected = [("v10-H", high), ("v10-S", robust), ("v9.1半仓对照", "control_v91_half")]
    out = ["# v10 两条路线：完整研究结果", "",
           "固定样本：2014-01-02 至 2026-09-11。单边佣金 0.01% + 滑点 0.10%，昨收盘信号次开盘成交；实际单位数和现金记账，延期成交后根据实际持仓继续决策。", "",
           "**本轮没有发现符合预定标准的高收益信号改进。v10-H 保留 v9.1 信号逻辑，交付为独立冻结实现，不宣称增加了收益来源。v10-S 减少了规则和换手，但其历史风险收益没有优于半仓 v9.1；不建议仅凭“稳健”名称替换现有策略。**", "",
           "## 所选版本与仓位对照", "",
           "| 版本 | 净年化 | 最大回撤 | 年化波动 | 平均非货币ETF占比 | 年均调仓日 | 双向年均金额换手 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for label, name in selected:
        if not name:
            continue
        m = r["summaries"][name]["full"]
        out.append("| %s | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f | %.2f倍 |" % (
            label, m["cagr"] * 100, m["max_dd"] * 100, m["volatility"] * 100,
            m["average_nonmoney_exposure"] * 100, m["rebalances_per_year"], m["turnover_per_year"]))
    out += ["", "v10-S 的非货币仓位包括国债ETF，不能把它与全权益风险直接等同。半仓对照在风险资产切换时重新分配50%/50%，持有期间允许比例漂移；其结果仅为仓位压缩参照，不能据此证明原策略没有选型偏差。", "",
            "## 选择顺序与分段", "",
            "先保存注册文件，再仅运行2014–2021，保存两个选择ID，最后运行后段与成本/邻域。所有区间都已在项目历史研究中见过，下面的validation不是真正未见样本外。", "",
            "| 版本 | 2014–2021开发年化 | 2022–2025已见验证年化 | 2026截至09-11累计收益 |",
            "|---|---:|---:|---:|"]
    for label, name in selected:
        if not name:
            continue
        s = r["summaries"][name]
        out.append("| %s | %.2f%% | %.2f%% | %+.2f%% |" % (
            label, s["development"]["cagr"] * 100, s["validation"]["cagr"] * 100,
            s["report_only_2026"]["total_return"] * 100))
    out += ["", "所选高收益ID：`%s`；所选稳健ID：`%s`。验证后未换赢家。原始登记和选择时间分别见 `registration.json`、`selection.json`。" % (high, robust), "",
            "## 全部10项结果（含失败项）", "",
            "| ID | 开发年化 | 开发回撤 | 全程年化 | 全程回撤 | 2022–2025年化 |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, s in r["summaries"].items():
        out.append("| %s | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% |" % (
            name, s["development"]["cagr"] * 100, s["development"]["max_dd"] * 100,
            s["full"]["cagr"] * 100, s["full"]["max_dd"] * 100, s["validation"]["cagr"] * 100))
    out += ["", "两个Top2长周期候选的开发期回撤超过50%，未通过预定稳健门槛。长周期、少参数本身不保证低风险；固定五份额的分散是所选稳健版本的重要结构差异。", "",
            "## 逐年收益", "", "| 年份 | v10-H | v10-S | v9.1半仓对照 |", "|---|---:|---:|---:|"]
    if high and robust:
        for y in r["summaries"][high]["full"]["yearly"]:
            vals = [r["summaries"][name]["full"]["yearly"][y] * 100 for _, name in selected]
            out.append("| %s | %+.2f%% | %+.2f%% | %+.2f%% |" % ((y if y != "2026" else "2026截至09-11"), *vals))
    out += ["", "各年度沿用同一条持仓路径，年度收益复利连乘等于全程净值，不是每年重新从现金开始。", "",
            "## 成本敏感性", "",
            "下表只改变单边滑点，佣金始终为单边1bp；1bp=0.01%。", "",
            "| 版本 | 5bp滑点年化 | 10bp主情景 | 20bp滑点年化 | 50bp滑点年化 |",
            "|---|---:|---:|---:|---:|"]
    for label, name in selected:
        if name:
            out.append("| %s | %s |" % (label, " | ".join("%.2f%%" % (r["costs"][name][str(k)]["full"]["cagr"] * 100) for k in (5, 10, 20, 50))))
    out += ["", "## 预定窗口邻域（只诊断，不重新选型）", "",
            "| 路线 | 窗口倍率 | 实际窗口 | 全程年化 | 最大回撤 |", "|---|---:|---|---:|---:|"]
    for n in r["neighbors"]:
        field = "score_windows" if n["config"]["kind"] == "legacy" else "lookbacks"
        m = n["metrics"]["full"]
        out.append("| %s | %.1f | %s | %.2f%% | %.2f%% |" % (
            n["route"], n["scale"], n["config"][field], m["cagr"] * 100, m["max_dd"] * 100))
    out += ["", "## 收益归因与覆盖", "",
            "每个资产的收盘单位数、隔夜与日内损益，以及佣金/滑点均按实际成交重建。各资产贡献转换为可加的对数增长，合计严格等于全程log净值；这是归因，不是删除该资产后的可交易回测。", "",
            "| 版本 | 最佳10日置零后的账面年化 | 最佳10日占净对数增长 | 延期调仓次数 |", "|---|---:|---:|---:|"]
    for label, name in selected:
        if name:
            s = r["summaries"][name]
            n = s["attribution"]["neutralize_best_days"][1]
            out.append("| %s | %.2f%% | %.2f%% | %d |" % (
                label, n["cagr"] * 100, n["share_net_log_growth"] * 100, s["full"]["deferred_rebalances"]))
    out += ["", "| ETF | 快照首日 | 首次可计算244观察期收益 | 首次满足旧规则270根预热 |", "|---|---|---|---|"]
    for code, coverage in r["coverage"].items():
        out.append("| %s | %s | %s | %s |" % (code, coverage["first"], coverage["first_244_return"], coverage["first_270_row_legacy_feature"]))
    out += ["", "各次月度信号的可用/不可用资产名单保存在 `paths.json` 的 policy_metadata.rebalance_events；早期未成熟资产的份额留货币，不回填预上市历史。", "",
            "## 与旧回测数字的区别", "",
            "本轮截至09-11，此前v9报告截至09-24；且本轮根据实际成交持仓重新决策，原报告为固定收盘目标路径重放。510500在2015-04-13、04-14缺行，新执行器保留待成交订单并于04-15开盘成交，导致两条路径在4个日期不同。佣金也严格按实际成交金额扣除。不能把本轮37.69%与旧报告37.13%的差解释成新规则增加了收益。", "",
            "冻结策略源码与原策略逻辑AST一致，同状态下独立指标与决策相同；2016–2021两个对照各1461日目标一致。2014–2021的4日执行差异保留在 `legacy_fidelity.json`，没有强行抹平。", "",
            "## 适用边界", "",
            "- 参数更少、月度调仓、现金回退限制本轮研究自由度，但不保证未来盈利或固定最大回撤。",
            "- 资产池仍来自事后可获得的ETF集合，并非完整历史存续基金数据库；复权数据也可能修订。",
            "- 尚未模拟盘口、真实限价可成交性、涨跌停封单、资金规模冲击、整手约束和实际申赎/溢价约束。",
            "- 月度信号不意味着月内有止损，趋势反转时仍会承受亏损。v10-S在2016、2018、2022年亏损。",
            "- 暂不接入现有每日推送或实盘账本，下一步应冻结规则并积累独立未来数据及可核查实际成交。",
            "", "本轮7个新候选、3个控制，30个额外成本情景、4个窗口诊断；全部结果保留，不把本轮预算冒称全项目独立试验数。", "",
            "![净值与回撤](equity_drawdown.svg)", ""]
    return "\n".join(out)


def plot(paths, report):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/v10-next-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    chosen = [("v10-H (v9.1 rules)", report["selection"]["high_return"], "#2455a4"),
              ("v10-S (annual trend)", report["selection"]["robust"], "#15806d"),
              ("v9.1 half exposure", "control_v91_half", "#757575")]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for label, name, color in chosen:
        if not name:
            continue
        daily = paths["results"][name]["daily"]
        dates = [datetime.fromisoformat(row[0]) for row in daily]
        values = [row[1] for row in daily]
        peak, dd = 1.0, []
        for v in values:
            peak = max(peak, v)
            dd.append((v / peak - 1) * 100)
        axes[0].plot(dates, values, label=label, color=color, linewidth=1.6)
        axes[1].plot(dates, dd, color=color, linewidth=1.2)
    for ax in axes:
        ax.axvspan(datetime(2022, 1, 1), datetime(2026, 1, 1), color="#2455a4", alpha=.05)
        ax.axvspan(datetime(2026, 1, 1), datetime(2026, 9, 11), color="#e9bb48", alpha=.12)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wealth (initial = 1, log scale)")
    axes[0].legend(loc="upper left", frameon=False)
    axes[0].set_title("Isolated v10 study | 2014-01-02 to 2026-09-11", loc="left", weight="bold")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(.09, .018, "Next-open actual holdings; 1bp commission + 10bp slippage per side.\n"
             "Shading: 2022-2025 known-history validation; 2026 report only. No clean historical OOS sample.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(OUT / "equity_drawdown.svg", bbox_inches="tight")
    fig.savefig(OUT / "equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    verify_protected()
    r = json.loads((OUT / "evaluation.json").read_text())
    p = profiles(r)
    dump(BASE / "profiles.json", p)
    (OUT / "REPORT.md").write_text(render(r), encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate", "period", "cagr", "total_return", "max_dd", "volatility", "sharpe", "average_nonmoney_exposure", "rebalances_per_year"])
        for name, s in r["summaries"].items():
            for period in ("full", "development", "validation", "report_only_2026"):
                m = s[period]
                writer.writerow([name, period] + [m[k] for k in ("cagr", "total_return", "max_dd", "volatility", "sharpe", "average_nonmoney_exposure", "rebalances_per_year")])
    plot(json.loads((OUT / "paths.json").read_text()), r)
    verify_protected()
    print("Frozen profiles:", ", ".join(v["name"] for v in p["variants"].values()))
    print("Report:", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
