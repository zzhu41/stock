"""Export frozen second-round candidates, including numerical near-ties."""
import csv
import hashlib
import json
import os

from .research import BASE, OUT, dump, sha, registered, protect
from .registry import get_candidate


def freeze_profiles(r):
    if registered() != r["registration"]:
        raise ValueError("Evaluation/source fingerprints differ")
    own = dict(r["registration"]["source_sha256"])
    own.update({p: sha(BASE / p) for p in ("cli.py", "report.py")})
    parent = dict(r["registration"]["parent_source_sha256"])
    parent["cli.py"] = sha(BASE.parent / "v10_next/cli.py")
    variants = {}
    for route, variant, name in (("high_return", "growth", "v10-H round2"), ("robust", "robust", "v10-S round2")):
        selected = r["selection"][route]
        if selected is None:
            continue
        config = json.loads(json.dumps(get_candidate(selected)))
        variants[variant] = dict(name=name, selected_candidate=selected, config=config,
                                 config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
                                 scope="Frozen research candidate, not deployed",
                                 selection_period="2014-2021 only; later known history did not change selection",
                                 numerical_targets=r["primary_acceptance"],
                                 promotion_assessment=("Near-tie with v9.2; no demonstrated incremental-return advantage"
                                                       if variant == "growth" else "Historical 30% target met; lower overfitting is not established"))
    return dict(variants=variants, own_source_sha256=own, parent_source_sha256=parent,
                prices_manifest_sha256=r["registration"]["prices_manifest_sha256"],
                artifact_sha256={p: sha(BASE / p) for p in ("results/evaluation.json", "results/selection.json", "qvix_manifest.json")})


def markdown(r):
    h, s = r["selection"]["high_return"], r["selection"]["robust"]
    main = [("v9.1对照", "c_v91"), ("v9.2对照", "c_v92"), ("v10-H第二轮", h), ("v10-S第二轮", s)]
    delta = r["primary_acceptance"]["full"]["high_difference"]
    out = ["# 第二轮：H超越v9.2、S年化至少30%的研究结果", "",
           "**S 找到了达到历史净年化30%%门槛、回撤较低的候选。H仅数值上高出v9.2约%.6f个百分点，经济上几乎持平，没有足够证据宣称收益升级。**" % (delta * 100), "",
           "固定区间2014-01-02至2026-09-11；主口径为收盘决策、次日开盘实际成交，单边佣金0.01%＋滑点0.10%。所有策略使用同一份价格、同一执行引擎和费用；本轮不更新或改动此前版本。", "",
           "## 公平对照", "",
           "| 版本 | 次开盘成本后年化 | 最大回撤 | 2022–2025净年化 | 2026截至09-11累计收益 | 理想同收盘年化 |",
           "|---|---:|---:|---:|---:|---:|"]
    for label, name in main:
        if not name:
            continue
        m = r["summaries"][name]
        out.append("| %s | %.5f%% | %.2f%% | %.2f%% | %+.2f%% | %.2f%% |" % (
            label, m["full"]["cagr"] * 100, m["full"]["max_dd"] * 100,
            m["validation"]["cagr"] * 100, m["report_only_2026"]["total_return"] * 100,
            r["ideal_close"][name]["full"]["cagr"] * 100))
    out += ["", "理想同收盘单边佣金仍为0.01%、滑点为0，是读取完整收盘信息后按该收盘价成交的反事实诊断；不用于本轮选型，不能与次开盘结果混为一谈。**H在理想收盘口径没有超过v9.2**；若用户要求两种口径都超过，本轮H不满足。", "",
            "## 冻结的两条路线", "",
            "- H：`%s`。保留原深跌和QVIX通道，删除量能通道，使用固定WLS25/30分数均值。少一个事件通道不等于已经证明更少过拟合；QVIX日期与发布时间仍有限制。" % h,
            "- S：`%s`。v9.1底座只把WLS25改为WLS30，其余规则和池不变，保留原深跌抄底；这是参数稳健候选，**没有减少全部规则数量或消除危机事件依赖**。" % s,
            "", "选择在2014–2021计算后即写入selection.json；2022–2025和2026结果随后揭示。主验收和强化检查的布尔值保留原协议定义，没有事后新增1pp门槛或更换冠军；不过数值上严格大于，并不自动等于有经济意义或统计支持。", "",
            "## 所有9项（含未选中候选）", "",
            "| ID | 开发期年化 | 开发期回撤 | 全程年化 | 全程回撤 | 2022–2025年化 |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, d in r["summaries"].items():
        out.append("| %s | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% |" % (
            name, d["development"]["cagr"] * 100, d["development"]["max_dd"] * 100,
            d["full"]["cagr"] * 100, d["full"]["max_dd"] * 100, d["validation"]["cagr"] * 100))
    out += ["", "例如`h_qvix_only`全程收益高于本轮冻结的H，但它不是开发期按协议选出的版本，不能看完后段后换它当‘通过验证的赢家’。它保留为已知历史上的备选假设，未获得新的独立验证。", "",
            "## 成本与窗口敏感性", "",
            "| 版本 | 单边10bp滑点 | 20bp滑点 | 50bp滑点 |", "|---|---:|---:|---:|"]
    for label, name in main:
        if name:
            out.append("| %s | %s |" % (label, " | ".join("%.2f%%" % (r["costs"][name][str(bp)]["full"]["cagr"] * 100) for bp in (10, 20, 50))))
    out += ["", "滑点之外均再计单边1bp佣金；30%是主情景历史目标，不是任何成本下都成立。", "",
            "| 路线 | 全部同族窗口倍率 | 全程年化 | 最大回撤 |", "|---|---:|---:|---:|"]
    for n in r["neighbors"]:
        out.append("| %s | %.1f | %.2f%% | %.2f%% |" % (
            n["route"], n["scale"], n["metrics"]["full"]["cagr"] * 100,
            n["metrics"]["full"]["max_dd"] * 100))
    out += ["", "邻域仅诊断，不把较好的邻居重新发布成赢家。WLS30此前已经被研究过，本轮重新比较不能称全新发现。", "",
            "## H的增量不确定性", "",
            "按同步日期、循环连续区块，配对重采样10,000次；以下为H减v9.2的年化收益差区间（百分点）：", "",
            "| 区块长度 | 实际差 | 条件95%区间 |", "|---|---:|---:|"]
    for b in r["high_paired_bootstrap"]:
        lo, hi = b["conditional_percentile_95"]
        out.append("| %d日 | %+.6f | [%+.2f, %+.2f] |" % (
            b["block_size"], b["observed_cagr_difference"] * 100, lo * 100, hi * 100))
    out += ["", "区间全部跨零，而且没有校正历次选型偏差；不能解释为PBO、未来收益概率或严格的样本外显著性。2026仅报告期内H亏损，也没有触发事后补规则。", "",
            "## 逐年收益（主成本口径）", "",
            "| 年份 | v9.2 | v10-H第二轮 | v10-S第二轮 |", "|---|---:|---:|---:|"]
    if h and s:
        for year in r["summaries"]["c_v92"]["full"]["yearly"]:
            vals = [r["summaries"][name]["full"]["yearly"][year] * 100 for name in ("c_v92", h, s)]
            out.append("| %s | %+.2f%% | %+.2f%% | %+.2f%% |" % (year if year != "2026" else "2026截至09-11", *vals))
    out += ["", "## 数据、计时和证据边界", "",
            "- 原策略、第一轮v10及行情/账本共177个保护文件哈希未变；第二轮只新增本目录。",
            "- 价格沿用第一轮截止09-11快照。QVIX截至该日最后可用行是09-09，09-10与09-11按严格当日规则停用QVIX通道，不前填。历史实际发布时间仍无法核实；next-open只能假设前日值此前已公开。",
            "- 同收盘诊断使用同一单位数/费用求解器，并将五个完整持有收益区间对应到收盘入场时钟；不是多持有一天来增加收益。无缺行短窗的目标与旧lab已核对。",
            "- 独立账户组合初始均分，此后财富占比漂移；没有账户间补仓、周期重置、杠杆或相反交易免佣。逐账户状态、份额和待成交指令均保留。",
            "- 原ETF池、短周期规则和稀有抄底事件仍有历史选择偏差；全部区间均已被研究过，不能称干净样本外。",
            "- 日线模拟未解决盘口、涨跌停封单、资金冲击、整手交易、复权修订和QDII溢价成交等问题。",
            "", "本轮登记2个对照、7个候选；另外18个成本情景、9个同收盘诊断、4个邻域，没有按结果追加候选。完整路径及精确资产/成本归因保存在paths.json/evaluation.json。", "",
            "方法背景：[回测过拟合概率](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)、[Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)；不能用重复挑选出来的高收益直接推断未来可靠性。", "",
            "![净值与回撤](equity_drawdown.svg)", ""]
    return "\n".join(out)


def plot(r, paths):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/v10-round2-matplotlib")
    from datetime import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    rows = [("v9.2 control", "c_v92", "#777777"),
            ("v10-H round 2", r["selection"]["high_return"], "#2455a4"),
            ("v10-S round 2", r["selection"]["robust"], "#15806d")]
    for label, name, color in rows:
        if not name:
            continue
        ds = paths[name]["daily"]
        dates = [datetime.fromisoformat(x[0]) for x in ds]
        values = [x[1] for x in ds]
        peak, drawdowns = 1, []
        for nav in values:
            peak = max(peak, nav)
            drawdowns.append((nav / peak - 1) * 100)
        axs[0].plot(dates, values, color=color, label=label, linewidth=1.6)
        axs[1].plot(dates, drawdowns, color=color, linewidth=1.2)
    for ax in axs:
        ax.axvspan(datetime(2022, 1, 1), datetime(2026, 1, 1), color="#2455a4", alpha=.05)
        ax.axvspan(datetime(2026, 1, 1), datetime(2026, 9, 11), color="#e9bb48", alpha=.12)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Wealth (initial = 1, log scale)")
    axs[0].set_title("V10 round 2 | Net next-open execution | 2014-2026-09-11", loc="left", weight="bold")
    axs[0].legend(frameon=False)
    axs[1].set_ylabel("Drawdown (%)")
    axs[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(.09, .02, "1bp commission + 10bp slippage per side. H is economically tied with v9.2.\n"
             "Known-history validation in blue; 2026 report-only in yellow. No clean historical OOS sample.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(OUT / "equity_drawdown.svg", bbox_inches="tight")
    fig.savefig(OUT / "equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    protect()
    r = json.loads((OUT / "evaluation.json").read_text())
    dump(BASE / "profiles.json", freeze_profiles(r))
    (OUT / "REPORT.md").write_text(markdown(r), encoding="utf-8")
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "period", "net_cagr", "total_return", "max_drawdown", "volatility"])
        for name, rows in r["summaries"].items():
            for period in ("full", "development", "validation", "report_only_2026"):
                m = rows[period]
                w.writerow([name, period, m["cagr"], m["total_return"], m["max_dd"], m["volatility"]])
    plot(r, json.loads((OUT / "paths.json").read_text()))
    protect()
    print("Frozen second-round research profiles and REPORT.md written")


if __name__ == "__main__":
    main()
