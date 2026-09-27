"""Disclose both the frozen developmental choice and labelled hindsight results."""
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import json
import os

from .data import BASE, UNIVERSE, protect, sha, load_histories
from .scan import OUT, fingerprints, dump
from v10_next.metrics import attribution


def load():
    report = json.loads((OUT / "evaluation.json").read_text())
    configs = json.loads((OUT / "registry.json").read_text())["candidates"]
    diagnostics = json.loads((OUT / "search_diagnostics.json").read_text())
    followup = json.loads((OUT / "regime_followup.json").read_text())
    paths = json.loads((OUT / "selected_paths.json").read_text())
    return report, {c["id"]: c for c in configs}, diagnostics, followup, paths


def profiles(r, configs, f):
    if r["fingerprints"] != fingerprints():
        raise ValueError("Registered implementation changed")
    own = dict(r["fingerprints"]["source_sha256"])
    own.update({name: sha(BASE / name) for name in ("cli.py", "report.py", "regime_followup.py")})
    deps = dict(r["fingerprints"]["dependencies_sha256"])
    deps["v10_next/cli.py"] = sha(BASE.parent / "v10_next/cli.py")
    variants = {}
    entries = [("growth", r["selection"]["champion"], "v10-H development choice",
                "development_2014_2021_only", "Research only: failed later relative-performance validation"),
               ("exploratory", r["hindsight_winner"]["id"], "v10-H hindsight maximum",
                "full_history_2014_2026_hindsight", "Exploratory: high drawdown, no clean OOS confirmation"),
               ("regime", f["candidate"]["id"], "v10-H open-stock exploratory candidate",
                "post_search_best_open_stock_mode_on_full_history", "Exploratory: not the frozen development champion")]
    for variant, name, label, method, status in entries:
        if name:
            c = deepcopy(configs[name])
            variants[variant] = dict(name=label, selected_by=method, status=status, config=c,
                                     candidate_hash=c["candidate_hash"], not_deployed=True, clean_oos=False)
    return dict(variants=variants, source_sha256=own, dependencies_sha256=deps,
                artifact_sha256={p: sha(BASE / p) for p in ("data_manifest.json", "results/evaluation.json",
                    "results/selection.json", "results/search_diagnostics.json", "results/regime_followup.json")})


def rules(config):
    p = config["params"]
    score = ("WLS%s / VOL20" % "/".join(map(str, config["score_windows"]))
             if config["score_mode"] == "wls" else config["score_mode"])
    return ["- A股池：" + "、".join("%s(%s)" % (UNIVERSE[c][0], c) for c in config["stock_pool"]),
            "- 免A股体制门的资产：" + "、".join("%s(%s)" % (UNIVERSE[c][0], c) for c in config["global_pool"]),
            "- 黄金518880、货币511880；满仓单标的，无杠杆。",
            "- 排名：%s；体制模式：`%s`。" % (score, config["regime_mode"]),
            "- 熊市普通进场门：%.1f%%；急跌退出：%.1f%%；过热阈值：%.1f%%。" % (
                p["bear_enter_mom"] * 100, p["panic_drop"] * 100, p["overheat"] * 100),
            "- 分池缓冲：%s；统一缓冲：%.1f%%；永不空仓：%s。" % (p["pool_buffer"], p["buffer"] * 100, p["never_empty"]),
            "- 深跌抄底：%s；开启时MOM5≤−8%%且价格比自身MA250低超过20%%（价格<0.8×MA250），实际买入后持有5个交易收益区间。" % bool(config["channels"])]


def markdown(r, configs, d, f):
    rows = {x["id"]: x for x in r["results"]}
    champion, best, regime = r["selection"]["champion"], r["hindsight_winner"]["id"], f["candidate"]["id"]
    chosen = [("v9.2对照", "c_v92"), ("开发期冻结选择", champion),
              ("全历史回看最高", best), ("熊市开放A股探索候选", regime)]
    allrows = dict(rows, **r["controls"])
    out = ["# v10-H大规模搜索：5,156配置及去牛熊消融", "",
           "**找到了更高的历史回测，但没有找到能够据此宣布可靠升级的强H。** 全历史最高净年化39.31%伴随46.52%最大回撤；另一小池、WLS30、熊市开放A股的探索候选为39.08%/−26.88%。开发期预先选中的版本在后段失败，不能用后来更好看的候选替换它后声称验证成功。", "",
           "固定共同区间2014-01-02至2026-09-11。主口径：收盘信号、次日开盘实际持仓反馈、单边佣金0.01%＋滑点0.10%，初始和全部实际买卖收费，无杠杆。全部历史均已被项目研究过，不是干净样本外。", "",
           "## 核心比较", "",
           "| 配置 | 全程净年化 | 最大回撤 | 2022–2025年化 | 2026截至09-11累计 | 理想同收盘年化 |",
           "|---|---:|---:|---:|---:|---:|"]
    for label, name in chosen:
        if not name:
            continue
        row = allrows[name]
        ideal = f["ideal_close"]["cagr"] if name == regime else r["ideal_close"][name]["cagr"]
        out.append("| %s | %.2f%% | %.2f%% | %.2f%% | %+.2f%% | %.2f%% |" % (
            label, row["full"]["cagr"] * 100, row["full"]["max_dd"] * 100,
            row["validation"]["cagr"] * 100, row["report_only_2026"]["total_return"] * 100, ideal * 100))
    out += ["", "理想同收盘是完整收盘数据生成信号后仍按同一收盘价成交的反事实诊断，含1bp佣金、无滑点；不可把它与主口径混淆，也不等同14:50实盘。", "",
            "## 枚举范围与选择记录", "",
            "原始5,164项，语义去重后5,156项；所有候选先开发期、再全程复算，共10,312条主候选回测，另有控制与诊断。没有只保留赢家。",
            "开发期按2014–2017、2018–2021及全开发区间三项净年化均超过v9.2筛选，再按开发净CAGR固定排序。只有2项合格，且开发期持仓/收益路径完全相同：差别只是是否保留2023年才有效的513120，不能把它们当两份独立确认。冠军由预定ID字典序决定。", "",
            "开发冻结者为v9.1底座增加房地产ETF512200。开发年化38.30%，但2022–2025只有28.36%，低于同期v9.2的42.20%。这项失败完整保留。其开发前半段对v9.2的优势来自已有v9.1规则差异，不能全归功于新增房地产ETF。", "",
            "本轮5,156配置对应%d条开发期不同收益路径、%d条全程不同路径；配置数不是独立假设数。" % (r["unique_return_paths_development"], r["unique_return_paths_full"]), "",
            "| 家族 | 全程最高净年化 | 对应最大回撤 |", "|---|---:|---:|"]
    for family, row in r["family_best"].items():
        out.append("| %s | %.2f%% | %.2f%% |" % (family, row["full"]["cagr"] * 100, row["full"]["max_dd"] * 100))
    out += ["", "各家族最高者都是回看挑选，表格只是搜索全景，不是七次独立验证。", "",
            "## 用户提出的牛熊门问题：保持其他参数不变", "",
            "以下以原v9.1池、WLS25及其他参数为锚，比较已经登记的配对配置：", "",
            "| 唯一体制开关变化 | 净年化 | 最大回撤 |", "|---|---:|---:|"]
    labels = {"ma250": "原MA250体制", "open_stock": "启用原bear_open_stock开关", "always_bull": "取消体制，固定原牛市路径", "all_assets": "取消体制，股票/跨境/黄金统一竞赛"}
    for mode, row in f["original_v91_anchor_ablation"].items():
        out.append("| %s | %.2f%% | %.2f%% |" % (labels[mode], row["full"]["cagr"] * 100, row["full"]["max_dd"] * 100))
    out += ["", "这里直接去门控没有改善原v9.1。39.08%的探索候选还同时缩小了池、把WLS25改为30，因此不能说‘只去掉牛熊门就把37.69%提升到39.08%’。", "",
            "`open_stock`沿用旧开关的完整语义：沪深300体制仍用于熊市7%门槛；股票与免体制门资产竞赛，黄金转为备胎/兜底，不再作为原熊市常规竞赛成员。它既不是完全不看体制，也不是保持黄金角色不变的纯A股放行。`always_bull`取消的是沪深300MA250体制判断；开启抄底时仍使用各资产自身年线距离。", "",
            "在所有能够找到已登记、完全同参数MA250对照的配置中：", "",
            "| 模式 | 已匹配对照数 | 全程收益提高数 | 配对年化差中位数 |", "|---|---:|---:|---:|"]
    for mode, p in f["matched_regime_pairs"].items():
        out.append("| %s | %d | %d | %+.2f个百分点 |" % (mode, p["matched"], p["higher_full_cagr"], p["median_full_difference"] * 100))
    out += ["", "每种模式其余40项缺少完全一致的已登记对照，没有事后补造它们以扩大样本。这些配对也相关、也已见历史，不能作为独立因果试验。", "",
            "## 三个交付配置的完整规则", ""]
    for label, name in chosen[1:]:
        if name:
            out += ["### " + label + "：`" + name + "`", ""] + rules(configs[name]) + [""]
    out += ["## 成本与已登记窗口", "",
            "| 配置 | 单边10bp滑点 | 20bp滑点 | 50bp滑点 |", "|---|---:|---:|---:|"]
    for label, name in chosen:
        if name:
            cost = f["costs"] if name == regime else r["costs"][name]
            out.append("| %s | %.2f%% | %.2f%% | %.2f%% |" % (
                label, allrows[name]["full"]["cagr"] * 100, cost["20"]["full"]["cagr"] * 100, cost["50"]["full"]["cagr"] * 100))
    out += ["", "各情景均另计单边1bp佣金。开发TopK、回看最高者和3控制按原登记补了12个成本情景及5个同收盘情景；冠军另有2个预定窗口诊断。", "",
            "按用户去牛熊问题，看完全程后另对open_stock探索候选补了20/50bp及同收盘共3个诊断。记录单独保存在regime_followup.json，不改注册、冻结选择或候选数，也不补发独立验证标签。以下20/25/30/40窗口均来自原有登记矩阵，只是后验查看，不是新调参：", "",
            "| 小池开放熊市A股的WLS窗口 | 全程净年化 | 最大回撤 |", "|---|---:|---:|"]
    for n in f["registered_window_neighbors"]:
        out.append("| %d | %.2f%% | %.2f%% |" % (n["window"], n["metrics"]["full"]["cagr"] * 100, n["metrics"]["full"]["max_dd"] * 100))
    out += ["", "## 搜索校正与后续表现", "",
            "对2014–2021全部5,156列真实净收益做同步循环区块、逐列零均值重定心的White-style最大统计量诊断，不只检查冠军或TopK：", "",
            "| 区块长度 | 抽样次数 | 候选列数 | p值 |", "|---|---:|---:|---:|"]
    for t in d["white_style"]:
        out.append("| %d日 | %d | %d | %.3f |" % (t["block_size"], t["draws"], t["tested_candidate_count"], t["p_value"]))
    out += ["", "这个保守的全家族检验没有提供超基准证据；p值不是过拟合概率，更不是未来获利概率。它也未校正此前未完整记录的搜索，不是完整Hansen SPA或PBO。", "",
            "固定四折的前段选最高、后段查看结果如下。该流程与主champion的双子段资格规则不同，且只是既有候选收益流重选，没有重建跨策略换仓成本，不能当可交易策略：", "",
            "| 后续区间 | 训练选中模型年化 | 同期v9.2年化 |", "|---|---:|---:|"]
    for fold in d["walkforward"]["folds"]:
        out.append("| %s—%s | %.2f%% | %.2f%% |" % (fold["test_start"], fold["test_end"], fold["selected_test_cagr"] * 100, fold["benchmark_test_cagr"] * 100))
    out += ["", "四折均落后。2018–2025收益流拼接年化约30.07%，基准41.16%；这进一步说明历史训练赢家可能退化，不能因为全历史最高曲线好看就宣布模型已经可靠。", "",
            "## 逐年收益（主成本口径）", "",
            "| 年份 | v9.2 | 开发冻结者 | 全历史最高者 | 开放熊市小池探索者 |", "|---|---:|---:|---:|---:|"]
    for year in r["controls"]["c_v92"]["yearly"]:
        values = [allrows[n]["yearly"][year] * 100 for _, n in chosen]
        out.append("| %s | %s |" % (year if year != "2026" else "2026截至09-11", " | ".join("%+.2f%%" % v for v in values)))
    out += ["", "## 边界与论文", "",
            "- 24资产是事后可获得的ETF集合，上市预热处理并不消除存活和选池偏差。没有填造上市前数据或缺行情成交。",
            "- 主模型是次日开盘压力口径，不代表用户14:50执行的实际收益；同收盘理想结果也不能替代分钟数据验证。",
            "- 未重建盘口深度、涨跌停封单、资金冲击、整手限制或QDII溢价成交；历史复权也可能修订。",
            "- QVIX只有日期没有历史发布时间，缺同日值时关闭该通道；搜索候选只用原深跌或关闭抄底，没有追加QVIX参数扫描。",
            "- 旧v9三版、已有两轮研究及真实数据/账本共209个文件哈希保持不变。新配置只做研究输出，未部署或推送。", "",
            "论文已阅读并用于设计研究：[Time Series Momentum](https://fairmodel.econ.yale.edu/ec439/mosk.pdf)、[趋势的长历史证据](https://images.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/AQR-JPM-Fall-2017.pdf)、[交易成本与买入/持有门槛](https://www.nber.org/papers/w20721)、[White Reality Check](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf)、[Hansen SPA作者页](https://reinhardhansen.github.io/research.html)。逐篇结论、近年复现研究和不能外推的部分见../LITERATURE.md。TSM按资产自身趋势构造，不要求用沪深300一个指标禁止所有A股入场；这里去门控是否有效由配对消融检验，论文不为某个ETF池背书。", "",
            "![净值与回撤](equity_drawdown.svg)", ""]
    return "\n".join(out)


def plot(r, f, paths):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/v10-search-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    regime_path = json.loads((OUT / "regime_candidate_path.json").read_text())
    entries = [("v9.2 control", paths["controls"]["c_v92"], "#777777"),
               ("Development champion (failed later)", paths["candidates"][r["selection"]["champion"]], "#9a649f"),
               ("Hindsight maximum (high drawdown)", paths["candidates"][r["hindsight_winner"]["id"]], "#d07732"),
               ("Open-stock small pool (exploratory)", regime_path, "#187c76")]
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for label, result, color in entries:
        dates = [datetime.fromisoformat(x[0]) for x in result["daily"]]
        values = [x[1] for x in result["daily"]]
        peak, dds = 1.0, []
        for value in values:
            peak = max(peak, value)
            dds.append((value / peak - 1) * 100)
        axs[0].plot(dates, values, color=color, label=label, linewidth=1.5)
        axs[1].plot(dates, dds, color=color, linewidth=1.1)
    for ax in axs:
        ax.axvspan(datetime(2022, 1, 1), datetime(2026, 1, 1), color="#2455a4", alpha=.05)
        ax.axvspan(datetime(2026, 1, 1), datetime(2026, 9, 11), color="#e9bb48", alpha=.12)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Wealth (initial = 1, log scale)")
    axs[0].set_title("5,156-configuration H search | Same-cost next-open execution", loc="left", weight="bold")
    axs[0].legend(loc="upper left", frameon=False, fontsize=9)
    axs[1].set_ylabel("Drawdown (%)")
    axs[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(.08, .018, "2014-01-02 to 2026-09-11. Per side: 1bp commission + 10bp slippage. No leverage.\n"
             "Hindsight-selected lines are exploratory. Prior history was already researched; none is clean OOS.", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(OUT / "equity_drawdown.svg", bbox_inches="tight")
    fig.savefig(OUT / "equity_drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    protect()
    r, configs, d, f, paths = load()
    dump(BASE / "profiles.json", profiles(r, configs, f))
    (OUT / "REPORT.md").write_text(markdown(r, configs, d, f), encoding="utf-8")
    with (OUT / "all_candidates.csv").open("w", newline="", encoding="utf-8-sig") as out:
        w = csv.writer(out)
        w.writerow(["id", "family", "regime", "development_cagr", "full_cagr", "full_max_dd", "validation_cagr", "2026_total_return", "trade_days_per_year", "score", "windows", "stock_pool", "global_pool"])
        for row in sorted(r["results"], key=lambda x: (-x["full"]["cagr"], x["id"])):
            c = configs[row["id"]]
            w.writerow([row["id"], row["family"], c["regime_mode"], row["development"]["cagr"], row["full"]["cagr"], row["full"]["max_dd"], row["validation"]["cagr"], row["report_only_2026"]["total_return"], row["trade_days_per_year"], c["score_mode"], c["score_windows"], c["stock_pool"], c["global_pool"]])
    h = load_histories()
    attribution_rows = {name: attribution(path, h) for name, path in paths["candidates"].items()}
    attribution_rows[f["candidate"]["id"]] = attribution(json.loads((OUT / "regime_candidate_path.json").read_text()), h)
    dump(OUT / "selected_attribution.json", attribution_rows)
    plot(r, f, paths)
    protect()
    print("Profiles, complete report, candidate CSV and chart written")


if __name__ == "__main__":
    main()
