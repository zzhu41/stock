"""Publish the corrected research result and keep the invalidated result explicit."""
from copy import deepcopy
import json

import numpy as np

from .data import BASE, sha, protect
from .registry import candidate_hash, controls
from .scan import dump


OUT = BASE / "corrected_results"


def pct(x):
    return "{:+,.2f}%".format(100 * x)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in rows])


def main():
    evaluation = json.loads((OUT / "evaluation.json").read_text())
    diagnostics = json.loads((OUT / "search_diagnostics.json").read_text())
    registry = json.loads((OUT / "registry.json").read_text())
    configs = {c["id"]: c for c in registry["candidates"]}
    rows = {r["id"]: r for r in evaluation["results"]}
    rows.update(evaluation["detailed_summaries"])
    paths = json.loads((OUT / "selected_paths.json").read_text())
    selected = evaluation["selection"]
    hindsight_id = evaluation["hindsight_winner"]["id"]
    hindsight_controls = configs[hindsight_id].get("prior_control_ids", [])
    primary, guarded = selected["primary_return_champion"], selected["risk_guarded_candidate"]
    names = {"c_v9": "v9", "c_v91": "v9.1", "c_v92": "v9.2", primary: "新H：收益优先"}
    if guarded and guarded != primary:
        names[guarded] = "新H：风险门槛候选"
    ids = ["c_v9", "c_v91", "c_v92", primary] + ([guarded] if guarded and guarded != primary else [])
    original = json.loads((BASE / "results/evaluation.json").read_text())
    bad = original["selection"]["primary_return_champion"]
    main_rows = [[names[n], pct(rows[n]["metrics"]["full"]["total_return"]),
                  pct(rows[n]["metrics"]["full"]["cagr"]), pct(rows[n]["metrics"]["full"]["max_dd"]),
                  rows[n]["switches"]] for n in ids]
    fulltable = table(["版本", "累计收益", "年化", "最大回撤", "换仓次数"], main_rows)
    train_difference = 100 * (rows[primary]["metrics"]["selection"]["cagr"] - rows["c_v92"]["metrics"]["selection"]["cagr"])
    full_difference = 100 * (rows[primary]["metrics"]["full"]["cagr"] - rows["c_v92"]["metrics"]["full"]["cagr"])
    years = sorted(rows[primary]["metrics"]["yearly"])
    annual = table(["年份"] + [names[n] for n in ids],
                   [[y + ("（截至09-24）" if y == "2026" else "")] +
                    [pct(rows[n]["metrics"]["yearly"][y]) for n in ids] for y in years])
    periods = table(["版本", "2014–2025年化（选型）", "2022–2025年化（也参与筛选）", "2026累计（仅报告）"],
                    [[names[n], pct(rows[n]["metrics"]["selection"]["cagr"]),
                      pct(rows[n]["metrics"]["recent_2022_2025"]["cagr"]),
                      pct(rows[n]["metrics"]["report_only_2026"]["total_return"])] for n in ids])
    fees = table(["版本", "单边1bp年化", "单边5bp年化", "5bp最大回撤"],
                 [[names[n], pct(rows[n]["metrics"]["full"]["cagr"]),
                   pct(rows[n]["fee5"]["full"]["cagr"]), pct(rows[n]["fee5"]["full"]["max_dd"])] for n in ids])
    rules = []
    for name in dict.fromkeys(n for n in (primary, guarded) if n):
        c = configs[name]
        rules.append("- **%s** `%s`：A股池 `%s`；跨境池 `%s`；黄金 `%s`、货币 `%s`；排名 `%s %s`；体制 `%s`；通道 `%s`；统一缓冲 `%s`、池别缓冲 `%s`。风险仓位 `%s`，单一ETF轮动。" % (
            names[name], name, ", ".join(c["stock_pool"]), ", ".join(c["global_pool"]), c["gold"], c["cash"],
            c["score_mode"], c["score_windows"], c["regime_mode"], c["channels"], c["params"]["buffer"],
            c["params"]["pool_buffer"], c["risk_weight"]))
    neighbors = table(["母配置", "窗口", "全程年化", "最大回撤"],
                      [[names[x["parent"]], str(x["windows"]), pct(x["metrics"]["metrics"]["full"]["cagr"]),
                        pct(x["metrics"]["metrics"]["full"]["max_dd"])] for x in evaluation["window_neighbors"]])
    by_hash = {c["candidate_hash"]: c for c in configs.values()}
    ablations = []
    for label, mutate in [
        ("WLS20", lambda c: c.update(score_mode="wls", score_windows=[20])),
        ("WLS25", lambda c: c.update(score_mode="wls", score_windows=[25])),
        ("WLS30", lambda c: c.update(score_mode="wls", score_windows=[30])),
        ("WLS35", lambda c: c.update(score_mode="wls", score_windows=[35])),
        ("仅移除510880", lambda c: c.update(stock_pool=[x for x in c["stock_pool"] if x != "510880"])),
        ("移除三只新增红利", lambda c: c.update(stock_pool=[x for x in c["stock_pool"] if x not in ("510880", "515080", "515100")])),
        ("恢复原池别缓冲", lambda c: c["params"].update(buffer=.02, pool_buffer={"stock": .02, "global": .03, "gold": .03})),
        ("统一2%缓冲", lambda c: c["params"].update(buffer=.02, pool_buffer={})),
    ]:
        c = deepcopy(configs[primary]); mutate(c)
        h = candidate_hash(c)
        if h in by_hash and h != configs[primary]["candidate_hash"]:
            n = by_hash[h]["id"]
            ablations.append(dict(label=label, id=n, metrics=rows[n]["metrics"]))
    ablation_table = table(["对主候选的单项变化", "选型年化", "全程年化", "全程最大回撤"],
                          [[x["label"], pct(x["metrics"]["selection"]["cagr"]), pct(x["metrics"]["full"]["cagr"]),
                            pct(x["metrics"]["full"]["max_dd"])] for x in ablations])
    regimes = []
    for regime in sorted({c["regime_mode"] for c in configs.values()}):
        family = [c for c in configs.values() if c["regime_mode"] == regime]
        best = max(family, key=lambda c: rows[c["id"]]["metrics"]["selection"]["cagr"])
        m = rows[best["id"]]["metrics"]
        regimes.append(dict(regime=regime, count=len(family), id=best["id"], metrics=m))
    regime_table = table(["牛熊门模式", "候选数", "该组选型冠军年化", "同配置全程年化"],
                        [[x["regime"], x["count"], pct(x["metrics"]["selection"]["cagr"]),
                          pct(x["metrics"]["full"]["cagr"])] for x in regimes])
    concentration = {}
    control_configs = {c["id"]: c for c in controls()}
    for n in dict.fromkeys([primary, "c_v92"] + ([guarded] if guarded else [])):
        returns = np.asarray(paths[n]["returns"])
        logs = np.log1p(returns)
        holdings = paths[n]["holdings"]
        by_asset, by_role, fee_log = {}, {}, 0.0
        cfg = configs.get(n, control_configs.get(n))
        for i in range(1, len(logs)):
            fee = float(np.log(.9998)) if holdings[i] != holdings[i - 1] else 0.0
            fee_log += fee
            code = holdings[i - 1] or "uninvested"
            role = cfg["asset_roles"].get(code, "uninvested")
            by_asset[code] = by_asset.get(code, 0.0) + float(logs[i]) - fee
            by_role[role] = by_role.get(role, 0.0) + float(logs[i]) - fee
        if not np.isclose(sum(by_asset.values()) + fee_log, logs.sum(), atol=1e-10):
            raise AssertionError("Log contribution reconciliation failed")
        concentration[n] = dict(top_positive_days_log_share={str(k): float(np.sort(logs)[-k:].sum() / logs.sum()) for k in (1, 5, 10)},
                                crash_buys=len(paths[n]["policy_metadata"]["crash_buys"]),
                                nonmoney_exposure=rows[n]["nonmoney_exposure"],
                                by_asset_log_contribution=by_asset, by_role_log_contribution=by_role,
                                fee_log_contribution=fee_log, total_log_return=float(logs.sum()))
    dump(OUT / "report_data.json", dict(comparison=main_rows, ablations=ablations, regime_family_winners=regimes,
                                       concentration=concentration, protected_files_verified=protect()))
    stat_text = "; ".join("块长%d：p=%.4f" % (x["block_size"], x["p_value"]) for x in diagnostics["white_style"])
    concentration_table = table(["版本", "最大10个正日占净对数增长", "抄底次数", "收盘非货币仓位天数占比"],
                                [[names[n], pct(x["top_positive_days_log_share"]["10"]), x["crash_buys"],
                                  pct(x["nonmoney_exposure"])] for n, x in concentration.items()])
    bridge = table(["同一规则", "原加减式复权输入年化", "重建输入年化"],
                   [[names.get(n, "此前54.75%候选"), pct(original["detailed_summaries"][n]["metrics"]["full"]["cagr"]),
                     pct(rows[n]["metrics"]["full"]["cagr"])] for n in ["c_v9", "c_v91", "c_v92", bad]])
    report = """# v10-H 继续迭代：复权核查与独立重算

期间统一为 **2014-01-02—2026-09-24**。以下主表使用未复权价格、公告份额折算和反推现金分红重建的总回报收盘指数。
原同收盘成交、首日免费、之后换仓净值乘0.9998、244交易日年化的约定保留；改变的是存在收益比例失真的行情输入。

## 本轮结果

主候选选型期年化仅比同数据v9.2高 **{train_difference:+.4f}个百分点**；全程差异为 **{full_difference:+.4f}个百分点**。
本轮没有得到足以认定为可靠收益升级的证据，原先54.75%的红利池优势也未保留下来。冻结配置是研究交付，不表示正式替换v9.2。
全6544项的全历史最高者为 `{hindsight_id}`，它对应已有控制 `{hindsight_controls}`；这是原规则的复现，不能改名当作新的H成果。

{fulltable}

收益优先和风险门槛的选择{same}。风险门槛是研究筛选条件，不等于未来回撤上限或无过拟合。

{rules}

完整参数在 [profiles.json](../profiles.json)。本轮没有修改原v9三版、以前的H/S、生产推送或账户持仓。

## 为什么不能交付最先跑出的54.75%

旧输入下的冠军加入三只红利ETF、使用WLS25/统一2%缓冲/深跌+QVIX通道，年化54.75%。
但510880在2014-01-22未复权价格1.581→1.614，真实价格上涨2.09%；加减式前复权0.228→0.261却得到14.47%。
2014-01-21每份0.059元现金分红有[基金公告](https://pdf.dfcfw.com/pdf/H2_AN201401140004972122_1.pdf)支持。
这不是新的交易优势，而是直接对加减式复权价取比例造成的失真。原54.75%结果已标记失效并完整保留。

旧冠军2014年收益251.54%，510880占当年净对数增长76.32%；固定原路径排除2014后，它的年化44.28%低于旧v9.2的47.63%。
这只是事后归因，不是重新开仓回测。详细证据见 [归因](../results/attribution_audit.json)、[原始报价](../results/price_audit)、[状态](../results/STATUS.json)。

同一规则仅更换数据后的桥接对照如下；这项差异不可宣传为策略迭代收益：

{bridge}

## 分年、分段及费用

{periods}

{annual}

2026为截至09-24的累计收益，不把不足一年累计收益当全年收益。所有分段都沿用连续持仓路径，未按年重置。

{concentration_table}

逐资产/角色的可加和对数贡献及手续费对账见[report_data.json](report_data.json)；这些是固定路径归因，不是独立资产策略收益。

{fees}

5bp是手续费参数压力；仍是同一收盘价格，不代表已覆盖盘口、滑点、市场冲击和14:50报价变化。

## 消融与参数敏感性

{ablation_table}

以上只读取同一已注册集合中的相应配置，没有观察结果后新添候选。若某项改动不在登记集合内，就不补造结果。

{neighbors}

这只是窗口局部敏感性；即使相邻窗口都赚钱，也不证明泛化。

{regime_table}

上表回答是否试过去掉/放松牛熊门：`open_stock`允许熊市A股进入，`always_bull`固定牛市，`all_assets`改为全资产竞争。
各组候选数量、池子和其他规则并不完全相同，所以这是组内选型冠军比较，不能当作仅牛熊门一项变化的因果消融。

## 筛选与过拟合边界

共6544个唯一配置；按2014–2025选型，并把2022–2025表现及费用/回撤门槛用于风险候选筛选。
本轮合格{eligible}个。2026虽未进入选择函数，但历史此前已经看过，**不是干净样本外测试**。

对新的完整6544列选型收益矩阵重新做White-style检验：{stat_text}，各1000次、同日联合抽样。
该检验是条件性的、可能保守的未学生化诊断，不是SPA/PBO；p值不是未来失败概率。
无论本次赢家回测多高，都不能据此宣称已消除过拟合。收益/风险和执行可用性需要继续独立观察。

## 数据修正仍有的具体限制

九次份额折算比例及首个新单位交易日经过[原公告核对](../results/corporate_actions_verified.json)，包括停牌后复牌的情形。
现金分红主要由同源未复权与前复权的仿射差值重建，保留价格精度区间；只有列出的分红例子独立核对了公告，未完成全部现金事件的第三方审计。
510500另用[经公告及完整表交叉核对的四笔分红](../results/verified_cash_510500.json)直接重建，避免供应商微小复权残差被误认成现金。
515100在2023-12-21每份派现0.437元的大额事件亦经[原公告核对](../results/extra_cash_verification.json)；派现金额不是额外市场涨幅。
收益指数假设除息日收盘立即、免费再投资，未建模实际支付日等待、整数份额和税费。
成交量转换为相同份额单位，避免拆分造成假放量。原11只与新增13只使用同一方法，未只对候选作有利修正。
`corrected_snapshots`的open列为close占位，**禁止用于次日开盘或盘中撮合**；本批未给出修正后的次日开盘收益。
真实14:50的信号、当时可得QVIX、报价及成交效果尚未验证。当前入口只输出带日期的历史模型结果，不发送订单或推送。

复现命令与文件导航见 [README](../README.md)，原登记及修正理由见 [PROTOCOL](../PROTOCOL.md)、[CORRECTION_PROTOCOL](../CORRECTION_PROTOCOL.md)，论文依据见 [LITERATURE](../LITERATURE.md)。
""".format(fulltable=fulltable, train_difference=train_difference, full_difference=full_difference,
           hindsight_id=hindsight_id, hindsight_controls=hindsight_controls,
           same="选中同一配置（不是两次独立验证）" if primary == guarded else "对应不同配置",
           rules="\n".join(rules), bridge=bridge, periods=periods, annual=annual, fees=fees,
           concentration_table=concentration_table,
           ablation_table=ablation_table, neighbors=neighbors, regime_table=regime_table,
           eligible=selected["risk_eligible_count"], stat_text=stat_text)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    # Core fingerprints are frozen by the corrected registration; delivery files
    # extend them without rewriting either search registration.
    fp = evaluation["fingerprints"]
    source = dict(fp["source_sha256"])
    source.update({p: sha(BASE / p) for p in ("cli.py", "report.py", "matrix_archive.py")})
    dependencies = dict(fp["dependency_sha256"])
    for p in ("v10_next/cli.py", "v10_next/metrics.py", "v10_next/execution.py", "v10_next/strategy.py", "v10_next/legacy.py"):
        dependencies[p] = sha(BASE.parent / p)
    variants = {}
    for variant, name, role in (("growth", primary, "primary_return_champion"),
                                ("guarded", guarded, "risk_guarded_candidate"),
                                ("exploratory", evaluation["hindsight_winner"]["id"], "full_history_hindsight_best")):
        if name:
            is_reference = bool(configs[name].get("prior_control_ids"))
            variants[variant] = dict(name=("Existing v9.2 reference / " if is_reference else "v10-H research candidate / ") + variant, selected_by=role,
                                     status="Existing control, not a new strategy" if is_reference else "No material improvement; full-period below corrected v9.2",
                                     is_existing_reference=is_reference,
                                     data_view="corrected_total_return_close", config=configs[name],
                                     candidate_hash=configs[name]["candidate_hash"])
    artifacts = ["corrected_manifest.json", "qvix_manifest.json", "protected_manifest.json",
                 "results/corporate_actions_verified.json", "results/verified_cash_510500.json",
                 "results/extra_cash_verification.json", "results/STATUS.json"]
    artifacts += ["corrected_results/" + p for p in ("registration.json", "selection.json", "evaluation.json", "search_diagnostics.json")]
    dump(BASE / "profiles.json", dict(variants=variants, source_sha256=source, dependency_sha256=dependencies,
                                     artifact_sha256={p: sha(BASE / p) for p in artifacts},
                                     data_view="corrected_total_return_close", not_deployed=True,
                                     warning="Original 54.75% winner invalidated as economic-return evidence"))
    print(fulltable)
    print("Report and frozen profiles saved; prior files verified:", protect())


if __name__ == "__main__":
    main()
