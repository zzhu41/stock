"""Compare the discussed H/S with v9 versions using the ORIGINAL close engine.

This deliberately retains its fee convention: 1bp each side approximated as
2bp per switch, no slippage, no initial-entry fee. No production file is edited.
"""
import hashlib
import json
from pathlib import Path

import backtest
import strategy
from strategy_versions import backtest_kwargs, load_local_histories, BASE_CODES
from v9_robustness import wls_window


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "v10_search/results"
START, END = "2014-01-01", "2026-09-24"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_report = json.loads((ROOT / "reports/correctness_review.json").read_text())
    for key in ("data_sha256", "code_sha256"):
        for name, expected in original_report[key].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Archived baseline inputs changed: " + name)
    hp = json.loads((ROOT / "v10_search/profiles.json").read_text())["variants"]["regime"]["config"]
    sp = json.loads((ROOT / "v10_round2/profiles.json").read_text())["variants"]["robust"]["config"]
    if hp["regime_mode"] != "open_stock" or hp["score_windows"] != [30] or sp["score_windows"] != [30]:
        raise ValueError("The discussed H/S profile has changed; do not silently substitute versions")
    histories = {c: [r for r in rows if r[0] <= END] for c, rows in load_local_histories().items()}
    if any(rows[-1][0] != END for rows in histories.values()):
        raise ValueError("All comparison histories must cover " + END)
    calendar = [r[0] for r in histories["510300"]]
    base_stock = list(sp["stock_pool"])
    base_global = list(sp["global_pool"])
    original_rank = strategy.rank
    cache = {}
    current_window = [25]
    allowed = [set(BASE_CODES)]

    def ranked(h, on_date=None, live_prices=None):
        if h is not histories or live_prices is not None:
            raise ValueError("Only the fixed comparison snapshot may be cached")
        key = (current_window[0], on_date)
        if key not in cache:
            cache[key] = original_rank(h, on_date=on_date)
        return [(code, dict(ind)) for code, ind in cache[key] if code in allowed[0]]

    strategy.rank = ranked
    results = {}
    try:
        cases = [("v9", backtest_kwargs("v9"), base_stock, base_global, 25),
                 ("v9.1", backtest_kwargs("v9.1"), base_stock, base_global, 25),
                 ("v10-H", hp["params"], hp["stock_pool"], hp["global_pool"], 30),
                 ("v10-S", backtest_kwargs("v9.1", **sp["overrides"]), sp["stock_pool"], sp["global_pool"], 30)]
        for label, params, stocks, globals_, window in cases:
            strategy.STOCK_POOL = list(stocks)
            strategy.GLOBAL_POOL = list(globals_)
            allowed[0] = set(stocks) | set(globals_) | {"518880", "510300"}
            current_window[0] = window
            with wls_window(window):
                r = backtest.backtest(histories, calendar, start=START, end=END, **params)
            m = dict(nav=r["nav"], total_return=r["nav"] - 1, ann=r["ann"], max_dd=r["max_dd"],
                     sessions=len(r["daily"]), yearly=dict(backtest.yearly(r["daily"])))
            if label in ("v9", "v9.1"):
                archived = next(x for x in original_report["results"] if x["version"] == label and x["start"] == START)["scenarios"]["same_close_ideal"]
                for key in ("nav", "ann", "max_dd"):
                    if abs(m[key] - archived[key]) > 1e-10:
                        raise AssertionError("Original engine no longer matches archive: " + label + " " + key)
            results[label] = m
            print(label, "total %+.2f%% annual %.2f%%" % (m["total_return"] * 100, m["ann"] * 100), flush=True)
    finally:
        strategy.rank = original_rank
    # Reuse the already-run v9.2 result only after ALL its original code and data
    # hashes were checked above. Avoid importing lab's global hooks into H/S.
    old = next(x for x in original_report["results"] if x["version"] == "v9.2" and x["start"] == START)["scenarios"]["same_close_ideal"]
    results["v9.2"] = dict(nav=old["nav"], total_return=old["nav"] - 1, ann=old["ann"], max_dd=old["max_dd"],
                            sessions=results["v9"]["sessions"], yearly=old["yearly"])
    report = dict(start="2014-01-02", end=END,
                  convention="Original same-close engine: 1bp per side, 2bp per switch, no slippage or initial-entry fee",
                  h_id=hp["id"], s_id=sp["id"],
                  h_description="Small pool, WLS30, bear_open_stock=True; the 39.08% next-open exploratory candidate",
                  data_sha256=original_report["data_sha256"], original_code_sha256=original_report["code_sha256"],
                  script_sha256=digest(Path(__file__)),
                  archived_v9_v91_exact_parity=True, v92_reused_after_all_hashes_matched=True, results=results)
    path = OUT / "original_close_comparison_20260924.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 原版收盘口径：v9三版与当前H/S", "",
             "统一2014-01-02至2026-09-24。当天收盘数据出信号、当天收盘价成交；沿用原引擎单边万一/换仓双边万二，无滑点、不计首笔建仓费。", "",
             "H特指最近讨论的39.08%次开盘收益的小池开放熊市A股候选，S为原v9.1池的WLS30候选。", "",
             "| 版本 | 累计总收益 | 年化收益 | 期末本金倍数 | 最大回撤 |", "|---|---:|---:|---:|---:|"]
    for label in ("v9", "v9.1", "v9.2", "v10-H", "v10-S"):
        m = results[label]
        lines.append("| %s | %+.2f%% | %.2f%% | %.2f倍 | %.2f%% |" % (
            label, m["total_return"] * 100, m["ann"] * 100, m["nav"], m["max_dd"] * 100))
    lines += ["", "v9/v9.1重新计算后与原档案一致。v9.2使用原档案同区间结果，已验证全部原代码及行情哈希一致。H/S使用同一原引擎新算，没有修改生产文件或参数。", "",
              "本表是理想收盘回测，不代表14:50实际可成交收益，也不消除样本内选型偏差。全历史最高收益候选（原报告截至09-11理想收盘53.21%）是另一个配置，不能悄悄替换此处的H。", ""]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
