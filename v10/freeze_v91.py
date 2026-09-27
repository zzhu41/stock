# -*- coding: utf-8 -*-
"""v9.1 正确性修复后的离线回归基线，固定截止日和数据指纹。
旧 v91_baseline.json 保留为修复前档案，不作为当前引擎的正确性标准。

用法:
  python3.8 v10/freeze_v91.py          # 校验模式: 与基线对比
  python3.8 v10/freeze_v91.py write    # 写入/更新基线(仅在建基线时用一次)
"""
import hashlib
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from backtest import backtest  # noqa: E402
from strategy_versions import backtest_kwargs, load_local_histories  # noqa: E402

BASELINE = os.path.join(BASE, "v10", "v91_corrected_baseline.json")


def fingerprint(end=None):
    histories = load_local_histories()
    end = end or min(rows[-1][0] for rows in histories.values())
    histories = {c: [r for r in rows if r[0] <= end] for c, rows in histories.items()}
    hashes = {c: hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
              for c, rows in histories.items()}
    calendar = [r[0] for r in histories["510300"]]
    res = backtest(histories, calendar, start="2014-01-01", end=end, **backtest_kwargs("v9.1"))
    # 逐日 (日期,净值8位,持仓) + 全部换仓记录 的指纹
    payload = {
        "daily": [[d, round(nav, 8), h] for d, nav, h in res["daily"]],
        "trades": [[d, frm, to, round(nav, 8)] for d, frm, to, nav in res["trades"]],
        "crash_buys": res["crash_buys"],
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return payload, digest, res, hashes, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("check", "write"), default="check")
    parser.add_argument("--end")
    args = parser.parse_args()
    base = None
    if args.mode == "check":
        with open(BASELINE, encoding="utf-8") as f:
            base = json.load(f)
    end = args.end or (base["end"] if base else None)
    payload, digest, res, hashes, end = fingerprint(end)
    print("v9.1 指纹: %s" % digest)
    print("年化 %+.2f%% | 回撤 %.2f%% | 夏普 %.2f | 换手 %d | 日数 %d"
          % (res["ann"] * 100, res["max_dd"] * 100, res["sharpe"],
             res["switches"], len(res["daily"])))
    if args.mode == "write":
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump({"schema": 2, "end": end, "data_sha256": hashes,
                       "reason": "2026-09-27: mandatory panic exits and missing-bar accounting corrected",
                       "sha256": digest, "payload": payload}, f, ensure_ascii=False)
        print("基线已写入 %s" % BASELINE)
        return
    if hashes != base["data_sha256"] or end != base["end"]:
        print("❌ 数据快照或截止日变化，不能作为代码回归比较；请核对复权修订后再显式建基线")
        sys.exit(1)
    if base["sha256"] == digest:
        print("✅ 回归通过: v9.1 输出与基线逐日一致")
    else:
        cur = payload["daily"]
        ref = base["payload"]["daily"]
        diff = [i for i in range(min(len(cur), len(ref)))
                if cur[i] != ref[i]]
        print("❌ 回归失败: 指纹不符; 日数 %d vs %d, 首个差异下标 %s"
              % (len(cur), len(ref), diff[0] if diff else "(长度不同)"))
        if diff:
            i = diff[0]
            print("  首个差异: %s vs %s" % (cur[i], ref[i]))
        sys.exit(1)


if __name__ == "__main__":
    main()
