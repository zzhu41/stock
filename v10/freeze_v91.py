# -*- coding: utf-8 -*-
"""v9.1 回归基线冻结/校验: 任何 strategy.py/backtest.py 改动后跑一遍,
保证实盘默认行为逐日逐字节不变。

用法:
  python3.8 v10/freeze_v91.py          # 校验模式: 与基线对比
  python3.8 v10/freeze_v91.py write    # 写入/更新基线(仅在建基线时用一次)
"""
import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from market_data import UNIVERSE, fetch_history  # noqa: E402
from backtest import backtest  # noqa: E402

BASELINE = os.path.join(BASE, "v10", "v91_baseline.json")


def fingerprint():
    histories = {c: fetch_history(c) for c in UNIVERSE}
    calendar = [r[0] for r in histories["510300"]]
    res = backtest(histories, calendar, start="2014-01-01")
    # 逐日 (日期,净值8位,持仓) + 全部换仓记录 的指纹
    payload = {
        "daily": [[d, round(nav, 8), h] for d, nav, h in res["daily"]],
        "trades": [[d, frm, to, round(nav, 8)] for d, frm, to, nav in res["trades"]],
        "crash_buys": res["crash_buys"],
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return payload, digest, res


def main():
    payload, digest, res = fingerprint()
    print("v9.1 指纹: %s" % digest)
    print("年化 %+.2f%% | 回撤 %.2f%% | 夏普 %.2f | 换手 %d | 日数 %d"
          % (res["ann"] * 100, res["max_dd"] * 100, res["sharpe"],
             res["switches"], len(res["daily"])))
    if len(sys.argv) > 1 and sys.argv[1] == "write":
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump({"sha256": digest, "payload": payload}, f, ensure_ascii=False)
        print("基线已写入 %s" % BASELINE)
        return
    with open(BASELINE, encoding="utf-8") as f:
        base = json.load(f)
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
