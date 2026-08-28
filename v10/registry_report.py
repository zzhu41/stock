# -*- coding: utf-8 -*-
"""实验注册表报告: 累计假设数(过拟合预算)与历史批次一览。

过拟合预算 = 累计检验过的假设数。多重检验直觉: 若每个变体有 p 的概率纯靠运气过关,
测 N 个就期望收获 N*p 个假阳性 —— 幸存组件数应与该预算对照看。
历史基线(2026-08-28 前的 15 轮, 凭记忆档案): ~250 个变体, 存活 6 个组件
(panic4%/永不空仓/WLS25/熊门7%/深跌抄底/分池缓冲)。
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(BASE, "results", "registry.jsonl")
HISTORICAL = 250          # v10 实验室建立前的历史变体数(约数, 见 strategy.py 各轮注释)
HISTORICAL_SURVIVORS = 6


def main():
    batches = []
    if os.path.exists(REGISTRY):
        with open(REGISTRY, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    batches.append(json.loads(line))
    v10_variants = sum(b["n_variants"] for b in batches)
    total = HISTORICAL + v10_variants
    print("== 过拟合预算 ==")
    print("历史 15 轮(档案): ~%d 变体, 存活 %d 组件" % (HISTORICAL, HISTORICAL_SURVIVORS))
    print("v10 实验室批次  : %d 批, %d 变体" % (len(batches), v10_variants))
    print("累计假设数      : ~%d" % total)
    print()
    for b in batches:
        print("  %s | %-18s | %d 变体 | 任一起点赢基线: %s"
              % (b["ts"], b["batch"], b["n_variants"],
                 ",".join(b["winners_any_start"]) or "无"))


if __name__ == "__main__":
    main()
