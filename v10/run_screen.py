# -*- coding: utf-8 -*-
"""v10 初筛矩阵驱动: 2 进程并行(fork 继承快照, 免重复加载), 结果写 results/screen.jsonl。
用法: python3.8 run_screen.py            # 全部变体 × 2014/2016 双起点
     python3.8 run_screen.py validate a,b,c   # 指定变体 × 6 起点全验证
"""
import json
import os
import sys
import time
from multiprocessing import Pool

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import lab


def job(args):
    v, s = args
    try:
        return lab.run_variant(v, s)
    except Exception as e:
        return {"variant": v, "start": s, "error": repr(e)}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "validate":                     # validate a,b,c → 6 起点
        names = sys.argv[2].split(",")
        starts = lab.STARTS
        out_name = "validate.jsonl"
    elif mode == "screen":                     # 全部变体 × 2014/2016
        names = list(lab.VARIANTS)
        starts = lab.STARTS[:2]
        out_name = "screen.jsonl"
    else:                                      # <out_name> <a,b,c> [n_starts]
        out_name = mode if mode.endswith(".jsonl") else mode + ".jsonl"
        names = list(lab.VARIANTS) if sys.argv[2] == "all" else sys.argv[2].split(",")
        starts = lab.STARTS[:int(sys.argv[3])] if len(sys.argv) > 3 else lab.STARTS[:2]
    lab.init_data()  # fork 前加载, 子进程继承
    jobs = [(v, s) for v in names for s in starts]
    print("共 %d 个任务 (%d 变体 × %d 起点), 2 进程并行" % (len(jobs), len(names), len(starts)))
    t0 = time.time()
    with Pool(2) as p:
        results = p.map(job, jobs)
    out = os.path.join(BASE, "results", out_name)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("耗时 %.0fs, 写入 %s" % (time.time() - t0, out))

    # 实验注册表: 每批次追加一行(过拟合预算追踪 —— 累计假设数即多重检验消耗)
    base_map = {r["start"]: r for r in results if r.get("variant") == "baseline"}
    winners = sorted({r["variant"] for r in results
                      if "error" not in r and r["variant"] != "baseline"
                      and r["start"] in base_map and r["nav"] > base_map[r["start"]]["nav"]})
    reg = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "batch": out_name,
           "n_variants": len(names), "n_jobs": len(jobs), "starts": list(starts),
           "winners_any_start": winners}
    with open(os.path.join(BASE, "results", "registry.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(reg, ensure_ascii=False) + "\n")

    # 汇总表: 按 2014 起点终值相对基线排序(无基线行时按终值排序)
    base = {r["start"]: r for r in results if r.get("variant") == "baseline"}
    rows = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    for e in errs:
        print("!! %(variant)s %(start)s: %(error)s" % e)
    rows.sort(key=lambda r: -(r["nav"] / base[r["start"]]["nav"]) if r["start"] in base else -r["nav"])
    print("\n%-14s %-10s %9s %7s %7s %6s %6s %5s" %
          ("变体", "起点", "终值x基线", "年化%", "回撤%", "夏普", "换手/年", "抄底"))
    for r in rows:
        ratio = "%8.3fx" % (r["nav"] / base[r["start"]]["nav"]) if r["start"] in base else "        -"
        print("%-14s %-10s %s %+7.2f %7.2f %6.3f %6.1f %5d" %
              (r["variant"], r["start"], ratio,
               r["ann"], r["dd"], r["sharpe"], r["sw_per_year"], r["crash_buys"]))


if __name__ == "__main__":
    main()
