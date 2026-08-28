# -*- coding: utf-8 -*-
"""读 results/*.jsonl, 对照基线出对比表: 终值比/年化差/回撤差/逐年差。
用法: python3.8 analyze.py screen [validate]"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    rows = {}
    with open(os.path.join(BASE, "results", "%s.jsonl" % name), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[(r["variant"], r["start"])] = r
    return rows


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "screen"
    rows = load(name)
    starts = sorted({s for _, s in rows})
    variants = sorted({v for v, _ in rows})
    base = {s: rows[("baseline", s)] for s in starts if ("baseline", s) in rows}

    print("== %s | 相对基线(2014起逐年差, pp) ==" % name)
    hdr = "%-14s" % "变体" + "".join("%10s" % s[:4] for s in starts) + "   弱年明细"
    print(hdr)
    for v in variants:
        if v == "baseline":
            continue
        cells = []
        for s in starts:
            r = rows.get((v, s))
            if not r or "error" in r:
                cells.append("%10s" % "ERR")
            elif s not in base:
                cells.append("%10s" % "-")
            else:
                ratio = r["nav"] / base[s]["nav"]
                cells.append("%9.3fx" % ratio + ("*" if ratio > 1 else " "))
        # 逐年差(2014 起点口径)
        r14, b14 = rows.get((v, "2014-01-01")), base.get("2014-01-01")
        detail = ""
        if r14 and b14 and "error" not in r14:
            diffs = []
            for y in sorted(b14["yearly"]):
                d = r14["yearly"].get(y, 0) - b14["yearly"][y]
                if abs(d) >= 3.0:
                    diffs.append("%s:%+.0f" % (y[2:], d))
            detail = " ".join(diffs)
        print("%-14s%s   %s" % (v, "".join(cells), detail))
    # 基线参考行
    print("\nbaseline     " + "".join("%10s" % ("%.1f%%" % base[s]["ann"]) for s in starts if s in base))


if __name__ == "__main__":
    main()
