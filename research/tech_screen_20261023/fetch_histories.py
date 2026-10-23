"""Read public Tencent adjusted daily bars into an isolated research snapshot.

No trading, credentials, production cache updates, or existing strategy imports.
Usage: python .../fetch_histories.py --universe .../technology_universe.csv
"""
import argparse
import csv
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
END = "2026-10-22"
START = "2024-01-01"
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
BENCHMARKS = {"000688": "sh", "399006": "sz", "000300": "sh"}


def exchange(code):
    if code in BENCHMARKS:
        return BENCHMARKS[code]
    if code.startswith(("4", "8", "92")):
        return "bj"
    if code.startswith("6"):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    return "bj"


def fetch_one(code):
    code = str(code).zfill(6)
    symbol = exchange(code) + code
    raw_path = BASE / "raw_history" / (symbol + ".json")
    csv_path = BASE / "history" / (symbol + ".csv")
    if raw_path.exists():
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        err = None
        for attempt in range(3):
            try:
                params = {"param": "%s,day,%s,%s,640,qfq" % (symbol, START, END)}
                response = requests.get(URL, params=params, timeout=18,
                                        headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                payload = response.json()
                node = (payload.get("data") or {}).get(symbol) or {}
                if not (node.get("qfqday") or node.get("day")):
                    raise ValueError("empty daily bars")
                raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                break
            except Exception as exc:
                err = "%s: %s" % (type(exc).__name__, str(exc)[:180])
                if attempt < 2:
                    time.sleep(0.6 * (attempt + 1))
        else:
            return {"code": code, "symbol": symbol, "error": err}
    node = (payload.get("data") or {}).get(symbol) or {}
    adjustment = "qfq" if node.get("qfqday") else "raw_index_or_fallback"
    data = node.get("qfqday") or node.get("day") or []
    rows = []
    for row in data:
        if row[0] > END:
            continue
        values = [float(x) for x in row[1:6]]
        if len(values) != 5 or not all(math.isfinite(x) for x in values):
            raise ValueError("invalid numeric bar: " + code)
        if min(values[:4]) <= 0 or values[4] < 0:
            raise ValueError("invalid price/volume: " + code)
        if rows and row[0] <= rows[-1][0]:
            raise ValueError("duplicate/out-of-order date: " + code)
        rows.append([row[0]] + values)
    if not rows:
        return {"code": code, "symbol": symbol, "error": "no bars before cutoff"}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "close", "high", "low", "volume_raw"])
        writer.writerows(rows)
    return {"code": code, "symbol": symbol, "rows": len(rows), "first": rows[0][0],
            "last": rows[-1][0], "adjustment": adjustment,
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for folder in ("raw_history", "history"):
        (BASE / folder).mkdir(parents=True, exist_ok=True)
    with open(args.universe, encoding="utf-8-sig", newline="") as handle:
        universe = list(csv.DictReader(handle))
    codes = sorted(set(str(r["code"]).zfill(6) for r in universe) | set(BENCHMARKS))
    results = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(fetch_one, c): c for c in codes}
        for task in as_completed(pending):
            try:
                result = task.result()
            except Exception as exc:
                result = {"code": pending[task], "error": repr(exc)[:200]}
            results.append(result)
            if len(results) % 40 == 0 or len(results) == len(codes):
                print("history %d/%d; errors=%d; %.1fs" %
                      (len(results), len(codes), sum("error" in r for r in results),
                       time.time() - started), flush=True)
    manifest = {"downloaded_at_local": datetime.now().isoformat(), "end": END,
                "start_requested": START, "source": URL,
                "universe_input": args.universe, "requested": len(codes),
                "success": sum("error" not in r for r in results),
                "results": sorted(results, key=lambda r: r["code"])}
    (BASE / "history_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved isolated history manifest", flush=True)


if __name__ == "__main__":
    main()
