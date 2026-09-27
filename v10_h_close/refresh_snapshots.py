"""Update only this new research snapshot, never the production price cache."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path

import market_data
from v10_search.data import UNIVERSE


BASE = Path(__file__).resolve().parent
TARGET_END = "2026-09-24"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = BASE / "snapshots"
    market_data.DATA_DIR = str(folder)
    market_data.UNIVERSE.update(UNIVERSE)
    market_data._today = lambda: TARGET_END
    pending, results = [], {}
    original = {}
    for code in UNIVERSE:
        path = folder / (code + ".csv")
        with path.open(newline="") as f:
            rows = list(csv.reader(f))
        original[code] = dict(sha256=digest(path), first=rows[0][0], last=rows[-1][0], rows=len(rows))
        if rows[-1][0] < TARGET_END:
            pending.append(code)
        else:
            results[code] = dict(refreshed=False, last=rows[-1][0])
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(market_data.fetch_history, code): code for code in pending}
        for future in as_completed(jobs):
            code = jobs[future]
            try:
                rows = future.result()
                results[code] = dict(refreshed=True, last=rows[-1][0], rows=len(rows))
                print(code, "updated through", rows[-1][0], flush=True)
            except Exception as exc:
                results[code] = dict(refreshed=False, error=repr(exc), last=original[code]["last"])
                print(code, "refresh failed", type(exc).__name__, flush=True)
    common_end = min(r["last"] for r in results.values())
    manifest = {}
    for code in UNIVERSE:
        path = folder / (code + ".csv")
        with path.open(newline="") as f:
            rows = [r for r in csv.reader(f) if r and r[0] <= common_end]
        if rows[-1][0] != common_end:
            raise ValueError("Last common date has no bar: " + code)
        with path.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
        manifest[code] = dict(sha256=digest(path), first=rows[0][0], last=rows[-1][0], rows=len(rows),
                               seed=original[code], update=results[code])
    report = dict(target_end=TARGET_END, end=common_end, assets=manifest,
                  source="Tencent public adjusted daily open/close/volume, overlap checked by existing loader",
                  existing_cache_untouched=True, failed_updates=[c for c, r in results.items() if "error" in r])
    (BASE / "snapshot_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Frozen common end:", common_end, "assets:", len(manifest), flush=True)


if __name__ == "__main__":
    main()
