"""Fetch Tencent daily bars for the pool-extension candidate ETF (513130).

Same vendor endpoint and page logic as v10_h_close/audit_prices.py, but writes
only inside this experiment directory. No frozen/production file is touched.
"""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import urllib.request

OUT = Path(__file__).resolve().parent / "data"


def fetch(code, adjustment, start="2020-01-01", end="2026-09-24"):
    if len(code) != 6 or not code.isdigit():
        raise ValueError("Expected a six-digit ETF code")
    symbol = ("sh" if code.startswith("5") else "sz") + code
    mode = "" if adjustment == "raw" else adjustment
    key = (mode + "day") if mode else "day"
    rows, pages, cursor = [], [], end
    while True:
        url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param="
               + "%s,day,%s,%s,640,%s" % (symbol, start, cursor, mode))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        page = payload["data"][symbol].get(key, [])
        if not page:
            if rows:
                break
            raise ValueError("Missing requested adjustment " + key)
        if page != sorted(page, key=lambda r: r[0]) or page[-1][0] > cursor:
            raise ValueError("Invalid date order")
        pages.append(dict(url=url, payload=payload))
        rows = page + rows
        if len(page) < 640 or page[0][0] <= start:
            break
        cursor = (datetime.strptime(page[0][0], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("Duplicate dates")
    out = dict(code=code, adjustment=adjustment,
               retrieved_at=datetime.now(timezone.utc).isoformat(), rows=rows)
    target = OUT / ("%s_%s_%s_%s.json" % (code, adjustment, start, end))
    target.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print(code, adjustment, len(rows), rows[0][:3], "->", rows[-1][:3], flush=True)
    return rows


if __name__ == "__main__":
    for adjustment in ("raw", "qfq"):
        fetch("513130", adjustment)
