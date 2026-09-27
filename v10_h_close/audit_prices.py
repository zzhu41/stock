"""Fetch independent adjustment views for auditing, without replacing snapshots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--codes", nargs="+", default=["510880"])
    p.add_argument("--start", default="2013-01-01")
    p.add_argument("--end", default="2015-06-30")
    p.add_argument("--adjustments", nargs="+", choices=("raw", "qfq", "hfq"), default=["raw", "qfq", "hfq"])
    p.add_argument("--full-history", action="store_true")
    args = p.parse_args()
    directory = Path(__file__).resolve().parent / "results" / "price_audit"
    directory.mkdir(exist_ok=True)
    def fetch(code):
        if len(code) != 6 or not code.isdigit():
            raise ValueError("Expected a six-digit ETF code")
        symbol = ("sh" if code.startswith("5") else "sz") + code
        for adjustment in args.adjustments:
            mode = "" if adjustment == "raw" else adjustment
            key = (mode + "day") if mode else "day"
            rows, pages, end = [], [], args.end
            while True:
                url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param="
                       + "%s,day,%s,%s,640,%s" % (symbol, args.start, end, mode))
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                page = payload["data"][symbol].get(key, [])
                if not page:
                    if rows:
                        break
                    raise ValueError("Missing requested adjustment " + key)
                if page != sorted(page, key=lambda r: r[0]) or page[-1][0] > end:
                    raise ValueError("Invalid date order")
                pages.append(dict(url=url, payload=payload))
                rows = page + rows
                if not args.full_history or len(page) < 640 or page[0][0] <= args.start:
                    break
                end = (datetime.strptime(page[0][0], "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            if len({r[0] for r in rows}) != len(rows):
                raise ValueError("Duplicate dates")
            out = dict(code=code, adjustment=adjustment,
                       retrieved_at=datetime.now(timezone.utc).isoformat(),
                       payload=pages[0]["payload"], pages=pages, rows=rows)
            target = directory / ("%s_%s_%s_%s.json" % (code, adjustment, args.start, args.end))
            target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
            dates = {r[0]: r for r in rows}
            print(code, adjustment, len(rows), rows[0][:3], rows[-1][:3], flush=True)
            for date in ("2014-01-20", "2014-01-21", "2014-01-22", "2014-03-12", "2014-03-13", "2014-04-04", "2014-04-08"):
                if date in dates:
                    print(" ", dates[date][:3], flush=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(fetch, args.codes))


if __name__ == "__main__":
    main()
