"""Read-only backtests and dated research signals for v10-H / v10-S.

python3.8 -B -m v10_next.cli backtest growth
python3.8 -B -m v10_next.cli backtest robust
python3.8 -B -m v10_next.cli signal robust --end 2026-09-11
"""
import argparse
import csv
import json
import math
from pathlib import Path

from .data import START, load_histories
from .execution import run
from .metrics import summarize
from .strategy import load_profile, make_policy, required_codes


def read_prices(directory, codes):
    """Read explicitly supplied snapshots; never refresh caches or request quotes."""
    histories = {}
    for code in codes:
        path = directory / (code + ".csv")
        with path.open(newline="", encoding="utf-8") as f:
            rows = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(f) if r]
        if not rows or [r[0] for r in rows] != sorted({r[0] for r in rows}):
            raise ValueError("Empty, unsorted or duplicate history: " + code)
        if any(not all(math.isfinite(v) for v in r[1:]) or min(r[1:3]) <= 0 or r[3] < 0 for r in rows):
            raise ValueError("Invalid price or volume: " + code)
        histories[code] = rows
    return histories


def evaluate(variant, data_dir=None, start=START, end=None, slippage_bps=10):
    profile = load_profile(variant)
    codes = required_codes(profile)
    available = read_prices(Path(data_dir), codes) if data_dir else load_histories()
    histories = {c: available[c] for c in codes}
    latest_common = min(rows[-1][0] for rows in histories.values())
    end = end or latest_common
    if end > latest_common:
        raise ValueError("Requested end exceeds common price coverage: " + latest_common)
    calendar = [r[0] for r in histories["510300"]]
    if end not in set(calendar):
        raise ValueError("End must be an observed trading date, not a presumed live date")
    policy = make_policy(variant, histories, calendar)
    result = run(histories, calendar, policy, start, end, fee=.0001, slippage=slippage_bps / 10000)
    return profile, policy, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("backtest", "signal"))
    parser.add_argument("variant", choices=("growth", "robust"))
    parser.add_argument("--data-dir", type=Path, help="Explicit read-only open/close/volume CSV directory")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", help="Observed signal/backtest date; defaults to common local data end")
    parser.add_argument("--slippage-bps", type=float, default=10)
    args = parser.parse_args()
    profile, policy, result = evaluate(args.variant, args.data_dir, args.start, args.end, args.slippage_bps)
    if args.command == "backtest":
        output = dict(version=profile["name"], status=profile["status"], metrics=summarize(result),
                      commission_per_side=.0001, slippage_per_side=args.slippage_bps / 10000,
                      caution="Historical model portfolio; no live account, order submission or future guarantee.")
    else:
        state = result["final_state"]
        output = dict(version=profile["name"], status=profile["status"], signal_date=state["date"],
                      source="Supplied dated CSV histories, not a real-time quote feed",
                      model_holdings_weights=state["weights"],
                      pending_target_weights=state["pending_target"],
                      pending_signal_date=state["pending_signal_date"],
                      action="hold actual model units" if state["pending_target"] is None else "target for a later available open",
                      execution="Next available trading-day open after signal; missing bars can defer",
                      caution="Research-model instruction only; it does not reflect the user's actual account or place an order.")
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
