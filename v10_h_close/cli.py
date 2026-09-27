"""Read-only H research, preserving close execution and identifying the frozen data vintage."""
import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path

from v10_next.cli import read_prices
from v10_round2.v92 import QvixSeries
from v10_search.features import FeatureBank
from v10_search.fast_execution import prepare
from .data import BASE, START, load_histories, load_fear, sha
from .registry import candidate_hash, required_codes
from .engine import run
from .policy import ClosePolicy


def load_profile(variant):
    profiles = json.loads((BASE / "profiles.json").read_text())
    if variant not in profiles["variants"]:
        raise ValueError("No frozen candidate for " + variant)
    for name, expected in profiles["source_sha256"].items():
        if sha(BASE / name) != expected:
            raise ValueError("Close-clock implementation changed: " + name)
    for name, expected in profiles["dependency_sha256"].items():
        if sha(BASE.parent / name) != expected:
            raise ValueError("Dependency changed: " + name)
    for name, expected in profiles["artifact_sha256"].items():
        if sha(BASE / name) != expected:
            raise ValueError("Frozen research artifact changed: " + name)
    p = profiles["variants"][variant]
    if candidate_hash(p["config"]) != p["candidate_hash"]:
        raise ValueError("Frozen configuration changed")
    return deepcopy(p)


def evaluate(variant, end=None, data_dir=None, qvix_file=None):
    p = load_profile(variant)
    config = p["config"]
    codes = required_codes(config)
    if data_dir and p.get("data_view") == "corrected_total_return_close":
        raise ValueError("Corrected profiles require a frozen reconstructed data vintage; arbitrary raw/QFQ CSVs are not total-return indices")
    if data_dir:
        histories = read_prices(Path(data_dir), codes)
        if "qvix" in config["channels"] and not qvix_file:
            raise ValueError("External price histories require an explicit --qvix-file for this profile")
    else:
        if p.get("data_view") == "corrected_total_return_close":
            from .corrected_scan import load_histories as load_corrected
            histories, _ = load_corrected()
        else:
            histories, _ = load_histories()
        histories = {c: histories[c] for c in codes}
    calendar = [r[0] for r in histories["510300"]]
    common_end = min(rows[-1][0] for rows in histories.values())
    end = end or common_end
    if end > common_end or end not in set(calendar):
        raise ValueError("End must be a dated observation no later than " + common_end)
    if qvix_file:
        with Path(qvix_file).open(newline="", encoding="utf-8") as f:
            q = QvixSeries([(r[0], float(r[1])) for r in csv.reader(f) if r])
        fear = tuple(q.state(date)["active"] for date in calendar)
    elif data_dir:
        fear = (False,) * len(calendar)
    else:
        fear = load_fear(calendar)
    bank = FeatureBank(histories, calendar)
    frame = prepare(histories, calendar)
    policy = ClosePolicy(config, bank, frame, fear, trace=True)
    result = run(frame, policy, START, end, fee=.0001, capture_daily=True, capture_trades=True)
    result["policy_metadata"] = policy.metadata
    return p, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("backtest", "signal"))
    parser.add_argument("variant", choices=("growth", "guarded", "exploratory"))
    parser.add_argument("--end")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--qvix-file", type=Path)
    args = parser.parse_args()
    p, r = evaluate(args.variant, args.end, args.data_dir, args.qvix_file)
    out = dict(version=p["name"], candidate_id=p["config"]["id"], selected_by=p["selected_by"],
               not_deployed=True, clean_oos=False, data_date=r["final_state"]["date"],
               clock="Original ideal same-close, first session free, later switch NAV times 0.9998",
               source="Dated CSV model portfolio, not real-time quotes or the actual account")
    out["research_status"] = p.get("status", "Research only")
    out["is_existing_reference"] = p.get("is_existing_reference", False)
    if p.get("data_view") == "corrected_total_return_close":
        out["data_view"] = "Raw-price/corporate-action reconstructed close total-return index"
        out["dividend_assumption"] = "Instant free close reinvestment on ex-date; payment delay and lot rounding not modeled"
        out["next_open_supported"] = False
    if args.command == "backtest":
        out["metrics"] = {k: r[k] for k in ("nav", "ann", "max_dd", "sharpe", "switches", "yearly")}
    else:
        out["historical_close_target"] = r["final_state"]["holding"]
        out["order_submission"] = False
        out["warning"] = "This describes an already observed historical close, not a tradable 14:50 order instruction"
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
