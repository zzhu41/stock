"""Frozen research-only H profiles: development-selected or hindsight-labelled."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from v10_next.cli import read_prices
from v10_round2.v92 import load_qvix
from .data import BASE, START, END, load_histories, sha
from .features import FeatureBank
from .fast_execution import prepare, run
from .policy import SearchPolicy
from .registry import candidate_hash, required_codes


def load_profile(variant):
    frozen = json.loads((BASE / "profiles.json").read_text())
    if variant not in frozen["variants"]:
        raise ValueError("No frozen research profile: " + variant)
    for name, expected in frozen["source_sha256"].items():
        if sha(BASE / name) != expected:
            raise ValueError("Search implementation changed: " + name)
    for name, expected in frozen["dependencies_sha256"].items():
        if sha(BASE.parent / name) != expected:
            raise ValueError("Frozen dependency changed: " + name)
    for name, expected in frozen["artifact_sha256"].items():
        if sha(BASE / name) != expected:
            raise ValueError("Search result/snapshot manifest changed: " + name)
    profile = frozen["variants"][variant]
    if candidate_hash(profile["config"]) != profile["candidate_hash"]:
        raise ValueError("Frozen strategy configuration changed")
    return deepcopy(profile)


def evaluate(variant, end=None, data_dir=None):
    p = load_profile(variant)
    codes = required_codes(p["config"])
    histories = read_prices(Path(data_dir), codes) if data_dir else load_histories()
    histories = {c: histories[c] for c in codes}
    calendar = [r[0] for r in histories["510300"]]
    common_end = min(rows[-1][0] for rows in histories.values())
    end = end or common_end
    if end > common_end or end not in set(calendar):
        raise ValueError("End must be a dated observation no later than " + common_end)
    bank = FeatureBank(histories, calendar)
    frame = prepare(histories, calendar)
    if set(p["config"]["channels"]) - {"deep"}:
        raise ValueError("Published search profiles cannot silently use an old external fear feed")
    policy = SearchPolicy(p["config"], bank, frame=frame, trace=True)
    result = run(frame, policy, START, end, fee=.0001, slippage=.001,
                 capture_daily=True, capture_trades=True)
    result["policy_metadata"] = policy.metadata
    return p, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("backtest", "signal"))
    parser.add_argument("variant", choices=("growth", "exploratory", "regime"))
    parser.add_argument("--end")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    p, r = evaluate(args.variant, args.end, args.data_dir)
    out = dict(version=p["name"], status=p["status"], selected_by=p["selected_by"],
               not_deployed=True, clean_oos=False,
               candidate_id=p["config"]["id"], date=r["final_state"]["date"],
               source="Dated CSV research-model account, not real-time quotes or actual brokerage positions")
    if args.command == "backtest":
        out["metrics"] = {k: r[k] for k in ("nav", "ann", "max_dd", "sharpe", "yearly", "trade_count", "fill_count", "deferred_count")}
    else:
        out["model_weights"] = r["final_state"]["weights"]
        out["pending_target"] = r["final_state"]["pending_target"]
        out["pending_signal_date"] = r["final_state"]["pending_signal_date"]
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
