"""Read-only second-round frozen research models; no live account or orders."""
import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from v10_next.cli import read_prices
from v10_next.metrics import summarize
from .research import BASE, Simulator, sha
from .v92 import QvixSeries


def load_profile(variant):
    frozen = json.loads((BASE / "profiles.json").read_text())
    if variant not in frozen["variants"]:
        raise ValueError("No selected frozen version for " + variant)
    for name, expected in frozen["own_source_sha256"].items():
        if sha(BASE / name) != expected:
            raise ValueError("Second-round implementation changed: " + name)
    for name, expected in frozen["parent_source_sha256"].items():
        if sha(BASE.parent / "v10_next" / name) != expected:
            raise ValueError("First-round dependency changed: " + name)
    for name, expected in frozen["artifact_sha256"].items():
        if sha(BASE / name) != expected:
            raise ValueError("Frozen research artifact changed: " + name)
    if sha(BASE.parent / "v10_next/data_manifest.json") != frozen["prices_manifest_sha256"]:
        raise ValueError("Price snapshot manifest changed")
    profile = frozen["variants"][variant]
    raw = json.dumps(profile["config"], sort_keys=True, ensure_ascii=False).encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != profile["config_sha256"]:
        raise ValueError("Frozen candidate config changed")
    return deepcopy(profile)


def codes_for(config):
    if config["kind"] == "accounts":
        return tuple(sorted({code for c in config["components"] for code in codes_for(c)}))
    return tuple(dict.fromkeys(list(config["stock_pool"]) + list(config["global_pool"]) + ["518880", "511880", "510300"]))


def requires_qvix(config):
    return any(requires_qvix(c) for c in config["components"]) if config["kind"] == "accounts" else "qvix" in config.get("channels", ())


def evaluate(variant, end=None, data_dir=None, qvix_file=None):
    profile = load_profile(variant)
    histories, qvix = None, None
    if data_dir is not None:
        histories = read_prices(Path(data_dir), codes_for(profile["config"]))
        if requires_qvix(profile["config"]) and qvix_file is None:
            raise ValueError("External prices for this profile require an explicit --qvix-file; the old frozen series is not a live feed")
    if qvix_file is not None:
        with Path(qvix_file).open(newline="", encoding="utf-8") as f:
            qvix = QvixSeries([(r[0], float(r[1])) for r in csv.reader(f) if r])
    simulator = Simulator(histories=histories, qvix=qvix)
    common_end = min(rows[-1][0] for rows in simulator.histories.values())
    end = end or common_end
    if end > common_end or end not in set(simulator.calendar):
        raise ValueError("End must be an observed date no later than common price coverage: " + common_end)
    return profile, simulator.simulate(profile["config"], end=end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("backtest", "signal"))
    parser.add_argument("variant", choices=("growth", "robust"))
    parser.add_argument("--end")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--qvix-file", type=Path)
    args = parser.parse_args()
    profile, result = evaluate(args.variant, args.end, args.data_dir, args.qvix_file)
    out = dict(version=profile["name"], selected_candidate=profile["selected_candidate"],
               status="frozen_research_model_not_deployed", signal_date=result["final_state"]["date"],
               source="Dated CSV model portfolio, not live quotes or the user's account")
    if args.command == "backtest":
        out["metrics"] = summarize(result)
    else:
        state = result["final_state"]
        out["model_weights"] = state["weights"]
        if state.get("independent_accounts"):
            out["account_states"] = state["account_states"]
            out["pending_account_instructions"] = state["pending_account_instructions"]
            out["instruction_scope"] = "Independent accounts; never merge targets into a free rebalanced portfolio"
        else:
            out["pending_target"] = state["pending_target"]
            out["pending_signal_date"] = state["pending_signal_date"]
        out["qvix_unavailable_dates"] = result["policy_metadata"].get("qvix_unavailable_dates", [])[-5:]
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
