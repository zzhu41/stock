"""Repeat the frozen family on cash-flow-corrected close indices, in a new result directory."""
import argparse
from copy import deepcopy
import csv
import json

import numpy as np

from . import scan
from .data import BASE, START, ASSET_ORDER, sha, protect
from .registry import candidate_hash, controls


OUT = BASE / "corrected_results"
MANIFEST = BASE / "corrected_manifest.json"
EXTRA_SOURCES = ("corrected_scan.py", "build_corrected_data.py", "total_return_audit.py", "CORRECTION_PROTOCOL.md")


def load_histories():
    m = json.loads(MANIFEST.read_text())
    out = {}
    for code in ASSET_ORDER:
        path = BASE / "corrected_snapshots" / (code + ".csv")
        if sha(path) != m["assets"][code]["sha256"]:
            raise ValueError("Corrected data changed: " + code)
        with path.open(newline="") as f:
            rows = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(f) if r]
        if not rows or rows[-1][0] != m["end"]:
            raise ValueError("Corrected data coverage differs")
        out[code] = rows
    return out, m["end"]


def fingerprints():
    fp = scan.fingerprints()
    fp["source_sha256"].update({p: sha(BASE / p) for p in EXTRA_SOURCES})
    fp["corrected_manifest_sha256"] = sha(MANIFEST)
    fp["action_manifest_sha256"] = sha(BASE / "results/corporate_actions_verified.json")
    fp["verified_cash_510500_sha256"] = sha(BASE / "results/verified_cash_510500.json")
    return fp


def configure():
    # Process-local routing only. Never edit the earlier scanner or artifacts.
    scan.OUT = OUT
    scan.load_histories = load_histories
    OUT.mkdir(exist_ok=True)


def develop(workers):
    protect()
    if (OUT / "selection.json").exists():
        raise RuntimeError("Corrected selection is already frozen")
    registry = json.loads((BASE / "results/registry.json").read_text())
    configs = registry["candidates"]
    scan.dump(OUT / "registry.json", registry)
    fp = fingerprints()
    scan.dump(OUT / "registration.json", dict(registered_at=scan.stamp(), fingerprints=fp,
              registry_sha256=sha(OUT / "registry.json"), candidate_count=len(configs),
              end="2026-09-24", fee=scan.FEE, stress_fee=scan.STRESS_FEE,
              primary_clock="original_same_close_with_corrected_total_return_indices",
              parent_registration_sha256=sha(BASE / "results/registration.json"),
              reason="Additive adjusted price ratios exaggerated cash-dividend assets; same candidate family rerun",
              clean_oos=False, next_open_permitted=False))
    scan.setup(configs)
    fee_checks = scan.verify_fee_transform(configs)
    dates = [d for d in scan._FRAME.dates if START <= d <= scan.TRAIN_END]
    metrics, paths = {}, {}
    for c in controls():
        result = scan.simulate(c, scan.TRAIN_END, details=True)
        paths[c["id"]] = result
        metrics[c["id"]] = scan.collect(c, result, dates)[0]
        print("CORRECTED CONTROL", c["id"], metrics[c["id"]]["metrics"]["selection"], flush=True)
    rows, dates = scan.batch(configs, scan.TRAIN_END, "selection", workers)
    if fingerprints() != fp:
        raise AssertionError("Corrected inputs changed during selection")
    scan.dump(OUT / "development.json", dict(results=rows, controls=metrics, fee_transform_checks=fee_checks))
    scan.dump(OUT / "development_controls.json", paths)
    chosen = scan.select(rows, configs, metrics["c_v92"])
    chosen.update(selected_at=scan.stamp(), registration_sha256=sha(OUT / "registration.json"),
                  development_sha256=sha(OUT / "development.json"),
                  return_matrix_sha256=sha(OUT / "selection_returns.npy"),
                  fee5_matrix_sha256=sha(OUT / "selection_fee5_returns.npy"),
                  candidate_ids=[c["id"] for c in configs],
                  selected_from="2014-2025 corrected-close known history; 2026 not used by this ranking")
    scan.dump(OUT / "selection.json", chosen)
    print("CORRECTED FROZEN", chosen["primary_return_champion"], chosen["risk_guarded_candidate"],
          "risk eligible", chosen["risk_eligible_count"], "protected", protect(), flush=True)


def evaluate(workers):
    registration = json.loads((OUT / "registration.json").read_text())
    chosen = json.loads((OUT / "selection.json").read_text())
    assert fingerprints() == registration["fingerprints"]
    for filename, key in (("registration.json", "registration_sha256"), ("development.json", "development_sha256"),
                          ("selection_returns.npy", "return_matrix_sha256"), ("selection_fee5_returns.npy", "fee5_matrix_sha256")):
        assert sha(OUT / filename) == chosen[key]
    assert sha(OUT / "registry.json") == registration["registry_sha256"]
    configs = json.loads((OUT / "registry.json").read_text())["candidates"]
    by_id = {c["id"]: c for c in configs}
    scan.setup(configs)
    rows, dates = scan.batch(configs, scan._END, "full", workers)
    for suffix in ("_returns.npy", "_fee5_returns.npy"):
        before = np.load(OUT / ("selection" + suffix), mmap_mode="r")
        after = np.load(OUT / ("full" + suffix), mmap_mode="r")
        if not np.allclose(before, after[:len(before)], rtol=1e-11, atol=1e-12):
            raise AssertionError("Future data changed earlier corrected paths")
    hindsight = sorted(rows, key=lambda r: (-r["metrics"]["full"]["cagr"], r["id"]))[0]
    original_choice = json.loads((BASE / "results/selection.json").read_text())["primary_return_champion"]
    ids = list(dict.fromkeys(n for n in (chosen["primary_return_champion"], chosen["risk_guarded_candidate"],
                                        hindsight["id"], original_choice) if n))
    paths, detailed = {}, {}
    for c in controls() + [by_id[n] for n in ids]:
        result = scan.simulate(c, scan._END, details=True)
        paths[c["id"]] = result
        detailed[c["id"]] = scan.collect(c, result, dates)[0]
    neighbors = []
    for name in dict.fromkeys(n for n in (chosen["primary_return_champion"], chosen["risk_guarded_candidate"]) if n):
        c = by_id[name]
        for scale in (.9, 1.1) if c["score_windows"] else ():
            altered = deepcopy(c)
            altered["score_windows"] = [max(2, round(w * scale)) for w in c["score_windows"]]
            altered["candidate_hash"] = candidate_hash(altered)
            altered["id"] = "neighbor_" + altered["candidate_hash"][:20]
            neighbors.append(dict(parent=name, scale=scale, windows=altered["score_windows"],
                                  metrics=scan.collect(altered, scan.simulate(altered), dates)[0]))
    report = dict(completed_at=scan.stamp(), end=scan._END, registration=registration,
                  selection=chosen, selection_sha256=sha(OUT / "selection.json"), fingerprints=fingerprints(),
                  results=rows, detailed_summaries=detailed, hindsight_winner=hindsight,
                  hindsight_label="Full-history hindsight maximum; no replacement of frozen pre-2026 selections",
                  original_input_winner_id=original_choice, window_neighbors=neighbors,
                  next_open_pressure=None, next_open_limitation="CSV open is a placeholder; raw shares/cash execution ledger needed",
                  full_return_matrix_sha256=sha(OUT / "full_returns.npy"),
                  full_fee5_matrix_sha256=sha(OUT / "full_fee5_returns.npy"),
                  pre2026_prefixes_unchanged=True, protected_files_verified=protect())
    assert report["fingerprints"] == registration["fingerprints"]
    scan.dump(OUT / "evaluation.json", report)
    scan.dump(OUT / "selected_paths.json", paths)
    print("CORRECTED FULL", {k: {"cagr": v["metrics"]["full"]["cagr"], "dd": v["metrics"]["full"]["max_dd"]}
                             for k, v in detailed.items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("develop", "evaluate", "statistics"))
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = p.parse_args()
    configure()
    if args.stage == "statistics":
        scan.statistics()
    elif args.stage == "develop":
        develop(args.workers)
    else:
        evaluate(args.workers)


if __name__ == "__main__":
    main()
