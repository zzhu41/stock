"""Reproduce frozen selected/baseline paths with independent Python accounting.

Read the frozen feature cache without calling build(), scan(), or protect().
The independent reference shares feature inputs; it does not independently
reconstruct market data, dividend adjustments, or technical indicators.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

# This import limits BLAS to one thread before NumPy is imported.
from .diagnostics import BASE, dump, load_stage, sha
import numpy as np
from .native import Simulator
from .reference import run_reference


def verify(stage="refinements"):
    loaded = load_stage(stage)
    registration = loaded["registration"]
    fingerprints = registration["fingerprints"]
    watched = {}
    for name, expected in fingerprints["sources"].items():
        path = BASE / name
        if sha(path) != expected:
            raise ValueError("Frozen core changed: " + str(path))
        watched[str(path)] = expected
    cache = BASE / "cache" / "features.npz"
    if sha(cache) != fingerprints["feature_cache_sha256"]:
        raise ValueError("Frozen feature cache changed")
    meta_path = cache.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    if meta["fingerprints"] != fingerprints["features"]:
        raise ValueError("Feature metadata differs from stage registration")
    watched[str(cache)] = sha(cache)
    watched[str(meta_path)] = sha(meta_path)
    watched[str(BASE / "reference.py")] = sha(BASE / "reference.py")
    watched[str(Path(__file__))] = sha(__file__)
    with np.load(cache, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    simulator = Simulator(arrays, meta)
    ids = list(dict.fromkeys(loaded["selected_ids"] + [loaded["baseline_id"]]))
    configs = [loaded["candidates"][loaded["ids"].index(candidate_id)] for candidate_id in ids]
    fee5_path = loaded["directory"] / "full_fee5_returns.npy"
    evaluation = json.loads((loaded["directory"] / "evaluation.json").read_text())
    if sha(fee5_path) != evaluation["full_fee5_sha256"]:
        raise ValueError("Saved fee5 matrix changed")
    saved_fee5 = np.load(fee5_path, mmap_mode="r", allow_pickle=False)
    if saved_fee5.shape != loaded["full"].shape:
        raise ValueError("Fee5 matrix shape differs")
    watched[str(fee5_path)] = evaluation["full_fee5_sha256"]
    start, end = registration["start"], registration["end"]
    fees = (0., registration["fee"], registration["stress_fee"])
    cases = []
    nofee = {}
    for fee in fees:
        native = simulator.run(configs, start=start, end=end, fee=fee, workers=1)
        for local_index, config in enumerate(configs):
            candidate_id = config["id"]
            row_index = loaded["ids"].index(candidate_id)
            reference = run_reference(arrays, meta, config, start=start, end=end, fee=fee)
            if reference["dates"] != loaded["full_dates"] or native["dates"] != loaded["full_dates"]:
                raise AssertionError("Reference/native/full saved calendars disagree")
            np.testing.assert_array_equal(reference["holdings"], native["holdings"][local_index])
            np.testing.assert_allclose(reference["returns"], native["returns"][local_index], rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(reference["summary"], native["summary"][local_index], rtol=1e-10, atol=1e-11)
            holdings = [meta["assets"][value] if value >= 0 else None for value in reference["holdings"]]
            if holdings != loaded["paths"][candidate_id]["holdings"]:
                raise AssertionError("Fresh reference holdings differ from frozen selected path")
            if fee == 0:
                nofee[candidate_id] = reference
            else:
                np.testing.assert_array_equal(reference["holdings"], nofee[candidate_id]["holdings"])
                charged = np.r_[False, reference["holdings"][1:] != reference["holdings"][:-1]]
                predicted = (1 + nofee[candidate_id]["returns"]) * np.where(charged, 1 - 2 * fee, 1.) - 1
                np.testing.assert_allclose(reference["returns"], predicted, rtol=1e-11, atol=1e-12)
            saved = loaded["full"][row_index] if fee == registration["fee"] else saved_fee5[row_index] if fee == registration["stress_fee"] else None
            if saved is not None:
                np.testing.assert_allclose(reference["returns"], saved, rtol=1e-11, atol=1e-12)
            cases.append(dict(id=candidate_id, fee_per_side=fee, observations=len(reference["returns"]),
                exact_holdings=True, exact_native_returns=bool(np.array_equal(reference["returns"], native["returns"][local_index])),
                max_native_return_error=float(np.max(np.abs(reference["returns"] - native["returns"][local_index]))),
                max_native_summary_error=float(np.max(np.abs(reference["summary"] - native["summary"][local_index]))),
                exact_saved_returns=bool(np.array_equal(reference["returns"], saved)) if saved is not None else None,
                max_saved_return_error=float(np.max(np.abs(reference["returns"] - saved))) if saved is not None else None,
                first_evaluation_return=float(reference["returns"][0]),
                fee_formula_verified=(fee != 0),
                summary=reference["summary"].tolist()))
            print(candidate_id, "fee", fee, "return error", cases[-1]["max_native_return_error"], flush=True)
    for path, expected in watched.items():
        if sha(path) != expected:
            raise ValueError("Input/source changed during verification: " + path)
    for name, expected in loaded["sources"].items():
        if sha(loaded["directory"] / name) != expected:
            raise ValueError("Frozen stage changed during verification: " + name)
    result = dict(stage=stage, completed_at=datetime.now(timezone.utc).isoformat(), passed=True,
        start=loaded["full_dates"][0], end=loaded["full_dates"][-1], candidate_ids=ids,
        independent_decision_and_accounting=True, shared_feature_inputs=True,
        scope="Frozen primary/guarded and v9.2; full daily holdings/returns/all 10 summaries at fees 0, 1, 5 bp; saved main/fee5 matrices; fee-only repricing from independent zero-fee path.",
        limitations=["Shared frozen features and source prices; separate indicator/data audits are still required.",
                     "Agreement verifies implementation fidelity, not strategy profitability or resistance to overfitting."],
        cases=cases, stage_input_sha256=loaded["sources"], source_and_cache_sha256=watched)
    destination = loaded["directory"] / "selected_reference_fidelity.json"
    dump(destination, result)
    print(destination, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="refinements")
    args = parser.parse_args()
    verify(args.stage)
