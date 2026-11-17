"""Immutable-history / completed-close research data for one frozen V12-R2.

No network calls; no writes outside this directory.  Import has no file I/O.
The ``features.vol20`` slot ALREADY contains prior_max20_60 decision risk.
Do not pass this bundle through risk_view/view_for_config a second time.
Only the chosen WLS25 / four-own-quote score lane is exposed.  This module
does not claim that other score definitions or volatility contexts exist.

``load_dataset() -> (arrays, metadata, base_config)`` is directly compatible
with v10_deep.reference.DatedValues and v12.reference.run_reference.
Run ``python -B research/v12_iteration_20261117/data_bundle.py`` to build and
verify all four archived fee/lag scenarios. Missing bundles build on first load.
"""
from bisect import bisect_right
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

ARRAY_PATH = HERE / "dataset.npz"
META_PATH = HERE / "dataset_metadata.json"
FROZEN_END = "2026-09-24"
CANDIDATE_ID = "v12r2_08b522c2e1aa291f054d"
INVALID_SCORE = -1e100
EXTENSION_FIELDS = ("close", "valid", "ret1", "volume_ratio", "mom5",
                    "mom20", "mom60", "vol20", "ma180", "ma250")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _archive_inputs():
    from v12_live.profile import load_profile
    from v12.exante_features import view_for_config

    profile = load_profile()  # Verifies the frozen model and dependency pins.
    config = profile["config"]
    assert profile["record"]["id"] == CANDIDATE_ID
    release = _json(ROOT / "v12_r2/release_receipt.json")
    files = ("results/features.npz", "results/feature_metadata.json",
             "results/paths.npz", "results/path_metadata.json")
    hashes = {}
    for name in files:
        digest = _sha(ROOT / "v12_r2" / name)
        if digest != release["file_sha256"][name]:
            raise ValueError("Frozen R2 data changed: " + name)
        hashes["v12_r2/" + name] = digest
    source_meta = _json(ROOT / "v12_r2/results/feature_metadata.json")
    with np.load(ROOT / "v12_r2/results/features.npz", allow_pickle=False) as z:
        source_arrays = {name: z[name] for name in z.files}
    effective = view_for_config(source_arrays, source_meta, config)
    lane = source_meta["score_names"].index(config["score"])
    arrays = {"features": effective["features"].copy(),
              "scores": source_arrays["scores"][lane:lane+1].copy(),
              "orders": source_arrays["orders"][lane:lane+1].copy(),
              "fear": source_arrays["fear"].copy(),
              "years": source_arrays["years"].copy()}
    meta = deepcopy(source_meta)
    meta["score_names"] = [config["score"]]
    return arrays, meta, config, hashes


def _check_archived_paths(arrays, meta, config):
    from v12.reference import run_reference

    path_meta = _json(ROOT / "v12_r2/results/path_metadata.json")
    candidate = path_meta["ids"].index(CANDIDATE_ID)
    audits = {}
    with np.load(ROOT / "v12_r2/results/paths.npz", allow_pickle=False) as saved:
        for name, lag, fee in (("close_1bp", 0, .0001), ("close_11bp", 0, .0011),
                               ("lag1_1bp", 1, .0001), ("lag1_11bp", 1, .0011)):
            result = run_reference(arrays, meta, dict(config, lag=lag),
                                   "2014-01-02", FROZEN_END, fee)
            for field in ("returns", "holdings", "summary"):
                np.testing.assert_array_equal(result[field], saved[name+"__"+field][candidate])
            audits[name] = dict(bit_exact_returns=True, bit_exact_holdings=True,
                                bit_exact_summary=True, observations=len(result["dates"]),
                                final_nav=float(result["summary"][0]),
                                max_drawdown=float(result["summary"][2]))
    return audits


def build_dataset():
    """Build a pinned close-data bundle; mutate only dataset* in this directory."""
    from v10_live.data import _load_seed, _read_state
    from v12_live.policy import latest_features

    arrays, meta, config, sources = _archive_inputs()
    assert meta["dates"][-1] == FROZEN_END
    frozen_count = len(meta["dates"])
    frozen_copies = {name: value.copy() for name, value in arrays.items()}
    seed = _load_seed()
    cache_path = ROOT / "data/v10_live/completed.json"
    cache_hash_before = _sha(cache_path)
    completed = _read_state(cache_path, seed)  # Validates cached checksum and date/asset axes.
    cache_hash_after = _sha(cache_path)
    if cache_hash_after != cache_hash_before:
        raise RuntimeError("Completed cache changed during snapshot; retry build")
    sources["data/v10_live/completed.json"] = cache_hash_before
    sources["v10_h_close/corrected_manifest.json"] = seed["manifest_hash"]
    for code in seed["tr"]:
        for relative in (
            "v10_h_close/corrected_snapshots/" + code + ".csv",
            "v10_h_close/results/price_audit/" + code + "_raw_2010-01-01_" + seed["end"] + ".json",
        ):
            sources[relative] = _sha(ROOT / relative)
    for relative in ("v12_r2/release_receipt.json", "v12_r2/registered_candidates.json",
                     "v12_live/policy.py", "v12_live/profile.py", "v12_live/constants.py",
                     "v10_live/data.py", "v10_live/policy.py", "v12/reference.py",
                     "v10_deep/reference.py", "v12/exante_features.py", "v11/features.py"):
        sources[relative] = _sha(ROOT / relative)
    if seed["end"] != FROZEN_END:
        raise ValueError("Unexpected seed/archive boundary")
    extension_dates = list(completed["calendar"])
    if extension_dates and extension_dates[0] <= FROZEN_END:
        raise ValueError("Extension overlaps frozen dates")
    rows = {code: [(r[0], float(r[2]), float(r[3])) for r in
                   seed["tr"][code] + completed["assets"][code]["tr"]]
            for code in seed["tr"]}
    axes = {code: [r[0] for r in history] for code, history in rows.items()}
    fields = {name: i for i, name in enumerate(meta["feature_names"])}
    extra = np.full((len(extension_dates), len(meta["assets"]), len(fields)), np.nan)
    extra[:, :, fields["valid"]] = 0.0
    scores = np.full((1, len(extension_dates), len(meta["assets"])), INVALID_SCORE)
    missing_quotes = []
    for i, date in enumerate(extension_dates):
        for ai, code in enumerate(meta["assets"]):
            if code not in rows:
                continue
            prefix = rows[code][:bisect_right(axes[code], date)]
            values = latest_features(prefix, date, code)
            for field in EXTENSION_FIELDS:
                extra[i, ai, fields[field]] = values[field]
            if not np.isfinite(values["close"]):
                missing_quotes.append(dict(date=date, asset=code))
            if values["valid"] > .5:
                scores[0, i, ai] = values["score"]
    arrays["features"] = np.ascontiguousarray(np.concatenate((arrays["features"], extra)))
    arrays["scores"] = np.ascontiguousarray(np.concatenate((arrays["scores"], scores), axis=1))
    extra_orders = np.argsort(-scores, axis=2, kind="stable").astype(np.int32)
    arrays["orders"] = np.ascontiguousarray(np.concatenate((arrays["orders"], extra_orders), axis=1))
    arrays["fear"] = np.concatenate((arrays["fear"], np.zeros(len(extension_dates), dtype=np.int32)))
    arrays["years"] = np.concatenate((arrays["years"], np.asarray([int(d[:4]) for d in extension_dates], dtype=np.int32)))
    meta["dates"] = list(meta["dates"]) + extension_dates
    meta["shape"] = list(arrays["features"].shape)
    # Appending the extension must leave every historical value unchanged,
    # including NaNs and sentinel values. The effective risk view is the baseline.
    for key in ("features", "fear", "years"):
        np.testing.assert_array_equal(arrays[key][:frozen_count], frozen_copies[key])
    for key in ("scores", "orders"):
        np.testing.assert_array_equal(arrays[key][:, :frozen_count], frozen_copies[key])
    audit = _check_archived_paths(arrays, meta, config)
    meta.update(
        dataset_schema=1, candidate_id=CANDIDATE_ID, base_config=deepcopy(config),
        risk_view_applied=True, risk_context_applied="prior_max20_60",
        score_names=[config["score"]],
        r2_score_specs=[dict(name=config["score"], window=25, smooth=4)],
        frozen_end=FROZEN_END, frozen_observations=frozen_count,
        extension_start=extension_dates[0] if extension_dates else None,
        completed_through=completed["last_completed"], extension_observations=len(extension_dates),
        extension_assets=list(rows), unsupported_extension_assets=[c for c in meta["assets"] if c not in rows],
        extension_feature_names=list(EXTENSION_FIELDS),
        unsupported_extension_features=[f for f in meta["feature_names"] if f not in EXTENSION_FIELDS],
        missing_extension_quotes=missing_quotes, source_sha256=sources,
        frozen_prefix_bit_exact=True, frozen_scenario_reproduction=audit,
        data_boundary_notes=[
            "features.vol20 already stores previous actual-own-quote max(VOL20,VOL60). Never apply risk_view again.",
            "The active score is unchanged WLS25 divided by its original VOL20, averaged over four actual own quotes.",
            "Original 24-asset axis is retained for exact archive holding indices; only the original live 11 assets have new observations.",
            "Unsupported extension asset closes are NaN, validity zero, scores invalid; no synthesized quotes or returns.",
            "Unsupported extension feature columns are NaN. Use only declared extension_feature_names for cross-boundary research.",
            "Cash has close data but is excluded from ranking. Fear is false in extension; baseline crash_mask=5 never uses QVIX.",
            "Dates come from verified completed benchmark observations, including 2026-09-28; no calendar fabrication.",
            "Post-release data are completed closes, not contemporaneous 14:50 execution marks. The current incomplete day is excluded.",
            "Historical selection has already seen 2021 and 2026; this research is not clean out-of-sample validation.",
        ])
    HERE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARRAY_PATH, **arrays)
    meta["dataset_sha256"] = _sha(ARRAY_PATH)
    meta["builder_sha256"] = _sha(Path(__file__))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    return arrays, meta, deepcopy(config)


def load_dataset(rebuild=False):
    """Read the pinned bundle, building only if absent or explicitly requested.

    A later live-cache update never silently changes an existing bundle.
    Rebuild explicitly to adopt a new completed-data snapshot.
    """
    if rebuild or not ARRAY_PATH.exists() or not META_PATH.exists():
        return build_dataset()
    meta = _json(META_PATH)
    if _sha(ARRAY_PATH) != meta["dataset_sha256"]:
        raise ValueError("Research dataset bytes differ from metadata")
    with np.load(ARRAY_PATH, allow_pickle=False) as z:
        arrays = {name: np.ascontiguousarray(z[name]) for name in z.files}
    return arrays, meta, deepcopy(meta["base_config"])


if __name__ == "__main__":
    _, metadata, _ = build_dataset()
    print(json.dumps({key: metadata[key] for key in (
        "candidate_id", "frozen_end", "completed_through", "extension_observations",
        "shape", "frozen_prefix_bit_exact", "frozen_scenario_reproduction",
        "missing_extension_quotes")}, ensure_ascii=False, indent=2))
