"""Selective frozen-model identity validation, invoked only during decisions.

The user selected this fixed exploratory result for observation. Its old
research primary=None and failed qualification are preserved verbatim.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .constants import CANDIDATE_ID,CANDIDATE_HASH,STRATEGY_ID

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "v12_r2"
RELEASE_SHA256 = "f05425fd1e4f599f780d89d785ce93074e1d3af2fe7ea4c8832ad7e9058582eb"
RESEARCH_FILES = ("registered_candidates.json", "registration.json", "results/selection.json",
                  "frozen_inputs.json", "data.py", "features.py", "research.py")
MODEL_DEPENDENCIES = ("v12/reference.py", "v12/schema.py", "v12/diagnostic_features.py",
    "v12/exante_features.py", "v11/features.py", "v10_deep/reference.py", "v10_deep/features.py",
    "v10_deep/schema.py", "v10_deep/registry.py", "v10_h_close/corrected_manifest.json")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_profile():
    """Read the one approved identity, never choose a winner or read an account."""
    release_path=RESEARCH/"release_receipt.json"
    if _sha(release_path)!=RELEASE_SHA256:
        raise ValueError("V12-R2 frozen release receipt changed")
    release=json.loads(release_path.read_text())
    for name in RESEARCH_FILES:
        if _sha(RESEARCH/name)!=release["file_sha256"][name]:
            raise ValueError("V12-R2 frozen model/profile changed: "+name)
    old=json.loads((RESEARCH/"frozen_inputs.json").read_text())["sha256"]
    for name in MODEL_DEPENDENCIES:
        if _sha(ROOT/name)!=old[name]:
            raise ValueError("V12-R2 frozen model dependency changed: "+name)
    selection=json.loads((RESEARCH/"results/selection.json").read_text())
    if (selection["primary"] is not None or selection["balanced_reference"] is not None
            or selection["exploratory"]["highest_cagr"]!=CANDIDATE_ID):
        raise ValueError("V12-R2 frozen research status/identity changed")
    registry=json.loads((RESEARCH/"registered_candidates.json").read_text())
    record=next((r for r in registry["records"] if r["id"]==CANDIDATE_ID),None)
    if record is None or record["hash"]!=CANDIDATE_HASH:
        raise ValueError("V12-R2 fixed candidate missing or hash changed")
    # The exact schema includes actual WLS window/smoothing; a display label
    # or aliased native score index is insufficient to identify the strategy.
    from v12_r2.research import identity
    if identity(record["config"],record["feature_spec"])!=CANDIDATE_HASH:
        raise ValueError("V12-R2 semantic configuration changed")
    if record["feature_spec"]!=dict(window=25,smooth=4,ma_window=180):
        raise ValueError("V12-R2 feature definition changed")
    return dict(record=deepcopy(record),config=deepcopy(record["config"]),strategy_id=STRATEGY_ID,
        selected_by="explicit_user_designation",research_primary=None,research_balanced_reference=None,
        research_qualified=False,order_submission=False)
