"""Public factories for the two frozen v10 research releases.

This module reads only files under v10_next and supplied price histories. It
does not import, patch, or write the existing production strategy modules.
"""
import hashlib
import json
from pathlib import Path

from .candidates import build_from_config
from .data import BASE, Features
from .legacy import factory


def load_profile(variant):
    path = BASE / "profiles.json"
    if not path.exists():
        raise RuntimeError("Run and review the bounded study before freezing release profiles")
    profiles = json.loads(path.read_text(encoding="utf-8"))
    if variant not in profiles["variants"]:
        raise ValueError("Variant must be growth or robust")
    for name, key in (("data_manifest.json", "data_manifest_sha256"),
                      ("results/evaluation.json", "evaluation_sha256"),
                      ("results/selection.json", "selection_sha256")):
        artifact = BASE / name
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != profiles[key]:
            raise ValueError("Frozen research artifact changed: " + name)
    for name, expected in profiles["implementation_sha256"].items():
        if hashlib.sha256((BASE / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen v10 implementation changed: " + name)
    profile = profiles["variants"][variant]
    raw = json.dumps(profile["config"], ensure_ascii=False, sort_keys=True).encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != profile["config_sha256"]:
        raise ValueError("Frozen profile config changed")
    return profile


def required_codes(profile):
    c = profile["config"]
    risk = list(c["pool"]) if c["kind"] == "monthly_momentum" else list(c["stock_pool"]) + list(c["global_pool"]) + [c["gold"]]
    # CSI300 supplies the calendar, including for monthly strategies.
    return tuple(dict.fromkeys(risk + [c["cash"], "510300"]))


def make_policy(variant, histories, calendar):
    profile = load_profile(variant)
    missing = set(required_codes(profile)) - set(histories)
    if missing:
        raise ValueError("Required histories missing: " + ", ".join(sorted(missing)))
    features = Features(histories)
    return build_from_config(profile["config"], factory(features, calendar), features)
