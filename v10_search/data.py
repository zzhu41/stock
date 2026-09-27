"""Frozen 24-ETF universe and protection of every previous strategy artifact."""
import csv
import hashlib
import json
import math
from pathlib import Path

from v10_next.frozen.metadata import UNIVERSE as ORIGINAL_UNIVERSE

BASE = Path(__file__).resolve().parent
START, END = "2014-01-01", "2026-09-11"
UNIVERSE = dict(ORIGINAL_UNIVERSE)
UNIVERSE.update({
    "510880": ("上证红利ETF", "sh", "stock"),
    "515080": ("中证红利ETF", "sh", "stock"),
    "515100": ("红利低波100ETF", "sh", "stock"),
    "159928": ("消费ETF", "sz", "stock"),
    "512010": ("医药ETF", "sh", "stock"),
    "512070": ("非银ETF", "sh", "stock"),
    "512880": ("证券ETF", "sh", "stock"),
    "512800": ("银行ETF", "sh", "stock"),
    "512200": ("房地产ETF", "sh", "stock"),
    "513030": ("德国ETF", "sh", "global"),
    "513520": ("日经ETF", "sh", "global"),
    "159985": ("豆粕ETF", "sz", "global"),
    "511010": ("国债ETF", "sh", "global"),
})
# 'global' is the engine's exemption from the domestic-equity regime gate;
# commodity/bond instruments are NOT being described as cross-border equities.
ASSET_ORDER = tuple(ORIGINAL_UNIVERSE) + tuple(sorted(set(UNIVERSE) - set(ORIGINAL_UNIVERSE)))
CASH, GOLD, BENCHMARK = "511880", "518880", "510300"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protect():
    files = json.loads((BASE / "protected_manifest.json").read_text())["sha256"]
    changed = [p for p, h in files.items() if not (BASE.parent / p).is_file() or sha(BASE.parent / p) != h]
    if changed:
        raise AssertionError("Previous strategy/research files changed: " + ", ".join(changed))
    return len(files)


def load_histories():
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    out = {}
    for code in ASSET_ORDER:
        path = BASE / "snapshots" / (code + ".csv")
        if sha(path) != manifest[code]["sha256"]:
            raise ValueError("Frozen input changed: " + code)
        with path.open(newline="") as f:
            rows = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(f) if r]
        if not rows or rows[-1][0] != END or [r[0] for r in rows] != sorted({r[0] for r in rows}):
            raise ValueError("Invalid date coverage: " + code)
        if any(not all(math.isfinite(v) for v in r[1:]) or min(r[1:3]) <= 0 or r[3] < 0 for r in rows):
            raise ValueError("Invalid price/volume: " + code)
        out[code] = rows
    return out
