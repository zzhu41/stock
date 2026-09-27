"""This batch's frozen snapshots and verification of all prior work."""
import csv
import hashlib
import json
import math
from pathlib import Path

from v10_search.data import ASSET_ORDER, UNIVERSE, CASH, GOLD, BENCHMARK


BASE = Path(__file__).resolve().parent
START = "2014-01-01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protect():
    files = json.loads((BASE / "protected_manifest.json").read_text())["sha256"]
    bad = [name for name, expected in files.items()
           if not (BASE.parent / name).is_file() or sha(BASE.parent / name) != expected]
    if bad:
        raise AssertionError("A previous strategy/research file changed: " + ", ".join(bad))
    return len(files)


def load_histories():
    manifest = json.loads((BASE / "snapshot_manifest.json").read_text())
    histories = {}
    for code in ASSET_ORDER:
        path = BASE / "snapshots" / (code + ".csv")
        expected = manifest["assets"][code]
        if sha(path) != expected["sha256"]:
            raise ValueError("Frozen input changed: " + code)
        with path.open(newline="") as f:
            rows = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(f) if r]
        if not rows or rows[-1][0] != manifest["end"] or [r[0] for r in rows] != sorted({r[0] for r in rows}):
            raise ValueError("Invalid date axis: " + code)
        if any(not all(math.isfinite(x) for x in r[1:]) or min(r[1:3]) <= 0 or r[3] < 0 for r in rows):
            raise ValueError("Invalid price/volume: " + code)
        histories[code] = rows
    return histories, manifest["end"]


def load_fear(calendar):
    from v10_round2.v92 import QvixSeries
    path = BASE / "snapshots/qvix50.csv"
    manifest = json.loads((BASE / "qvix_manifest.json").read_text())
    if sha(path) != manifest["sha256"]:
        raise ValueError("QVIX snapshot changed")
    with path.open(newline="") as f:
        rows = [(r[0], float(r[1])) for r in csv.reader(f) if r]
    series = QvixSeries(rows)
    return tuple(series.state(date)["active"] for date in calendar)
