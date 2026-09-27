"""Immutable local snapshots and dated features for the isolated v10 study."""
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path

from .frozen import strategy
from .frozen.metadata import CASH, STOCK_POOL, GLOBAL_POOL


BASE = Path(__file__).resolve().parent
START = "2014-01-01"
END = "2026-09-11"


def verify_protected():
    """Check the existing worktree, including prior uncommitted repairs."""
    manifest = json.loads((BASE / "protected_manifest.json").read_text())
    mismatches = []
    for name, expected in manifest["sha256"].items():
        path = BASE.parent / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(name)
    if mismatches:
        raise AssertionError("Existing files changed during isolated research: " + ", ".join(mismatches))
    return len(manifest["sha256"])


def load_histories():
    manifest = json.loads((BASE / "data_manifest.json").read_text())
    histories = {}
    for code, expected in manifest.items():
        path = BASE / "data" / (code + ".csv")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["snapshot_sha256"]:
            raise ValueError("Frozen data fingerprint changed: " + code)
        with path.open(newline="") as f:
            rows = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(f) if r]
        if [r[0] for r in rows] != sorted({r[0] for r in rows}):
            raise ValueError("Unsorted or duplicate dates: " + code)
        if any(not all(math.isfinite(x) for x in r[1:]) or min(r[1:3]) <= 0 or r[3] < 0 for r in rows):
            raise ValueError("Invalid price/volume: " + code)
        histories[code] = rows
    return histories


def configure(params, stock_pool=STOCK_POOL, global_pool=GLOBAL_POOL):
    # Frozen presets explicitly contain every original engine parameter. Only
    # strategy globals with matching names are assigned; engine knobs stay here.
    for key, value in params.items():
        name = key.upper()
        if hasattr(strategy, name):
            setattr(strategy, name, dict(value) if isinstance(value, dict) else value)
    strategy.STOCK_POOL = list(stock_pool)
    strategy.GLOBAL_POOL = list(global_pool)
    strategy.MIN_ROWS = max(strategy.MA_BULL,
                            max(strategy.MOM_WINDOWS) + 1 + strategy.SKIP_DAYS,
                            strategy.MOM_MAIN + 1 + strategy.SKIP_DAYS) + strategy.VOL_DAYS


def wls_score(closes, window, volatility):
    values = closes[-window:]
    n = len(values)
    weights = list(range(1, n + 1))
    total = sum(weights)
    mx = sum(i * w for i, w in enumerate(weights)) / total
    my = sum(v * w for v, w in zip(values, weights)) / total
    xy = sum(w * (i - mx) * (v - my) for i, (v, w) in enumerate(zip(values, weights)))
    xx = sum(w * (i - mx) ** 2 for i, w in enumerate(weights))
    return xy / xx / my * 250 / volatility if xx > 0 and my > 0 and volatility > 0 else 0.0


class Features:
    """All slicing uses bisect_right(date); later snapshot rows are inaccessible."""
    def __init__(self, histories):
        self.histories = histories
        self.dates = {c: [r[0] for r in rows] for c, rows in histories.items()}
        self.closes = {c: [r[2] for r in rows] for c, rows in histories.items()}
        self.volumes = {c: [r[3] for r in rows] for c, rows in histories.items()}
        self.cache = {}
        self.presets = json.loads((BASE / "frozen/presets.json").read_text())

    def last_index(self, date, code):
        ds = self.dates.get(code, [])
        i = bisect.bisect_right(ds, date) - 1
        return i if i >= 0 and ds[i] == date else None

    def cash_available(self, date):
        return self.last_index(date, CASH) is not None

    def __call__(self, date, code, lookbacks):
        i = self.last_index(date, code)
        if i is None or not lookbacks or i < max(lookbacks):
            return None
        cs = self.closes[code]
        returns = {n: cs[i] / cs[i - n] - 1 for n in lookbacks}
        return dict(score=sum(returns.values()) / len(returns), returns=returns, available=True)

    def table(self, date, pool, windows=(25,)):
        table = []
        for code in pool:
            i = self.last_index(date, code)
            if i is None or code == CASH:
                continue
            key = (date, code, tuple(windows))
            if key not in self.cache:
                # All registered high-return candidates share price features;
                # only WLS windows change scores. Gates are applied in decide.
                cs = self.closes[code][:i + 1]
                vs = self.volumes[code][:i + 1]
                ind = strategy.indicators(cs, vs)
                if ind is not None and tuple(windows) != (25,):
                    ind["score"] = sum(wls_score(cs, w, ind["vol"]) for w in windows) / len(windows)
                self.cache[key] = ind
            ind = self.cache[key]
            if ind is not None:
                table.append((code, dict(ind)))
        table.sort(key=lambda item: item[1]["score"], reverse=True)
        return table
