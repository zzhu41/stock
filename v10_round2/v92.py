"""Isolated v9.2 control and deep-drop + QVIX ablation for round two.

The signal uses a dated QVIX closing observation, never a previous observation
silently labelled as today's. Historical publication timestamps are unknown:
research assumes that close's value is available by next-open execution. This
does not establish that it was available at 14:50 or exactly at the close.

No original strategy or first-round file is modified. Snapshot creation is an
explicit local-only operation; importing this module has no file/network I/O.
"""
import bisect
import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from v10_next.data import configure
from v10_next.frozen import strategy
from v10_next.frozen.metadata import CASH, GLOBAL_POOL, STOCK_POOL
from v10_next.legacy import LegacyPolicy


BASE = Path(__file__).resolve().parent
END = "2026-09-11"
ALL_CHANNELS = ("deep", "qvix", "volume")
QVIX_Z, QVIX_M5, QVIX_DEPTH = 2.5, -.04, .20
VOLUME_RATIO, VOLUME_M5, VOLUME_DEPTH = 2.0, -.04, .10


def freeze_qvix():
    """Copy the existing local CSV prefix once; never refresh or change a source."""
    source = BASE.parent / "data/qvix50.csv"
    snapshot = BASE / "data/qvix50.csv"
    manifest_path = BASE / "qvix_manifest.json"
    if snapshot.exists() or manifest_path.exists():
        load_qvix()  # Reject partial or changed snapshots instead of overwriting.
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = source.read_bytes()
    with source.open(newline="", encoding="utf-8") as f:
        rows = [(r[0], float(r[1])) for r in csv.reader(f) if r and r[0] <= END]
    QvixSeries(rows)  # validate before writing either file
    if not rows:
        raise ValueError("Local QVIX has no observations before the fixed research cutoff")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    manifest = dict(source="data/qvix50.csv", source_sha256=hashlib.sha256(raw).hexdigest(),
                    snapshot="data/qvix50.csv", snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    rows=len(rows), first_date=rows[0][0], last_date=rows[-1][0], cutoff=END,
                    publication_timestamps_available=False,
                    limitation="Dated closes only; next-open availability is an assumption, not verified publication timing.")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_qvix():
    snapshot = BASE / "data/qvix50.csv"
    manifest = json.loads((BASE / "qvix_manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != manifest["snapshot_sha256"]:
        raise ValueError("Frozen QVIX fingerprint changed")
    with snapshot.open(newline="", encoding="utf-8") as f:
        rows = [(r[0], float(r[1])) for r in csv.reader(f) if r]
    if (len(rows) != manifest["rows"] or not rows or rows[-1][0] != manifest["last_date"]
            or rows[-1][0] > END or manifest["cutoff"] != END):
        raise ValueError("Frozen QVIX coverage differs from its manifest")
    return QvixSeries(rows)


class QvixSeries:
    """Population z-score against up to 250 strictly previous rows, minimum 120."""
    def __init__(self, rows):
        self.rows = tuple((d, float(v)) for d, v in rows)
        self.dates = tuple(d for d, _ in self.rows)
        self.values = tuple(v for _, v in self.rows)
        if tuple(sorted(set(self.dates))) != self.dates:
            raise ValueError("QVIX dates must be unique and ascending")
        if any(not math.isfinite(v) or v <= 0 for v in self.values):
            raise ValueError("QVIX observations must be finite and positive")
        self._cache = {}

    def state(self, date):
        if date not in self._cache:
            i = bisect.bisect_right(self.dates, date) - 1
            out = dict(requested_date=date, date=self.dates[i] if i >= 0 else None,
                       value=self.values[i] if i >= 0 else None, z=None,
                       active=False, available=False, previous_samples=0, note="missing_same_day_qvix")
            if i >= 0 and self.dates[i] == date:
                window = self.values[max(0, i - 250):i]
                out["previous_samples"] = len(window)
                if len(window) < 120:
                    out["note"] = "fewer_than_120_previous_qvix_rows"
                else:
                    mean = sum(window) / len(window)
                    std = (sum((v - mean) ** 2 for v in window) / len(window)) ** .5
                    z = (self.values[i] - mean) / std if std > 0 else 0.0
                    out.update(z=z, active=z >= QVIX_Z, available=True, note="")
            self._cache[date] = out
        return dict(self._cache[date])


def trigger_channels(ind, volume_ratio, fear, channels=ALL_CHANNELS,
                     deep_m5=-.08, deep_depth=.20):
    """The exact union used by the original v9.2 shadow and lab, including edges."""
    selected = set(channels)
    if selected - set(ALL_CHANNELS):
        raise ValueError("Unknown crash channel")
    hit = []
    if "deep" in selected and deep_m5 < 0 and ind["mom5"] <= deep_m5 and ind["dist_ma250"] < -deep_depth:
        hit.append("deep")
    if "qvix" in selected and fear and ind["mom5"] <= QVIX_M5 and ind["dist_ma250"] < -QVIX_DEPTH:
        hit.append("qvix")
    if "volume" in selected and volume_ratio >= VOLUME_RATIO and ind["mom5"] <= VOLUME_M5 \
            and ind["dist_ma250"] < -VOLUME_DEPTH:
        hit.append("volume")
    return hit


class V92Policy(LegacyPolicy):
    """Same v9.1 ordinary rules, with channel-union entries and fill-based locks."""
    def __init__(self, config, features, calendar, qvix):
        candidate = dict(id="control_v92", base_version="v9.1", stock_pool=tuple(STOCK_POOL),
                         global_pool=tuple(GLOBAL_POOL), score_windows=(25,), risk_weight=1.0)
        candidate.update(deepcopy(config))
        self.channels = tuple(candidate.get("channels", ALL_CHANNELS))
        if len(set(self.channels)) != len(self.channels) or set(self.channels) - set(ALL_CHANNELS):
            raise ValueError("Invalid or duplicate crash channels")
        self.qvix = qvix
        if "qvix" in self.channels and not callable(getattr(qvix, "state", None)):
            raise TypeError("QVIX channel requires a dated QvixSeries")
        super().__init__(candidate, features, calendar)
        if not 0 < self.risk_weight <= 1:
            raise ValueError("Crash policy risk weight must be positive and at most one")
        self.metadata.update(channels=list(self.channels), qvix_unavailable_dates=[],
                             publication_timestamps_verified=False)

    def volume_ratio(self, date, code):
        i = self.features.last_index(date, code)
        if i is None or i < 20:
            return 0.0
        volumes = self.features.volumes[code]
        prior = volumes[i - 20:i]
        average = sum(prior) / 20
        return volumes[i] / average if average > 0 else 0.0

    def __call__(self, date, observed_histories, state):
        configure(self.params, self.stock_pool, self.global_pool)
        weights = state["weights"]
        risky = [c for c, w in weights.items() if c != CASH and w > 1e-9]
        if len(risky) > 1:
            raise ValueError("v9.2 control expects at most one risk asset")
        holding = risky[0] if risky else CASH if weights.get(CASH, 0) > 1e-9 else None
        i = self.index[date]
        if self.pending_crash and holding == self.pending_crash["code"]:
            entry = state["holding_since"].get(holding)
            if entry is None or entry < self.pending_crash["signal_date"]:
                raise AssertionError("Crash entry was not confirmed by an actual fill")
            self.lock_code = holding
            self.lock_entry_index = self.index[entry]
            self.metadata["crash_events"].append(dict(self.pending_crash, fill_date=entry))
            self.pending_crash = None
        elif self.pending_crash and state.get("execution_deferred"):
            self.metadata["trace"].append(dict(date=date, target=self.pending_crash["code"],
                                                reason="crash_order_deferred"))
            return None
        elif self.pending_crash:
            raise AssertionError("Crash order neither filled nor explicitly deferred")

        qstate = (self.qvix.state(date) if "qvix" in self.channels else
                  dict(active=False, available=False, date=None, z=None, note="qvix_channel_disabled"))
        if "qvix" in self.channels and not qstate["available"]:
            self.metadata["qvix_unavailable_dates"].append(date)
        table = self.features.table(date, self.pool, self.windows)
        entry = state["holding_since"].get(holding)
        age = i - self.index[entry] + 1 if entry else 0
        target, reason = strategy.decide(table, holding, age)
        if self.lock_code and holding == self.lock_code and i - self.lock_entry_index < self.params["crash_lock"] - 1:
            target, reason = holding, "confirmed_crash_lock"
        else:
            self.lock_code = self.lock_entry_index = None
            for code, ind in table:
                if code not in self.pool or code == holding:
                    continue
                ratio = self.volume_ratio(date, code)
                hit = trigger_channels(ind, ratio, qstate["active"], self.channels,
                                       self.params["crash_mom5"], self.params["crash_below_ma"])
                if hit:
                    target, reason = code, "crash_union_entry:" + "+".join(hit)
                    self.pending_crash = dict(signal_date=date, code=code, channels=hit,
                                               mom5=ind["mom5"], dist_ma250=ind["dist_ma250"],
                                               volume_ratio20=ratio, qvix_date=qstate["date"], qvix_z=qstate["z"])
                    break

        self.metadata["trace"].append(dict(date=date, target=target, reason=reason,
                                            qvix_available=qstate["available"], qvix_date=qstate["date"],
                                            qvix_z=qstate["z"], qvix_note=qstate["note"]))
        if target == holding and not state.get("pending_target"):
            return None
        if target == CASH:
            return {CASH: 1.0} if self.features.cash_available(date) else {}
        target_weights = {target: self.risk_weight}
        if self.risk_weight < 1 and self.features.cash_available(date):
            target_weights[CASH] = 1 - self.risk_weight
        return target_weights


def factory(config, features, calendar, qvix):
    return V92Policy(config, features, calendar, qvix)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-qvix", action="store_true", help="Create the new local snapshot once, without network access")
    args = parser.parse_args()
    if args.freeze_qvix:
        print(json.dumps(freeze_qvix(), ensure_ascii=False, indent=2))
    else:
        parser.error("Use --freeze-qvix for the explicit local snapshot step")
