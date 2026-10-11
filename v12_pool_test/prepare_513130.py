"""Build the 513130 (Hang Seng Tech ETF) total-return history and feature columns.

Isolation guarantees:
- reads frozen artifacts read-only (v12/inputs, v12_r2/results, corrected snapshots);
- writes only inside v12_pool_test/;
- re-implements the exact frozen feature arithmetic from v10_deep.features and
  v12.diagnostic_features, then proves bit-level equivalence on an existing asset
  (513100) before trusting the new 513130 column.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v10_deep.features import _regression, _rolling_mean  # noqa: E402
from v12.diagnostic_features import _raw_scores  # noqa: E402
from v10_h_close.total_return_audit import reconstruct_total_return  # noqa: E402

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
CODE = "513130"

FEATURE_NAMES = json.loads((ROOT / "v12/inputs/features.json").read_text())["feature_names"]
F = {n: i for i, n in enumerate(FEATURE_NAMES)}
MOM_WINDOWS = (3, 5, 10, 15, 20, 30, 40, 60, 90, 120, 180, 240)
VOL_WINDOWS = (10, 20, 40, 60, 120)
MA_WINDOWS = (80, 100, 120, 150, 180, 200, 250, 300, 360)


def rebuild_total_return():
    """Zero-corporate-action reconstruction; 513130 has no dividends/splits
    (vendor exposes no qfq/hfq series; issuer records confirm none)."""
    target = DATA / (CODE + "_corrected.csv")
    if target.exists():
        with target.open(newline="") as f:
            return [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(f) if r]
    raw = json.loads((DATA / "513130_raw_2020-01-01_2026-09-24.json").read_text())["rows"]
    raw_rows = [(r[0], float(r[1]), float(r[2]), float(r[5])) for r in raw]
    # qfq == raw expresses the verified empty action table; cash_events={} is the
    # caller-verified complete calendar (no dividend, no split since 2021-05-24).
    rebuilt = reconstruct_total_return(raw_rows, raw_rows, (), code=CODE, cash_events={})
    with target.open("w", newline="") as f:
        csv.writer(f).writerows(rebuilt["rows"])
    print("REBUILT", CODE, len(rebuilt["rows"]), "rows,",
          rebuilt["diagnostics"]["method"], "cash events:", len(rebuilt["cash_events"]))
    return rebuilt["rows"]


def feature_columns(rows, calendar):
    """Exact v10_deep.features arithmetic for one asset -> (calendar, 36) block.

    rows: (date, placeholder, TR close, volume); calendar: benchmark date axis.
    """
    date_lookup = {d: i for i, d in enumerate(calendar)}
    prices = np.asarray([r[2] for r in rows])
    volumes = np.asarray([r[3] for r in rows])
    own = np.asarray([i for i, r in enumerate(rows) if r[0] in date_lookup], dtype=int)
    idx = np.asarray([date_lookup[rows[i][0]] for i in own], dtype=int)
    ret = np.concatenate(([0.0], prices[1:] / prices[:-1] - 1))
    arrays = {"close": prices, "ret1": ret, "valid": (np.arange(len(rows)) >= 269).astype(float)}
    for w in MOM_WINDOWS:
        values = np.full(len(rows), np.nan)
        values[w:] = prices[w:] / prices[:-w] - 1
        arrays["mom%d" % w] = values
    for w in VOL_WINDOWS:
        avg = _rolling_mean(ret, w)
        arrays["vol%d" % w] = np.sqrt(np.maximum(0, _rolling_mean(ret * ret, w) - avg * avg))
    for w in MA_WINDOWS:
        arrays["ma%d" % w] = prices / _rolling_mean(prices, w) - 1
    for w in (20, 60, 120):
        values = np.full(len(rows), np.nan)
        for j in range(w - 1, len(rows)):
            values[j] = prices[j] / max(prices[j - w + 1:j + 1]) - 1
        arrays["dd%d" % w] = values
    volume_average = np.roll(_rolling_mean(volumes, 20), 1)
    arrays["volume_ratio"] = np.divide(volumes, volume_average, out=np.zeros(len(rows)), where=volume_average > 0)
    arrays["volume_ratio"][:20] = np.nan
    path_length = _rolling_mean(np.concatenate(([0.0], np.abs(np.diff(prices)))), 20) * 20
    change = np.abs(prices - np.roll(prices, 20))
    arrays["er20"] = np.divide(change, path_length, out=np.ones(len(rows)), where=path_length > 0)
    arrays["er20"][:20] = np.nan
    for w in (120, 250):
        ema = np.empty(len(rows))
        ema[0] = prices[0]
        alpha = 2.0 / (w + 1)
        for j in range(1, len(rows)):
            ema[j] = alpha * prices[j] + (1 - alpha) * ema[j - 1]
        arrays["ema%d" % w] = prices / ema - 1
    block = np.full((len(calendar), len(FEATURE_NAMES)), np.nan)
    block[:, F["valid"]] = 0.0
    for name, values in arrays.items():
        block[idx, F[name]] = values[own]
    # Score lanes on the asset's own quoted-row series.
    scores = {}
    raw25 = _raw_scores(prices, 25)  # exact frozen list-sum WLS25/VOL20
    scores["wls25_v20"] = (idx, np.asarray(raw25, dtype=float)[own])
    mean, slope, unused = _regression(prices, 20)
    trend = np.full(len(rows), np.nan)
    trend[19:] = slope / mean * 250
    wls20 = np.divide(trend, arrays["vol20"], out=np.zeros(len(rows)), where=arrays["vol20"] > 0)
    series = wls20[own]
    smooth = np.full(len(own), np.nan)
    smooth[2:] = (series[2:] + series[1:-1] + series[:-2]) / 3
    scores["wls20_smooth3"] = (idx, smooth)
    return block, scores, own


def r2_lane(rows, calendar, window, smooth):
    """v12.diagnostic_features.make_view lane arithmetic for one asset."""
    date_lookup = {d: i for i, d in enumerate(calendar)}
    prices = np.asarray([r[2] for r in rows])
    raw = _raw_scores(prices, window)
    smoothed = _rolling_mean(raw, smooth) if smooth > 1 else raw
    required = max(270, window + smooth - 1)
    lane = np.full(len(calendar), -1e100)
    for j, r in enumerate(rows):
        if r[0] not in date_lookup:
            continue
        if j + 1 >= required and np.isfinite(smoothed[j]):
            lane[date_lookup[r[0]]] = smoothed[j]
    return lane


def selfcheck(rows, calendar, code, frozen_features, frozen_scores, ai):
    """Prove this module's arithmetic reproduces frozen columns for an old asset."""
    block, scores, own = feature_columns(rows, calendar)
    quote_last = max(i for i, r in enumerate(rows) if r[0] <= "2026-09-11")
    legacy_end_date = rows[quote_last][0]
    checks = []
    # Non-overwritten features must match bit-for-bit over the whole history.
    for name in ("close", "ret1", "mom3", "mom10", "mom30", "mom60", "mom240",
                 "vol10", "vol40", "vol60", "vol120", "ma80", "ma180", "ma360",
                 "dd20", "dd60", "dd120", "ema120", "ema250", "er20", "valid"):
        same = np.array_equal(block[:, F[name]], frozen_features[:, ai, F[name]], equal_nan=True)
        checks.append((name, same))
    # Legacy-overwritten features: identical once the legacy bank ends (2026-09-11).
    # Vendor/legacy float paths differ at ~1 ulp; accept bit-equal or <=1e-12 abs.
    tail = np.asarray([d > legacy_end_date for d in calendar])
    for name in ("mom5", "mom20", "vol20", "ma250", "volume_ratio"):
        a, b = block[tail, F[name]], frozen_features[tail, ai, F[name]]
        same = bool(np.array_equal(a, b, equal_nan=True) or
                    np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True))
        checks.append((name + "@post-legacy", same))
    # Frozen lanes are -1e100 wherever the frozen mask (close/valid) removed the
    # raw value; compare only where the frozen lane kept an observation.
    idx, values = scores["wls25_v20"]
    frozen_lane = frozen_scores[0][:, ai]
    frozen_valid = np.isfinite(frozen_lane[idx]) & (frozen_lane[idx] > -1e90)
    checks.append(("wls25_v20", bool(frozen_valid.any()) and
                   bool(np.array_equal(values[frozen_valid], frozen_lane[idx][frozen_valid]))))
    idx20, smooth20 = scores["wls20_smooth3"]
    frozen20 = frozen_scores[1][:, ai]
    frozen20_valid = np.isfinite(frozen20[idx20]) & (frozen20[idx20] > -1e90)
    checks.append(("wls20_smooth3", bool(frozen20_valid.any()) and
                   bool(np.array_equal(smooth20[frozen20_valid], frozen20[idx20][frozen20_valid]))))
    failed = [name for name, ok in checks if not ok]
    print("SELFCHECK", code, "legacy bank ends", legacy_end_date,
          "checks:", len(checks), "failed:", failed if failed else "none", flush=True)
    if failed:
        for name, ok in checks:
            if not ok:
                col = F.get(name.split("@")[0])
                if col is not None:
                    a, b = block[:, col], frozen_features[:, ai, col]
                    diff = np.flatnonzero(~(a == b) & ~(np.isnan(a) & np.isnan(b)))
                    print("  first diffs", name, [calendar[i] for i in diff[:5]],
                          a[diff[:3]], b[diff[:3]])
        raise SystemExit("Self-check failed: arithmetic does not reproduce frozen columns")


def main():
    from v12_r2.data import verify_inputs, load_base
    verify_inputs()
    arrays, meta, histories = load_base()
    calendar = list(meta["dates"])
    rows = rebuild_total_return()
    if rows[-1][0] != calendar[-1]:
        raise ValueError("513130 must extend to the frozen end date")
    # Prove equivalence on an existing asset first (513100 has full history).
    ai100 = list(meta["assets"]).index("513100")
    selfcheck(histories["513100"], calendar, "513100", arrays["features"], arrays["scores"], ai100)
    block, scores, own = feature_columns(rows, calendar)
    lanes = {}
    for window in (20, 25, 30):
        for s in (3, 4, 5):
            lanes["v12r2_wls%d_s%d" % (window, s)] = r2_lane(rows, calendar, window, s)
    out = DATA / "513130_columns.npz"
    np.savez_compressed(
        out, features=block,
        wls25_idx=scores["wls25_v20"][0], wls25_val=scores["wls25_v20"][1],
        wls20s3_idx=scores["wls20_smooth3"][0], wls20s3_val=scores["wls20_smooth3"][1],
        **{"lane_" + k: v for k, v in lanes.items()})
    print("SAVED", out, "quoted rows in calendar:", len(own), "of", len(rows), flush=True)


if __name__ == "__main__":
    main()
