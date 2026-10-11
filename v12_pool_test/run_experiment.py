"""V12-R2 pool-extension experiment: add 513130 (Hang Seng Tech ETF) to the pool.

Baseline uses frozen 24-asset matrices bit-for-bit; the experiment appends one
513130 column built by prepare_513130.py (self-checked against frozen columns).
The deployed R2 candidate config is untouched except global_pool += 513130.

Nothing outside v12_pool_test/ is written. Frozen artifacts are hash-checked
before use; the frozen R2 paths are the baseline arbitrator.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v12_r2.data import BASE as R2_BASE, ROOT as R2_ROOT, START, END, SCENARIOS  # noqa: E402
from v12_r2.data import verify_inputs, load_base, read, sha, dump  # noqa: E402
from v12_r2.features import view as r2_view, effective_config  # noqa: E402
from v12.reference import run_reference  # noqa: E402
from v10_deep.scan import metrics  # noqa: E402

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
CODE = "513130"
R2_ID = "v12r2_08b522c2e1aa291f054d"  # deployed V12-R2 candidate


def load_frozen_r2():
    verify_inputs()
    arrays, meta, histories = load_base()
    receipt = read("v12_r2/results/feature_receipt.json")
    if sha(R2_ROOT / "v12_r2/results/features.npz") != receipt["array_sha256"]:
        raise ValueError("Frozen R2 feature matrix changed")
    if sha(R2_ROOT / "v12_r2/results/feature_metadata.json") != receipt["metadata_sha256"]:
        raise ValueError("Frozen R2 feature metadata changed")
    expanded = dict(np.load(R2_ROOT / "v12_r2/results/features.npz", allow_pickle=False))
    meta_r2 = json.loads((R2_ROOT / "v12_r2/results/feature_metadata.json").read_text())
    return arrays, meta, histories, expanded, meta_r2


def r2_record():
    registry = read("v12_r2/registered_candidates.json")
    return next(r for r in registry["records"] if r["id"] == R2_ID)


def load_513130_column(calendar, score_names):
    z = np.load(DATA / "513130_columns.npz")
    block = z["features"]
    close = block[:, 0]
    valid = block[:, 1]
    mask = (~np.isfinite(close)) | (valid == 0)
    by_name = {}
    w25 = np.full(len(calendar), -1e100)
    w25[z["wls25_idx"]] = z["wls25_val"]
    by_name["wls25_v20"] = w25
    w20 = np.full(len(calendar), -1e100)
    w20[z["wls20s3_idx"]] = z["wls20s3_val"]
    by_name["wls20_smooth3"] = w20
    # Original lanes follow the frozen investability mask; R2 lanes follow
    # make_view semantics (required observations + finite raw value only).
    for lane in (by_name["wls25_v20"], by_name["wls20_smooth3"]):
        lane[mask] = -1e100
        lane[~np.isfinite(lane)] = -1e100
    for window in (20, 25, 30):
        for s in (3, 4, 5):
            name = "v12r2_wls%d_s%d" % (window, s)
            lane = z["lane_" + name].copy()
            lane[~np.isfinite(lane)] = -1e100
            by_name[name] = lane
    missing = [n for n in score_names if n not in by_name]
    if missing:
        raise ValueError("No 513130 column for frozen lanes: " + ",".join(missing))
    return block, np.asarray([by_name[n] for n in score_names])


def extend(arrays24, meta24, block, lanes24):
    t, a = len(meta24["dates"]), len(meta24["assets"])
    if block.shape != (t, len(meta24["feature_names"])) or lanes24.shape != (len(meta24["score_names"]), t):
        raise ValueError("513130 column shape mismatch")
    features = np.concatenate([arrays24["features"], block[:, None, :]], axis=1)
    scores = np.concatenate([arrays24["scores"], lanes24[:, :, None]], axis=2)
    orders = np.argsort(-scores, axis=2, kind="stable").astype(np.int32)
    out = dict(features=np.ascontiguousarray(features), scores=np.ascontiguousarray(scores),
               orders=np.ascontiguousarray(orders), fear=arrays24["fear"], years=arrays24["years"])
    meta = deepcopy(meta24)
    meta["assets"] = list(meta24["assets"]) + [CODE]
    meta["pool_extension"] = dict(added=CODE, role="global_pool variant only",
                                  frozen_columns="bit-for-bit from v12_r2/results/features.npz")
    return out, meta


def run(arrays, meta, record, pool_add=(), lag=0, fee=0.0001, pool_field="global_pool"):
    config = effective_config(record, lag=lag)
    if pool_add:
        config = deepcopy(config)
        config[pool_field] = list(config[pool_field]) + list(pool_add)
    v = r2_view(arrays, meta, record)
    # r2_view masks validity by the active lane; reuse it for both baselines.
    result = run_reference(v, meta, config, start=START, end=END, fee=fee)
    return config, result


def yearly(result):
    returns, dates = result["returns"], result["dates"]
    return {y: float(np.expm1(np.log1p(returns[np.asarray([d[:4] == y for d in dates])]).sum()))
            for y in sorted({d[:4] for d in dates})}


def hst_usage(result, meta):
    ai = meta["assets"].index(CODE)
    days = np.flatnonzero(result["holdings"] == ai)
    spans = []
    if len(days):
        start = prev = days[0]
        for d in days[1:]:
            if d == prev + 1:
                prev = d
                continue
            spans.append((start, prev))
            start = prev = d
        spans.append((start, prev))
    out = []
    for start, prev in spans:
        seg = result["returns"][start:prev + 1]
        out.append(dict(entry=result["dates"][start], exit=result["dates"][prev],
                        days=prev - start + 1,
                        return_pct=float(100 * (np.prod(1 + seg) - 1))))
    return dict(days=int(len(days)), spans=out)


def main():
    arrays, meta, histories, expanded, meta_r2 = load_frozen_r2()
    record = r2_record()
    calendar = list(meta_r2["dates"])

    # 1) Baseline fidelity: pure-Python oracle on frozen matrices must reproduce
    #    the frozen close_1bp path of the deployed candidate bit-for-bit.
    pm = json.loads((R2_ROOT / "v12_r2/results/path_metadata.json").read_text())
    row = pm["ids"].index(R2_ID)
    with np.load(R2_ROOT / "v12_r2/results/paths.npz", allow_pickle=False) as z:
        frozen_returns = z["close_1bp__returns"][row]
        frozen_holdings = z["close_1bp__holdings"][row]
    config, base = run(expanded, meta_r2, record)
    same_returns = bool(np.array_equal(base["returns"], frozen_returns))
    same_holdings = bool(np.array_equal(base["holdings"], frozen_holdings))
    print("BASELINE close_1bp exact returns:", same_returns, "holdings:", same_holdings, flush=True)
    if not (same_returns and same_holdings):
        diff = np.flatnonzero(base["returns"] != frozen_returns)
        print("  first diff days:", [base["dates"][i] for i in diff[:5]])
        raise SystemExit("Oracle does not reproduce the frozen path; aborting experiment")

    # 2) Extended 25-asset matrix: frozen columns + one 513130 column.
    block, lanes24 = load_513130_column(calendar, list(meta_r2["score_names"]))
    arrays25, meta25 = extend(expanded, meta_r2, block, lanes24)

    # Sanity: on the frozen pool the extended matrix must reproduce the same path.
    config_b, base25 = run(arrays25, meta25, record)
    print("EXTENDED-MATRIX same-pool exact:", bool(np.array_equal(base25["returns"], frozen_returns)),
          flush=True)

    # 3) Scenarios x variants.
    rows = {}
    traces = {}
    for lag, fee in [(s[1], s[2]) for s in (("close_1bp", 0, .0001), ("close_11bp", 0, .0011),
                                           ("lag1_1bp", 1, .0001), ("lag1_11bp", 1, .0011))]:
        name = ("lag1" if lag else "close") + "_%dbp" % (fee * 10000)
        for label, pool, field in (("base", (), "global_pool"), ("base25", (), "global_pool"),
                                   ("hst_global", (CODE,), "global_pool"),
                                   ("hst_stock", (CODE,), "stock_pool")):
            arrs, mt = (expanded, meta_r2) if label == "base" else (arrays25, meta25)
            config, result = run(arrs, mt, record, pool_add=pool, lag=lag, fee=fee, pool_field=field)
            m = metrics(result["returns"])
            y = yearly(result)
            rows.setdefault(name, {})[label] = dict(
                total_return=m["total_return"], cagr=m["cagr"], max_dd=m["max_dd"],
                sharpe=m["sharpe"], switches=int(result["summary"][3]),
                crashes=int(result["summary"][7]), yearly=y)
            if label in ("hst_global", "hst_stock"):
                traces.setdefault(name, {})[label] = dict(usage=hst_usage(result, meta25))
    out = dict(candidate=R2_ID, config=config, period=[START, END],
               baseline_exact=same_returns, scenarios=rows, hst=traces)
    dump(BASE / "results/pool_513130.json", out)

    # 4) Console summary.
    print("\n=== V12-R2 pool test: +513130 Hang Seng Tech ===")
    for name, variants in rows.items():
        for label in ("hst_global", "hst_stock"):
            b, h = variants["base"], variants[label]
            print("[%s] %-11s cagr %6.2f%% -> %6.2f%%   total %9.1f%% -> %9.1f%%   maxDD %.2f%% -> %.2f%%   switches %d -> %d" % (
                name, label, 100 * b["cagr"], 100 * h["cagr"], 100 * b["total_return"], 100 * h["total_return"],
                100 * b["max_dd"], 100 * h["max_dd"], b["switches"], h["switches"]))
        print()
    usage = traces["close_1bp"]["hst_global"]["usage"]
    print("HST(global) usage close_1bp: %d days, %d spans" % (usage["days"], len(usage["spans"])))
    for s in usage["spans"]:
        print("  %s -> %s  %2dd  %+.2f%%" % (s["entry"], s["exit"], s["days"], s["return_pct"]))
    print("\nYearly (close_1bp):")
    yb = rows["close_1bp"]["base"]["yearly"]
    yg = rows["close_1bp"]["hst_global"]["yearly"]
    ys = rows["close_1bp"]["hst_stock"]["yearly"]
    for y in sorted(yb):
        print("  %s  %8.2f%% | global %8.2f%% (%+6.2fpp) | stock %8.2f%% (%+6.2fpp)" % (
            y, 100 * yb[y], 100 * yg[y], 100 * (yg[y] - yb[y]), 100 * ys[y], 100 * (ys[y] - yb[y])))


if __name__ == "__main__":
    main()
