"""Dynamic-pool experiment on the frozen V12-R2 matrix (read-only).

Question: instead of a fixed pool, admit off-pool assets dynamically, because
different ETFs lead in different periods. The 13 off-pool assets already have
frozen features and R2 score lanes, so no new data is needed.

Variants (off-pool assets get global-pool semantics when admitted = most
favourable treatment: exempt from the bear gate, 3% buffer):
- dyn_top1:    each day admit the single highest-scored off-pool asset.
- dyn_monthly: on the first session of each month admit the two off-pool
               assets with the strongest mom20; membership lasts the month.

The frozen engine loop is copied verbatim from v12.reference.run_reference;
only the Policy class is swapped. Frozen files are never modified.
"""
import json
import math
import sys
from bisect import bisect_left, bisect_right
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v12_r2.data import START, END, verify_inputs, load_base, read, sha, dump  # noqa: E402
from v12_r2.features import view as r2_view, effective_config  # noqa: E402
from v12.reference import Policy as FrozenPolicy  # noqa: E402
from v10_deep.reference import Portfolio, DatedValues, CASH, GOLD  # noqa: E402
from v10_deep.scan import metrics  # noqa: E402

BASE = Path(__file__).resolve().parent
R2_ID = "v12r2_08b522c2e1aa291f054d"
FIXED_POOL = {"159915", "510300", "510500", "512400", "512890", "563300", "588080",
              "513100", "513120", "518880", "511880"}


def extend_r2_lane(arrays, meta, histories):
    """Compute v12r2_wls25_s4 for the 13 off-pool assets (frozen build masked them
    to INVALID because RISK_ASSETS covers only the 10 tradeable risk assets).

    Self-check: the same arithmetic must reproduce the frozen lane bit-for-bit
    on the 10 in-pool assets before any off-pool value is trusted.
    """
    from v12_pool_test.prepare_513130 import r2_lane
    si = meta["score_names"].index("v12r2_wls25_s4")
    frozen = arrays["scores"][si]
    calendar = list(meta["dates"])
    tradeable = [c for c in meta["assets"] if c in FIXED_POOL and c != "511880"]
    out = np.array(frozen, copy=True)
    for code in tradeable:
        ai = meta["assets"].index(code)
        mine = r2_lane(histories[code], calendar, 25, 4)
        if not np.array_equal(mine, frozen[:, ai]):
            diff = np.flatnonzero(mine != frozen[:, ai])
            raise SystemExit("R2 lane rebuild mismatch on %s at %s" %
                             (code, [calendar[i] for i in diff[:3]]))
    print("R2 lane rebuild: bit-exact on all 10 in-pool assets", flush=True)
    added = []
    for ai, code in enumerate(meta["assets"]):
        if code in FIXED_POOL:
            continue
        out[:, ai] = r2_lane(histories[code], calendar, 25, 4)
        added.append(code)
    scores = np.array(arrays["scores"], copy=True)
    scores[si] = out
    orders = np.array(arrays["orders"], copy=True)
    orders[si] = np.argsort(-scores[si], axis=1, kind="stable").astype(np.int32)
    view_arrays = dict(arrays)
    view_arrays["scores"] = np.ascontiguousarray(scores)
    view_arrays["orders"] = np.ascontiguousarray(orders)
    return view_arrays, added


class DynamicPoolPolicy(FrozenPolicy):
    """Frozen policy plus a daily/monthly off-pool admission list (global semantics)."""

    def __init__(self, data, config, off_pool, dates, mode):
        self.off_pool = set(off_pool)
        self.dates = list(dates)
        self.mode = mode
        self._month, self._monthly = None, set()
        super().__init__(data, config)

    def _admitted(self, signal_day):
        d = self.data
        candidates = [c for c in self.off_pool if d.informed(signal_day, c)]
        if self.mode == "top1":
            if not candidates:
                return set()
            best = max(candidates, key=lambda c: d.score(signal_day, c))
            return {best} if d.score(signal_day, best) > 0 else set()
        # monthly: two strongest mom20, reselected on the month's first session
        month = self.dates[signal_day][:7]
        if month != self._month:
            ranked = [c for c in candidates
                      if math.isfinite(self.momentum(signal_day, c, 20))]
            ranked.sort(key=lambda c: -self.momentum(signal_day, c, 20))
            self._monthly = set(ranked[:2])
            self._month = month
        return set(self._monthly)

    def intent(self, execution_day, account, can_sell):
        signal_day = execution_day - self.config["lag"]
        admitted = self._admitted(signal_day) if signal_day >= 0 else set()
        if not admitted:
            return super().intent(execution_day, account, can_sell)
        saved = (self.globals, self.trade_pool, self.safe_pool)
        self.globals = self.globals | admitted
        self.trade_pool = self.stocks | self.globals | {GOLD}
        self.safe_pool = self.globals | {GOLD}
        try:
            return super().intent(execution_day, account, can_sell)
        finally:
            self.globals, self.trade_pool, self.safe_pool = saved


def run_dynamic(arrays, meta, config, mode, start=START, end=END, fee=.0001):
    """Verbatim copy of v12.reference.run_reference with a pluggable policy."""
    dates = meta["dates"]
    lo = bisect_left(dates, start)
    hi = bisect_right(dates, end) - 1
    data = DatedValues(arrays, meta, config)
    off_pool = [c for c in meta["assets"] if c not in FIXED_POOL]
    policy = DynamicPoolPolicy(data, config, off_pool, dates, mode)
    account = Portfolio()
    returns, holdings = [], []
    for day in range(lo, hi + 1):
        previous_nav = account.nav
        current_price = data.price(day, account.holding)
        can_sell = account.holding is None or math.isfinite(current_price)
        if not can_sell:
            account.missing += 1
        if day > lo:
            if account.holding is not None and can_sell:
                account.nav *= current_price / account.mark
                account.mark = current_price
            account.age += 1
        target, panic, crash = policy.intent(day, account, can_sell)
        if target is not None and target != account.holding:
            purchase_price = data.price(day, target)
            if not can_sell or not math.isfinite(purchase_price):
                account.blocked += 1
            else:
                if panic and account.holding is not None and config["panic_cooldown"] > 0:
                    policy.cooldown_until[account.holding] = day + config["panic_cooldown"] + 1
                if day > lo:
                    account.nav *= 1 - 2 * fee
                    account.switches += 1
                account.entries += 1
                account.holding, account.mark = target, purchase_price
                account.age, account.asset_peak = 0, purchase_price
                policy.rotation.clear()
                if crash:
                    account.lock_until = day + config["crash_lock"]
                    account.crashes += 1
                policy.after_fill(account)
        if not math.isfinite(account.nav) or account.nav <= 0:
            raise ArithmeticError("Invalid NAV")
        account.nav_peak = max(account.nav_peak, account.nav)
        account.max_dd = min(account.max_dd, account.nav / account.nav_peak - 1)
        returns.append(account.nav / previous_nav - 1)
        holdings.append(data.asset_index[account.holding] if account.holding is not None else -1)
    return dict(returns=np.asarray(returns), holdings=np.asarray(holdings, dtype=np.int32),
                dates=dates[lo:hi + 1],
                summary=(account.nav, account.switches, account.crashes))


def yearly(result):
    returns, dates = result["returns"], result["dates"]
    return {y: float(np.expm1(np.log1p(returns[np.asarray([d[:4] == y for d in dates])]).sum()))
            for y in sorted({d[:4] for d in dates})}


def off_pool_usage(result, meta):
    out = {}
    for ai, code in enumerate(meta["assets"]):
        if code in FIXED_POOL:
            continue
        days = int(np.sum(result["holdings"] == ai))
        if days:
            out[code] = days
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main():
    verify_inputs()
    arrays, meta, histories = load_base()
    receipt = read("v12_r2/results/feature_receipt.json")
    if sha(ROOT / "v12_r2/results/features.npz") != receipt["array_sha256"]:
        raise ValueError("Frozen R2 feature matrix changed")
    expanded = dict(np.load(ROOT / "v12_r2/results/features.npz", allow_pickle=False))
    meta_r2 = json.loads((ROOT / "v12_r2/results/feature_metadata.json").read_text())
    registry = read("v12_r2/registered_candidates.json")
    record = next(r for r in registry["records"] if r["id"] == R2_ID)
    expanded, added = extend_r2_lane(expanded, meta_r2, histories)
    print("off-pool lanes computed for:", ",".join(added), flush=True)
    view = r2_view(expanded, meta_r2, record)

    rows, usage = {}, {}
    for lag, fee in ((0, .0001), (0, .0011), (1, .0001), (1, .0011)):
        name = ("lag1" if lag else "close") + "_%dbp" % (fee * 10000)
        for label, mode in (("base", None), ("dyn_top1", "top1"), ("dyn_monthly", "monthly")):
            config = effective_config(record, lag=lag)
            if mode is None:
                from v12.reference import run_reference
                r = run_reference(view, meta_r2, config, start=START, end=END, fee=fee)
                result = dict(returns=r["returns"], holdings=r["holdings"], dates=r["dates"],
                              summary=(float(r["summary"][0]), int(r["summary"][3]), int(r["summary"][7])))
            else:
                result = run_dynamic(view, meta_r2, config, mode, fee=fee)
            m = metrics(result["returns"])
            rows.setdefault(name, {})[label] = dict(
                total_return=m["total_return"], cagr=m["cagr"], max_dd=m["max_dd"],
                switches=result["summary"][1], crashes=result["summary"][2], yearly=yearly(result))
            if mode and name == "close_1bp":
                usage[label] = off_pool_usage(result, meta_r2)
    dump(BASE / "results/dynamic_pool.json", dict(candidate=R2_ID, scenarios=rows, usage=usage))

    print("\n=== dynamic pool on frozen 24-asset matrix (13 off-pool candidates) ===")
    for name, variants in rows.items():
        for label in ("dyn_top1", "dyn_monthly"):
            b, h = variants["base"], variants[label]
            print("[%s] %-11s cagr %6.2f%% -> %6.2f%%   total %9.1f%% -> %9.1f%%   maxDD %.2f%% -> %.2f%%   switches %d -> %d" % (
                name, label, 100 * b["cagr"], 100 * h["cagr"], 100 * b["total_return"], 100 * h["total_return"],
                100 * b["max_dd"], 100 * h["max_dd"], b["switches"], h["switches"]))
        print()
    print("off-pool holdings days (close_1bp):", json.dumps(usage, ensure_ascii=False))
    print("\nYearly (close_1bp):")
    yb = rows["close_1bp"]["base"]["yearly"]
    for y in sorted(yb):
        line = "  %s  %8.2f%%" % (y, 100 * yb[y])
        for label in ("dyn_top1", "dyn_monthly"):
            yv = rows["close_1bp"][label]["yearly"][y]
            line += " | %s %8.2f%% (%+6.2fpp)" % (label, 100 * yv, 100 * (yv - yb[y]))
        print(line)


if __name__ == "__main__":
    main()
