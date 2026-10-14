"""Dynamic-pool experiment v2: in-pool membership dynamics (read-only).

The 2026-10-11 experiment (dynamic_pool.py) tested admitting OFF-pool assets
(boundary expansion) and both variants collapsed. This round tests the
complementary question: dynamically SHRINKING or re-shaping the fixed 11-asset
pool itself, using only frozen features (mom60/vol20 are already in the
matrix, so no new data and no lane extension are needed).

Variants (membership rules apply to the 9 risk assets = 7 stocks + 2 globals;
GOLD stays available as the fallback asset and CASH is never a pool member):
- top4/top6/top8 : on the month's first session keep only the K strongest
                   mom60 risk assets for that month.
- drop1/drop2    : each month ban the K weakest mom60 risk assets.
- hivol2         : each month ban the 2 highest-vol20 risk assets.
- neg_ban_daily  : daily, any risk asset with mom20 <= 0 is banned that day.
- daily_top6     : daily, keep only the 6 strongest mom60 risk assets.
- bear_gold_only : bear regime competition is reduced to GOLD (+CASH exits).
- rand2 (x10)    : placebo, each month bans 2 random risk assets (seeded).

The frozen engine loop is copied verbatim from v12.reference.run_reference;
only the Policy class is swapped. Frozen files are never modified.
"""
import json
import math
import random
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v12_r2.data import START, END, verify_inputs, load_base, read, sha, dump  # noqa: E402
from v12_r2.features import view as r2_view, effective_config  # noqa: E402
from v12.reference import Policy as FrozenPolicy, run_reference  # noqa: E402
from v10_deep.reference import Portfolio, DatedValues, CASH, GOLD  # noqa: E402
from v10_deep.scan import metrics  # noqa: E402

BASE = Path(__file__).resolve().parent
R2_ID = "v12r2_08b522c2e1aa291f054d"
RISK9 = ("159915", "510300", "510500", "512400", "512890", "563300", "588080",
         "513100", "513120")


class MembershipPolicy(FrozenPolicy):
    """Frozen policy plus a per-day banned set carved out of the risk pools."""

    def __init__(self, data, config, dates, mode, param, seed=0):
        self.dates = list(dates)
        self.mode, self.param, self.seed = mode, param, seed
        self._month, self._monthly_ban = None, set()
        self.ban_days = {}  # code -> days banned (diagnostics)
        super().__init__(data, config)

    def _rated(self, day, field):
        out = {}
        for code in RISK9:
            if self.data.informed(day, code):
                value = self.data.value(day, code, field)
                if math.isfinite(value):
                    out[code] = value
        return out

    def _monthly_banned(self, day):
        month = self.dates[day][:7]
        if month == self._month:
            return self._monthly_ban
        self._month = month
        mode, k = self.mode, self.param
        field = "mom60"
        if ":" in mode:  # robustness: alternative ranking window, e.g. top6:mom120
            mode, field = mode.split(":")
        if mode.startswith("top"):
            rated = self._rated(day, field)
            keep = {c for c, _ in sorted(rated.items(), key=lambda kv: -kv[1])[:k]}
            banned = {c for c in rated if c not in keep}
        elif mode.startswith("drop"):
            rated = self._rated(day, field)
            banned = {c for c, _ in sorted(rated.items(), key=lambda kv: kv[1])[:k]}
        elif mode == "hivol":
            rated = self._rated(day, "vol20")
            banned = {c for c, _ in sorted(rated.items(), key=lambda kv: -kv[1])[:k]}
        elif mode == "rand":
            rng = random.Random("%d:%s" % (self.seed, month))
            banned = set(rng.sample(sorted(RISK9), k))
        else:
            raise ValueError(mode)
        self._monthly_ban = banned
        return banned

    def _banned(self, day):
        if self.mode == "neg_daily":
            return {c for c in RISK9
                    if self.data.informed(day, c) and self.momentum(day, c, 20) <= 0}
        if self.mode == "daily_top":
            rated = self._rated(day, "mom60")
            keep = {c for c, _ in sorted(rated.items(), key=lambda kv: -kv[1])[:self.param]}
            return {c for c in rated if c not in keep}
        return self._monthly_banned(day)

    def intent(self, execution_day, account, can_sell):
        signal_day = execution_day - self.config["lag"]
        if self.mode == "bear_gold":
            # regime_confirm == 1 in the frozen R2 config, so the extra resolve
            # below is idempotent: the first call flips/keeps state, the second
            # (inside super().intent) only re-clears the confirmation counter.
            bull = signal_day >= 0 and self.regime.resolve(
                self.data, signal_day, self.config, self.stocks)
            banned = set() if bull else set(RISK9)
        else:
            banned = self._banned(signal_day) if signal_day >= 0 else set()
        if not banned:
            return super().intent(execution_day, account, can_sell)
        for code in banned:
            self.ban_days[code] = self.ban_days.get(code, 0) + 1
        saved = (self.stocks, self.globals, self.trade_pool, self.safe_pool)
        self.stocks = self.stocks - banned
        self.globals = self.globals - banned
        self.trade_pool = self.stocks | self.globals | {GOLD}
        self.safe_pool = self.globals | {GOLD}
        try:
            return super().intent(execution_day, account, can_sell)
        finally:
            self.stocks, self.globals, self.trade_pool, self.safe_pool = saved


def run_membership(arrays, meta, config, dates, mode, param, seed=0,
                   start=START, end=END, fee=.0001):
    """Verbatim copy of v12.reference.run_reference with a pluggable policy."""
    lo = bisect_left(dates, start)
    hi = bisect_right(dates, end) - 1
    data = DatedValues(arrays, meta, config)
    policy = MembershipPolicy(data, config, dates, mode, param, seed)
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
                dates=dates[lo:hi + 1], ban_days=policy.ban_days,
                summary=(account.nav, account.switches, account.crashes))


def yearly(result):
    returns, dates = result["returns"], result["dates"]
    return {y: float(np.expm1(np.log1p(returns[np.asarray([d[:4] == y for d in dates])]).sum()))
            for y in sorted({d[:4] for d in dates})}


VARIANTS = [
    ("top4", "top4", 4), ("top6", "top6", 6), ("top8", "top8", 8),
    ("drop1", "drop1", 1), ("drop2", "drop2", 2),
    ("hivol2", "hivol", 2),
    ("neg_ban_daily", "neg_daily", 0),
    ("daily_top6", "daily_top", 6),
    ("bear_gold_only", "bear_gold", 0),
]
SCENARIOS = ((0, .0001), (0, .0011), (1, .0001), (1, .0011))


def main():
    verify_inputs()
    arrays, meta, _histories = load_base()
    receipt = read("v12_r2/results/feature_receipt.json")
    if sha(ROOT / "v12_r2/results/features.npz") != receipt["array_sha256"]:
        raise ValueError("Frozen R2 feature matrix changed")
    expanded = dict(np.load(ROOT / "v12_r2/results/features.npz", allow_pickle=False))
    meta_r2 = json.loads((ROOT / "v12_r2/results/feature_metadata.json").read_text())
    registry = read("v12_r2/registered_candidates.json")
    record = next(r for r in registry["records"] if r["id"] == R2_ID)
    view = r2_view(expanded, meta_r2, record)
    dates = meta_r2["dates"]

    rows, bans = {}, {}
    for lag, fee in SCENARIOS:
        name = ("lag1" if lag else "close") + "_%dbp" % (fee * 10000)
        config = effective_config(record, lag=lag)
        base = run_reference(view, meta_r2, config, start=START, end=END, fee=fee)
        bm = metrics(base["returns"])
        rows.setdefault(name, {})["base"] = dict(
            total_return=bm["total_return"], cagr=bm["cagr"], max_dd=bm["max_dd"],
            switches=int(base["summary"][3]), crashes=int(base["summary"][7]),
            yearly=yearly(dict(returns=base["returns"], dates=base["dates"])))
        for label, mode, param in VARIANTS:
            r = run_membership(view, meta_r2, config, dates, mode, param, fee=fee)
            m = metrics(r["returns"])
            rows[name][label] = dict(
                total_return=m["total_return"], cagr=m["cagr"], max_dd=m["max_dd"],
                switches=r["summary"][1], crashes=r["summary"][2], yearly=yearly(r))
            if name == "close_1bp":
                bans[label] = dict(sorted(r["ban_days"].items(), key=lambda kv: -kv[1]))
        print("scenario %s done" % name, flush=True)

    placebos = []
    for seed in range(10):
        config = effective_config(record, lag=0)
        r = run_membership(view, meta_r2, config, dates, "rand", 2, seed=seed, fee=.0001)
        m = metrics(r["returns"])
        placebos.append(dict(seed=seed, cagr=m["cagr"], total_return=m["total_return"],
                             max_dd=m["max_dd"], switches=r["summary"][1]))
        print("placebo seed %d done" % seed, flush=True)

    dump(BASE / "results/dynamic_pool_v2.json",
         dict(candidate=R2_ID, scenarios=rows, ban_days=bans, placebo_rand2=placebos))

    print("\n=== dynamic pool v2: in-pool membership dynamics (frozen matrix) ===")
    for name, variants in rows.items():
        b = variants["base"]
        print("[%s] base cagr %6.2f%%  maxDD %.2f%%  switches %d" %
              (name, 100 * b["cagr"], 100 * b["max_dd"], b["switches"]))
        for label, _, _ in VARIANTS:
            h = variants[label]
            print("  %-14s cagr %6.2f%% (%+7.2fpp)  maxDD %7.2f%%  switches %4d  crashes %d" % (
                label, 100 * h["cagr"], 100 * (h["cagr"] - b["cagr"]),
                100 * h["max_dd"], h["switches"], h["crashes"]))
        print()
    pc = [p["cagr"] for p in placebos]
    print("placebo rand2 (close_1bp, 10 seeds): cagr min %6.2f%%  median %6.2f%%  max %6.2f%%  vs base %6.2f%%" % (
        100 * min(pc), 100 * sorted(pc)[len(pc) // 2], 100 * max(pc),
        100 * rows["close_1bp"]["base"]["cagr"]))
    print("\nYearly (close_1bp):")
    yb = rows["close_1bp"]["base"]["yearly"]
    header = "  year  %10s" % "base"
    for label, _, _ in VARIANTS:
        header += " %10s" % label[:10]
    print(header)
    for y in sorted(yb):
        line = "  %s  %9.2f%%" % (y, 100 * yb[y])
        for label, _, _ in VARIANTS:
            line += " %9.2f%%" % (100 * rows["close_1bp"][label]["yearly"][y])
        print(line)
    print("\nban days (close_1bp):")
    for label, _, _ in VARIANTS:
        if bans.get(label):
            print("  %-14s %s" % (label, json.dumps(bans[label])))


if __name__ == "__main__":
    main()
