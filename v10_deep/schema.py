"""Explicit, bounded policy configuration; labels do not create independent trials."""
from copy import deepcopy
import hashlib
import json
from .data import ASSET_ORDER, CASH, GOLD, BENCHMARK
from .features import F

PARAMETERS = (
    "score", "mom", "exit_mom", "fast_mom", "ma", "regime", "regime_hyst", "regime_confirm",
    "buffer_mode", "buffer", "global_buffer", "gold_buffer", "rank_keep", "min_hold", "switch_confirm",
    "bull_entry", "bear_entry", "exit_floor", "panic_mode", "panic", "overheat", "fallback_mode", "fallback_floor",
    "crash_mask", "crash_lock", "deep_mom", "deep_below", "relaxed_mom", "volume_below", "volume_ratio",
    "crash_guard", "crash_pick", "crash_stock_only", "trail_stop", "lag", "stock_mask", "global_mask",
    "breadth_threshold", "trend_buffer", "panic_cooldown"
)
P = {n: i for i, n in enumerate(PARAMETERS)}
STOCK = ("159915","588080","510300","510500","563300","512400","512890")
GLOBAL = ("513100","513120")


def baseline():
    return dict(score="wls25_v20", mom=20, exit_mom=20, fast_mom=5, ma="ma250", regime="ma",
                regime_hyst=0., regime_confirm=1, buffer_mode="momentum", buffer=.02, global_buffer=.03, gold_buffer=.03,
                rank_keep=1, min_hold=0, switch_confirm=1, bull_entry=0., bear_entry=.07, exit_floor=0.,
                panic_mode="fixed", panic=.04, overheat=.4, fallback_mode="rank", fallback_floor=-9.9,
                crash_mask=7, crash_lock=5, deep_mom=-.08, deep_below=.20, relaxed_mom=-.04,
                volume_below=.10, volume_ratio=2., crash_guard="always", crash_pick="score", crash_stock_only=0,
                trail_stop=0., lag=0, stock_pool=list(STOCK), global_pool=list(GLOBAL), breadth_threshold=.5,
                trend_buffer=0., panic_cooldown=0)


ENUMS = dict(regime={"ma":0,"open_stock":1,"always_bull":2,"all_assets":3,"breadth":4,"dual":5},
             buffer_mode={"momentum":0,"score_gap":1,"score_relative":2,"rank":3},
             panic_mode={"fixed":0,"volatility":1}, fallback_mode={"rank":0,"cash":1,"low_vol":2},
             crash_guard={"always":0,"held_negative":1,"candidate_better":2,"held_shock":3,"cash_only":4},
             crash_pick={"score":0,"deepest":1,"least_volatile":2})


def semantic(c):
    return {k: deepcopy(v) for k,v in c.items() if k not in ("id","hash","families","parents","stage")}


def identifier(c):
    return hashlib.sha256(json.dumps(semantic(c),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def encode(c, meta):
    if c['lag'] not in (0,1):raise ValueError('Only same-close or one-close delay is registered')
    if len(set(c['stock_pool']))!=len(c['stock_pool']) or len(set(c['global_pool']))!=len(c['global_pool']):
        raise ValueError('Duplicate pool members')
    if set(c['stock_pool']) & set(c['global_pool']) or set(c['stock_pool']+c['global_pool']) & {CASH,GOLD}:
        raise ValueError('Asset roles must be disjoint; cash/gold roles are fixed')
    result = []
    for name in PARAMETERS:
        if name == "score": value=meta["score_names"].index(c[name])
        elif name in ("mom","exit_mom","fast_mom"): value=F["mom%d" % c[name]]
        elif name == "ma": value=F[c[name]]
        elif name.endswith("_mask") and name != "crash_mask":
            value=sum(1 << ASSET_ORDER.index(code) for code in c[name.replace("_mask","_pool")])
        elif name in ENUMS: value=ENUMS[name][c[name]]
        else:value=c[name]
        result.append(float(value))
    return result


def native_header():
    lines=["enum Feature {"+", ".join("F_"+n.upper()+"="+str(i) for n,i in F.items())+"};",
           "enum Parameter {"+", ".join("P_"+n.upper()+"="+str(i) for n,i in P.items())+"};",
           "constexpr int NF=%d, NP=%d, CASH=%d, GOLD=%d, BENCH=%d;" % (len(F),len(P),ASSET_ORDER.index(CASH),ASSET_ORDER.index(GOLD),ASSET_ORDER.index(BENCHMARK))]
    return "\n".join(lines)+"\n"
