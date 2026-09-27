"""Register 36 fixed portfolio controls, freeze S on 2014-25, then report 2026.

Run with ``python3.8 -B -m v10_deep.portfolio_scan``. This runner is deliberately
single-process. It reads an already validated feature cache without rebuilding
it, never reads open prices, and never resimulates the archived v9 controls.
"""
import bisect
from copy import deepcopy
from datetime import datetime, timezone
import gc
import gzip
import io
import json
from pathlib import Path
import time

import numpy as np

from .data import BASE, ROOT, START, END, TRAIN_END, dump, protect, sha
from .portfolios import build_topk_policy, prices_from_features, registry, run_weights


OUT = BASE / "results/portfolios"
CONTROL_DIR = BASE / "results/mechanisms"
FEES = {"primary": .0001, "fee5": .0005}
SOURCE_FILES = ("portfolio_scan.py", "portfolios.py", "features.py", "data.py")
PERIODS = {"selection": (START, TRAIN_END), "early_2014_2017": (START, "2017-12-31"),
           "middle_2018_2021": ("2018-01-01", "2021-12-31"),
           "recent_2022_2025": ("2022-01-01", TRAIN_END),
           "report_only_2026": ("2026-01-01", END), "full": (START, END)}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def metrics(returns):
    """Existing study convention: 244 sessions/year; include boundary-day P&L."""
    r = np.asarray(returns, dtype=np.float64)
    if not len(r) or not np.isfinite(r).all() or np.any(r <= -1):
        raise ValueError("Expected a nonempty finite positive-NAV return path")
    nav = np.cumprod(1 + r)
    factor, std = float(nav[-1]), float(r.std())
    ann = float(np.expm1(np.log1p(r).sum() * 244 / len(r)))
    dd = float(np.min(nav / np.maximum.accumulate(np.r_[1., nav])[1:] - 1))
    return dict(sessions=len(r), nav=factor, total_return=factor - 1, cagr=ann,
                max_dd=dd, volatility=std * 244 ** .5,
                sharpe=float(r.mean() / std * 244 ** .5) if std else 0.,
                calmar=ann / abs(dd) if dd else 0.)


def summarize(returns, dates):
    out = {}
    for name, (start, end) in PERIODS.items():
        lo, hi = bisect.bisect_left(dates, start), bisect.bisect_right(dates, end)
        if hi > lo:
            out[name] = metrics(returns[lo:hi])
    out["yearly"] = {}
    for year in sorted({date[:4] for date in dates}):
        lo = bisect.bisect_left(dates, year + "-01-01")
        hi = bisect.bisect_right(dates, year + "-12-31")
        out["yearly"][year] = metrics(returns[lo:hi])["total_return"]
    return out


def load_features_readonly():
    """Read two required score families; refuse to repair or overwrite a cache."""
    meta = json.loads((BASE / "cache/features.json").read_text())
    paths = {"source": BASE / "features.py", "data": ROOT / "v10_h_close/corrected_manifest.json",
             "legacy_features": ROOT / "v10_search/features.py",
             "legacy_strategy": ROOT / "v10_next/frozen/strategy.py"}
    for key, expected in meta["fingerprints"].items():
        path = paths.get(key, ROOT / key)
        if not path.is_file() or sha(path) != expected:
            raise AssertionError("Feature input/source fingerprint changed: " + key)
    needed = ("wls25_v20", "blend_20_40_60")
    indices = [meta["score_names"].index(name) for name in needed]
    with np.load(BASE / "cache/features.npz", allow_pickle=False) as loaded:
        arrays = dict(features=loaded["features"], scores=loaded["scores"][indices])
    meta = deepcopy(meta)
    meta["score_names"] = list(needed)
    return arrays, meta


def fingerprint(meta):
    manifest = json.loads((ROOT / "v10_h_close/corrected_manifest.json").read_text())
    inputs = {}
    for code, record in manifest["assets"].items():
        path = ROOT / "v10_h_close/corrected_snapshots" / (code + ".csv")
        inputs[code] = sha(path)
        if inputs[code] != record["sha256"]:
            raise AssertionError("Corrected snapshot changed: " + code)
    return dict(sources={name: sha(BASE / name) for name in SOURCE_FILES},
                features=meta["fingerprints"], corrected_snapshot_sha256=inputs,
                corrected_data_sha256=sha(ROOT / "v10_h_close/corrected_manifest.json"),
                feature_cache_sha256=sha(BASE / "cache/features.npz"),
                feature_metadata_sha256=sha(BASE / "cache/features.json"),
                protected_manifest_sha256=sha(BASE / "protected_manifest.json"))


def archive_sources():
    return {name: sha(CONTROL_DIR / name) for name in
            ("selection.json", "development.json", "evaluation.json", "registration.json")}


def archived_controls(phase, fp):
    """Read the three already frozen controls, without invoking any strategy."""
    registration = json.loads((CONTROL_DIR / "registration.json").read_text())
    for key in ("corrected_data_sha256", "feature_cache_sha256", "features"):
        if registration["fingerprints"][key] != fp[key]:
            raise AssertionError("Archived controls use different inputs: " + key)
    if (registration["start"], registration["train_end"], registration["end"],
        registration["fee"], registration["stress_fee"]) != (START, TRAIN_END, END, .0001, .0005):
        raise AssertionError("Archived controls use a different period or cost convention")
    selection = json.loads((CONTROL_DIR / "selection.json").read_text())
    ids = selection["controls"]
    artifact = "development.json" if phase == "selection" else "evaluation.json"
    result = json.loads((CONTROL_DIR / artifact).read_text())
    rows = {row["id"]: row for row in result["rows"] if row["id"] in ids.values()}
    if len(rows) != 3:
        raise AssertionError("Missing archived v9/v9.1/v9.2 control")
    return {version: deepcopy(rows[identifier]) for version, identifier in ids.items()}


def choose(rows, max_drawdown):
    """All comparisons use development rows only; no full-period input accepted."""
    checks, eligible = {}, []
    for row in rows:
        train, fee5 = row["metrics"]["selection"], row["fee5"]["selection"]
        flags = dict(multiple_members=row["topk"] >= 2, cagr_30=train["cagr"] >= .30,
                     drawdown_at_most_v92=abs(train["max_dd"]) <= max_drawdown,
                     fee5_cagr_25=fee5["cagr"] >= .25,
                     recent_cagr_25=row["metrics"]["recent_2022_2025"]["cagr"] >= .25)
        checks[row["id"]] = flags
        if all(flags.values()):
            eligible.append(row)
    eligible.sort(key=lambda row: (abs(row["metrics"]["selection"]["max_dd"]),
                                   -row["metrics"]["selection"]["cagr"],
                                   row["selection_switches"], row["id"]))
    champions = sorted(rows, key=lambda row: (-row["metrics"]["selection"]["cagr"], row["id"]))
    return dict(selected_s=eligible[0]["id"] if eligible else None,
                primary=champions[0]["id"], primary_role="portfolio return champion; does not replace single-ETF H",
                qualified_count=len(eligible), qualified_ids=[row["id"] for row in eligible],
                checks=checks, selected_at=stamp(), selection_period=[START, TRAIN_END],
                clean_oos=False, s_drawdown_limit=max_drawdown,
                tie_break=["minimum abs selection drawdown", "maximum selection CAGR",
                           "fewer selection rebalances after first evaluation day", "ascending fixed ID"])


def run_one(config, arrays, meta, prices, end, fee):
    policy = build_topk_policy(config, arrays, meta)
    result = run_weights(prices, meta["dates"], policy, START, end, fee=fee)
    r = np.asarray(result["returns"], dtype=np.float64)
    navs = np.asarray([row[1] for row in result["daily"]], dtype=np.float64)
    if r[0] != 0 or not np.allclose(navs, np.cumprod(1 + r), rtol=1e-12, atol=1e-12):
        raise AssertionError("NAV/return accounting or initial free-day convention failed")
    for day in result["daily"]:
        if any(w < 0 for w in day[2].values()) or sum(day[2].values()) > 1 + 1e-10:
            raise AssertionError("Invalid realized weights")
    for trade in result["trades"]:
        cost = 0. if trade["initial_session_free"] else trade["nav_before"] * fee * trade["turnover"]
        if not np.isclose(trade["model_fee"], cost, rtol=1e-10, atol=1e-12):
            raise AssertionError("Charged fee does not match realized weight turnover")
    return result


def summary_row(config, index, primary, stress, dates):
    return dict(id=config["id"], hash=config["hash"], index=index, topk=config["topk"],
                score=config["score"], pool=config["pool"], regime=config["regime"],
                metrics=summarize(primary["returns"], dates), fee5=summarize(stress["returns"], dates),
                switches=primary["switches"], fee5_switches=stress["switches"],
                selection_switches=sum(not t["initial_session_free"] and t["date"] <= TRAIN_END
                                       for t in primary["trades"]),
                blocked=primary["diagnostics"]["blocked_rebalance_count"],
                missing_held=len(primary["diagnostics"]["missing_held_bars"]),
                total_turnover=primary["diagnostics"]["total_turnover"],
                fee5_total_turnover=stress["diagnostics"]["total_turnover"])


def dump_gzip(path, obj):
    """Reproducible lossless daily paths, including actual weights and fills."""
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as output:
                json.dump(obj, output, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                output.write("\n")


def report(evaluation, controls):
    rows = {row["id"]: row for row in evaluation["rows"]}
    chosen = evaluation["selection"]
    lines = ["# Fixed portfolio diversification experiment", "",
             "36 configurations were registered before their simulations. The S choice uses 2014–2025 only; "
             "the already known 2026 tail is report-only and is not a clean holdout.", "",
             "S: `%s`; eligible candidates: %d. Portfolio return champion: `%s`. "
             "This champion does not replace the single-ETF H choice." %
             (chosen["selected_s"], chosen["qualified_count"], chosen["primary"]), "",
             "Close-only corrected TR indices; immediate free ex-date dividend reinvestment. "
             "Open columns are placeholders and are never executed. First evaluation day is free; "
             "thereafter cost is pre-rebalance NAV × fee × L1 security-weight turnover. "
             "Both 1 bp and 5 bp paths are freshly simulated. Units drift until membership changes. "
             "A missing necessary quote discards the entire rebalance without pending orders.", "",
             "Top1 here is the no-crash, no momentum-gap-buffer control for this portfolio family; it is not v9.2. "
             "All k values can fall back to one asset. Selection requires configured k ≥ 2, not two actual members on every date.", "",
             "| Candidate | Train CAGR | Train DD | Full CAGR | Full DD | Full 5 bp CAGR | 2022–25 CAGR |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    named = [(name, row) for name, row in controls.items()]
    selected_ids = list(dict.fromkeys([chosen["selected_s"], chosen["primary"]]))
    named += [(identifier, rows[identifier]) for identifier in selected_ids if identifier]
    for name, row in named:
        m, f = row["metrics"], row["fee5"]
        lines.append("| %s | %.4f%% | %.4f%% | %.4f%% | %.4f%% | %.4f%% | %.4f%% |" %
                     (name, 100*m["selection"]["cagr"], 100*m["selection"]["max_dd"],
                      100*m["full"]["cagr"], 100*m["full"]["max_dd"],
                      100*f["full"]["cagr"], 100*m["recent_2022_2025"]["cagr"]))
    lines += ["", "The 36-row development/evaluation files retain every candidate, yearly returns, and period statistics. "
              "Each path archive contains both fee runs with daily NAV/weights, returns, signals, actual model fills, "
              "and missing-price diagnostics. Full runs preserve both training return prefixes exactly. "
              "The three old controls are copied from the prior frozen mechanisms artifacts, not resimulated.", "",
              "This is adaptive research on previously inspected history. Registration and a frozen selection "
              "prevent changing this 36-member set after its results; they do not erase prior selection bias "
              "or prove expected live performance.", ""]
    return "\n".join(lines)


def run():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Portfolio output already exists; do not overwrite frozen registrations/results")
    before = protect()
    arrays, meta = load_features_readonly()
    fp, archive_hashes = fingerprint(meta), archive_sources()
    controls = archived_controls("selection", fp)
    drawdown_limit = abs(controls["v9.2"]["metrics"]["selection"]["max_dd"])
    if abs(drawdown_limit - .24164829) > 1e-8:
        raise AssertionError("Archived v9.2 training drawdown differs from the specified S limit")
    configs = registry()
    if configs["count"] != 36 or len({c["hash"] for c in configs["candidates"]}) != 36:
        raise AssertionError("The fixed 36-config registry changed")
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "registered_candidates.json", configs)
    registration = dict(registered_at=stamp(), candidate_count=36, start=START, train_end=TRAIN_END, end=END,
                        fees=FEES, fingerprints=fp, controls_source_directory=str(CONTROL_DIR.relative_to(ROOT)),
                        controls_source_sha256=archive_hashes,
                        registry_sha256=sha(OUT / "registered_candidates.json"), clean_oos=False,
                        selection_rule=dict(topk_at_least=2, selection_cagr_at_least=.30,
                            selection_abs_dd_at_most=drawdown_limit, selection_fee5_cagr_at_least=.25,
                            recent_2022_2025_cagr_at_least=.25,
                            sort=["abs selection DD ascending", "selection CAGR descending",
                                  "charged selection rebalances ascending", "fixed ID ascending"], no_eligible=None),
                        champion_rule="max selection primary CAGR across all 36; fixed ID breaks ties; independent of single H",
                        execution_clock="original same-close TR index units, no pending orders",
                        fee_formula="preNAV * fee * sum(abs(target security weight - marked security weight)); first day free",
                        allowed_price_column="close", open_is_placeholder_and_forbidden=True,
                        no_crash_entries=True, no_legacy_momentum_gap_buffer=True,
                        full_period_access="only after selection.json has been written",
                        report_only_tail="2026 was already inspected historically; not clean out of sample",
                        initial_protected_file_count=before)
    dump(OUT / "registration.json", registration)
    print("REGISTERED 36", sha(OUT / "registration.json"), flush=True)
    prices = prices_from_features(arrays, meta)
    train_dates = [date for date in meta["dates"] if START <= date <= TRAIN_END]
    train = {label: np.empty((36, len(train_dates)), dtype=np.float64) for label in FEES}
    rows, started = [], time.monotonic()
    for index, config in enumerate(configs["candidates"]):
        results = {label: run_one(config, arrays, meta, prices, TRAIN_END, fee) for label, fee in FEES.items()}
        for label in FEES:
            train[label][index] = results[label]["returns"]
        rows.append(summary_row(config, index, results["primary"], results["fee5"], train_dates))
        print("portfolio selection %d/36 %.1fs" % (index + 1, time.monotonic() - started), flush=True)
    del results
    for label, values in train.items():
        name = "selection_returns.npy" if label == "primary" else "selection_fee5_returns.npy"
        np.save(OUT / name, values, allow_pickle=False)
    dump(OUT / "development.json", dict(rows=rows, dates=train_dates, controls=controls,
                                        all_fee_paths_freshly_simulated=True))
    chosen = choose(rows, drawdown_limit)
    chosen.update(registration_sha256=sha(OUT / "registration.json"),
                  development_sha256=sha(OUT / "development.json"),
                  selection_returns_sha256=sha(OUT / "selection_returns.npy"),
                  selection_fee5_sha256=sha(OUT / "selection_fee5_returns.npy"))
    if fingerprint(meta) != fp or archive_sources() != archive_hashes:
        raise AssertionError("Inputs or source artifacts changed before selection freeze")
    dump(OUT / "selection.json", chosen)
    selection_hash = sha(OUT / "selection.json")
    print("FROZEN S", chosen["selected_s"], "qualified", chosen["qualified_count"],
          "portfolio primary", chosen["primary"], flush=True)
    # No full-period portfolio result is generated until the above immutable file exists.
    full_dates = [date for date in meta["dates"] if START <= date <= END]
    full = {label: np.empty((36, len(full_dates)), dtype=np.float64) for label in FEES}
    rows, paths = [], {}
    (OUT / "paths").mkdir()
    for index, config in enumerate(configs["candidates"]):
        results = {label: run_one(config, arrays, meta, prices, END, fee) for label, fee in FEES.items()}
        for label in FEES:
            full[label][index] = results[label]["returns"]
            if not np.array_equal(full[label][index, :len(train_dates)], train[label][index]):
                raise AssertionError("Full-period run changed a training prefix: " + config["id"] + " " + label)
        rows.append(summary_row(config, index, results["primary"], results["fee5"], full_dates))
        path = OUT / "paths" / (config["id"] + ".json.gz")
        dump_gzip(path, dict(id=config["id"], config_hash=config["hash"], fees=FEES,
                            selection_sha256=selection_hash, primary=results["primary"], fee5=results["fee5"]))
        paths[config["id"]] = dict(path=str(path.relative_to(OUT)), sha256=sha(path))
        print("portfolio full %d/36 %.1fs" % (index + 1, time.monotonic() - started), flush=True)
    del results
    gc.collect()
    for label, values in full.items():
        name = "full_returns.npy" if label == "primary" else "full_fee5_returns.npy"
        np.save(OUT / name, values, allow_pickle=False)
    controls = archived_controls("full", fp)
    if fingerprint(meta) != fp or archive_sources() != archive_hashes or sha(OUT / "selection.json") != selection_hash:
        raise AssertionError("Source inputs or frozen selection changed during evaluation")
    evaluation = dict(rows=rows, dates=full_dates, controls=controls, selection=chosen,
                      selection_sha256=selection_hash, registration_sha256=chosen["registration_sha256"],
                      fingerprints=fp, controls_source_sha256=archive_hashes, paths=paths,
                      full_returns_sha256=sha(OUT / "full_returns.npy"),
                      full_fee5_sha256=sha(OUT / "full_fee5_returns.npy"),
                      all_72_fee_paths_freshly_simulated=True, all_72_training_prefixes_exact=True,
                      candidate_count=36, protected_files_verified=protect(), completed_at=stamp(), clean_oos=False)
    dump(OUT / "evaluation.json", evaluation)
    (OUT / "REPORT.md").write_text(report(evaluation, controls))
    print("COMPLETE", OUT, "selection", selection_hash, flush=True)
    lookup = {row["id"]: row for row in rows}
    for identifier in dict.fromkeys((chosen["selected_s"], chosen["primary"])):
        if identifier:
            print("RESULT", identifier, json.dumps(lookup[identifier], sort_keys=True), flush=True)
    return evaluation


if __name__ == "__main__":
    run()
