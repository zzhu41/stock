"""Fixed cash-amount sensitivity views; never modify a frozen price vintage.

``build_scenario(name) -> (histories, metadata)`` reads and verifies the existing
v10_h_close vintage. ``scenario_histories`` is the pure in-memory equivalent.
Rows are (date, close_index, close_index, normalized_volume): open remains an
explicit placeholder and MUST NOT be used for open/intraday execution.

These scenarios change cash only at already identified event dates. They do not
bound missed events, execution costs, or strategy returns after signal changes.
"""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

from v10_h_close.total_return_audit import adjustment_scales


SCENARIOS = ("cash_lower", "cash_upper", "official_515100")
OFFICIAL_CODE = "515100"
OFFICIAL_DATE = "2025-12-23"
OFFICIAL_CASH = .1053
OFFICIAL_SOURCE = "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2025-12-18/515100_20251218_0CQV.pdf"
DEFAULT_SOURCE = Path(__file__).resolve().parent.parent / "v10_h_close"


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def _content_hash(value):
    body = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _rows(values, code, label):
    result = []
    for row in values:
        if len(row) != 4 or not isinstance(row[0], str):
            raise ValueError("Expected date/open/close/volume: %s %s" % (code, label))
        date, opening, close, volume = row[0], float(row[1]), float(row[2]), float(row[3])
        if (not all(math.isfinite(x) for x in (opening, close, volume))
                or min(opening, close) <= 0 or volume < 0):
            raise ValueError("Invalid price or volume: %s %s %s" % (code, label, date))
        result.append((date, opening, close, volume))
    dates = [r[0] for r in result]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Dates must be nonempty, unique and ascending: " + code)
    return result


def _cash_table(events, dates, code, scenario):
    table, metadata = {}, []
    valid_dates = set(dates[1:])
    for event in events:
        date = event["date"]
        if date not in valid_dates or date in table:
            raise ValueError("Cash event must be a unique quoted date after the first bar: %s %s" % (code, date))
        point, lower, upper = (float(event[k]) for k in ("cash_per_old_share", "lower", "upper"))
        if not all(math.isfinite(x) for x in (point, lower, upper)) or not 0 <= lower <= point <= upper:
            raise ValueError("Invalid cash interval: %s %s" % (code, date))
        amount = lower if scenario == "cash_lower" else upper if scenario == "cash_upper" else point
        provenance = "frozen_manifest_" + ("lower" if scenario == "cash_lower" else
                                           "upper" if scenario == "cash_upper" else "point")
        sources = deepcopy(event.get("sources", []))
        if scenario == "official_515100" and code == OFFICIAL_CODE and date == OFFICIAL_DATE:
            if not any(math.isclose(point, x, rel_tol=0, abs_tol=1e-12) for x in (.105, OFFICIAL_CASH)):
                raise ValueError("Unexpected frozen cash at the official 515100 replacement date")
            amount, provenance = OFFICIAL_CASH, "independently_checked_official_cash"
            sources = [dict(url=OFFICIAL_SOURCE, announcement_date="2025-12-18",
                            record_date="2025-12-22", ex_date=OFFICIAL_DATE,
                            payment_date="2025-12-26", cash_per_ten_shares="1.053")]
        table[date] = (point, amount)
        metadata.append(dict(date=date, base_cash_per_old_share=point,
                             scenario_cash_per_old_share=amount, lower=lower, upper=upper,
                             amount_changed=amount != point, provenance=provenance,
                             base_provenance=event.get("provenance", "frozen_manifest"),
                             sources=sources))
    return table, metadata


def scenario_histories(manifest, raw_histories, corrected_histories, name,
                       benchmark="510300"):
    """Pure fixed-scenario construction, returning ``(histories, metadata)``.

    Input histories contain four-column rows (date, open, close, volume), not
    Tencent's six-column arrays. All assets/dates and baseline volume are kept.
    No input is mutated. Future split scales affect the *unit convention* for
    volume, while event returns use only the ratio applicable at the current
    bar; future cash never changes an earlier TR prefix.
    """
    if name not in SCENARIOS:
        raise ValueError("Unknown fixed precision scenario: " + str(name))
    assets = manifest["assets"]
    if set(assets) != set(raw_histories) or set(assets) != set(corrected_histories):
        raise ValueError("Manifest/raw/corrected asset sets differ")
    if benchmark not in assets:
        raise ValueError("Missing benchmark calendar asset")
    if name == "official_515100" and OFFICIAL_CODE not in assets:
        raise ValueError("Official 515100 scenario requires that asset")
    histories, asset_meta, changed_codes = {}, {}, []
    official_found = False
    for code, asset in assets.items():
        raw = _rows(raw_histories[code], code, "raw")
        base_rows = _rows(corrected_histories[code], code, "corrected")
        dates = [r[0] for r in raw]
        if dates != [r[0] for r in base_rows]:
            raise ValueError("Raw/corrected date axes differ: " + code)
        if (len(raw) != asset["rows"] or dates[0] != asset["first_date"]
                or dates[-1] != asset["last_date"] or dates[-1] != manifest["end"]):
            raise ValueError("Frozen coverage differs: " + code)
        if any(r[1] != r[2] for r in base_rows):
            raise ValueError("Expected explicit close-only corrected rows: " + code)
        scales, splits = adjustment_scales(dates, asset["split_events"])
        cash, events = _cash_table(asset["cash_events"], dates, code, name)
        if code == OFFICIAL_CODE and OFFICIAL_DATE in cash:
            official_found = True
        changed = any(e["amount_changed"] for e in events)
        if changed:
            changed_codes.append(code)
        baseline_value = scenario_value = base_rows[0][2]
        changed_so_far = False
        output, maximum_baseline_error = [], 0.0
        for i, (raw_row, base_row, scale) in enumerate(zip(raw, base_rows, scales)):
            expected_volume = raw_row[3] / scale
            if not math.isclose(expected_volume, base_row[3], rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError("Frozen normalized volume disagrees with verified splits: %s %s" % (code, dates[i]))
            if i:
                split = scales[i] / scales[i - 1]
                original_cash, scenario_cash = cash.get(dates[i], (0.0, 0.0))
                changed_so_far = changed_so_far or original_cash != scenario_cash
                previous_raw_close = raw[i - 1][2]
                baseline_value *= (split * raw_row[2] + original_cash) / previous_raw_close
                scenario_value *= (split * raw_row[2] + scenario_cash) / previous_raw_close
            if not math.isfinite(scenario_value) or scenario_value <= 0:
                raise ArithmeticError("Invalid reconstructed precision scenario: " + code)
            error = abs(baseline_value - base_row[2]) / max(abs(base_row[2]), 1e-300)
            maximum_baseline_error = max(maximum_baseline_error, error)
            if error > 1e-11:
                raise ValueError("Manifest point events do not reconstruct frozen close: %s %s" % (code, dates[i]))
            # Unaffected assets retain exact baseline floats. Changed assets are
            # rebuilt with raw event returns; volumes always retain exact floats.
            value = scenario_value if changed_so_far else base_row[2]
            output.append((dates[i], value, value, base_row[3]))
        histories[code] = output
        asset_meta[code] = dict(rows=len(output), first_date=dates[0], last_date=dates[-1],
                               changed=changed, splits=splits, cash_events=events,
                               point_reconstruction_max_relative_error=maximum_baseline_error,
                               raw_content_sha256=_content_hash(raw),
                               baseline_content_sha256=_content_hash(base_rows),
                               scenario_content_sha256=_content_hash(output))
    if name == "official_515100" and not official_found:
        raise ValueError("Official 515100 event is absent from the frozen cash calendar")
    calendar = [r[0] for r in histories[benchmark]]
    metadata = dict(schema=1, scenario=name, end=manifest["end"], benchmark=benchmark,
                    calendar=calendar, calendar_sha256=_content_hash(calendar),
                    asset_order=list(assets), changed_codes=changed_codes,
                    input_manifest_content_sha256=_content_hash(manifest),
                    histories_content_sha256=_content_hash(histories), assets=asset_meta,
                    data_view="fixed_cash_amount_sensitivity_close_total_return",
                    open_policy="placeholder_equals_total_return_close", usable_for_next_open=False,
                    volume_policy="exact_frozen_normalized_volume_unchanged",
                    reinvestment="Immediate free reinvestment at ex-date close; actual payment delay not simulated",
                    scenario_scope="Cash amounts on existing event dates only; no new assets, dates, split ratios or strategy parameters",
                    official_source=OFFICIAL_SOURCE if name == "official_515100" else None,
                    limitations=["Bounds are conditional on detected event dates and declared splits, not missed-action guarantees.",
                                 "Lower/upper asset paths do not bound adaptive strategy returns; recompute signals and holdings.",
                                 "These are preset sensitivity diagnostics, not independently validated market data or new selection trials.",
                                 "Prior examined history remains contaminated; no clean OOS or execution claim."])
    return histories, metadata


def build_scenario(name, source_dir=None):
    """Read/verify the frozen vintage, construct in memory, never write files."""
    if name not in SCENARIOS:
        raise ValueError("Unknown fixed precision scenario: " + str(name))
    source = Path(source_dir) if source_dir is not None else DEFAULT_SOURCE
    manifest_path = source / "corrected_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    source_files = {"corrected_manifest.json": hashlib.sha256(manifest_bytes).hexdigest()}
    raw, corrected = {}, {}
    for code, asset in manifest["assets"].items():
        raw_name = "results/price_audit/%s_raw_2010-01-01_%s.json" % (code, manifest["end"])
        corrected_name = "corrected_snapshots/%s.csv" % code
        for relative, expected in ((raw_name, asset["raw_sha256"]),
                                   (corrected_name, asset["sha256"])):
            observed = _sha(source / relative)
            if observed != expected:
                raise ValueError("Frozen source hash changed: " + relative)
            source_files[relative] = observed
        vendor_rows = json.loads((source / raw_name).read_text())["rows"]
        raw[code] = [(r[0], float(r[1]), float(r[2]), float(r[5])) for r in vendor_rows]
        with (source / corrected_name).open(newline="") as f:
            corrected[code] = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(f) if r]
    for relative, key in (("results/corporate_actions_verified.json", "action_manifest_sha256"),
                          ("results/verified_cash_510500.json", "verified_cash_510500_sha256")):
        if key in manifest:
            observed = _sha(source / relative)
            if observed != manifest[key]:
                raise ValueError("Frozen action source hash changed: " + relative)
            source_files[relative] = observed
    histories, metadata = scenario_histories(manifest, raw, corrected, name)
    # Source hashes describe exactly the files read, not a regenerated vintage.
    metadata.update(source_directory=str(source.resolve()), source_file_sha256=source_files,
                    constructor_source_sha256=_sha(Path(__file__)),
                    split_scale_source_sha256=_sha(DEFAULT_SOURCE / "total_return_audit.py"))
    return histories, metadata
