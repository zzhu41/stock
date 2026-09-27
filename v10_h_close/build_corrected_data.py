"""Build a separately hashed, close-only total-return data vintage for all 24 ETFs."""
import csv
from datetime import datetime, timezone
import json

from .data import BASE, ASSET_ORDER, protect, sha
from .total_return_audit import reconstruct_total_return


def build():
    protect()
    target = BASE / "corrected_snapshots"
    manifest_path = BASE / "corrected_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Corrected vintage is frozen; do not overwrite")
    target.mkdir(exist_ok=True)
    action_path = BASE / "results/corporate_actions_verified.json"
    actions = json.loads(action_path.read_text())["actions"]
    cash_path = BASE / "results/verified_cash_510500.json"
    verified_cash = json.loads(cash_path.read_text())["cash_events"]
    original_manifest = json.loads((BASE / "snapshot_manifest.json").read_text())
    end = original_manifest["end"]
    assets = {}
    for code in ASSET_ORDER:
        raw_path = BASE / "results/price_audit" / (code + "_raw_2010-01-01_" + end + ".json")
        q_path = BASE / "snapshots" / (code + ".csv")
        if sha(q_path) != original_manifest["assets"][code]["sha256"]:
            raise ValueError("Original QFQ snapshot changed")
        raw = json.loads(raw_path.read_text())["rows"]
        # Tencent array order is date/open/close/high/low/volume.
        raw_rows = [(r[0], float(r[1]), float(r[2]), float(r[5])) for r in raw]
        with q_path.open(newline="") as f:
            q_rows = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(f) if r]
        if [r[0] for r in raw_rows] != [r[0] for r in q_rows]:
            raise ValueError("Raw/QFQ date axes differ: " + code)
        if any(abs(a[3] - b[3]) > 1e-6 for a, b in zip(raw_rows, q_rows)):
            raise ValueError("Raw/QFQ volume observations differ: " + code)
        events = [x for x in actions if x["code"] == code]
        rebuilt = reconstruct_total_return(raw_rows, q_rows, events, code=code,
                                           cash_events=verified_cash if code == "510500" else None)
        path = target / (code + ".csv")
        with path.open("w", newline="") as f:
            csv.writer(f).writerows(rebuilt["rows"])
        assets[code] = dict(sha256=sha(path), raw_sha256=sha(raw_path), qfq_sha256=sha(q_path),
                            rows=len(rebuilt["rows"]), first_date=rebuilt["rows"][0][0],
                            last_date=rebuilt["rows"][-1][0], split_events=events,
                            cash_events=rebuilt["cash_events"], diagnostics=rebuilt["diagnostics"])
        print("CORRECTED DATA", code, len(rebuilt["rows"]), "cash events", len(rebuilt["cash_events"]),
              "splits", len(events), flush=True)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), end=end, assets=assets,
                    action_manifest_sha256=sha(action_path), original_manifest_sha256=sha(BASE / "snapshot_manifest.json"),
                    verified_cash_510500_sha256=sha(cash_path),
                    method_source_sha256=sha(BASE / "total_return_audit.py"),
                    builder_source_sha256=sha(BASE / "build_corrected_data.py"),
                    data_view="raw_price_and_corporate_actions_reconstructed_close_total_return",
                    cash_dividends="Inferred from affine QFQ differences with explicit rounding bounds; not all independently verified",
                    reinvestment="Immediate free close reinvestment on the ex-date; payment-date delay not modeled",
                    open_column="close placeholder; prohibited for next-open or intraday execution",
                    volume="Raw volume expressed in final share units using verified future split products",
                    clean_oos=False, protected_files_verified=protect())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    build()
