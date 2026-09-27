"""Frozen inputs and a new protection baseline after authorized UI changes."""
import hashlib
import json
from pathlib import Path
import subprocess

from v10_h_close.corrected_scan import load_histories
from v10_h_close.data import load_fear
from v10_search.data import ASSET_ORDER, UNIVERSE, CASH, GOLD, BENCHMARK

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
START, END, TRAIN_END = "2014-01-02", "2026-09-24", "2025-12-31"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def freeze_protected():
    path = BASE / "protected_manifest.json"
    if path.exists():
        return protect()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=str(ROOT)).decode().split("\0")
    files = {n: sha(ROOT / n) for n in names if n and not n.startswith("v10_deep/") and (ROOT / n).is_file()}
    for p in (ROOT / "data").glob("*.csv"):
        files[str(p.relative_to(ROOT))] = sha(p)
    dump(path, dict(git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip(),
                    sha256=files, note="New baseline after authorized bot/README changes; prior manifests remain untouched."))
    return len(files)


def protect():
    manifest = json.loads((BASE / "protected_manifest.json").read_text())
    changed = [n for n, expected in manifest["sha256"].items() if not (ROOT / n).is_file() or sha(ROOT / n) != expected]
    if changed:
        raise AssertionError("Previous work changed: " + ", ".join(changed))
    return len(manifest["sha256"])


def inputs():
    histories, end = load_histories()
    if end != END:
        raise ValueError("Unexpected common cutoff")
    calendar = [r[0] for r in histories[BENCHMARK]]
    return histories, calendar, load_fear(calendar)
