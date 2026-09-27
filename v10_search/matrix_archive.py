"""Lossless archives for the complete float64 matrices; no candidate filtering."""
import argparse
import gzip
import json
import os
import shutil
from pathlib import Path

from .data import BASE, sha

OUT = BASE / "results"
NAMES = ("development_returns.npy", "full_returns.npy")


def pack():
    records = {}
    for name in NAMES:
        source, target = OUT / name, OUT / (name + ".gz")
        with source.open("rb") as inp, target.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as archive:
                shutil.copyfileobj(inp, archive, length=1024 * 1024)
        records[name] = dict(raw_sha256=sha(source), archive_sha256=sha(target),
                             raw_bytes=source.stat().st_size, archive_bytes=target.stat().st_size)
        print("Archived", name, records[name]["archive_bytes"], flush=True)
    (OUT / "matrix_archives.json").write_text(json.dumps(records, indent=2) + "\n")


def restore():
    records = json.loads((OUT / "matrix_archives.json").read_text())
    for name in NAMES:
        expected = records[name]
        raw, archive = OUT / name, OUT / (name + ".gz")
        if raw.exists():
            if sha(raw) != expected["raw_sha256"]:
                raise ValueError("Existing matrix differs; refusing overwrite: " + name)
            continue
        if sha(archive) != expected["archive_sha256"]:
            raise ValueError("Archive fingerprint differs: " + name)
        temporary = OUT / (name + ".restore.tmp")
        with gzip.open(archive, "rb") as inp, temporary.open("wb") as out:
            shutil.copyfileobj(inp, out, length=1024 * 1024)
        if sha(temporary) != expected["raw_sha256"]:
            raise ValueError("Restored matrix hash differs: " + name)
        os.replace(temporary, raw)
        print("Restored", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "restore"))
    args = parser.parse_args()
    (pack if args.action == "pack" else restore)()
