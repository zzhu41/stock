"""Lossless archives for this batch's four complete return matrices.

Run explicitly after matrix-producing/statistics workers finish:
  python3 -m v10_h_close.matrix_archive pack
  python3 -m v10_h_close.matrix_archive verify
  python3 -m v10_h_close.matrix_archive restore

Packing never deletes a source matrix. Restoration verifies both compressed
and uncompressed SHA256/size, and never overwrites a different existing matrix.
All hashing/compression is streamed, so large matrices need no extra RAM copy.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile


OUT = Path(__file__).resolve().parent / "results"
NAMES = ("selection_returns.npy", "selection_fee5_returns.npy",
         "full_returns.npy", "full_fee5_returns.npy")
MANIFEST = "matrix_archives.json"
CHUNK = 1024 * 1024


def _fingerprint(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as source:
        while True:
            block = source.read(CHUNK)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _copy(source, target, maximum_bytes=None):
    digest, size = hashlib.sha256(), 0
    while True:
        block = source.read(CHUNK)
        if not block:
            break
        size += len(block)
        if maximum_bytes is not None and size > maximum_bytes:
            raise ValueError("Restored matrix exceeds the recorded byte count")
        digest.update(block)
        target.write(block)
    return digest.hexdigest(), size


def _temporary(directory, name):
    fd, path = tempfile.mkstemp(prefix="." + name + ".", suffix=".tmp", dir=str(directory))
    return fd, Path(path)


def _remove(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _load_records(directory):
    records = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if set(records) != set(NAMES):
        raise ValueError("Archive manifest must describe exactly the four registered matrices")
    for name, record in records.items():
        for key in ("raw_sha256", "archive_sha256"):
            value = record.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid digest in archive manifest: " + name)
        for key in ("raw_bytes", "archive_bytes"):
            value = record.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Invalid byte count in archive manifest: " + name)
    return records


def pack(directory=OUT):
    """Stage deterministic gzip archives, then publish their receipt; keep originals."""
    directory = Path(directory)
    if any(not (directory / name).is_file() for name in NAMES):
        raise ValueError("All four complete source matrices must exist before packing")
    records, staged = {}, []
    manifest_tmp = None
    try:
        for name in NAMES:
            source = directory / name
            fd, temporary = _temporary(directory, name + ".gz")
            staged.append((temporary, directory / (name + ".gz")))
            with os.fdopen(fd, "wb") as raw, source.open("rb") as inp:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as output:
                    raw_hash, raw_size = _copy(inp, output)
            if _fingerprint(source) != (raw_hash, raw_size):
                raise ValueError("Source matrix changed while being archived: " + name)
            archive_hash, archive_size = _fingerprint(temporary)
            records[name] = dict(raw_sha256=raw_hash, archive_sha256=archive_hash,
                                 raw_bytes=raw_size, archive_bytes=archive_size,
                                 compression="gzip", gzip_mtime=0)
        fd, manifest_tmp = _temporary(directory, MANIFEST)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(records, output, indent=2, sort_keys=True)
            output.write("\n")
        for temporary, target in staged:
            os.replace(str(temporary), str(target))
        os.replace(str(manifest_tmp), str(directory / MANIFEST))
        return records
    finally:
        for temporary, _ in staged:
            _remove(temporary)
        if manifest_tmp is not None:
            _remove(manifest_tmp)


def verify(directory=OUT):
    """Check every archive and all raw matrices which are currently present."""
    directory = Path(directory)
    records = _load_records(directory)
    status = {}
    for name in NAMES:
        record = records[name]
        archive, raw = directory / (name + ".gz"), directory / name
        if _fingerprint(archive) != (record["archive_sha256"], record["archive_bytes"]):
            raise ValueError("Archive fingerprint differs: " + name)
        if raw.exists() and _fingerprint(raw) != (record["raw_sha256"], record["raw_bytes"]):
            raise ValueError("Existing matrix differs; refusing overwrite: " + name)
        status[name] = "present_verified" if raw.exists() else "archive_verified_raw_missing"
    return status


def restore(directory=OUT):
    """Restore absent matrices only, with verification before publishing each file."""
    directory = Path(directory)
    records = _load_records(directory)
    status = verify(directory)  # Fail on any existing conflict before restoring anything.
    for name in NAMES:
        if status[name] == "present_verified":
            continue
        record = records[name]
        target = directory / name
        fd, temporary = _temporary(directory, name + ".restore")
        try:
            with os.fdopen(fd, "wb") as output, gzip.open(str(directory / (name + ".gz")), "rb") as inp:
                restored = _copy(inp, output, maximum_bytes=record["raw_bytes"])
            if restored != (record["raw_sha256"], record["raw_bytes"]):
                raise ValueError("Restored matrix fingerprint differs: " + name)
            # Same-directory hard-link creation is atomic and refuses to clobber
            # a concurrently created target. Only the verified inode is exposed.
            try:
                os.link(str(temporary), str(target))
                status[name] = "restored_verified"
            except FileExistsError:
                if _fingerprint(target) != (record["raw_sha256"], record["raw_bytes"]):
                    raise ValueError("Matrix appeared during restore and differs: " + name)
                status[name] = "present_verified"
        finally:
            _remove(temporary)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "archive", "verify", "restore"))
    args = parser.parse_args()
    action = pack if args.action in ("pack", "archive") else verify if args.action == "verify" else restore
    print(json.dumps(action(), indent=2, sort_keys=True))
