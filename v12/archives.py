"""Lossless V12 result archives, with a stdlib-only cold-clone restore path.

Archive every raw results/**/*.npz and each raw .json strictly over 5 MiB.
Existing .gz artifacts (including concentration shards) are never recompressed.
pack/restore write only inside V12; verify does not create or modify files.
No operation deletes a raw file or overwrites conflicting archives/targets.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zlib


BASE = Path(__file__).resolve().parent
JSON_THRESHOLD_BYTES = 5 * 1024 * 1024
MANIFEST_NAME = "path_archives.json"
CHUNK = 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _base(value):
    base = Path(value)
    _require(base.is_dir() and not base.is_symlink(), "V12 base must be an existing regular directory")
    return base.resolve()


def _path(base, relative, results_only=False):
    _require(isinstance(relative, str) and relative and not any(c in relative for c in "\x00\r\n"),
             "Archive path must be a nonempty single-line relative path")
    value = Path(relative)
    _require(not value.is_absolute() and ".." not in value.parts and value.as_posix() == relative,
             "Absolute, traversal or noncanonical archive path")
    if results_only:
        _require(len(value.parts) > 1 and value.parts[0] == "results", "Raw/archive paths must stay in results")
    target = base / value
    current = base
    for part in value.parts:
        current = current / part
        _require(not current.is_symlink(), "Symlink archive paths are not supported")
    _require(base in target.resolve().parents, "Archive path escapes V12")
    return target


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover(base):
    directory = _path(base, "results")
    if not directory.exists():
        return []
    selected = []
    for root, directories, files in os.walk(str(directory), followlinks=False):
        directories.sort(); files.sort()
        for name in directories:
            child = Path(root) / name
            _require(not child.is_symlink(), "Symlink result directories are not supported")
        for name in files:
            suffix = Path(name).suffix
            if suffix not in (".npz", ".json"):
                continue
            path = Path(root) / name
            relative = path.relative_to(base).as_posix()
            path = _path(base, relative, results_only=True)
            _require(path.is_file(), "Archival input must be a regular file")
            if suffix == ".npz" or path.stat().st_size > JSON_THRESHOLD_BYTES:
                selected.append(relative)
    return sorted(selected)


def _load_manifest(base):
    path = _path(base, MANIFEST_NAME)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    _require(value.get("schema") == 1 and value.get("json_threshold_bytes") == JSON_THRESHOLD_BYTES,
             "Unknown archive manifest schema or threshold")
    rows = value.get("archives")
    _require(isinstance(rows, list) and rows, "Archive manifest cannot be empty")
    seen = set()
    for row in rows:
        _require(isinstance(row, dict), "Invalid archive entry")
        raw, compressed = row.get("path"), row.get("archive")
        _path(base, raw, results_only=True); _path(base, compressed, results_only=True)
        _require(raw not in seen and compressed == raw + ".gz", "Duplicate or mismatched archive destination")
        seen.add(raw)
        kind = Path(raw).suffix.lstrip(".")
        _require(kind in ("npz", "json") and row.get("kind") == kind, "Invalid archive file kind")
        for name in ("raw_sha256", "archive_sha256"):
            _require(isinstance(row.get(name), str) and re.fullmatch(r"[0-9a-f]{64}", row[name]), "Invalid SHA256 in manifest")
        for name in ("raw_bytes", "archive_bytes"):
            _require(type(row.get(name)) is int and row[name] >= 0, "Invalid byte count in manifest")
        _require(row["archive_bytes"] > 0 and (kind != "json" or row["raw_bytes"] > JSON_THRESHOLD_BYTES),
                 "Archive entry violates the registered file-size rule")
    _require(rows == sorted(rows, key=lambda row: row["path"]), "Archive manifest paths must be sorted")
    return value


def _gzip_raw_identity(path, output=None):
    digest, size = hashlib.sha256(), 0
    try:
        with gzip.open(str(path), "rb") as incoming:
            for chunk in iter(lambda: incoming.read(CHUNK), b""):
                digest.update(chunk); size += len(chunk)
                if output is not None:
                    output.write(chunk)
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError("Invalid gzip archive: " + str(path)) from exc
    return digest.hexdigest(), size


def _verify_entry(base, row):
    raw = _path(base, row["path"], results_only=True)
    archive = _path(base, row["archive"], results_only=True)
    _require(archive.is_file(), "Missing archive: " + row["archive"])
    _require(archive.stat().st_size == row["archive_bytes"] and sha(archive) == row["archive_sha256"],
             "Compressed archive hash/size differs: " + row["archive"])
    digest, size = _gzip_raw_identity(archive)
    _require(digest == row["raw_sha256"] and size == row["raw_bytes"], "Decompressed raw hash/size differs: " + row["path"])
    _require(sha(archive) == row["archive_sha256"], "Archive changed during verification")
    if raw.exists():
        _require(raw.is_file() and raw.stat().st_size == row["raw_bytes"] and sha(raw) == row["raw_sha256"],
                 "Existing raw file differs; refusing overwrite: " + row["path"])
        return True
    return False


def verify(base=BASE):
    """Verify gzip and raw-byte hashes even when all raw files are absent."""
    base = _base(base)
    manifest_path = _path(base, MANIFEST_NAME)
    manifest_hash = sha(manifest_path)
    manifest = _load_manifest(base)
    listed = {row["path"] for row in manifest["archives"]}
    unregistered = set(_discover(base)) - listed
    _require(not unregistered, "Unregistered archival inputs appeared: " + ", ".join(sorted(unregistered)))
    present = sum(_verify_entry(base, row) for row in manifest["archives"])
    _require(sha(manifest_path) == manifest_hash, "Archive manifest changed during verification")
    return dict(files=len(listed), raw_present=present, raw_missing=len(listed) - present,
        raw_bytes=sum(row["raw_bytes"] for row in manifest["archives"]),
        archive_bytes=sum(row["archive_bytes"] for row in manifest["archives"]),
        manifest_sha256=manifest_hash, verified=True)


@contextmanager
def _lock(base):
    directory = _path(base, "cache")
    directory.mkdir(exist_ok=True)
    with _path(base, "cache/archives.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _write_new(path, raw):
    """Atomic creation without replacing a concurrent pre-existing file."""
    descriptor, name = tempfile.mkstemp(prefix=".archive-manifest-", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(str(temporary), str(path))
    finally:
        temporary.unlink()


def _ignore_pattern(relative):
    # Escape wildmatch metacharacters and spaces so one large file does not
    # accidentally ignore similarly named ordinary JSON evidence.
    return "/" + re.sub(r"([\\*?\[\]#! ])", r"\\\1", relative)


def _ensure_gitignore(base, rows):
    path = _path(base, ".gitignore")
    before = path.read_bytes() if path.exists() else None
    text = before.decode("utf-8") if before is not None else ""
    lines = text.splitlines()
    additions = [rule for rule in ("cache/", "results/**/*.npz", "results/*.npz") if rule not in lines]
    additions += [_ignore_pattern(row["path"]) for row in rows
                  if row["kind"] == "json" and _ignore_pattern(row["path"]) not in lines]
    if not additions:
        return False
    new = text + ("\n" if text and not text.endswith("\n") else "")
    new += "\n# V12 lossless archives: exact large-JSON paths; compressed files stay tracked.\n"
    new += "\n".join(additions) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=".archive-ignore-", dir=str(base))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(new.encode("utf-8")); stream.flush(); os.fsync(stream.fileno())
        current = path.read_bytes() if path.exists() else None
        _require(current == before, ".gitignore changed during archive preparation; refusing overwrite")
        _path(base, ".gitignore")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _existing_archive(base, row):
    archive = _path(base, row["archive"], results_only=True)
    if not archive.exists():
        return False
    _require(archive.is_file(), "Existing archive is not a regular file")
    digest, size = _gzip_raw_identity(archive)
    _require(digest == row["raw_sha256"] and size == row["raw_bytes"],
             "Existing archive has different contents; refusing overwrite: " + row["archive"])
    row.update(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)
    return True


def _compress_new(base, row):
    source = _path(base, row["path"], results_only=True)
    target = _path(base, row["archive"], results_only=True)
    descriptor, name = tempfile.mkstemp(prefix=".archive-gzip-", dir=str(target.parent))
    temporary = Path(name)
    digest, size = hashlib.sha256(), 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            with source.open("rb") as incoming, gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0, compresslevel=6) as zipped:
                for chunk in iter(lambda: incoming.read(CHUNK), b""):
                    digest.update(chunk); size += len(chunk); zipped.write(chunk)
            output.flush(); os.fsync(output.fileno())
        _require(digest.hexdigest() == row["raw_sha256"] and size == row["raw_bytes"], "Raw file changed while being compressed")
        _path(base, row["archive"], results_only=True)
        try:
            os.link(str(temporary), str(target))
        except FileExistsError:
            _existing_archive(base, row)  # Matching concurrent content is safe; different bytes are never overwritten.
        row.update(archive_sha256=sha(target), archive_bytes=target.stat().st_size)
    finally:
        temporary.unlink()


def pack(base=BASE):
    """Freeze a complete manifest only after all intended diagnostics finish."""
    base = _base(base)
    with _lock(base):
        manifest_path = _path(base, MANIFEST_NAME)
        if manifest_path.exists():
            result = verify(base)
            rows = _load_manifest(base)["archives"]
            result.update(already_packed=True, gitignore_updated=_ensure_gitignore(base, rows))
            return result
        selected = _discover(base)
        _require(selected, "No raw result matrices or large JSON files to archive")
        rows = []
        for name in selected:
            source = _path(base, name, results_only=True)
            row = dict(path=name, archive=name + ".gz", kind=source.suffix[1:],
                       raw_sha256=sha(source), raw_bytes=source.stat().st_size)
            _existing_archive(base, row)  # Validate every old archive before creating any new one.
            rows.append(row)
        for row in rows:
            if "archive_sha256" not in row:
                _compress_new(base, row)
        for row in rows:
            _verify_entry(base, row)
        _require(selected == _discover(base), "Archive input set changed while packing")
        manifest = dict(schema=1, json_threshold_bytes=JSON_THRESHOLD_BYTES, archives=rows,
            format="gzip wrapping exact original file bytes", existing_gzip_recompressed=False,
            raw_files_deleted=False, scope="Every results NPZ plus raw results JSON strictly over 5 MiB")
        _write_new(manifest_path, (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        updated = _ensure_gitignore(base, rows)
        return dict(files=len(rows), raw_present=len(rows), raw_missing=0,
            raw_bytes=sum(row["raw_bytes"] for row in rows), archive_bytes=sum(row["archive_bytes"] for row in rows),
            manifest_sha256=sha(manifest_path), verified=True, already_packed=False, gitignore_updated=updated)


def restore(base=BASE):
    """Restore only missing raw files, after validating the whole archive set."""
    base = _base(base)
    with _lock(base):
        verification = verify(base)  # Reject conflicting existing files before creating any raw output.
        rows = _load_manifest(base)["archives"]
        restored = 0
        for row in rows:
            target, archive = _path(base, row["path"], True), _path(base, row["archive"], True)
            if target.exists():
                _require(target.is_file() and target.stat().st_size == row["raw_bytes"] and sha(target) == row["raw_sha256"],
                         "Existing raw file changed; refusing overwrite: " + row["path"])
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix=".archive-restore-", dir=str(target.parent))
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    digest, size = _gzip_raw_identity(archive, output)
                    output.flush(); os.fsync(output.fileno())
                _require(digest == row["raw_sha256"] and size == row["raw_bytes"]
                         and sha(archive) == row["archive_sha256"], "Restore bytes differ from the frozen archive")
                _require(sha(base / MANIFEST_NAME) == verification["manifest_sha256"], "Manifest changed while restoring")
                _path(base, row["path"], True)
                try:
                    os.link(str(temporary), str(target)); restored += 1
                except FileExistsError:
                    _require(target.is_file() and target.stat().st_size == row["raw_bytes"] and sha(target) == row["raw_sha256"],
                             "Concurrent raw file differs; refusing overwrite: " + row["path"])
            finally:
                temporary.unlink()
        verification.update(restored=restored, raw_present=len(rows), raw_missing=0)
        _require(sha(base / MANIFEST_NAME) == verification["manifest_sha256"], "Manifest changed while restoring")
        return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "verify", "restore"))
    args = parser.parse_args()
    try:
        result = {"pack": pack, "verify": verify, "restore": restore}[args.command]()
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps(dict(command=args.command, error=str(exc)), ensure_ascii=False))
        raise SystemExit(1)
    print(json.dumps(dict(command=args.command, **result), ensure_ascii=False, indent=2))
