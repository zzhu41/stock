"""Small, valid float64 NPY fixtures only; never touch real result matrices."""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from v10_h_close.matrix_archive import MANIFEST, NAMES, pack, restore, verify


def npy_bytes(seed):
    # Standard NumPy v1.0 header, built without needing NumPy in this utility.
    header = repr(dict(descr="<f8", fortran_order=False, shape=(2, 3))).encode("ascii")
    header += b" " * ((-10 - len(header) - 1) % 16) + b"\n"
    return (b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
            + struct.pack("<6d", 0.0, -0.0, seed + .125, -1e-12, .5, 1.0))


class MatrixArchiveTests(unittest.TestCase):
    def fixtures(self, directory):
        payloads = {name: npy_bytes(i) for i, name in enumerate(NAMES)}
        for name, payload in payloads.items():
            (directory / name).write_bytes(payload)
        return payloads

    def test_round_trip_is_byte_exact_and_pack_keeps_all_originals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            payloads = self.fixtures(root)
            records = pack(root)
            for name in NAMES:
                self.assertEqual((root / name).read_bytes(), payloads[name])
                self.assertEqual(records[name]["raw_sha256"], hashlib.sha256(payloads[name]).hexdigest())
                with gzip.open(str(root / (name + ".gz")), "rb") as archive:
                    self.assertEqual(archive.read(), payloads[name])
            # Remove only tiny fixtures to exercise reconstruction, never real outputs.
            (root / NAMES[0]).unlink()
            (root / NAMES[2]).unlink()
            status = restore(root)
            self.assertEqual(status[NAMES[0]], "restored_verified")
            self.assertEqual(status[NAMES[1]], "present_verified")
            for name in NAMES:
                self.assertEqual((root / name).read_bytes(), payloads[name])
            self.assertTrue(all(x == "present_verified" for x in verify(root).values()))
            self.assertFalse(any(p.name.endswith(".tmp") for p in root.iterdir()))

    def test_archives_are_deterministic_with_zero_mtime_and_no_filename(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            first = pack(root)
            first_receipt = (root / MANIFEST).read_bytes()
            mtimes = {name: (root / name).stat().st_mtime_ns for name in NAMES}
            second = pack(root)
            self.assertEqual(first, second)
            self.assertEqual((root / MANIFEST).read_bytes(), first_receipt)
            for name in NAMES:
                raw = (root / (name + ".gz")).read_bytes()
                self.assertEqual(struct.unpack("<I", raw[4:8])[0], 0)
                self.assertFalse(raw[3] & 8)  # FNAME does not contain a temporary filename.
                self.assertEqual((root / name).stat().st_mtime_ns, mtimes[name])

    def test_modified_archive_is_rejected_before_any_restore(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            pack(root)
            (root / NAMES[0]).unlink()
            archive = root / (NAMES[0] + ".gz")
            archive.write_bytes(archive.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "Archive fingerprint"):
                restore(root)
            self.assertFalse((root / NAMES[0]).exists())
            self.assertFalse(any(p.name.endswith(".tmp") for p in root.iterdir()))

    def test_uncompressed_hash_is_checked_even_when_compressed_hash_matches(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            records = pack(root)
            (root / NAMES[0]).unlink()
            records[NAMES[0]]["raw_sha256"] = "0" * 64
            (root / MANIFEST).write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, "Restored matrix fingerprint"):
                restore(root)
            self.assertFalse((root / NAMES[0]).exists())
            self.assertFalse(any(p.name.endswith(".tmp") for p in root.iterdir()))

    def test_restore_refuses_to_overwrite_a_different_existing_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixtures(root)
            pack(root)
            changed = b"a different research result"
            (root / NAMES[0]).write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "Existing matrix differs"):
                restore(root)
            self.assertEqual((root / NAMES[0]).read_bytes(), changed)

    def test_incomplete_source_set_does_not_publish_partial_archives(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            payloads = self.fixtures(root)
            (root / NAMES[-1]).unlink()
            with self.assertRaisesRegex(ValueError, "All four"):
                pack(root)
            self.assertFalse((root / MANIFEST).exists())
            self.assertFalse(list(root.glob("*.gz")))
            for name in NAMES[:-1]:
                self.assertEqual((root / name).read_bytes(), payloads[name])


if __name__ == "__main__":
    unittest.main()
