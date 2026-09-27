"""Reproduction integrity, including refusal to overwrite a corrupt path."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v11 import artifacts


class ArchiveTests(unittest.TestCase):
    def test_exact_roundtrip_and_existing_corruption_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'results/example/paths.npz'
            source.parent.mkdir(parents=True);original=b'original exact bytes\x00\xff'*100
            source.write_bytes(original)
            with patch.object(artifacts,'BASE',root),patch.object(artifacts,'MANIFEST',root/'path_archives.json'):
                self.assertEqual(artifacts.pack(),1)
                source.unlink()
                self.assertEqual(artifacts.restore(),1)
                self.assertEqual(source.read_bytes(),original)
                self.assertEqual(artifacts.restore(),0)
                source.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'refusing overwrite'):artifacts.restore()
                self.assertEqual(source.read_bytes(),b'corrupt')

    def test_external_paths_and_bad_archive_refuse_before_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'results/a.npz';source.parent.mkdir();source.write_bytes(b'a')
            with patch.object(artifacts,'BASE',root),patch.object(artifacts,'MANIFEST',root/'path_archives.json'):
                for value in ('../escape','/absolute'):
                    with self.assertRaises(ValueError):artifacts._path(value)
                artifacts.pack();source.unlink()
                archive=source.with_suffix('.npz.gz');archive.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'Archive hash differs'):artifacts.restore()
                self.assertFalse(source.exists())


if __name__=='__main__':unittest.main()
