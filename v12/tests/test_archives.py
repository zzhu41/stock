"""Exact-byte archives and hostile/cold filesystem cases, all in temporary dirs."""
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from v12 import archives as a


def json_bytes(size):
    prefix,suffix=b'{"data":"',b'"}\n'
    return prefix+b'x'*(size-len(prefix)-len(suffix))+suffix


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v12-archive-test-');self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)/'v12';self.base.mkdir()
        folder=self.base/'results/main';folder.mkdir(parents=True)
        self.matrix=folder/'paths.npz'
        np.savez_compressed(self.matrix,returns=np.asarray([[0.,.01,-.02]],dtype=np.float64))
        self.large=folder/'large.json';self.large.write_bytes(json_bytes(a.JSON_THRESHOLD_BYTES+17))
        self.edge=folder/'exact_threshold.json';self.edge.write_bytes(json_bytes(a.JSON_THRESHOLD_BYTES))
        self.small=folder/'small.json';self.small.write_text('{ "preserve_spacing": true }\n')
        self.old_gzip=folder/'concentration_close_1bp.json.gz'
        with gzip.open(self.old_gzip,'wb') as stream:stream.write(b'{"already_compressed":true}\n')
        self.ignore=self.base/'.gitignore'
        self.ignore.write_text('# user rule\ncustom/cache/\nresults/**/*.npz\nresults/**/*.npy\n')

    def manifest(self):return json.loads((self.base/a.MANIFEST_NAME).read_text())

    def snapshot(self,base=None):
        base=base or self.base
        return {str(p.relative_to(base)):a.sha(p) for p in base.rglob('*') if p.is_file() and not p.is_symlink()}

    def test_scope_threshold_exact_roundtrip_and_idempotent_pack(self):
        raw={p.relative_to(self.base).as_posix():p.read_bytes() for p in (self.matrix,self.large)}
        existing=a.sha(self.old_gzip);ignore_before=self.ignore.read_text()
        result=a.pack(self.base);self.assertEqual(result['files'],2)
        rows=self.manifest()['archives'];self.assertEqual({row['path'] for row in rows},set(raw))
        self.assertEqual(a.sha(self.old_gzip),existing)
        self.assertFalse(Path(str(self.edge)+'.gz').exists());self.assertFalse(Path(str(self.small)+'.gz').exists())
        self.assertTrue(self.ignore.read_text().startswith(ignore_before))
        self.assertIn('/results/main/large.json',self.ignore.read_text())
        self.assertNotIn('/results/main/exact_threshold.json',self.ignore.read_text())
        for row in rows:
            with gzip.open(self.base/row['archive'],'rb') as stream:self.assertEqual(stream.read(),raw[row['path']])
        before=self.snapshot();again=a.pack(self.base)
        self.assertTrue(again['already_packed']);self.assertEqual(before,self.snapshot())
        for name in raw:(self.base/name).unlink()
        self.assertEqual(a.verify(self.base)['raw_missing'],2)
        self.assertEqual(a.restore(self.base)['restored'],2)
        for name,value in raw.items():self.assertEqual((self.base/name).read_bytes(),value)
        self.assertEqual(a.restore(self.base)['restored'],0)

    def test_fresh_clone_verify_requires_no_raw_files_or_numpy_inputs(self):
        a.pack(self.base)
        clone=Path(self.temp.name)/'clone';clone.mkdir()
        shutil.copyfile(self.base/a.MANIFEST_NAME,clone/a.MANIFEST_NAME)
        for row in self.manifest()['archives']:
            p=clone/row['archive'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(self.base/row['archive'],p)
        before=self.snapshot(clone)
        result=a.verify(clone)
        self.assertEqual(result['raw_missing'],2);self.assertEqual(result['raw_present'],0)
        self.assertEqual(before,self.snapshot(clone));self.assertFalse((clone/'cache').exists())
        self.assertEqual(a.restore(clone)['restored'],2)
        for row in self.manifest()['archives']:self.assertEqual(a.sha(clone/row['path']),row['raw_sha256'])

    def test_existing_raw_corruption_refuses_before_other_missing_files_are_restored(self):
        a.pack(self.base);self.matrix.unlink();self.large.write_bytes(b'different original')
        before=self.large.read_bytes()
        with self.assertRaisesRegex(ValueError,'Existing raw file differs'):a.restore(self.base)
        self.assertFalse(self.matrix.exists());self.assertEqual(self.large.read_bytes(),before)

    def test_corrupt_gzip_refuses_without_creating_or_replacing_raw(self):
        a.pack(self.base);self.matrix.unlink()
        archive=Path(str(self.matrix)+'.gz');archive.write_bytes(b'bad gzip bytes')
        with self.assertRaisesRegex(ValueError,'Compressed archive'):a.verify(self.base)
        with self.assertRaises(ValueError):a.restore(self.base)
        self.assertFalse(self.matrix.exists())

    def test_resealed_compressed_hash_does_not_hide_wrong_decompressed_bytes(self):
        a.pack(self.base);manifest=self.manifest();row=next(r for r in manifest['archives'] if r['kind']=='npz')
        archive=self.base/row['archive']
        with gzip.open(archive,'wb') as stream:stream.write(b'wrong decompressed original')
        row.update(archive_sha256=a.sha(archive),archive_bytes=archive.stat().st_size)
        (self.base/a.MANIFEST_NAME).write_text(json.dumps(manifest))
        self.matrix.unlink()
        with self.assertRaisesRegex(ValueError,'Decompressed raw'):a.restore(self.base)
        self.assertFalse(self.matrix.exists())

    def test_preexisting_matching_archive_reused_conflicting_archive_never_overwritten(self):
        target=Path(str(self.matrix)+'.gz')
        with gzip.open(target,'wb') as stream:stream.write(self.matrix.read_bytes())
        matching=a.sha(target);a.pack(self.base);self.assertEqual(a.sha(target),matching)
        manifest_before=(self.base/a.MANIFEST_NAME).read_bytes()
        self.matrix.write_bytes(b'changed raw after freeze')
        with self.assertRaisesRegex(ValueError,'Existing raw file differs'):a.pack(self.base)
        self.assertEqual((self.base/a.MANIFEST_NAME).read_bytes(),manifest_before)
        self.assertEqual(a.sha(target),matching)

    def test_preflight_conflict_leaves_other_archives_and_manifest_uncreated(self):
        target=Path(str(self.matrix)+'.gz')
        with gzip.open(target,'wb') as stream:stream.write(b'unrelated original')
        before=target.read_bytes()
        with self.assertRaisesRegex(ValueError,'Existing archive has different'):a.pack(self.base)
        self.assertEqual(target.read_bytes(),before)
        self.assertFalse(Path(str(self.large)+'.gz').exists())
        self.assertFalse((self.base/a.MANIFEST_NAME).exists())

    def test_new_npz_after_manifest_cannot_silently_extend_frozen_scope(self):
        a.pack(self.base);before=(self.base/a.MANIFEST_NAME).read_bytes()
        added=self.base/'results/later.npz';np.savez_compressed(added,x=np.asarray([1.]))
        with self.assertRaisesRegex(ValueError,'Unregistered archival inputs'):a.pack(self.base)
        with self.assertRaisesRegex(ValueError,'Unregistered archival inputs'):a.verify(self.base)
        self.assertEqual((self.base/a.MANIFEST_NAME).read_bytes(),before)
        self.assertFalse(Path(str(added)+'.gz').exists())

    def test_absolute_traversal_manifest_duplicates_and_symlink_escapes_rejected(self):
        for value in ('/tmp/escape.npz','results/../escape.npz','./results/main/paths.npz'):
            with self.subTest(value=value),self.assertRaises(ValueError):a._path(self.base,value,True)
        outside=Path(self.temp.name)/'outside';outside.mkdir();(outside/'secret.npz').write_bytes(b'synthetic outside content')
        link=self.base/'results/escape.npz';link.symlink_to(outside/'secret.npz')
        with self.assertRaisesRegex(ValueError,'Symlink'):a.pack(self.base)
        self.assertFalse((outside/'secret.npz.gz').exists());link.unlink()
        link=self.base/'results/linked_directory';link.symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Symlink'):a.pack(self.base)
        link.unlink();a.pack(self.base)
        manifest=self.manifest();manifest['archives'].append(dict(manifest['archives'][0]))
        (self.base/a.MANIFEST_NAME).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'Duplicate'):a.verify(self.base)

    def test_restore_refuses_a_symlink_destination_even_if_content_matches(self):
        a.pack(self.base);outside=Path(self.temp.name)/'other.npz';outside.write_bytes(self.matrix.read_bytes())
        self.matrix.unlink();self.matrix.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'Symlink'):a.restore(self.base)
        self.assertTrue(self.matrix.is_symlink())

    def test_gitignore_uses_literal_large_json_paths_and_keeps_user_patterns(self):
        special=self.base/'results/main/report[1] #!.json';special.write_bytes(self.large.read_bytes())
        a.pack(self.base)
        pattern=a._ignore_pattern('results/main/report[1] #!.json')
        self.assertIn(pattern,self.ignore.read_text())
        self.assertIn('custom/cache/',self.ignore.read_text());self.assertIn('results/**/*.npy',self.ignore.read_text())
        binary=shutil.which('git')
        if binary:
            subprocess.run([binary,'init','-q',str(self.base)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            expected='results/main/report[1] #!.json'
            other='results/main/report1 #!.json'
            response=subprocess.run([binary,'-C',str(self.base),'check-ignore','--no-index','--stdin'],
                input=expected+'\n'+other+'\n',universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertEqual(response.returncode,0,response.stderr)
            self.assertEqual(response.stdout.splitlines(),[expected])


if __name__=='__main__':unittest.main()
