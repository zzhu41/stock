"""Archive and restore exact research path bytes; never overwrite corruption."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile

from .data import BASE, dump, sha

MANIFEST=BASE/'path_archives.json'


def _path(relative):
    value=Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('Archive path must stay within v11')
    target=BASE/value
    if BASE.resolve() not in target.resolve().parents:
        raise ValueError('Archive destination escapes v11')
    return target


def pack():
    if MANIFEST.exists():raise ValueError('Path archive manifest already frozen')
    entries=[]
    for source in sorted((BASE/'results').rglob('*.npz')):
        target=source.with_suffix(source.suffix+'.gz')
        if target.exists():raise ValueError('Archive already exists: '+str(target))
        with source.open('rb') as incoming,target.open('xb') as output:
            with gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0) as compressed:
                shutil.copyfileobj(incoming,compressed)
        entries.append(dict(path=str(source.relative_to(BASE)),archive=str(target.relative_to(BASE)),
                            sha256=sha(source),archive_sha256=sha(target),bytes=source.stat().st_size))
    if not entries:raise ValueError('No completed path matrices to archive')
    dump(MANIFEST,dict(schema=1,archives=entries,format='gzip wrapping exact original NumPy archive bytes'))
    return len(entries)


def restore():
    manifest=json.loads(MANIFEST.read_text());count=0
    for row in manifest['archives']:
        target,archive=_path(row['path']),_path(row['archive'])
        if sha(archive)!=row['archive_sha256']:raise ValueError('Archive hash differs: '+row['archive'])
        if target.exists():
            if sha(target)!=row['sha256']:raise ValueError('Existing matrix is corrupt; refusing overwrite: '+row['path'])
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix='.restore-',dir=str(target.parent))
        try:
            with os.fdopen(descriptor,'wb') as output,gzip.open(str(archive),'rb') as incoming:
                shutil.copyfileobj(incoming,output)
            if Path(temporary).stat().st_size!=row['bytes'] or sha(temporary)!=row['sha256']:
                raise ValueError('Restored matrix hash/size differs: '+row['path'])
            # Hard link is atomic and refuses a concurrent destination overwrite.
            os.link(temporary,str(target));count+=1
        finally:
            Path(temporary).unlink()
    return count


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('pack','restore'))
    args=parser.parse_args();print(json.dumps(dict(command=args.command,files=pack() if args.command=='pack' else restore())))
