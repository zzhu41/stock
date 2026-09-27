"""Lossless matrix archives for the new stages; optional verified pruning."""
import argparse
import gzip
import hashlib
import json
from .data import BASE
from v10_h_close.matrix_archive import pack, verify, restore, NAMES

STAGES=('mechanisms','combinations','refinements','portfolios')


def archive(stage,prune=False):
    directory=BASE/'results'/stage
    receipt=pack(directory)
    verify(directory)
    for name,record in receipt.items():
        h=hashlib.sha256();size=0
        with gzip.open(str(directory/(name+'.gz')),'rb') as f:
            for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk);size+=len(chunk)
        if h.hexdigest()!=record['raw_sha256'] or size!=record['raw_bytes']:
            raise ValueError('Lossless archive round-trip failed: '+name)
    if prune:
        # Only these four newly generated copies; receipts and verified archives remain.
        for name in NAMES:(directory/name).unlink()
    return {'stage':stage,'raw_bytes':sum(r['raw_bytes'] for r in receipt.values()),
            'archive_bytes':sum(r['archive_bytes'] for r in receipt.values()),'round_trip_verified':True,'unpacked_copies_pruned':prune}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('pack','verify','restore'))
    p.add_argument('--stage',choices=STAGES+('all',),default='all');p.add_argument('--prune',action='store_true');a=p.parse_args()
    if a.prune and a.action!='pack':p.error('--prune is only valid when packing')
    for stage in STAGES if a.stage=='all' else (a.stage,):
        result=archive(stage,a.prune) if a.action=='pack' else (verify if a.action=='verify' else restore)(BASE/'results'/stage)
        print(json.dumps(result,ensure_ascii=False),flush=True)
