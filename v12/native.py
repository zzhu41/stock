"""V12 native fork; compilation and all generated files stay in V12 or an explicit temporary cache."""
import ctypes
from bisect import bisect_left,bisect_right
import fcntl
import hashlib
import json
import os
import subprocess
import numpy as np
from pathlib import Path
from v10_deep.data import START, END, ASSET_ORDER, dump
from v10_deep.features import FEATURE_NAMES
BASE = Path(__file__).resolve().parent
from .schema import native_header, encode


class Simulator:
    def __init__(self, arrays, meta, cache_dir=None):
        self.arrays,self.meta=arrays,meta
        T,A,S=len(meta['dates']),len(ASSET_ORDER),len(meta['score_names'])
        if list(meta['assets'])!=list(ASSET_ORDER) or meta['feature_names']!=FEATURE_NAMES:
            raise ValueError('Native schema requires the declared fixed asset and feature order')
        for name,shape,dtype in [('features',(T,A,len(FEATURE_NAMES)),np.float64),('scores',(S,T,A),np.float64),
                                  ('orders',(S,T,A),np.int32),('fear',(T,),np.int32)]:
            value=arrays[name]
            if value.shape!=shape or value.dtype!=dtype or not value.flags.c_contiguous:
                raise ValueError('Invalid native array layout: '+name)
        if arrays['orders'].min()<0 or arrays['orders'].max()>=A:raise ValueError('Invalid native ranking index')
        cache=Path(cache_dir) if cache_dir is not None else BASE/'cache';cache.mkdir(parents=True,exist_ok=True)
        source=(BASE/'native.cpp').read_bytes();header_text=native_header()
        fingerprint={'cpp':hashlib.sha256(source).hexdigest(),'header':hashlib.sha256(header_text.encode()).hexdigest(),
                     'flags':['-O3','-std=c++11','-shared','-fPIC','-fopenmp']}
        key=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()[:20]
        directory=cache/('native_'+key);directory.mkdir(exist_ok=True)
        so=directory/'native.so';receipt=directory/'build.json'
        with (cache/'compile.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not so.exists() or not receipt.exists() or json.loads(receipt.read_text())!=fingerprint:
                (directory/'schema.hpp').write_text(header_text);(directory/'native.cpp').write_bytes(source)
                temporary=directory/('native.%d.tmp.so'%os.getpid())
                subprocess.run(['g++']+fingerprint['flags']+['-I'+str(directory),str(directory/'native.cpp'),'-o',str(temporary)],check=True)
                os.replace(str(temporary),str(so));dump(receipt,fingerprint)
        self.build_fingerprint=fingerprint
        self.lib=ctypes.CDLL(str(so));self.lib.simulate.restype=ctypes.c_int
        self.lib.simulate.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]+[ctypes.c_int]*3+[ctypes.c_double,ctypes.c_int]+[ctypes.c_void_p]*3

    def run(self,configs,start=START,end=END,fee=.0001,workers=2,capture=True):
        if workers not in (1,2):raise ValueError('Research worker limit is one or two')
        dates=self.meta['dates'];lo=bisect_left(dates,start);hi=bisect_right(dates,end)-1
        if lo>hi or lo>=len(dates):raise ValueError('Empty native evaluation interval')
        encoded=np.ascontiguousarray([encode(c,self.meta) for c in configs],dtype=np.float64)
        count=len(configs);days=hi-lo+1
        returns=np.empty((count,days),dtype=np.float64) if capture else None
        holdings=np.empty((count,days),dtype=np.int32) if capture else None
        summary=np.empty((count,10),dtype=np.float64)
        def ptr(x):return x.ctypes.data if x is not None else None
        a=self.arrays
        code=self.lib.simulate(ptr(a['features']),ptr(a['scores']),ptr(a['orders']),ptr(a['fear']),len(dates),len(self.meta['assets']),
                               ptr(encoded),count,lo,hi,fee,workers,ptr(returns),ptr(holdings),ptr(summary))
        if code:raise RuntimeError('Native simulator rejected inputs: '+str(code))
        if not np.isfinite(summary).all() or (summary[:,0]<=0).any():raise ArithmeticError('Invalid research NAV')
        return dict(returns=returns,holdings=holdings,summary=summary,dates=dates[lo:hi+1])
