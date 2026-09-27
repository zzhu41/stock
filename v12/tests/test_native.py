"""Only synthetic flag1 cases and already-known real flag0 controls.

Both native compilers write exclusively into temporary directories. No search,
live data access, old cache writes, or real flag1 performance is permitted here.
"""
import ctypes
from bisect import bisect_left, bisect_right
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from v10_deep import schema as frozen_schema
from v10_deep.data import sha
from v10_deep.features import F
from v10_deep.reference import run_reference as frozen_reference
from v10_deep.tests.test_native import inputs, put, prices, finish, config, AI, A, B, X, CASH
from v12.native import Simulator
from v12.reference import run_reference
from v12.schema import encode, validate_locked_panic_exit

ROOT = Path(__file__).resolve().parents[2]


class IsolatedFrozenNative:
    """Compile exact original bytes/header in tmp; never touch its cache."""
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir()
        (directory/"native.cpp").write_bytes((ROOT/"v10_deep/native.cpp").read_bytes())
        (directory/"schema.hpp").write_text(frozen_schema.native_header())
        subprocess.run(["g++", "-O3", "-std=c++11", "-shared", "-fPIC", "-fopenmp",
                        "-I"+str(directory), str(directory/"native.cpp"), "-o", str(directory/"native.so")],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.lib = ctypes.CDLL(str(directory/"native.so"))
        self.lib.simulate.restype = ctypes.c_int
        self.lib.simulate.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]+[ctypes.c_int]*3+[ctypes.c_double,ctypes.c_int]+[ctypes.c_void_p]*3

    def run(self, arrays, meta, configs, fee=.0001, start=None, end=None):
        n,t,a = len(configs),len(meta["dates"]),len(meta["assets"])
        lo = bisect_left(meta["dates"],start) if start is not None else 0
        hi = bisect_right(meta["dates"],end)-1 if end is not None else t-1
        days = hi-lo+1
        encoded = np.ascontiguousarray([frozen_schema.encode(c,meta) for c in configs],dtype=np.float64)
        returns = np.empty((n,days),dtype=np.float64)
        holdings = np.empty((n,days),dtype=np.int32)
        summary = np.empty((n,10),dtype=np.float64)
        p = lambda values: values.ctypes.data
        rc = self.lib.simulate(p(arrays["features"]),p(arrays["scores"]),p(arrays["orders"]),p(arrays["fear"]),
             t,a,p(encoded),n,lo,hi,fee,1,p(returns),p(holdings),p(summary))
        if rc:
            raise AssertionError("Original native rejected test inputs")
        return dict(returns=returns,holdings=holdings,summary=summary)


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cache = Path(cls.tmp.name)/"v12_cache"
        cls.old = IsolatedFrozenNative(Path(cls.tmp.name)/"frozen")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def compare(self, arrays, meta, configs, fee=.0001):
        arrays = finish(arrays)
        result = Simulator(arrays,meta,cache_dir=self.cache).run(configs,
            start=meta["dates"][0],end=meta["dates"][-1],fee=fee,workers=1)
        references = []
        for i,c in enumerate(configs):
            reference = run_reference(arrays,meta,c,fee=fee)
            np.testing.assert_array_equal(result["holdings"][i],reference["holdings"])
            np.testing.assert_allclose(result["returns"][i],reference["returns"],rtol=0,atol=1e-14)
            np.testing.assert_allclose(result["summary"][i],reference["summary"],rtol=1e-12,atol=1e-12)
            references.append(reference)
        return result,references

    @staticmethod
    def crash_inputs(lag=0):
        arrays,meta = inputs(9)
        arrays["features"][1,AI[B],F["mom5"]] = -.1
        arrays["features"][1,AI[B],F["ma250"]] = -.3
        prices(arrays,B,[100.,100.,94.,90.,91.,92.,93.,94.,95.])
        # Competing crash on the panic date must not overwrite a cash exit.
        arrays["features"][2,AI[X],F["mom5"]] = -.1
        arrays["features"][2,AI[X],F["ma250"]] = -.3
        return arrays,meta,config(lag=lag,crash_mask=1,crash_lock=5)

    def test_schema_appends_only_one_flag_and_rejects_non_boolean_values(self):
        unused,meta = inputs()
        candidate = config()
        self.assertEqual(encode(candidate,meta),frozen_schema.encode(candidate,meta)+[0.])
        self.assertEqual(encode(dict(candidate,locked_panic_exit=1),meta)[-1],1.)
        for value in (-1,2,.5,None,"1",float("nan")):
            with self.assertRaises(ValueError):
                validate_locked_panic_exit(dict(candidate,locked_panic_exit=value))

    def test_active_lock_panic_exits_to_cash_and_cannot_be_replaced_by_new_crash(self):
        arrays,meta,c = self.crash_inputs()
        out,refs = self.compare(arrays,meta,[dict(c,locked_panic_exit=0),dict(c,locked_panic_exit=1)])
        self.assertEqual(list(out["holdings"][0,:4]),[AI[A],AI[B],AI[B],AI[B]])
        self.assertEqual(list(out["holdings"][1,:4]),[AI[A],AI[B],AI[CASH],AI[A]])
        self.assertTrue(refs[1]["trace"][2]["locked_panic_exit_requested"])
        self.assertFalse(refs[1]["trace"][2]["crash_requested"])
        self.assertEqual(refs[1]["trace"][2]["lock_until"],-1)
        self.assertEqual(out["summary"][1,7],1.)
        self.assertEqual(out["returns"][1,0],0.)
        self.assertAlmostEqual(np.prod(1+out["returns"][1,:3]),.94*.9998**2)

    def test_missing_cash_quote_preserves_lock_and_rechecks_next_daily_intent(self):
        arrays,meta,c = self.crash_inputs()
        prices(arrays,CASH,[100.,100.,np.nan,100.,100.,100.,100.,100.,100.])
        out,refs = self.compare(arrays,meta,[dict(c,locked_panic_exit=1)])
        self.assertEqual(out["holdings"][0,2],AI[B])
        self.assertFalse(refs[0]["trace"][2]["filled"])
        self.assertEqual(refs[0]["trace"][2]["lock_until"],6)
        self.assertEqual(out["holdings"][0,3],AI[CASH])
        self.assertEqual(refs[0]["trace"][3]["lock_until"],-1)
        self.assertEqual(out["summary"][0,5],1.)
        self.assertAlmostEqual(np.prod(1+out["returns"][0,:4]),.90*.9998**2)
        # No pending cash order is invented: if the shock is over when a
        # quote returns, the existing lock still governs the new daily intent.
        prices(arrays,B,[100.,100.,94.,94.,95.,96.,97.,98.,99.])
        out,refs = self.compare(arrays,meta,[dict(c,locked_panic_exit=1)])
        self.assertEqual(out["holdings"][0,3],AI[B])
        self.assertEqual(refs[0]["trace"][3]["lock_until"],6)
        self.assertFalse(refs[0]["trace"][3]["locked_panic_exit_requested"])

    def test_lag1_entry_date_shock_is_not_a_post_entry_stop(self):
        arrays,meta,c = self.crash_inputs(lag=1)
        prices(arrays,B,[100.,90.,81.,76.14,77.,78.,79.,80.,81.])
        out,refs = self.compare(arrays,meta,[dict(c,locked_panic_exit=1)])
        self.assertEqual(list(out["holdings"][0,:5]),[-1,AI[A],AI[B],AI[B],AI[CASH]])
        self.assertTrue(refs[0]["trace"][3]["panic"])
        self.assertFalse(refs[0]["trace"][3]["locked_panic_exit_requested"])
        self.assertEqual(refs[0]["trace"][3]["lock_until"],7)
        self.assertTrue(refs[0]["trace"][4]["locked_panic_exit_requested"])
        self.assertEqual(refs[0]["trace"][4]["lock_until"],-1)

    def test_missing_current_holding_cannot_sell_or_clear_lagged_lock(self):
        arrays,meta,c = self.crash_inputs(lag=1)
        prices(arrays,B,[100.,90.,81.,76.14,np.nan,70.,71.,72.,73.])
        out,refs = self.compare(arrays,meta,[dict(c,locked_panic_exit=1)])
        self.assertEqual(out["holdings"][0,4],AI[B])
        self.assertTrue(refs[0]["trace"][4]["locked_panic_exit_requested"])
        self.assertFalse(refs[0]["trace"][4]["filled"])
        self.assertEqual(refs[0]["trace"][4]["lock_until"],7)
        self.assertEqual(out["returns"][0,4],0.)
        self.assertAlmostEqual(out["returns"][0,5],70/76.14-1)
        self.assertEqual(out["holdings"][0,6],AI[CASH])
        self.assertEqual(refs[0]["trace"][6]["lock_until"],-1)

    def test_no_panic_keeps_original_lock_and_normal_panic_rules_stay_unchanged(self):
        arrays,meta,c = self.crash_inputs()
        prices(arrays,B,[100.]*9)
        put(arrays,B,"mom20",-.1)
        self.compare(arrays,meta,[dict(c,locked_panic_exit=0),dict(c,locked_panic_exit=1)])
        finished=finish(arrays)
        old=self.old.run(finished,meta,[c])
        new=Simulator(finished,meta,cache_dir=self.cache).run([dict(c,locked_panic_exit=1)],
            start=meta["dates"][0],end=meta["dates"][-1],workers=1)
        np.testing.assert_array_equal(old["holdings"],new["holdings"])
        arrays,meta=inputs(5)
        prices(arrays,A,[100.,94.,95.,96.,97.])
        variants=[config(locked_panic_exit=flag) for flag in (0,1)]
        out,refs=self.compare(arrays,meta,variants)
        np.testing.assert_array_equal(out["holdings"][0],out["holdings"][1])
        self.assertEqual(out["holdings"][1,1],AI[B])
        self.assertTrue(all(not r["locked_panic_exit_requested"] for r in refs[1]["trace"]))

    def test_existing_trailing_panic_is_reused_without_new_threshold(self):
        arrays,meta,c=self.crash_inputs()
        c.update(panic=0.,trail_stop=.05,locked_panic_exit=1)
        out,refs=self.compare(arrays,meta,[c])
        self.assertEqual(out["holdings"][0,2],AI[CASH])
        self.assertTrue(refs[0]["trace"][2]["panic"])

    def test_seeded_synthetic_variants_flag0_exact_and_flag1_oracle_future_prefix(self):
        rng=np.random.default_rng(812)
        arrays,meta=inputs(48)
        for code in (A,B,X):
            prices(arrays,code,100*np.exp(np.cumsum(rng.normal(0,.055,48))))
            for name in ("mom5","mom20","mom60","ma250"):
                put(arrays,code,name,rng.normal(-.05,.25,48))
            put(arrays,code,"vol20",rng.uniform(.01,.07,48))
            put(arrays,code,"volume_ratio",rng.uniform(.5,3.5,48))
            arrays["scores"][:,:,AI[code]]=rng.normal(0,20,(2,48))
        arrays["features"][[12,31],AI[B],F["close"]]=np.nan
        arrays["features"][[12,31],AI[B],F["valid"]]=0.
        arrays["fear"]=rng.integers(0,2,48,dtype=np.int32)
        variants=[config(lag=lag,crash_mask=mask,crash_lock=lock,min_hold=2,switch_confirm=confirm,
                         panic_mode=mode,panic=panic,locked_panic_exit=flag)
                  for lag,mask,lock,confirm,mode,panic in
                  ((0,7,5,1,"fixed",.04),(1,7,5,2,"fixed",.04),(0,5,3,2,"volatility",1.5),
                   (1,1,5,1,"volatility",2.),(0,0,5,1,"fixed",.04),(1,4,1,2,"fixed",.06))
                  for flag in (0,1)]
        out,unused=self.compare(arrays,meta,variants,fee=.0011)
        finished=finish(arrays)
        old=self.old.run(finished,meta,variants[::2],fee=.0011)
        for field in ("returns","holdings","summary"):
            np.testing.assert_array_equal(out[field][::2],old[field])
        changed=deepcopy(finished)
        changed["features"][30:,:,F["close"]]*=7
        changed["scores"][:,30:,:]*=-3
        future=Simulator(finish(changed),meta,cache_dir=self.cache).run(variants,
            start=meta["dates"][0],end=meta["dates"][29],fee=.0011,workers=1)
        for field in ("returns","holdings"):
            np.testing.assert_array_equal(out[field][:,:30],future[field])

    def test_real_frozen_H_v92_simple_only_flag0_match_original_native_every_day(self):
        """Explicitly never evaluate real histories with the new flag enabled."""
        from v10_deep.cli import load_profile
        profiles=json.loads((ROOT/"v10_deep/profiles.json").read_text())
        metadata_path=ROOT/"v10_deep/cache/features.json"
        arrays_path=ROOT/"v10_deep/cache/features.npz"
        self.assertEqual(sha(metadata_path),profiles["feature_metadata_sha256"])
        self.assertEqual(sha(arrays_path),profiles["feature_cache_sha256"])
        meta=json.loads(metadata_path.read_text())
        with np.load(arrays_path,allow_pickle=False) as archive:
            arrays={k:archive[k] for k in ("features","scores","orders","fear")}
        configs=[dict(load_profile(name)["config"],lag=lag,locked_panic_exit=0)
                 for name in ("growth","v92","simple") for lag in (0,1)]
        self.assertTrue(all(c["locked_panic_exit"]==0 for c in configs))
        start,end="2014-01-02","2026-09-24"
        for fee in (.0001,.0011):
            old=self.old.run(arrays,meta,configs,fee,start,end)
            new=Simulator(arrays,meta,cache_dir=self.cache).run(configs,
                start=start,end=end,fee=fee,workers=1)
            self.assertEqual(len(new["dates"]),3097)
            for field in ("returns","holdings","summary"):
                np.testing.assert_array_equal(new[field],old[field])
            for i,c in enumerate(configs):
                expected=frozen_reference(arrays,meta,c,start=start,end=end,fee=fee)
                actual=run_reference(arrays,meta,c,start=start,end=end,fee=fee)
                np.testing.assert_array_equal(actual["returns"],expected["returns"])
                np.testing.assert_array_equal(actual["holdings"],expected["holdings"])
                np.testing.assert_array_equal(actual["summary"],expected["summary"])
                np.testing.assert_array_equal(new["returns"][i],actual["returns"])
                np.testing.assert_array_equal(new["holdings"][i],actual["holdings"])


if __name__ == "__main__":
    unittest.main()
