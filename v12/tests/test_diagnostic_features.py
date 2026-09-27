"""Feature-only diagnostics: synthetic neighbors and unchanged frozen centers."""
from copy import deepcopy
import csv
from datetime import date, timedelta
import json
from pathlib import Path
import unittest

import numpy as np

from v10_deep.features import FEATURE_NAMES,F
from v10_deep.schema import baseline
from v12.diagnostic_features import make_view,INVALID

ASSETS=("159915","510300","511880")
CASH=ASSETS.index("511880")
ROOT=Path(__file__).resolve().parents[2]


def fixture(n=330,missing=()):
    dates=[(date(2020,1,1)+timedelta(days=i)).isoformat() for i in range(n)]
    cube=np.zeros((n,len(ASSETS),len(FEATURE_NAMES)),dtype=np.float64)
    cube[:,:,F["close"]]=np.nan
    cube[:,:,F["ma180"]]=.123
    cube[:,:,F["ma250"]]=.456
    scores=np.full((2,n,len(ASSETS)),INVALID,dtype=np.float64)
    histories={}
    for ai,code in enumerate(ASSETS):
        rows=[]
        for i,day in enumerate(dates):
            if ai==0 and i in missing:
                continue
            j=len(rows)
            close=100+.07*j+.5*np.sin(j*.27+ai)
            rows.append((day,close,close,100.+j))
            cube[i,ai,F["close"]]=close
            if j>=269 and ai!=CASH:
                cube[i,ai,F["valid"]]=1.
                scores[0,i,ai]=1000.+i+ai
                scores[1,i,ai]=-1000.-i-ai
        histories[code]=rows
    arrays=dict(features=cube,scores=scores,
                orders=np.argsort(-scores,axis=2,kind="stable").astype(np.int32),
                fear=np.zeros(n,dtype=np.int32))
    meta=dict(dates=dates,assets=list(ASSETS),feature_names=list(FEATURE_NAMES),
              score_names=["wls25_v20","wls20_smooth3"],shape=list(cube.shape))
    config=dict(baseline(),id="parent-id",hash="parent-hash")
    return arrays,meta,histories,config


def independent_score(prices,window):
    """Weighted least squares solved through linear algebra, independent of conv."""
    p=np.asarray(prices,dtype=float)
    y=p[-window:]
    x=np.arange(window,dtype=float)
    w=np.arange(1,window+1,dtype=float)
    design=np.column_stack((np.ones(window),x))
    intercept,slope=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
    weighted_mean=np.average(y,weights=w)
    r=p[-20:]/p[-21:-1]-1
    sigma=np.std(r,ddof=0)
    return slope/weighted_mean*250/sigma if sigma>0 else 0.


class DiagnosticFeaturesTests(unittest.TestCase):
    def test_centers_reuse_original_scores_features_and_config_exactly(self):
        arrays,meta,histories,config=fixture()
        before={k:v.copy() for k,v in arrays.items()}
        for name,ma,w,s in (("wls25_v20","ma250",25,1),("wls20_smooth3","ma180",20,3)):
            center=dict(config,score=name,ma=ma)
            # Exact centers intentionally do not need to recompute history.
            view,m,c=make_view(arrays,meta,{},center,window=w,smooth=s,ma_window=int(ma[2:]))
            for key in arrays:
                np.testing.assert_array_equal(view[key],arrays[key])
                self.assertFalse(np.shares_memory(view[key],arrays[key]))
            self.assertEqual(c,center)
            self.assertEqual(m["score_names"],meta["score_names"])
            self.assertEqual(m["v12_diagnostic_features"]["score_center_reused"],name)
        for key in arrays:np.testing.assert_array_equal(arrays[key],before[key])

    def test_new_scores_are_trailing_own_quotes_and_do_not_smooth_control_sentinels(self):
        arrays,meta,histories,config=fixture(missing=(270,275))
        original={k:v.copy() for k,v in arrays.items()}
        for window,smooth in ((25,2),(20,2),(20,4),(16,3),(24,3),(30,1)):
            with self.subTest(window=window,smooth=smooth):
                view,m,c=make_view(arrays,meta,histories,config,window=window,smooth=smooth)
                self.assertEqual(view["scores"].shape[0],3)
                np.testing.assert_array_equal(view["scores"][:2],arrays["scores"])
                np.testing.assert_array_equal(view["orders"][:2],arrays["orders"])
                self.assertEqual(c["score"],m["score_names"][-1])
                self.assertNotIn("id",c)
                lane=view["scores"][-1,:,0]
                rows=histories[ASSETS[0]]
                axis={d:i for i,d in enumerate(meta["dates"])}
                p=[r[2] for r in rows]
                for j in (269,270,271,len(rows)-1):
                    i=axis[rows[j][0]]
                    expected=sum(independent_score(p[:j+1-k],window) for k in range(smooth))/smooth
                    self.assertAlmostEqual(lane[i],expected,places=7)
                    self.assertEqual(view["features"][i,0,F["valid"]],1.)
                self.assertTrue(np.all(lane[:269]==INVALID))
                self.assertEqual(lane[270],INVALID)
                self.assertEqual(lane[275],INVALID)
                self.assertTrue(np.all(view["scores"][-1,:,CASH]==INVALID))
                self.assertTrue(np.all(view["features"][:,CASH,F["valid"]]==0.))
        for key in arrays:np.testing.assert_array_equal(arrays[key],original[key])

    def test_WLS25_unmasked_math_retains_exact_legacy_sum_arithmetic(self):
        from v92_plus_live.policy import latest_features
        arrays,meta,histories,config=fixture()
        view,m,c=make_view(arrays,meta,histories,config,window=25,smooth=2)
        rows=histories[ASSETS[0]]
        live_rows=[(r[0],r[2],r[3]) for r in rows]
        for j in (270,271,329):
            expected=(latest_features(live_rows[:j+1],rows[j][0])["score"]+
                      latest_features(live_rows[:j],rows[j-1][0])["score"])/2
            self.assertEqual(view["scores"][-1,j,0],expected)

    def test_extra_math_warmup_masks_only_view_validity(self):
        arrays,meta,histories,config=fixture(340)
        view,m,c=make_view(arrays,meta,histories,config,window=300,smooth=4)
        self.assertEqual(m["v12_diagnostic_features"]["required_observations"],303)
        self.assertEqual(arrays["features"][300,0,F["valid"]],1.)
        self.assertEqual(view["features"][301,0,F["valid"]],0.)
        self.assertEqual(view["features"][302,0,F["valid"]],1.)
        self.assertEqual(view["scores"][-1,301,0],INVALID)
        self.assertNotEqual(view["scores"][-1,302,0],INVALID)

    def test_MA_override_only_borrows_benchmark_ma180_and_preserves_all_crash_distances(self):
        arrays,meta,histories,config=fixture()
        for ma in (144,216,200,300):
            with self.subTest(ma=ma):
                view,m,c=make_view(arrays,meta,histories,config,ma_window=ma)
                expected=arrays["features"].copy()
                expected[:,1,F["ma180"]]=view["features"][:,1,F["ma180"]]
                np.testing.assert_array_equal(view["features"],expected)
                np.testing.assert_array_equal(view["features"][:,:,F["ma250"]],arrays["features"][:,:,F["ma250"]])
                np.testing.assert_array_equal(view["scores"],arrays["scores"])
                self.assertEqual(c["ma"],"ma180")
                self.assertTrue(np.isnan(view["features"][ma-2,1,F["ma180"]]))
                p=[r[2] for r in histories["510300"]]
                self.assertAlmostEqual(view["features"][-1,1,F["ma180"]],p[-1]/np.mean(p[-ma:])-1,places=14)
                d=m["v12_diagnostic_features"]
                self.assertEqual(d["actual_ma_window"],ma)
                self.assertEqual(d["source_config_ma"],"ma250")
                self.assertEqual(d["ma_slot"],"ma180")
                self.assertEqual(d["ma_overridden_asset"],"510300")

    def test_future_perturbations_cannot_change_prefix_and_prefix_input_ignores_later_bad_prices(self):
        arrays,meta,histories,config=fixture(350,missing=(280,))
        first,m,c=make_view(arrays,meta,histories,config,window=16,smooth=4,ma_window=216)
        changed=deepcopy(histories)
        cutoff=310
        for code,rows in changed.items():
            changed[code]=[r if r[0]<=meta["dates"][cutoff] else (r[0],r[1],r[2]*7,r[3]) for r in rows]
        second,unused,unused=make_view(arrays,meta,changed,config,window=16,smooth=4,ma_window=216)
        np.testing.assert_array_equal(first["features"][:cutoff+1],second["features"][:cutoff+1])
        np.testing.assert_array_equal(first["scores"][:,:cutoff+1],second["scores"][:,:cutoff+1])
        prefix={key:value[:,:cutoff+1] if key in ("scores","orders") else value[:cutoff+1] for key,value in arrays.items()}
        prefix_meta=deepcopy(meta);prefix_meta["dates"]=meta["dates"][:cutoff+1]
        for code,rows in changed.items():
            changed[code]=[r if r[0]<=meta["dates"][cutoff] else (r[0],r[1],float("nan"),r[3]) for r in rows]
        shorter,unused,unused=make_view(prefix,prefix_meta,changed,config,window=16,smooth=4,ma_window=216)
        np.testing.assert_array_equal(first["scores"][:,:cutoff+1],shorter["scores"])

    def test_invalid_requests_fail_instead_of_silently_selecting_another_feature(self):
        arrays,meta,histories,config=fixture(10)
        for kw in ({"window":1},{"smooth":0},{"ma_window":1},{"window":20.5},{"smooth":True}):
            with self.assertRaises(ValueError):make_view(arrays,meta,histories,config,**kw)
        broken=deepcopy(histories);del broken["159915"]
        with self.assertRaisesRegex(ValueError,"Missing frozen"):
            make_view(arrays,meta,broken,config,window=16)

    def test_real_frozen_center_lanes_are_bit_exact_without_computing_new_neighbors(self):
        """No engine or performance evaluation, including for altered views."""
        frozen=json.loads((ROOT/"v10_deep/profiles.json").read_text())
        meta=json.loads((ROOT/"v10_deep/cache/features.json").read_text())
        selected=[meta["score_names"].index(n) for n in ("wls25_v20","wls20_smooth3")]
        with np.load(ROOT/"v10_deep/cache/features.npz",allow_pickle=False) as archive:
            arrays={key:np.ascontiguousarray(archive[key][selected] if key in ("scores","orders") else archive[key]) for key in archive.files}
        meta["score_names"]=["wls25_v20","wls20_smooth3"]
        for role in ("simple","growth"):
            config=frozen["variants"][role]["config"]
            view,unused,c=make_view(arrays,meta,{},config)
            for key in arrays:np.testing.assert_array_equal(arrays[key],view[key])
            self.assertEqual(c,config)


if __name__=="__main__":
    unittest.main()
