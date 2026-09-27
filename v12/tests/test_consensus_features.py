"""Synthetic features only; never evaluate B candidate performance."""
from copy import deepcopy
from datetime import date,timedelta
import unittest

import numpy as np

from v10_deep.data import ASSET_ORDER
from v10_deep.features import FEATURE_NAMES,F
from v12.consensus_features import build_consensus,RISK_ASSETS,INVALID
from v92_plus_live.policy import latest_features

ASSETS=list(ASSET_ORDER)
AI={code:i for i,code in enumerate(ASSETS)}
A,B,C=RISK_ASSETS[:3]


def fixture(n=280,missing=()):
    dates=[(date(2020,1,1)+timedelta(days=i)).isoformat() for i in range(n)]
    cube=np.zeros((n,len(ASSETS),len(FEATURE_NAMES)),dtype=np.float64)
    cube[:,:,F["close"]]=np.nan
    scores=np.full((2,n,len(ASSETS)),INVALID,dtype=np.float64)
    histories={}
    for ai,code in enumerate(ASSETS):
        rows=[]
        for i,d in enumerate(dates):
            if code==A and i in missing:
                continue
            j=len(rows)
            p=100.+j*.07+.7*np.sin(.21*j+.1*ai)
            rows.append((d,p,p,100.+j))
            cube[i,ai,F["close"]]=p
            if j>=269 and code!="511880":
                cube[i,ai,F["valid"]]=1.
                scores[:,i,ai]=[ai+1.,20.-ai]
        histories[code]=rows
    arrays=dict(features=cube,scores=scores,orders=np.argsort(-scores,axis=2,kind="stable").astype(np.int32),
                fear=np.arange(n,dtype=np.int32)%2)
    meta=dict(dates=dates,assets=ASSETS,feature_names=FEATURE_NAMES,
              score_names=["wls25_v20","wls20_smooth3"])
    return arrays,meta,histories


def lane(arrays,meta,names,kind):
    return arrays["scores"][meta["score_names"].index(names[kind])]


class ConsensusFeatureTests(unittest.TestCase):
    def test_preserves_every_original_array_lane_and_metadata_without_aliasing(self):
        arrays,meta,histories=fixture()
        original={k:v.copy() for k,v in arrays.items()}
        old_meta=deepcopy(meta)
        result,m,names=build_consensus(arrays,meta,histories)
        self.assertEqual(set(names),{"equal_score","equal_rank","wls25_smooth3"})
        self.assertEqual(result["scores"].shape[0],5)
        for key in ("features","fear"):
            np.testing.assert_array_equal(result[key],arrays[key])
        for key in ("scores","orders"):
            np.testing.assert_array_equal(result[key][:2],arrays[key])
        for key in arrays:
            np.testing.assert_array_equal(arrays[key],original[key])
            self.assertFalse(np.shares_memory(arrays[key],result[key]))
        self.assertEqual(meta,old_meta)
        self.assertEqual(m["score_names"][:2],meta["score_names"])
        self.assertTrue(m["v12_consensus_features"]["native_active_score_validity_mask_required"])

    def test_arithmetic_mean_requires_both_scores_and_current_own_maturity(self):
        arrays,meta,histories=fixture(missing=(270,))
        arrays["scores"][:,269,AI[A]]=[10.,30.]
        arrays["scores"][0,271,AI[A]]=np.nan
        arrays["scores"][1,272,AI[A]]=INVALID
        # Deliberately inconsistent supplied validity must not create a quote
        # on the missing date or waive the 270-own-observation requirement.
        arrays["features"][:,AI[A],F["valid"]]=1.
        arrays["features"][270,AI[A],F["close"]]=100.
        arrays["scores"][:,268,AI[A]]=[10.,30.]
        arrays["scores"][:,270,AI[A]]=[10.,30.]
        out,m,names=build_consensus(arrays,meta,histories)
        values=lane(out,m,names,"equal_score")
        self.assertEqual(values[269,AI[A]],20.)
        for i in (268,270,271,272):self.assertEqual(values[i,AI[A]],INVALID)
        self.assertEqual(lane(out,m,names,"equal_rank")[271,AI[A]],INVALID)

    def test_average_rank_denominator_and_ties_use_only_common_original_risk_members(self):
        arrays,meta,histories=fixture()
        day=275
        arrays["features"][day,:,F["valid"]]=0.
        for code in (A,B,C,"159928","511880"):
            arrays["features"][day,AI[code],F["valid"]]=1.
        arrays["scores"][1,day,[AI[A],AI[B],AI[C]]]=[1.,1.,3.]
        arrays["scores"][0,day,[AI[A],AI[B],AI[C]]]=[3.,2.,1.]
        arrays["scores"][:,day,[AI["159928"],AI["511880"]]]=1e20
        out,m,names=build_consensus(arrays,meta,histories)
        ranks=lane(out,m,names,"equal_rank")
        np.testing.assert_allclose(ranks[day,[AI[A],AI[B],AI[C]]],[.75,(.5+2/3)/2,(1+1/3)/2],rtol=0,atol=1e-16)
        self.assertEqual(ranks[day,AI["159928"]],INVALID)
        self.assertEqual(ranks[day,AI["511880"]],INVALID)
        # Remove B from one component. Both ranks now use the common N=2,
        # giving A/C equal consensus scores and original asset order on ties.
        arrays["scores"][0,day,AI[B]]=np.nan
        out,m,names=build_consensus(arrays,meta,histories)
        ranks=lane(out,m,names,"equal_rank")
        self.assertEqual(ranks[day,AI[A]],.75)
        self.assertEqual(ranks[day,AI[C]],.75)
        order=out["orders"][m["score_names"].index(names["equal_rank"]),day]
        self.assertEqual(list(order[:2]),[AI[A],AI[C]])

    def test_no_members_and_one_member_ranks_are_not_backfilled(self):
        arrays,meta,histories=fixture()
        arrays["features"][275:277,:,F["valid"]]=0.
        arrays["features"][276,AI[A],F["valid"]]=1.
        out,m,names=build_consensus(arrays,meta,histories)
        values=lane(out,m,names,"equal_rank")
        self.assertTrue(np.all(values[275]==INVALID))
        self.assertEqual(values[276,AI[A]],1.)
        self.assertEqual(np.count_nonzero(values[276]>-1e90),1)

    def test_wls25_raw_three_own_quote_mean_uses_pre_maturity_values_and_skips_missing_bars(self):
        arrays,meta,histories=fixture(missing=(271,))
        out,m,names=build_consensus(arrays,meta,histories)
        values=lane(out,m,names,"wls25_smooth3")
        rows=histories[A]
        p=[r[2] for r in rows]
        # Independent weighted regression for the very first mature point;
        # two raw component scores precede the original lane's valid mask.
        def independent(end):
            y=np.asarray(p[end-24:end+1]);x=np.arange(25,dtype=float);w=np.arange(1,26,dtype=float)
            fit=np.linalg.lstsq(np.column_stack((np.ones(25),x))*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
            r=np.asarray(p[end-19:end+1])/np.asarray(p[end-20:end])-1.
            return fit[1]/np.average(y,weights=w)*250/np.std(r)
        self.assertAlmostEqual(values[269,AI[A]],sum(independent(j) for j in (267,268,269))/3,places=7)
        self.assertEqual(values[271,AI[A]],INVALID)
        # Later checks use the separately verified live legacy implementation.
        j=len(rows)-1
        old=[(r[0],r[2],r[3]) for r in rows]
        components=[latest_features(old[:j+1-k],rows[j-k][0])["score"] for k in (0,1,2)]
        self.assertAlmostEqual(values[-1,AI[A]],sum(components)/3,places=12)
        for kind in names:
            selected=lane(out,m,names,kind)
            self.assertTrue(np.all(selected[:,AI["511880"]]==INVALID))
            self.assertTrue(np.all(selected[:,AI["159928"]]==INVALID))

    def test_future_score_and_price_changes_do_not_change_any_prior_consensus(self):
        arrays,meta,histories=fixture(295)
        out,m,names=build_consensus(arrays,meta,histories)
        cutoff=278
        altered=deepcopy(arrays)
        altered["scores"][:,cutoff+1:,:]*=-5
        altered["features"][cutoff+1:,:,F["valid"]]=0.
        changed=deepcopy(histories)
        for code,rows in changed.items():
            changed[code]=[r if r[0]<=meta["dates"][cutoff] else (r[0],r[1],r[2]*8,r[3]) for r in rows]
        future,unused,unused=build_consensus(altered,meta,changed)
        np.testing.assert_array_equal(out["scores"][:,:cutoff+1],future["scores"][:,:cutoff+1])
        np.testing.assert_array_equal(out["orders"][:,:cutoff+1],future["orders"][:,:cutoff+1])

    def test_input_lane_order_is_explicit_and_missing_original_member_is_rejected(self):
        arrays,meta,histories=fixture()
        first,m,names=build_consensus(arrays,meta,histories)
        arrays["scores"]=arrays["scores"][::-1].copy()
        arrays["orders"]=arrays["orders"][::-1].copy()
        meta["score_names"]=meta["score_names"][::-1]
        second,m2,names2=build_consensus(arrays,meta,histories)
        for kind in names:np.testing.assert_array_equal(lane(first,m,names,kind),lane(second,m2,names2,kind))
        del histories[A]
        with self.assertRaisesRegex(ValueError,"Missing frozen history"):
            build_consensus(arrays,meta,histories)


if __name__=="__main__":
    unittest.main()
