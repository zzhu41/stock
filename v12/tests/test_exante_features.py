"""C feature tests only; no strategy returns or candidate selection."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from v10_deep.data import inputs,sha
from v12.data import load_inputs
from v12.consensus_features import build_consensus,INVALID,RISK_ASSETS
from v12.exante_features import build_exante,view_for_config,MULTI_NAME
from v12.tests.test_consensus_features import fixture,ASSETS,AI,A,F

ROOT=Path(__file__).resolve().parents[2]


class ExanteFeatureTests(unittest.TestCase):
    def test_append_fixed_multi_preserves_A_B_features_scores_and_order_lanes(self):
        arrays,meta,histories=fixture(285)
        original={k:v.copy() for k,v in arrays.items()}
        old_meta=deepcopy(meta)
        b,bm,bnames=build_consensus(arrays,meta,histories)
        c,cm,cnames=build_exante(arrays,meta,histories)
        self.assertEqual(c["scores"].shape[0],6)
        self.assertEqual(cnames["multi_10_20_40"],MULTI_NAME)
        for name,value in bnames.items():self.assertEqual(cnames[name],value)
        np.testing.assert_array_equal(c["scores"][:5],b["scores"])
        np.testing.assert_array_equal(c["orders"][:5],b["orders"])
        np.testing.assert_array_equal(c["features"],b["features"])
        np.testing.assert_array_equal(c["fear"],b["fear"])
        for key in arrays:np.testing.assert_array_equal(arrays[key],original[key])
        self.assertEqual(meta,old_meta)
        for code in set(ASSETS)-set(RISK_ASSETS):
            self.assertTrue(np.all(c["scores"][-1,:,AI[code]]==INVALID))

    def test_multi_matches_independent_trailing_regressions_and_270_quote_gate(self):
        arrays,meta,histories=fixture(285,missing=(270,))
        out,m,names=build_exante(arrays,meta,histories)
        values=out["scores"][-1,:,AI[A]]
        p=np.asarray([r[2] for r in histories[A]])
        r=p[-20:]/p[-21:-1]-1.
        sigma=np.std(r)
        parts=[]
        for window in (10,20,40):
            x=np.arange(window,dtype=float);w=np.arange(1,window+1,dtype=float);y=p[-window:]
            fitted=np.linalg.lstsq(np.column_stack((np.ones(window),x))*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
            parts.append(fitted[1]/np.average(y,weights=w)*250/sigma)
        self.assertAlmostEqual(values[-1],np.mean(parts),places=7)
        self.assertTrue(np.all(values[:269]==INVALID))
        self.assertNotEqual(values[269],INVALID)
        self.assertEqual(values[270],INVALID)
        self.assertNotEqual(values[271],INVALID)

    def test_prior_context_uses_previous_actual_quote_and_never_changes_rank_scores(self):
        arrays,meta,histories=fixture(285,missing=(272,))
        out,m,names=build_exante(arrays,meta,histories)
        ai=AI[A]
        out["features"][:,ai,F["vol20"]]=np.arange(285)*.001
        out["features"][:,ai,F["vol60"]]=np.arange(285)*.002
        out["features"][272,ai,F["vol20"]]=999.
        out["features"][272,ai,F["vol60"]]=999.
        original=out["features"].copy()
        config=dict(score=MULTI_NAME,risk_context="prior20")
        prior20=view_for_config(out,m,config)
        prior60=view_for_config(out,m,dict(config,risk_context="prior60"))
        maximum=view_for_config(out,m,dict(config,risk_context="prior_max20_60"))
        self.assertEqual(prior20["features"][273,ai,F["vol20"]],.271)
        self.assertEqual(prior60["features"][273,ai,F["vol20"]],.542)
        self.assertEqual(maximum["features"][273,ai,F["vol20"]],.542)
        self.assertTrue(np.isnan(prior20["features"][272,ai,F["vol20"]]))
        self.assertEqual(prior20["features"][272,ai,F["valid"]],0.)
        self.assertTrue(np.isnan(prior20["features"][0,ai,F["vol20"]]))
        for view in (prior20,prior60,maximum):
            np.testing.assert_array_equal(view["scores"],out["scores"])
            np.testing.assert_array_equal(view["orders"],out["orders"])
            for name,index in F.items():
                if name not in ("vol20","valid"):
                    np.testing.assert_array_equal(view["features"][:,:,index],out["features"][:,:,index])
        np.testing.assert_array_equal(out["features"],original)
        self.assertEqual(config["risk_context"],"prior20")

    def test_current_25smooth3_validity_is_exactly_B_mask_and_invalid_scores_stay_out(self):
        arrays,meta,histories=fixture()
        out,m,names=build_exante(arrays,meta,histories)
        score=names["wls25_smooth3"]
        values=out["scores"][m["score_names"].index(score)]
        expected=out["features"].copy()
        expected[:,:,F["valid"]]*=np.isfinite(values)&(values>-1e90)
        actual=view_for_config(out,m,dict(score=score))
        np.testing.assert_array_equal(actual["features"],expected)
        for bad in (np.nan,INVALID):
            changed=deepcopy(out)
            changed["scores"][m["score_names"].index(score),275,AI[A]]=bad
            result=view_for_config(changed,m,dict(score=score))
            self.assertEqual(result["features"][275,AI[A],F["valid"]],0.)
        with self.assertRaisesRegex(ValueError,"Unknown active score"):
            view_for_config(out,m,dict(score="unknown"))
        with self.assertRaisesRegex(ValueError,"Unknown volatility"):
            view_for_config(out,m,dict(score=score,risk_context="future20"))

    def test_future_perturbations_leave_score_and_prior_views_unchanged_in_prefix(self):
        arrays,meta,histories=fixture(300,missing=(278,))
        original,m,names=build_exante(arrays,meta,histories)
        changed=deepcopy(histories)
        cutoff=285
        for code,rows in changed.items():
            changed[code]=[r if r[0]<=meta["dates"][cutoff] else (r[0],r[1],r[2]*5,r[3]) for r in rows]
        future,m2,unused=build_exante(arrays,meta,changed)
        np.testing.assert_array_equal(original["scores"][:,:cutoff+1],future["scores"][:,:cutoff+1])
        future["features"][cutoff+1:,:,F["vol20"]]=999.
        future["features"][cutoff+1:,:,F["vol60"]]=888.
        for context in ("current20","prior20","prior60","prior_max20_60"):
            c=dict(score=MULTI_NAME,risk_context=context)
            left=view_for_config(original,m,c)
            right=view_for_config(future,m2,c)
            np.testing.assert_array_equal(left["features"][:cutoff+1],right["features"][:cutoff+1])

    def test_real_current_controls_are_exact_and_multi_matches_pinned_V11_lane_without_rebuild(self):
        """Read-only feature equality; never instantiate a simulator."""
        frozen=json.loads((ROOT/"v11/profiles.json").read_text())
        candidates=[]
        for path in (ROOT/"v11/cache").glob("features-*.json"):
            old=json.loads(path.read_text())
            if old.get("fingerprints")==frozen["feature_fingerprints"]:
                candidates.append((path,old))
        if not candidates:
            self.skipTest("Pinned V11 feature cache is absent; never rebuild an old cache here")
        self.assertEqual(len(candidates),1)
        path,old=candidates[0]
        self.assertEqual(sha(path.with_suffix(".npz")),frozen["feature_array_sha256"])
        self.assertEqual(old["cache_array_sha256"],frozen["feature_array_sha256"])
        self.assertEqual(old["score_names"][2:],[s["name"] for s in frozen["score_specs"]])
        arrays,meta,unused=load_inputs()
        histories,unused,unused=inputs()
        out,m,names=build_exante(arrays,meta,histories)
        for score in ("wls25_v20","wls20_smooth3"):
            current=view_for_config(out,m,dict(score=score))
            np.testing.assert_array_equal(current["features"],arrays["features"])
        self.assertEqual(m["dates"],old["dates"])
        self.assertEqual(m["assets"],old["assets"])
        with np.load(path.with_suffix(".npz"),allow_pickle=False) as archive:
            index=old["score_names"].index("v11_wls_10_20_40_mean_s1")
            np.testing.assert_array_equal(out["scores"][-1],archive["scores"][index])
            np.testing.assert_array_equal(out["orders"][-1],archive["orders"][index])


if __name__=="__main__":
    unittest.main()
