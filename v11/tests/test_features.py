"""Mathematical and causal feature regressions, with no candidate performance runs."""
from copy import deepcopy
from datetime import date, timedelta
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v11 import features as f

F = {name: i for i, name in enumerate(f.FEATURE_NAMES)}


def spec(name="new", estimator="wls", windows=(20,), aggregate="mean", smooth=3):
    return dict(name=name, estimator=estimator, windows=list(windows), aggregate=aggregate, smooth=smooth)


def bundle(days=325, missing=None, change_after=None, extra_drift=0.):
    missing = missing or {}
    dates = [(date(2010, 1, 1) + timedelta(days=i)).isoformat() for i in range(days)]
    assets = list(f.ASSET_ORDER)
    cube = np.full((days, len(assets), len(f.FEATURE_NAMES)), np.nan)
    cube[:, :, F["valid"]] = 0.
    scores = np.full((2, days, len(assets)), f.INVALID)
    histories = {}
    for ai, code in enumerate(assets):
        rows = []
        for day, date_string in enumerate(dates):
            if day in missing.get(code, set()):
                continue
            value = 50 * math.exp(.0004 * (ai + 1) * day + .018 * math.sin(.37 * day + ai * .21))
            if code not in f.ORIGINAL_RISK and code != f.CASH:
                value *= math.exp(extra_drift * day)
            if change_after is not None and day > change_after:
                value *= 1 + .05 * (day - change_after)
            rows.append((date_string, value, value, 1000. + day))
            own = len(rows)
            cube[day, ai, F["close"]] = value
            cube[day, ai, F["valid"]] = float(own >= 270 and code != f.CASH)
            cube[day, ai, F["vol20"]] = .01 + day * .00001 + ai * .000001
            cube[day, ai, F["vol60"]] = .02 - day * .00001 + ai * .000001
            if cube[day, ai, F["valid"]]:
                scores[:, day, ai] = [float(ai + 1), float(100 - ai)]
        histories[code] = rows
    arrays = dict(features=np.ascontiguousarray(cube), scores=np.ascontiguousarray(scores),
                  orders=np.argsort(-scores, axis=2, kind="stable").astype(np.int32),
                  fear=np.zeros(days, dtype=np.int32))
    meta = dict(dates=dates, assets=assets, feature_names=list(f.FEATURE_NAMES),
                score_names=["frozen_h", "frozen_v92"], fingerprints={"synthetic": "pinned"}, shape=list(cube.shape))
    return dict(arrays=arrays, meta=meta, histories=histories, profiles={}, fingerprints={"synthetic": "pinned"})


def scalar_score(prices, estimator, window):
    returns = [prices[i] / prices[i - 1] - 1 for i in range(len(prices) - 20, len(prices))]
    mean = sum(returns) / 20
    sigma = math.sqrt(sum((r - mean) ** 2 for r in returns) / 20)
    if estimator == "logmom":
        trend = math.log(prices[-1] / prices[-window - 1]) * 250 / window
    else:
        y = prices[-window:] if estimator == "wls" else [math.log(v) for v in prices[-window:]]
        weights = list(range(1, window + 1)) if estimator in ("wls", "logwls") else [1] * window
        mx = sum(w * i for i, w in enumerate(weights)) / sum(weights)
        my = sum(w * v for w, v in zip(weights, y)) / sum(weights)
        slope = sum(w * (i - mx) * (v - my) for i, (w, v) in enumerate(zip(weights, y))) / sum(w * (i - mx) ** 2 for i, w in enumerate(weights))
        trend = slope * 250 / my if estimator == "wls" else slope * 250
    return trend / sigma if sigma else 0.


class MathematicalFeatureTests(unittest.TestCase):
    def test_linear_and_log_momentum_formulas_match_independent_scalar_arithmetic(self):
        prices = [math.exp(.003 * i + .02 * math.sin(.61 * i)) for i in range(100)]
        for estimator in ("wls", "logwls", "olslog", "logmom"):
            for window in (20, 30, 61):
                with self.subTest(estimator=estimator, window=window):
                    values = f.score_series(prices, estimator, window)
                    for endpoint in (70, 99):
                        expected = scalar_score(prices[:endpoint + 1], estimator, window)
                        np.testing.assert_allclose(values[endpoint], expected, rtol=1e-11, atol=1e-10)
                    start = window if estimator == "logmom" else window - 1
                    self.assertTrue(np.isnan(values[:max(start, 19)]).all())

    def test_robust_slopes_recover_exact_line_and_reduce_one_outlier_influence(self):
        clean = np.asarray([.7 + .003 * i for i in range(45)])
        for estimator in (f._huber_slopes, f._theilsen_slopes):
            np.testing.assert_allclose(estimator(clean, 30), .003, rtol=1e-11, atol=1e-12)
        contaminated = clean.copy()
        contaminated[-1] += 1.5
        unused, ordinary = f._linear(contaminated, 30, False)
        for estimator in (f._huber_slopes, f._theilsen_slopes):
            robust = estimator(contaminated, 30)[-1]
            self.assertLess(abs(robust - .003), abs(ordinary[-1] - .003) / 5)
        self.assertEqual((f.HUBER_DELTA, f.HUBER_ITERATIONS), (1.345, 8))

    def test_short_and_zero_variance_sequences_do_not_fabricate_scores(self):
        for estimator in f.ESTIMATORS:
            short = f.score_series(np.ones(7), estimator, 30)
            self.assertTrue(np.isnan(short).all())
            flat = f.score_series(np.ones(50), estimator, 30)
            self.assertEqual(flat[-1], 0.)
        for values in ([1., 0.], [1., float("nan")]):
            with self.assertRaises(ValueError):
                f.score_series(values, "wls", 20)

    def test_average_percentile_ties_missing_and_singleton(self):
        values = np.asarray([[1., 2., 2., 999.], [np.nan, 3., f.INVALID, 999.]])
        eligible = np.asarray([[True, True, True, False], [True, True, True, False]])
        actual = f.percentile_ranks(values, eligible)
        np.testing.assert_allclose(actual[0, :3], [1/3, 5/6, 5/6])
        self.assertEqual(actual[1, 1], 1.)
        self.assertTrue(np.isnan(actual[:, 3]).all())
        self.assertTrue(np.isnan(actual[1, [0, 2]]).all())

    def test_required_counts_handle_logmom_extra_price_and_mature_rank_smoothing(self):
        self.assertEqual(f.required_observations(spec(windows=(293,), smooth=3)), 295)
        self.assertEqual(f.required_observations(spec(estimator="logmom", windows=(293,), smooth=3)), 296)
        self.assertEqual(f.required_observations(spec(aggregate="rank", smooth=3)), 272)
        self.assertEqual(f.required_observations(spec(aggregate="rank", smooth=5)), 274)
        self.assertEqual(f.required_observations(spec(aggregate="rank", estimator="logmom", windows=(293,), smooth=3)), 296)

    def test_spec_validation_forbids_hidden_weights_and_name_redefinition(self):
        invalid = [spec(windows=(20, 20)), spec(smooth=0), spec(estimator="magic"), spec(windows=(True,))]
        for value in invalid:
            with self.assertRaises(ValueError):
                f.canonical_specs([value])
        with self.assertRaises(ValueError):
            f.canonical_specs([spec(), spec(windows=(30,))])
        self.assertEqual(f.canonical_specs([spec(windows=(60,20))])[0]["windows"], [20,60])


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        cache = Path(f.BASE) / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test-features-", dir=str(cache))
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.base_patch = patch.object(f, "BASE", self.base)
        self.base_patch.start()
        self.addCleanup(self.base_patch.stop)

    def build(self, source, specs, **kwargs):
        with patch.object(f, "load_frozen", return_value=source):
            return f.build(specs, **kwargs)

    def test_frozen_arrays_preserved_and_own_observation_smoothing_skips_missing_bars(self):
        source = bundle(missing={"159915": {301, 305}})
        before = {key: value.copy() for key, value in source["arrays"].items()}
        arrays, meta = self.build(source, [spec()])
        for key in ("features", "fear"):
            np.testing.assert_array_equal(arrays[key], before[key])
        for key in ("scores", "orders"):
            np.testing.assert_array_equal(arrays[key][:2], before[key])
        for key, original in before.items():
            np.testing.assert_array_equal(source["arrays"][key], original)
        own = source["histories"]["159915"]
        ai = meta["assets"].index("159915")
        expected = np.mean(f.score_series([row[2] for row in own], "wls", 20)[-3:])
        self.assertAlmostEqual(arrays["scores"][-1, -1, ai], expected, places=11)
        self.assertEqual(arrays["scores"][-1, 305, ai], f.INVALID)
        self.assertEqual(arrays["observations"][-1, ai], len(own))
        self.assertEqual(meta["score_names"][:2], source["meta"]["score_names"])
        for key in ("features", "scores", "orders", "fear"):
            self.assertTrue(arrays[key].flags.c_contiguous)
        self.assertEqual(arrays["orders"].dtype, np.int32)
        np.testing.assert_array_equal(arrays["orders"][-1], np.argsort(-arrays["scores"][-1],axis=1,kind="stable"))

    def test_rank_ignores_all_extra_assets_and_needs_full_mature_smoothing(self):
        specifications = [spec(aggregate="rank", windows=(20,40))]
        arrays, meta = self.build(bundle(), specifications)
        changed, unused = self.build(bundle(extra_drift=.08), specifications)
        np.testing.assert_array_equal(arrays["scores"][-1], changed["scores"][-1])
        code = meta["assets"].index("159915")
        self.assertTrue(np.all(arrays["scores"][-1, :271, code] == f.INVALID))
        self.assertTrue(np.isfinite(arrays["scores"][-1, 271, code]))
        self.assertGreater(arrays["scores"][-1, 271, code], f.INVALID_CUTOFF)
        for code in set(meta["assets"]) - set(f.ORIGINAL_RISK):
            self.assertTrue(np.all(arrays["scores"][-1, :, meta["assets"].index(code)] == f.INVALID))

    def test_long_window_validity_prevents_native_neutral_fallback(self):
        definitions = [spec("regression", windows=(293,),smooth=3), spec("momentum",estimator="logmom",windows=(293,),smooth=3)]
        arrays, meta = self.build(bundle(), definitions)
        asset = meta["assets"].index("159915")
        for name, required in (("regression",295),("momentum",296)):
            index = meta["score_names"].index(name)
            self.assertEqual(meta["required_observations_by_score"][name], required)
            self.assertTrue(np.all(arrays["scores"][index,:required-1,asset] == f.INVALID))
            self.assertGreater(arrays["scores"][index,required-1,asset],f.INVALID_CUTOFF)
            view = f.risk_view(arrays,meta,"current20",score=name)
            self.assertEqual(view["features"][required-2,asset,F["valid"]],0)
            self.assertEqual(view["features"][required-1,asset,F["valid"]],1)
        damaged = {key:value.copy() for key,value in arrays.items()}
        index = meta["score_names"].index("regression")
        damaged["scores"][index,-1,asset] = np.nan
        self.assertEqual(f.risk_view(damaged,meta,"current20",score="regression")["features"][-1,asset,F["valid"]],0)

    def test_initial_calendar_rank_gap_is_masked_even_if_observation_count_is_mature(self):
        source = bundle()
        source["meta"]["dates"] = source["meta"]["dates"][-20:]
        for key in ("features","fear"):
            source["arrays"][key] = source["arrays"][key][-20:]
        for key in ("scores","orders"):
            source["arrays"][key] = source["arrays"][key][:,-20:]
        arrays,meta = self.build(source,[spec(aggregate="rank")])
        asset = meta["assets"].index("159915")
        self.assertGreater(arrays["observations"][0,asset],272)
        self.assertEqual(arrays["scores"][-1,0,asset],f.INVALID)
        view = f.risk_view(arrays,meta,"current20",score="new")
        self.assertEqual(view["features"][0,asset,F["valid"]],0)
        self.assertEqual(view["features"][2,asset,F["valid"]],1)

    def test_future_data_cannot_change_any_estimator_rank_or_prior_risk_prefix(self):
        definitions = [spec(name=estimator+aggregation,estimator=estimator,windows=(20,40),aggregate=aggregation)
                       for estimator in f.ESTIMATORS for aggregation in ("mean","rank")]
        left,meta = self.build(bundle(),definitions)
        right,right_meta = self.build(bundle(change_after=300),definitions)
        np.testing.assert_array_equal(left["scores"][:,:301],right["scores"][:,:301])
        np.testing.assert_array_equal(left["orders"][:,:301],right["orders"][:,:301])
        for context in ("current20","prior20","prior60","prior_max20_60"):
            np.testing.assert_array_equal(f.risk_view(left,meta,context,score="huberlogrank")["features"][:301],
                                          f.risk_view(right,right_meta,context,score="huberlogrank")["features"][:301])
        self.assertNotEqual(meta["cache_paths"],right_meta["cache_paths"])

    def test_previous_sigma_uses_previous_own_quote_and_never_mutates_controls(self):
        source=bundle(missing={"159915":{303}})
        arrays,meta=self.build(source,[])
        before=arrays["features"].copy();asset=meta["assets"].index("159915")
        current=f.risk_view(arrays,meta,"current20",score="frozen_h")
        np.testing.assert_array_equal(current["features"],before)
        for context,column in (("prior20","vol20"),("prior60","vol60")):
            view=f.risk_view(arrays,meta,context,score="frozen_h")
            self.assertEqual(view["features"][304,asset,F["vol20"]],before[302,asset,F[column]])
            self.assertTrue(np.isnan(view["features"][303,asset,F["vol20"]]))
            self.assertIs(view["scores"],arrays["scores"])
            self.assertIs(view["orders"],arrays["orders"])
        view=f.risk_view(arrays,meta,"prior_max20_60",score="frozen_h")
        self.assertEqual(view["features"][304,asset,F["vol20"]],max(before[302,asset,F["vol20"]],before[302,asset,F["vol60"]]))
        np.testing.assert_array_equal(arrays["features"],before)

    def test_cache_hit_source_spec_and_content_invalidation_and_integrity(self):
        source=bundle();definition=[spec()]
        arrays,meta=self.build(source,definition)
        with patch.object(f,"_compute",side_effect=AssertionError("Unexpected cache rebuild")):
            cached,cached_meta=self.build(source,definition)
        np.testing.assert_array_equal(cached["scores"],arrays["scores"])
        self.assertEqual(cached_meta["cache_paths"],meta["cache_paths"])
        changed_spec,other=self.build(source,[spec(windows=(30,))])
        self.assertNotEqual(other["cache_paths"],meta["cache_paths"])
        original_sha=f._sha
        def changed_source(path):
            return "0"*64 if Path(path)==Path(f.__file__) else original_sha(path)
        with patch.object(f,"_sha",side_effect=changed_source):
            unused,modified=self.build(source,definition)
        self.assertNotEqual(modified["cache_paths"],meta["cache_paths"])
        path=Path(meta["cache_paths"]["arrays"])
        path.write_bytes(path.read_bytes()+b'corruption')
        with self.assertRaisesRegex(ValueError,"integrity"):
            self.build(source,definition)
        rebuilt,unused=self.build(source,definition,force=True)
        np.testing.assert_array_equal(rebuilt["scores"],arrays["scores"])
        with self.assertRaisesRegex(ValueError,"overwrite"):
            self.build(source,[spec(name="frozen_h")])

    def test_build_does_not_change_mutable_root_strategy(self):
        import strategy
        before={key:deepcopy(value) for key,value in vars(strategy).items() if key.isupper() or key=="_state_bull"}
        self.build(bundle(),[spec(estimator="huberlog")])
        self.assertEqual(before,{key:value for key,value in vars(strategy).items() if key.isupper() or key=="_state_bull"})

    def test_real_frozen_h_and_v92_score_prefixes_are_preserved_without_running_returns(self):
        source=f.load_frozen()
        arrays,meta=self.build(source,[spec(name="v11_fidelity_wls20_smooth3")])
        for key in ("features","fear"):
            np.testing.assert_array_equal(arrays[key],source["arrays"][key])
        for key in ("scores","orders"):
            np.testing.assert_array_equal(arrays[key][:len(source["meta"]["score_names"])],source["arrays"][key])
        original_index=source["meta"]["score_names"].index("wls20_smooth3")
        ai=[meta["assets"].index(code) for code in f.ORIGINAL_RISK]
        original=source["arrays"]["scores"][original_index][:,ai]
        generated=arrays["scores"][-1][:,ai]
        np.testing.assert_allclose(generated,original,rtol=1e-13,atol=1e-12)
        for name in source["meta"]["score_names"]:
            view=f.risk_view(arrays,meta,"current20",score=name)
            # All relevant original-pool decision features remain exact; extra
            # research assets never enter this V11 trading universe.
            np.testing.assert_array_equal(view["features"][:,ai],source["arrays"]["features"][:,ai])


if __name__=="__main__":
    unittest.main()
