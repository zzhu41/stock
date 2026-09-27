"""Small synthetic portfolio checks only; no full-history candidate evaluation."""
from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v10_deep.portfolios import (CASH, build_topk_policy, prices_from_features,
                                 registry, run_weights, targets_as_weights)


DAYS = tuple("2020-01-%02d" % i for i in range(1, 13))


def small_features():
    assets = ["510300", "159915", "513100", "518880", CASH]
    names = ["close", "valid", "ma250", "mom20", "mom5", "ret1", "open"]
    cube = np.zeros((3, len(assets), len(names)), dtype=float)
    cube[:, :, names.index("close")] = 100
    cube[:, :, names.index("open")] = 9999  # Must never become a fill price.
    cube[:, :, names.index("valid")] = 1
    cube[:, assets.index(CASH), names.index("valid")] = 0
    cube[:, :, names.index("ma250")] = .10
    cube[:, :, names.index("mom20")] = .10
    cube[:, :, names.index("mom5")] = .02
    cube[:, :, names.index("ret1")] = .001
    cube[:, assets.index("159915"), names.index("close")] = [100, 120, 150]
    scores = np.asarray([[[999, 30, 20, 10, -1e100], [999, 25, 40, 10, -1e100], [999, 30, 20, 10, -1e100]],
                         [[999, 15, 20, 10, -1e100]] * 3], dtype=float)
    meta = dict(dates=list(DAYS[:3]), assets=assets, feature_names=names,
                score_names=["wls25_v20", "blend_20_40_60"])
    config = deepcopy(registry()["candidates"][0])
    config.update(id="synthetic", topk=2, stock_pool=["159915"], global_pool=["513100"],
                  regime="legacy_gate", score="wls25_v20")
    config.pop("hash", None)
    return dict(features=cube, scores=scores), meta, config


class PortfolioTests(unittest.TestCase):
    def test_membership_unchanged_preserves_units_and_drifting_weights(self):
        arrays, meta, config = small_features()
        before = arrays["features"].copy()
        policy = build_topk_policy(config, arrays, meta)
        result = run_weights(prices_from_features(arrays, meta), meta["dates"], policy, 0, 2)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["switches"], 0)
        self.assertEqual(result["diagnostics"]["total_model_fee"], 0)
        self.assertIsNone(result["signal_target"][1][1])
        self.assertIsNone(result["signal_target"][2][1])
        self.assertAlmostEqual(result["nav"], 1.25)
        self.assertAlmostEqual(result["final_state"]["units"]["159915"], .005)
        self.assertAlmostEqual(result["daily"][-1][2]["159915"], .6)
        np.testing.assert_array_equal(before, arrays["features"])

    def test_explicit_equal_weight_resets_are_charged_not_free(self):
        prices = {"A": [100, 120], "B": [100, 100]}
        result = run_weights(prices, DAYS[:2], lambda i, s: {"A": .5, "B": .5}, 0, 1)
        self.assertEqual(len(result["trades"]), 2)
        change = result["trades"][1]
        self.assertAlmostEqual(change["nav_before"], 1.1)
        self.assertAlmostEqual(change["turnover"], 1 / 11)
        self.assertAlmostEqual(change["model_fee"], .00001)
        self.assertAlmostEqual(result["nav"], 1.09999)
        self.assertAlmostEqual(result["daily"][-1][2]["A"], .5)

    def test_member_replacement_charges_the_whole_pretrade_weight_turnover(self):
        prices = {"A": [100, 120, 150], "B": [100, 100, 100], "C": [100, 100, 100]}
        targets = [{"A": .5, "B": .5}, None, {"B": .5, "C": .5}]
        result = run_weights(prices, DAYS[:3], lambda i, s: targets[i], 0, 2)
        change = result["trades"][-1]
        self.assertAlmostEqual(change["nav_before"], 1.25)
        self.assertAlmostEqual(change["turnover"], 1.2)
        self.assertAlmostEqual(result["nav"], 1.25 * (1 - .0001 * 1.2))
        self.assertEqual(set(result["final_state"]["units"]), {"B", "C"})
        self.assertFalse(result["diagnostics"]["live_cash_ledger"])

    def test_partial_allocation_leaves_real_model_cash_without_interest(self):
        prices = {"A": [100, 200]}
        result = run_weights(prices, DAYS[:2], lambda i, s: {"A": .4} if i == 0 else None, 0, 1)
        self.assertAlmostEqual(result["final_state"]["cash"], .6)
        self.assertAlmostEqual(result["nav"], 1.4)
        for bad in ({"A": 1.01}, {"A": -.1}, {"A": math.inf}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run_weights(prices, DAYS[:2], lambda i, s: bad, 0, 1)

    def test_missing_held_or_target_quote_discards_entire_rebalance(self):
        prices = {"A": [100, None, 130], "B": [100, 100, 100], "C": [100, 100, 100]}
        targets = [{"A": .5, "B": .5}, {"B": .5, "C": .5}, None]
        result = run_weights(prices, DAYS[:3], lambda i, s: targets[i], 0, 2)
        self.assertEqual(result["diagnostics"]["blocked_rebalance_count"], 1)
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(set(result["final_state"]["units"]), {"A", "B"})
        self.assertAlmostEqual(result["nav"], 1.15)
        self.assertEqual(result["diagnostics"]["total_model_fee"], 0)
        target_missing = {"A": [100, 110, 120], "B": [100, 100, 100], "C": [100, None, 100]}
        other = run_weights(target_missing, DAYS[:3], lambda i, s: targets[i], 0, 2)
        self.assertEqual(other["diagnostics"]["blocked_rebalance_count"], 1)
        self.assertEqual(set(other["final_state"]["units"]), {"A", "B"})
        self.assertAlmostEqual(other["nav"], 1.1)

    def test_unquoted_untouched_leg_is_not_fictitiously_traded_to_pay_fees(self):
        prices = {"A": [100, None], "B": [100, 100]}
        targets = [{"A": .5}, {"A": .5, "B": .5}]
        free = run_weights(prices, DAYS[:2], lambda i, s: targets[i], 0, 1, fee=0)
        self.assertEqual(free["diagnostics"]["blocked_rebalance_count"], 0)
        self.assertEqual([c["code"] for c in free["trades"][-1]["allocation_changes"]], ["B"])
        # With fees, enforcing the same final A weight would require a small
        # reduction in A units too. Its missing quote blocks the whole order.
        charged = run_weights(prices, DAYS[:2], lambda i, s: targets[i], 0, 1)
        self.assertEqual(charged["diagnostics"]["blocked_rebalance_count"], 1)
        self.assertAlmostEqual(charged["final_state"]["cash"], .5)

    def test_registry_is_fixed_no_crash_and_independent_between_calls(self):
        book = registry()
        self.assertEqual(book["count"], 36)
        self.assertEqual(len({c["id"] for c in book["candidates"]}), 36)
        self.assertEqual({c["topk"] for c in book["candidates"]}, {1, 2, 3})
        self.assertEqual({c["pool"] for c in book["candidates"]}, {"original", "broad", "macro"})
        self.assertTrue(all(c["crash_channels"] == [] for c in book["candidates"]))
        book["candidates"][0]["stock_pool"].clear()
        self.assertTrue(registry()["candidates"][0]["stock_pool"])

    def test_legacy_gate_all_assets_and_panic_have_explicit_distinct_behavior(self):
        arrays, meta, config = small_features()
        fields = {n: i for i, n in enumerate(meta["feature_names"])}
        ids = {c: i for i, c in enumerate(meta["assets"])}
        arrays["features"][0, ids["510300"], fields["ma250"]] = -.10
        arrays["features"][0, ids["513100"], fields["mom20"]] = .05
        arrays["features"][0, ids["518880"], fields["mom20"]] = .08
        state = dict(holdings=())
        gated = build_topk_policy(config, arrays, meta)(0, state)
        self.assertEqual(gated, {"518880": 1.0})
        config["regime"] = "all_assets"
        full = build_topk_policy(config, arrays, meta)(0, state)
        self.assertEqual(full, {"159915": .5, "513100": .5})
        self.assertNotIn("510300", full)  # This synthetic pool excludes the benchmark.
        arrays["features"][0, ids["159915"], fields["ret1"]] = -.05
        panic = build_topk_policy(config, arrays, meta)(0, dict(holdings=("159915",)))
        self.assertEqual(panic, {"513100": .5, "518880": .5})

    def test_policy_state_mutation_cannot_modify_owned_account_state(self):
        def policy(i, state):
            state["cash"] = -1e20
            state["units"]["A"] = 1e20
            state["current_quotes"]["A"] = 1
            return {"A": 1} if i == 0 else None
        result = run_weights({"A": [100, 110]}, DAYS[:2], policy, 0, 1)
        self.assertAlmostEqual(result["nav"], 1.1)
        self.assertAlmostEqual(result["final_state"]["units"]["A"], .01)
        self.assertEqual(result["final_state"]["cash"], 0)

    def test_top1_replays_a_short_native_baseline_code_path_every_day(self):
        # A synthetic 12-session native control, compiled only in a temporary
        # directory. No real feature cache or full-history strategy is touched.
        from v10_deep import native as native_module
        from v10_deep.data import ASSET_ORDER
        from v10_deep.features import F, FEATURE_NAMES
        from v10_deep.schema import baseline
        T, A = len(DAYS), len(ASSET_ORDER)
        cube = np.zeros((T, A, len(FEATURE_NAMES)), dtype=np.float64)
        score = np.zeros((1, T, A), dtype=np.float64)
        wanted = ["159915", "159915", "510300", "513100", "513100", "510300"] * 2
        for i in range(T):
            for ai, code in enumerate(ASSET_ORDER):
                cube[i, ai, F["close"]] = 100 + 3 * ai + .2 * i
                cube[i, ai, F["valid"]] = int(code != CASH)
                cube[i, ai, F["ma250"]] = .1
                cube[i, ai, F["mom20"]] = .3 if code == wanted[i] else .05
                cube[i, ai, F["mom5"]] = .02
                cube[i, ai, F["ret1"]] = .001
                cube[i, ai, F["vol20"]] = .01
                score[0, i, ai] = 100 if code == wanted[i] else 0
        cube[2, ASSET_ORDER.index("159915"), F["close"]] = np.nan
        cube[2, ASSET_ORDER.index("159915"), F["valid"]] = 0
        arrays = dict(features=cube, scores=score,
                      orders=np.argsort(-score, axis=2, kind="stable").astype(np.int32),
                      fear=np.zeros(T, dtype=np.int32))
        meta = dict(dates=list(DAYS), assets=list(ASSET_ORDER), feature_names=FEATURE_NAMES,
                    score_names=["wls25_v20"])
        source = (native_module.BASE / "native.cpp").read_bytes()
        with tempfile.TemporaryDirectory() as directory, patch.object(native_module, "BASE", Path(directory)):
            (Path(directory) / "native.cpp").write_bytes(source)
            native = native_module.Simulator(arrays, meta).run([baseline()], DAYS[0], DAYS[-1], workers=1)
        targets = [ASSET_ORDER[j] if j >= 0 else None for j in native["holdings"][0]]
        result = run_weights(prices_from_features(arrays, meta), DAYS, targets_as_weights(wanted), 0, T - 1)
        expected = np.cumprod(1 + native["returns"][0])
        np.testing.assert_allclose([row[1] for row in result["daily"]], expected, rtol=0, atol=1e-10)
        self.assertEqual([next(iter(row[2])) if row[2] else None for row in result["daily"]], targets)
        self.assertEqual(result["switches"], int(native["summary"][0, 3]))
        self.assertEqual(result["diagnostics"]["blocked_rebalance_count"], int(native["summary"][0, 5]))
        self.assertEqual(result["daily"][0][1], 1)


if __name__ == "__main__":
    unittest.main()
