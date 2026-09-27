"""Synthetic protocol tests. No real post-2021 data or model returns are loaded."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v11 import reselect
from v11.registry import canonical
from v10_deep.features import FEATURE_NAMES
from v10_deep.schema import baseline, identifier
from v10_deep.reference import run_reference, DatedValues
from v11.features import risk_view


ASSETS = ("159915", "588080", "510300", "510500", "563300", "512400", "512890",
          "513100", "513120", "518880", "511880")
A, B = ASSETS[:2]
DATES = ["2013-12-31", "2014-01-02", "2017-12-29", "2018-01-02", "2019-12-31",
         "2020-01-02", "2021-12-31", "2022-01-04", "2023-12-29", "2024-01-02",
         "2025-12-31", "2026-01-05"]
SCORES = ("wls20_smooth3", "wls25_v20", "toy_score_one", "toy_score_two", "toy_score_three")


def candidate(config, family):
    value = canonical(config)
    digest = identifier(value)
    return dict(value, id="v11_" + digest[:20], hash=digest, families=[family], parents=[], stage="synthetic")


def synthetic_registry():
    h = baseline()
    h.update(score=SCORES[0], ma="ma180", panic_mode="volatility", panic=1.5)
    values = [candidate(h, "control_h"), candidate(baseline(), "control_v92"),
              candidate(dict(baseline(), min_hold=2), "control_simple")]
    values += [candidate(dict(h, score=score), "robust_regression") for score in SCORES[2:]]
    return dict(schema=1, candidates=values, unique_count=len(values), score_specs=[],
                controls=dict(h=values[0]["id"], v92=values[1]["id"], simple=values[2]["id"]))


def synthetic_arrays():
    fields = {name: i for i, name in enumerate(FEATURE_NAMES)}
    features = np.zeros((len(DATES), len(ASSETS), len(FEATURE_NAMES)))
    scores = np.ones((len(SCORES), len(DATES), len(ASSETS)))
    for i in range(len(DATES)):
        for j, code in enumerate(ASSETS):
            close = 100 + 10 * i if code == A else 200 + 40 * i if code == B else 100.
            features[i, j, fields["close"]] = close
            features[i, j, fields["valid"]] = code != "511880"
            for name, value in (("mom20", .3 if code == B else .1), ("mom5", .01), ("mom60", .2),
                                ("ma180", .1), ("ma250", .1), ("vol20", .02), ("vol60", .02), ("volume_ratio", 1.)):
                features[i, j, fields[name]] = value
    scores[:2, :, ASSETS.index(A)] = 1000.
    scores[2:, :, ASSETS.index(B)] = 1000.
    arrays = dict(features=features, scores=scores, orders=np.argsort(-scores, axis=2, kind="stable").astype(np.int32),
                  fear=np.zeros(len(DATES), dtype=np.int32), observations=np.full((len(DATES), len(ASSETS)), 300))
    meta = dict(dates=list(DATES), assets=list(ASSETS), feature_names=list(FEATURE_NAMES), score_names=list(SCORES),
                required_observations_by_score={name: 270 for name in SCORES}, shape=list(features.shape))
    return arrays, meta


def evaluator(calls, root, reject_all=False):
    """Tiny deterministic training returns whose best score changes by cutoff."""
    def evaluate(candidates, arrays, meta, start, end, scenarios, workers):
        assert workers == 1
        assert meta["dates"][-1] <= end
        assert arrays["features"].shape[0] == len(meta["dates"])
        assert arrays["scores"].shape[1] == len(meta["dates"])
        year = end[:4]
        stage = "A" if sum(c["end"] == end for c in calls) == 0 else "B"
        assert (root / year / stage / "registration.json").exists()
        assert (root / year / stage / "registered_candidates.json").exists()
        if stage == "B":
            assert (root / year / "A/selection.json").exists()
        calls.append(dict(end=end, max_date=meta["dates"][-1], stage=stage, ids=[c["id"] for c in candidates]))
        days = [d for d in meta["dates"] if start <= d <= end]
        ranked = dict(zip(SCORES, (.002, .001, .005, .004, .003))) if end < "2019" else dict(zip(SCORES, (.002, .001, .001, .005, .006)))
        if reject_all:
            ranked.update({name: .0001 for name in SCORES if name != SCORES[0]})
        results = {}
        for name, lag, fee in scenarios:
            returns = np.zeros((len(candidates), len(days)))
            summary = np.zeros((len(candidates), 10))
            for i, config in enumerate(candidates):
                returns[i, 1:] = max(0., ranked[config["score"]] - fee - lag * .0002)
                result = reselect.metrics(returns[i])
                summary[i] = [result["nav"], result["cagr"], result["max_dd"], 0., 1., 0., 0., 0.,
                              sum(returns[i]), sum(returns[i] ** 2)]
            results[name] = dict(returns=returns, holdings=np.zeros_like(returns, dtype=np.int32), summary=summary)
        return results, days
    return evaluate


class ReselectionProtocolTests(unittest.TestCase):
    def test_real_run_requires_explicit_gate_before_loading_data(self):
        with patch.object(reselect, "register", side_effect=AssertionError("Must not load real data")):
            with self.assertRaisesRegex(RuntimeError, "authorization"):
                reselect.run()

    def test_training_blocks_adapt_to_each_registered_endpoint(self):
        self.assertEqual(reselect.training_blocks("2017-12-31"), (("early", "2014-01-02", "2017-12-31"),))
        self.assertEqual(reselect.training_blocks("2019-12-31")[-1], ("late", "2018-01-01", "2019-12-31"))
        self.assertEqual(reselect.training_blocks("2023-12-31")[-1], ("late", "2018-01-01", "2023-12-31"))

    def test_each_B_menu_comes_from_its_own_cutoff_A_results_and_is_registered_first(self):
        registry = synthetic_registry()
        arrays, meta = synthetic_arrays()
        with tempfile.TemporaryDirectory() as directory:
            root, calls, trained = Path(directory), [], []
            evaluate = evaluator(calls, root)
            for fold in reselect.FOLDS[:2]:
                trained.append(reselect.train_fold(registry, arrays, meta, fold, root / fold["train_end"][:4],
                    dict(synthetic=True), evaluator=evaluate))
            first, second = trained
            first_scores = {item["values"]["score"] for item in first["B"]["registry"]["components"]["score"]}
            second_scores = {item["values"]["score"] for item in second["B"]["registry"]["components"]["score"]}
            self.assertIn(SCORES[2], first_scores)
            self.assertNotIn(SCORES[4], first_scores)
            self.assertIn(SCORES[4], second_scores)
            self.assertNotIn(SCORES[2], second_scores)
            self.assertNotEqual(first_scores, second_scores)
            for item in trained:
                end = item["fold"]["train_end"]
                for label in ("A", "B"):
                    choice = item[label]["selection"]
                    self.assertEqual(choice["selection_period"], [reselect.START, end])
                    self.assertFalse(choice["used_H_fallback"])
                    self.assertTrue((item[label]["directory"] / "paths.npz").exists())
                self.assertTrue(item["H_control_training_parity"])
            self.assertEqual([(c["end"], c["stage"]) for c in calls],
                [("2017-12-31", "A"), ("2017-12-31", "B"), ("2019-12-31", "A"), ("2019-12-31", "B")])
            self.assertTrue(all(c["max_date"] <= c["end"] for c in calls))
            # A completed fold must be reproducible without another evaluation.
            with patch.object(reselect, "run_candidates", side_effect=AssertionError("Unexpected recomputation")):
                repeated = reselect.train_fold(registry, arrays, meta, reselect.FOLDS[0], root / "2017",
                    dict(synthetic=True), evaluator=lambda *a, **k: self.fail("recomputed frozen fold"))
            self.assertEqual(repeated["A"]["selection"], first["A"]["selection"])

    def test_no_eligible_candidate_falls_back_to_H_without_a_future_rescue(self):
        registry = synthetic_registry()
        arrays, meta = synthetic_arrays()
        with tempfile.TemporaryDirectory() as directory:
            root, calls = Path(directory), []
            result = reselect.train_fold(registry, arrays, meta, reselect.FOLDS[0], root / "2017",
                dict(synthetic=True), evaluator=evaluator(calls, root, reject_all=True))
            for label in ("A", "B"):
                self.assertIsNone(result[label]["selection"]["primary"])
                self.assertTrue(result[label]["selection"]["used_H_fallback"])
                self.assertEqual(result[label]["selection"]["chosen_id"], registry["controls"]["h"])

    def test_continuous_execution_preserves_boundary_return_and_fees_and_shares_choices(self):
        registry = synthetic_registry()
        arrays, meta = synthetic_arrays()
        configs = {c["score"]: c for c in registry["candidates"] if "robust_regression" in c["families"]}
        chosen = [configs[SCORES[2]]] + [configs[SCORES[4]]] * 3
        trained = []
        for fold, config in zip(reselect.FOLDS, chosen):
            training = dict(registry=registry, selection=dict(chosen_id=config["id"]))
            trained.append(dict(fold=fold, A=training, B=deepcopy(training)))
        definitions = reselect.schedules(registry, trained)
        self.assertTrue(all(definitions[label][0]["config"]["id"] == registry["controls"]["h"] for label in ("A", "B", "H")))
        before = deepcopy(definitions)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            summaries, actual = reselect.execute_schedules(definitions, arrays, meta, directory)
            self.assertEqual(definitions, before)
            for scenario, lag, fee in reselect.SCENARIOS:
                a = actual["A"][scenario]
                boundary = next(row for row in a["trace"] if row["date"] == "2018-01-02")
                self.assertEqual(boundary["previous_holding"], A)
                self.assertEqual(boundary["holding"], B)
                self.assertTrue(boundary["filled"])
                self.assertAlmostEqual(boundary["cost"], boundary["marked_nav"] * 2 * fee)
                late = next(row for row in a["trace"] if row["date"] == "2020-01-02")
                self.assertTrue(late["model_changed"])
                self.assertFalse(late["filled"])
                self.assertEqual(late["cost"], 0.)
                period = summaries["A"][scenario]["periods"]["fold_2018_2019"]
                self.assertAlmostEqual(period["prior_nav"], 120. / 110.)
                self.assertAlmostEqual(period["nav"], (130. / 120.) * (1 - 2 * fee) * (360. / 320.))
                self.assertEqual(period["switches"], 1)
                self.assertLessEqual(a["dates"][-1], "2025-12-31")
                with Path(summaries["A"][scenario]["trace_path"]).open() as stream:
                    self.assertIn("2018-01-02", stream.read())
                chosen_ids = [segment["config"]["id"] for segment in definitions["A"]]
                saved_schedule = json.loads((Path(directory) / "A" / scenario / "schedule.json").read_text())
                self.assertEqual([s["config"]["id"] for s in saved_schedule], chosen_ids)
                self.assertTrue(all(s["config"]["lag"] == lag for s in saved_schedule))
                h = deepcopy(definitions["H"][0]["config"])
                h["lag"] = lag
                view = risk_view(arrays, meta, h["risk_context"], score=h["score"])
                expected = run_reference(view, meta, h, start=reselect.START, end=reselect.END, fee=fee)
                np.testing.assert_array_equal(actual["H"][scenario]["returns"], expected["returns"])
                np.testing.assert_array_equal(actual["H"][scenario]["holdings"], expected["holdings"])


if __name__ == "__main__":
    unittest.main()
