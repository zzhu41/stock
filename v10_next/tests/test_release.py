"""Release integrity and public-factory checks; all mutation is in temp fixtures."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v10_next import cli
from v10_next import strategy as release
from v10_next.candidates import get_candidate


def config_digest(config):
    return hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class ReleaseTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        code = root / "engine.py"
        code.write_text("VALUE = 1\n", encoding="utf-8")
        config = get_candidate("r_m12_broad")
        profiles = {
            "variants": {"robust": {
                "name": "v10-S", "status": "frozen_research_candidate_not_deployed",
                "selected_candidate": config["id"], "config": config,
                "config_sha256": config_digest(config)}},
            "implementation_sha256": {"engine.py": hashlib.sha256(code.read_bytes()).hexdigest()},
        }
        for name, key in (("data_manifest.json", "data_manifest_sha256"),
                          ("results/evaluation.json", "evaluation_sha256"),
                          ("results/selection.json", "selection_sha256")):
            artifact = root / name
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("{}\n", encoding="utf-8")
            profiles[key] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        (root / "profiles.json").write_text(json.dumps(profiles), encoding="utf-8")
        return root, profiles

    def test_config_change_is_detected_without_touching_real_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root, profiles = self.fixture(directory)
            with patch.object(release, "BASE", root):
                profile = release.load_profile("robust")
                self.assertEqual(profile["config"]["lookbacks"], [244])
                profiles["variants"]["robust"]["config"]["slot_weight"] = .9
                (root / "profiles.json").write_text(json.dumps(profiles), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "profile config changed"):
                    release.load_profile("robust")

    def test_implementation_change_and_unknown_variant_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            with patch.object(release, "BASE", root):
                with self.assertRaisesRegex(ValueError, "growth or robust"):
                    release.load_profile("typo")
                (root / "engine.py").write_text("VALUE = 2\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "implementation changed: engine.py"):
                    release.load_profile("robust")

    def test_changed_research_manifest_evaluation_or_selection_is_rejected(self):
        for name in ("data_manifest.json", "results/evaluation.json", "results/selection.json"):
            with self.subTest(artifact=name), tempfile.TemporaryDirectory() as directory:
                root, _ = self.fixture(directory)
                (root / name).write_text('{"modified": true}\n', encoding="utf-8")
                with patch.object(release, "BASE", root), \
                        self.assertRaisesRegex(ValueError, "Frozen research artifact changed"):
                    release.load_profile("robust")

    def test_required_assets_include_calendar_cash_and_no_duplicates(self):
        robust = {"config": get_candidate("r_m12_broad")}
        self.assertEqual(release.required_codes(robust),
                         ("510300", "510500", "513100", "518880", "511010", "511880"))
        legacy = {"config": get_candidate("h_core_pool")}
        self.assertEqual(release.required_codes(legacy),
                         ("159915", "510300", "510500", "513100", "518880", "511880"))
        alternate = copy.deepcopy(robust)
        alternate["config"]["pool"] = ["518880", "518880"]
        self.assertEqual(release.required_codes(alternate), ("518880", "511880", "510300"))

    def test_missing_asset_fails_before_feature_construction(self):
        profile = {"config": get_candidate("r_m12_broad")}
        histories = {code: [] for code in release.required_codes(profile) if code != "511010"}
        with patch.object(release, "load_profile", return_value=profile), \
                patch.object(release, "Features") as features:
            with self.assertRaisesRegex(ValueError, "Required histories missing: 511010"):
                release.make_policy("robust", histories, [])
            features.assert_not_called()

    def test_factory_uses_frozen_profile_copy_without_rerunning_candidate_selection(self):
        profile = {"config": get_candidate("r_m12_broad")}
        profile["config"]["id"] = "frozen_profile_identifier"
        calls = []

        class Reader:
            def __call__(self, day, code, lookbacks):
                calls.append(tuple(lookbacks))
                value = .1 if code == "510300" else -.1
                return dict(score=value, returns={n: value for n in lookbacks}, available=True)

            def cash_available(self, day):
                return True

        histories = {code: [] for code in release.required_codes(profile)}
        with patch.object(release, "load_profile", return_value=profile), \
                patch.object(release, "Features", return_value=Reader()), \
                patch("v10_next.candidates.get_candidate", side_effect=AssertionError("must not reselect")):
            policy = release.make_policy("robust", histories, [])
        profile["config"]["lookbacks"] = [122]
        profile["config"]["slot_weight"] = .9
        target = policy("2026-09-01", {}, {"previous_date": "2026-08-31"})
        self.assertEqual(target, {"510300": .2, "511880": .8})
        self.assertEqual(set(calls), {(244,)})
        self.assertEqual(policy.metadata["candidate_id"], "frozen_profile_identifier")

    def test_signal_output_identifies_historical_model_date_and_account_scope(self):
        profile = dict(name="v10-S", status="frozen_research_candidate_not_deployed")
        result = {"final_state": dict(date="2026-09-11", weights={"511880": 1.0},
                                     pending_target=None, pending_signal_date=None)}
        output = io.StringIO()
        with patch.object(cli, "evaluate", return_value=(profile, object(), result)), \
                patch("sys.argv", ["v10_next.cli", "signal", "robust", "--end", "2026-09-11"]), \
                contextlib.redirect_stdout(output):
            cli.main()
        message = json.loads(output.getvalue())
        self.assertEqual(message["signal_date"], "2026-09-11")
        self.assertIn("not_deployed", message["status"])
        self.assertIn("not a real-time quote feed", message["source"])
        self.assertIn("does not reflect the user's actual account", message["caution"])
        self.assertIsNone(message["pending_target_weights"])


if __name__ == "__main__":
    unittest.main()
