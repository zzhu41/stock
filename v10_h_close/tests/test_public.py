"""Public close-clock contracts using temp profiles and two synthetic sessions."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from v10_h_close import cli
from v10_h_close.registry import candidate_hash, required_codes
from v10_search.registry import BASE_STOCK, DIVIDENDS, PRESETS, _candidate


DAYS = ("2026-09-23", "2026-09-24")


def selected_config():
    base = json.loads(PRESETS.read_text())["v9.1"]
    config = _candidate(base, "public_fixture", BASE_STOCK + DIVIDENDS, ("513100", "513120"),
                        window=25, overrides={"buffer": .02, "pool_buffer": {}})
    config.update(kind="v92", channels=["deep", "qvix"])
    config["candidate_hash"] = candidate_hash(config)
    config["id"] = "hc_" + config["candidate_hash"][:20]
    return config


def profiles():
    config = selected_config()
    return {
        "growth": dict(name="H-close primary", selected_by="selection_2014_2025_maximum_cagr",
                       config=deepcopy(config), candidate_hash=config["candidate_hash"]),
        "guarded": dict(name="H-close guarded", selected_by="selection_2014_2025_five_risk_guards",
                        config=deepcopy(config), candidate_hash=config["candidate_hash"]),
    }


def histories_for(config):
    return {code: [(DAYS[0], 1., 1., 100.), (DAYS[1], 1.1, 1.1, 100.)]
            for code in required_codes(config)}


class Schedule:
    def __init__(self):
        self.metadata = {"fixture": "synthetic only"}

    def __call__(self, index, holding, age, can_sell):
        return "159915" if index == 0 else "510300"


class PublicTests(unittest.TestCase):
    def fixture(self, directory):
        parent = Path(directory)
        root = parent / "v10_h_close"
        (root / "results").mkdir(parents=True)
        (parent / "v10_search").mkdir()
        (root / "engine.py").write_text("ENGINE = 1\n")
        (parent / "v10_search/policy.py").write_text("POLICY = 1\n")
        for name in ("snapshot_manifest.json", "qvix_manifest.json", "results/selection.json"):
            (root / name).write_text("{}\n")
        frozen = dict(
            variants=profiles(), source_sha256={"engine.py": cli.sha(root / "engine.py")},
            dependency_sha256={"v10_search/policy.py": cli.sha(parent / "v10_search/policy.py")},
            artifact_sha256={name: cli.sha(root / name) for name in
                             ("snapshot_manifest.json", "qvix_manifest.json", "results/selection.json")})
        (root / "profiles.json").write_text(json.dumps(frozen))
        return root, parent, frozen

    def test_primary_and_guarded_are_aliases_of_one_frozen_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _ = self.fixture(directory)
            with patch.object(cli, "BASE", root):
                primary, guarded = cli.load_profile("growth"), cli.load_profile("guarded")
                self.assertEqual(primary["config"], guarded["config"])
                self.assertEqual(primary["candidate_hash"], guarded["candidate_hash"])
                self.assertEqual(primary["config"]["id"], "hc_9dd904b5d548af3bd2f4")
                self.assertNotEqual(primary["selected_by"], guarded["selected_by"])
                primary["config"]["score_windows"] = [30]
                self.assertEqual(cli.load_profile("growth")["config"]["score_windows"], [25])
                with self.assertRaisesRegex(ValueError, "No frozen candidate"):
                    cli.load_profile("unknown")

    def test_hash_checks_reject_source_dependency_artifact_and_config_changes(self):
        cases = (("source", "Close-clock implementation changed"),
                 ("dependency", "Dependency changed"), ("prices", "Frozen research artifact changed"),
                 ("qvix", "Frozen research artifact changed"), ("selection", "Frozen research artifact changed"),
                 ("windows", "Frozen configuration changed"), ("buffer", "Frozen configuration changed"),
                 ("channels", "Frozen configuration changed"))
        for target, message in cases:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root, parent, frozen = self.fixture(directory)
                config = frozen["variants"]["growth"]["config"]
                if target in ("windows", "buffer", "channels"):
                    if target == "windows":
                        config["score_windows"] = [30]
                    elif target == "buffer":
                        config["params"]["pool_buffer"] = {"stock": .02, "global": .03, "gold": .03}
                    else:
                        config["channels"] = ["deep", "qvix", "volume"]
                    (root / "profiles.json").write_text(json.dumps(frozen))
                else:
                    changed = {"source": root / "engine.py", "dependency": parent / "v10_search/policy.py",
                               "prices": root / "snapshot_manifest.json", "qvix": root / "qvix_manifest.json",
                               "selection": root / "results/selection.json"}[target]
                    changed.write_text("changed\n")
                with patch.object(cli, "BASE", root), self.assertRaisesRegex(ValueError, message):
                    cli.load_profile("growth")

    def test_required_assets_include_all_dividends_both_globals_cash_and_benchmark(self):
        config = selected_config()
        codes = required_codes(config)
        self.assertEqual(set(codes), set(BASE_STOCK) | set(DIVIDENDS) |
                         {"513100", "513120", "518880", "511880", "510300"})
        self.assertEqual(len(codes), len(set(codes)))
        self.assertFalse(config["params"]["bear_open_stock"])
        self.assertEqual(config["regime_mode"], "ma250")

    def test_external_qvix_profile_cannot_fall_back_to_old_fear_data(self):
        profile = profiles()["growth"]
        with patch.object(cli, "load_profile", return_value=profile), \
                patch.object(cli, "read_prices", return_value=histories_for(profile["config"])), \
                patch.object(cli, "load_fear", side_effect=AssertionError("old feed must not be used")), \
                patch.object(cli, "FeatureBank") as bank:
            with self.assertRaisesRegex(ValueError, "explicit --qvix-file"):
                cli.evaluate("growth", end=DAYS[-1], data_dir="external-fixture")
            bank.assert_not_called()

    def test_complete_config_is_used_and_real_engine_retains_the_original_fee_clock(self):
        results = []
        for variant, profile in profiles().items():
            config = profile["config"]
            histories = histories_for(config)
            series = MagicMock()
            series.state.side_effect = lambda day: {"active": day == DAYS[-1]}
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                qvix_path = Path(directory) / "qvix.csv"
                qvix_path.write_text("2026-09-23,20\n2026-09-24,30\n")
                with patch.object(cli, "load_profile", return_value=profile), \
                        patch.object(cli, "read_prices", return_value=histories), \
                        patch.object(cli, "load_histories", side_effect=AssertionError("not the supplied snapshots")), \
                        patch.object(cli, "load_fear", side_effect=AssertionError("must use explicit QVIX")), \
                        patch.object(cli, "QvixSeries", return_value=series), \
                        patch.object(cli, "FeatureBank", return_value=object()), \
                        patch.object(cli, "ClosePolicy", return_value=Schedule()) as factory:
                    _, result = cli.evaluate(variant, end=DAYS[-1], data_dir=directory, qvix_file=qvix_path)
                passed = factory.call_args[0][0]
                self.assertEqual(passed, config)
                self.assertEqual(passed["score_windows"], [25])
                self.assertEqual(passed["params"]["buffer"], .02)
                self.assertEqual(passed["params"]["pool_buffer"], {})
                self.assertEqual(passed["channels"], ["deep", "qvix"])
                self.assertEqual(factory.call_args[0][3], (False, True))
                self.assertEqual(result["navs"][0], 1.0)
                self.assertAlmostEqual(result["nav"], 1.1 * .9998, places=14)
                self.assertEqual(result["switches"], 1)
                self.assertEqual(result["diagnostics"]["switch_fee_factor"], .9998)
                self.assertEqual(result["diagnostics"]["first_session_entry_fee"], 0.0)
                self.assertEqual(result["diagnostics"]["price_clock"], "original_same_close")
                results.append(result)
        self.assertEqual(results[0]["daily"], results[1]["daily"])
        self.assertEqual(results[0]["trades"], results[1]["trades"])

    def test_future_end_is_rejected_before_loading_features_or_submitting_anything(self):
        profile = profiles()["growth"]
        with patch.object(cli, "load_profile", return_value=profile), \
                patch.object(cli, "read_prices", return_value=histories_for(profile["config"])), \
                patch.object(cli, "FeatureBank") as bank, patch.object(cli, "run") as execute:
            with self.assertRaisesRegex(ValueError, "dated observation"):
                cli.evaluate("growth", end="2026-09-25", data_dir="fixture", qvix_file="unused-path")
            bank.assert_not_called()
            execute.assert_not_called()

    def test_signal_is_a_past_close_not_a_live_order_and_aliases_keep_one_candidate_id(self):
        result = {"final_state": {"date": DAYS[-1], "holding": "510880"}}
        messages = []
        for variant, profile in profiles().items():
            output = io.StringIO()
            with patch.object(cli, "evaluate", return_value=(profile, result)), \
                    patch("sys.argv", ["v10_h_close.cli", "signal", variant, "--end", DAYS[-1]]), \
                    contextlib.redirect_stdout(output):
                cli.main()
            message = json.loads(output.getvalue())
            self.assertEqual(message["data_date"], DAYS[-1])
            self.assertEqual(message["historical_close_target"], "510880")
            self.assertTrue(message["not_deployed"])
            self.assertFalse(message["clean_oos"])
            self.assertFalse(message["order_submission"])
            self.assertIn("not a tradable 14:50 order instruction", message["warning"])
            self.assertIn("first session free", message["clock"])
            self.assertIn("0.9998", message["clock"])
            self.assertIn("not real-time quotes or the actual account", message["source"])
            messages.append(message)
        self.assertEqual(messages[0]["candidate_id"], messages[1]["candidate_id"])
        self.assertNotEqual(messages[0]["selected_by"], messages[1]["selected_by"])


if __name__ == "__main__":
    unittest.main()
