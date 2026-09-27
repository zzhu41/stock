"""Frozen public API checks; all edited profiles and data live in temp fixtures."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from v10_round2 import cli
from v10_round2.registry import get_candidate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class PublicTests(unittest.TestCase):
    def fixture(self, directory):
        parent = Path(directory)
        root, first = parent / "v10_round2", parent / "v10_next"
        (root / "results").mkdir(parents=True)
        first.mkdir()
        (root / "cli.py").write_text("PUBLIC_API = 1\n")
        (first / "execution.py").write_text("EXECUTION = 1\n")
        (first / "data_manifest.json").write_text("{}\n")
        (root / "results/evaluation.json").write_text("{}\n")
        (root / "results/selection.json").write_text("{}\n")
        config = get_candidate("h_qvix_score25_30")
        frozen = dict(
            variants={"growth": dict(name="v10-H2", selected_candidate=config["id"],
                                      config=config, config_sha256=config_sha(config))},
            own_source_sha256={"cli.py": sha(root / "cli.py")},
            parent_source_sha256={"execution.py": sha(first / "execution.py")},
            artifact_sha256={name: sha(root / name) for name in
                             ("results/evaluation.json", "results/selection.json")},
            prices_manifest_sha256=sha(first / "data_manifest.json"))
        (root / "profiles.json").write_text(json.dumps(frozen))
        return root, first, frozen

    def test_profile_loads_an_independent_frozen_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _, _ = self.fixture(directory)
            with patch.object(cli, "BASE", root):
                profile = cli.load_profile("growth")
                self.assertEqual(profile["selected_candidate"], "h_qvix_score25_30")
                self.assertEqual(profile["config"]["score_windows"], [25, 30])
                profile["config"]["score_windows"] = [1]
                self.assertEqual(cli.load_profile("growth")["config"]["score_windows"], [25, 30])
                with self.assertRaisesRegex(ValueError, "No selected frozen version"):
                    cli.load_profile("typo")

    def test_config_source_dependency_artifact_and_manifest_changes_are_rejected(self):
        for target, message in (("config", "Frozen candidate config changed"),
                                ("own", "Second-round implementation changed"),
                                ("parent", "First-round dependency changed"),
                                ("evaluation", "Frozen research artifact changed"),
                                ("selection", "Frozen research artifact changed"),
                                ("prices", "Price snapshot manifest changed")):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root, first, frozen = self.fixture(directory)
                if target == "config":
                    frozen["variants"]["growth"]["config"]["risk_weight"] = .9
                    (root / "profiles.json").write_text(json.dumps(frozen))
                else:
                    path = {"own": root / "cli.py", "parent": first / "execution.py",
                            "evaluation": root / "results/evaluation.json",
                            "selection": root / "results/selection.json",
                            "prices": first / "data_manifest.json"}[target]
                    path.write_text("changed\n")
                with patch.object(cli, "BASE", root), self.assertRaisesRegex(ValueError, message):
                    cli.load_profile("growth")

    def test_required_codes_include_all_books_calendar_and_cash_once(self):
        single = get_candidate("c_v91")
        expected = set(single["stock_pool"]) | set(single["global_pool"]) | {"518880", "511880", "510300"}
        self.assertEqual(set(cli.codes_for(single)), expected)
        accounts = get_candidate("s_accounts25_30")
        codes = cli.codes_for(accounts)
        self.assertEqual(set(codes), expected)
        self.assertEqual(len(codes), len(expected))
        self.assertEqual(codes, tuple(sorted(expected)))

    def test_qvix_requirement_is_recursive_and_channel_specific(self):
        self.assertFalse(cli.requires_qvix(get_candidate("c_v91")))
        self.assertTrue(cli.requires_qvix(get_candidate("h_qvix_score25_30")))
        accounts = get_candidate("s_accounts25_30")
        self.assertFalse(cli.requires_qvix(accounts))
        accounts["components"] = (get_candidate("c_v91"), get_candidate("h_qvix_only"))
        self.assertTrue(cli.requires_qvix(accounts))
        deep_only = get_candidate("c_v92")
        deep_only["channels"] = ("deep",)
        self.assertFalse(cli.requires_qvix(deep_only))

    def test_external_growth_prices_cannot_fall_back_to_old_qvix(self):
        profile = {"config": get_candidate("h_qvix_score25_30")}
        with patch.object(cli, "load_profile", return_value=profile), \
                patch.object(cli, "read_prices", return_value={}), \
                patch.object(cli, "Simulator") as simulator:
            with self.assertRaisesRegex(ValueError, "explicit --qvix-file"):
                cli.evaluate("growth", end="2026-09-11", data_dir="unused")
            simulator.assert_not_called()

    def test_explicit_external_qvix_is_passed_to_the_simulator(self):
        profile = {"config": get_candidate("h_qvix_only")}
        histories = {"510300": [("2026-09-11", 1., 1., 1.)]}
        simulator = MagicMock(histories=histories, calendar=["2026-09-11"])
        simulator.simulate.return_value = {"marker": "synthetic"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qvix.csv"
            path.write_text("2026-09-11,25.0\n")
            with patch.object(cli, "load_profile", return_value=profile), \
                    patch.object(cli, "read_prices", return_value=histories), \
                    patch.object(cli, "Simulator", return_value=simulator) as constructor:
                _, result = cli.evaluate("growth", end="2026-09-11", data_dir=directory, qvix_file=path)
        self.assertEqual(result, {"marker": "synthetic"})
        self.assertIs(constructor.call_args[1]["histories"], histories)
        self.assertEqual(constructor.call_args[1]["qvix"].rows, (("2026-09-11", 25.0),))
        simulator.simulate.assert_called_once_with(profile["config"], end="2026-09-11")

    def test_external_legacy_accounts_need_no_qvix_and_future_end_is_rejected(self):
        profile = {"config": get_candidate("s_accounts25_30")}
        histories = {"510300": [("2026-09-11", 1., 1., 1.)]}
        simulator = MagicMock(histories=histories, calendar=["2026-09-11"])
        with patch.object(cli, "load_profile", return_value=profile), \
                patch.object(cli, "read_prices", return_value=histories), \
                patch.object(cli, "Simulator", return_value=simulator):
            cli.evaluate("robust", end="2026-09-11", data_dir="unused")
            with self.assertRaisesRegex(ValueError, "observed date"):
                cli.evaluate("robust", end="2026-09-25", data_dir="unused")

    def test_account_signal_preserves_book_instructions_and_drifted_wealth(self):
        profile = dict(name="v10-S2", selected_candidate="s_accounts25_30")
        state = dict(
            date="2026-09-11", weights={"A": .8, "B": .2}, independent_accounts=True,
            pending_target=None, pending_signal_date=None,
            account_states={"one": {"final_wealth_share": .8}, "two": {"final_wealth_share": .2}},
            pending_account_instructions={"one": {"signal_date": "2026-09-11", "target_weights": {"C": 1.0}}})
        result = dict(final_state=copy.deepcopy(state), policy_metadata={})
        output = io.StringIO()
        with patch.object(cli, "evaluate", return_value=(profile, result)), \
                patch("sys.argv", ["v10_round2.cli", "signal", "robust", "--end", "2026-09-11"]), \
                contextlib.redirect_stdout(output):
            cli.main()
        message = json.loads(output.getvalue())
        self.assertEqual(message["signal_date"], "2026-09-11")
        self.assertEqual(message["model_weights"], {"A": .8, "B": .2})
        self.assertEqual(message["account_states"], state["account_states"])
        self.assertEqual(message["pending_account_instructions"], state["pending_account_instructions"])
        self.assertNotIn("pending_target", message)
        self.assertIn("never merge targets", message["instruction_scope"])
        self.assertIn("not_deployed", message["status"])
        self.assertIn("not live quotes", message["source"])


if __name__ == "__main__":
    unittest.main()
