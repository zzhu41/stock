"""Small corrected-data routing tests; no real matrices or full backtests."""
import contextlib
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from v10_h_close import build_corrected_data, cli, corrected_scan as corrected, scan


DATES = ("2025-12-30", "2025-12-31", "2026-09-24")


def metric_row(name, selection=.60, full=.20):
    return dict(id=name, metrics={"selection": {"cagr": selection, "max_dd": -.2},
                "recent_2022_2025": {"cagr": selection}, "full": {"cagr": full, "max_dd": -.2},
                "report_only_2026": {"cagr": full * 100}},
                fee5={"selection": {"cagr": selection * .9}}, selection_switches=5, switches=999)


class CorrectedScanTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root / "results").mkdir()
        (root / "snapshots").mkdir()
        (root / "corrected_snapshots").mkdir()
        old = root / "snapshots/510300.csv"
        old.write_text("2026-09-24,999,999,1\n")
        path = root / "corrected_snapshots/510300.csv"
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows((d, v, v, 100.) for d, v in
                                        zip(DATES, (1., 1.123456789012345, 1.234567890123456)))
        manifest = dict(end=DATES[-1], assets={"510300": {"sha256": corrected.sha(path)}})
        manifest_path = root / "corrected_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        configurations = [dict(id=name, candidate_hash=name, channels=[], score_windows=[],
                               complexity={"active_optional_rule_count": 1})
                          for name in ("corrected_choice", "old_choice")]
        (root / "results/registry.json").write_text(json.dumps({"candidates": configurations}))
        (root / "results/registration.json").write_text('{"original_vintage": true}\n')
        (root / "results/selection.json").write_text('{"primary_return_champion": "old_choice"}\n')
        (root / "results/selection_returns.npy").write_bytes(b"not a corrected matrix; must never load")
        return root, manifest_path, configurations

    @contextlib.contextmanager
    def routed(self, root, manifest):
        with patch.object(corrected, "BASE", root), \
                patch.object(corrected, "OUT", root / "corrected_results"), \
                patch.object(corrected, "MANIFEST", manifest), \
                patch.object(corrected, "ASSET_ORDER", ("510300",)), \
                patch.object(scan, "OUT", root / "results"), \
                patch.object(scan, "load_histories", side_effect=AssertionError("old loader used")), \
                patch.object(scan, "_FRAME", None), patch.object(scan, "_END", None):
            corrected.configure()
            yield

    def test_configure_routes_only_process_local_scanner_output_and_loader(self):
        original_out, original_loader = scan.OUT, scan.load_histories
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, _ = self.fixture(directory)
            old_selection = (root / "results/selection.json").read_bytes()
            with self.routed(root, manifest):
                self.assertEqual(scan.OUT, root / "corrected_results")
                self.assertIs(scan.load_histories, corrected.load_histories)
                histories, end = scan.load_histories()
                self.assertEqual(end, DATES[-1])
                self.assertEqual(histories["510300"][-1][2], 1.234567890123456)
                self.assertNotEqual(histories["510300"][-1][2], 999)
            self.assertEqual((root / "results/selection.json").read_bytes(), old_selection)
        self.assertEqual(scan.OUT, original_out)
        self.assertIs(scan.load_histories, original_loader)

    def test_corrected_loader_preserves_precision_and_rejects_hash_or_tail_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest_path, _ = self.fixture(directory)
            with patch.object(corrected, "BASE", root), patch.object(corrected, "MANIFEST", manifest_path), \
                    patch.object(corrected, "ASSET_ORDER", ("510300",)):
                histories, _ = corrected.load_histories()
                self.assertEqual([r[0] for r in histories["510300"]], list(DATES))
                self.assertEqual(histories["510300"][1][1:3], (1.123456789012345, 1.123456789012345))
                path = root / "corrected_snapshots/510300.csv"
                path.write_text("2026-09-23,1,1,100\n")
                with self.assertRaisesRegex(ValueError, "Corrected data changed"):
                    corrected.load_histories()
                manifest = json.loads(manifest_path.read_text())
                manifest["assets"]["510300"]["sha256"] = corrected.sha(path)
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "coverage"):
                    corrected.load_histories()

    def test_builder_does_not_overwrite_an_existing_corrected_vintage(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, _ = self.fixture(directory)
            original = manifest.read_bytes()
            with patch.object(build_corrected_data, "BASE", root), \
                    patch.object(build_corrected_data, "protect", return_value=209), \
                    patch.object(build_corrected_data, "reconstruct_total_return") as helper:
                with self.assertRaisesRegex(RuntimeError, "frozen"):
                    build_corrected_data.build()
                helper.assert_not_called()
            self.assertEqual(manifest.read_bytes(), original)

    def test_development_and_evaluation_use_corrected_files_and_do_not_reselect_on_2026(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, configurations = self.fixture(directory)
            original_files = {p: p.read_bytes() for p in (root / "results").iterdir()}
            output = root / "corrected_results"
            rows = [metric_row("corrected_choice", .60, .20), metric_row("old_choice", .55, .90)]
            reference = metric_row("c_v92", .50, .50)
            fixed_fingerprint = {"data_vintage": "corrected-fixture"}
            seen_inputs, batches, simulated = [], [], []

            def setup(configs):
                self.assertEqual(configs, configurations)
                histories, end = scan.load_histories()
                self.assertEqual(histories["510300"][1][2], 1.123456789012345)
                seen_inputs.append(histories)
                scan._FRAME = SimpleNamespace(dates=tuple(r[0] for r in histories["510300"]))
                scan._END = end

            def batch(configs, end, label, workers):
                self.assertEqual(scan.OUT, output)
                self.assertEqual(end, scan.TRAIN_END if label == "selection" else DATES[-1])
                batches.append((label, end))
                dates = DATES[:2] if label == "selection" else DATES
                values = np.zeros((len(dates), len(configs)))
                if label == "full":
                    values[-1] = [.01, .90]  # Later data favor the old candidate, not the frozen selection.
                np.save(output / (label + "_returns.npy"), values)
                np.save(output / (label + "_fee5_returns.npy"), values)
                return deepcopy(rows), list(dates)

            def simulate(config, end=None, **kwargs):
                simulated.append((config["id"], end))
                return {"id": config["id"], "vintage": "corrected-fixture"}

            def collect(config, result, dates):
                self.assertEqual(result["vintage"], "corrected-fixture")
                selected = reference if config["id"] == "c_v92" else next(r for r in rows if r["id"] == config["id"])
                return deepcopy(selected), np.zeros(len(dates)), np.zeros(len(dates))

            numpy_load = np.load

            def corrected_matrix_only(path, *args, **kwargs):
                self.assertEqual(Path(path).parent, output)
                return numpy_load(path, *args, **kwargs)

            with self.routed(root, manifest), \
                    patch.object(corrected, "protect", return_value=209), \
                    patch.object(corrected, "fingerprints", return_value=fixed_fingerprint), \
                    patch.object(corrected, "controls", return_value=[{"id": "c_v92"}]), \
                    patch.object(scan, "setup", side_effect=setup), \
                    patch.object(scan, "verify_fee_transform", return_value=[{"synthetic": True}]), \
                    patch.object(scan, "simulate", side_effect=simulate), \
                    patch.object(scan, "collect", side_effect=collect), \
                    patch.object(scan, "batch", side_effect=batch), \
                    patch.object(scan, "next_open", side_effect=AssertionError("TR close placeholders are not opens")), \
                    patch.object(corrected.np, "load", side_effect=corrected_matrix_only), \
                    contextlib.redirect_stdout(io.StringIO()):
                corrected.develop(workers=1)
                frozen = (output / "selection.json").read_bytes()
                selection = json.loads(frozen)
                self.assertEqual(selection["primary_return_champion"], "corrected_choice")
                self.assertEqual(selection["risk_guarded_candidate"], "corrected_choice")
                self.assertIn("2026 not used", selection["selected_from"])
                registration = json.loads((output / "registration.json").read_text())
                self.assertFalse(registration["next_open_permitted"])
                self.assertEqual(registration["fingerprints"], fixed_fingerprint)
                self.assertTrue(all(end == scan.TRAIN_END for _, end in simulated))
                corrected.evaluate(workers=1)
            self.assertEqual(len(seen_inputs), 2)
            self.assertEqual(batches, [("selection", scan.TRAIN_END), ("full", DATES[-1])])
            self.assertEqual((output / "selection.json").read_bytes(), frozen)
            evaluation = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(evaluation["selection"]["primary_return_champion"], "corrected_choice")
            self.assertEqual(evaluation["hindsight_winner"]["id"], "old_choice")
            self.assertEqual(evaluation["original_input_winner_id"], "old_choice")
            self.assertIsNone(evaluation["next_open_pressure"])
            self.assertIn("placeholder", evaluation["next_open_limitation"])
            for path, raw in original_files.items():
                self.assertEqual(path.read_bytes(), raw)

    def profile(self, corrected_view=True):
        profile = dict(name="fixture", selected_by="fixture selection",
                       config=dict(id="fixture", stock_pool=["510300"], global_pool=["513100"],
                                   gold="518880", cash="511880", benchmark_code="510300", channels=["deep"]))
        if corrected_view:
            profile["data_view"] = "corrected_total_return_close"
        return profile

    def test_cli_corrected_view_routes_to_reconstructed_data_while_old_profiles_keep_old_loader(self):
        codes = ("510300", "513100", "518880", "511880")
        new_histories = {c: [(DATES[-1], 1., 1., 100.)] for c in codes}
        old_histories = {c: [(DATES[-1], 999., 999., 100.)] for c in codes}
        old_out, old_scan_loader = scan.OUT, scan.load_histories
        for corrected_view in (True, False):
            policy = MagicMock(metadata={})
            with self.subTest(corrected_view=corrected_view), \
                    patch.object(cli, "load_profile", return_value=self.profile(corrected_view)), \
                    patch.object(corrected, "load_histories", return_value=(new_histories, DATES[-1])) as new_loader, \
                    patch.object(cli, "load_histories", return_value=(old_histories, DATES[-1])) as old_loader, \
                    patch.object(cli, "load_fear", return_value=(False,)), \
                    patch.object(cli, "FeatureBank", return_value=object()) as bank, \
                    patch.object(cli, "prepare", return_value=object()), \
                    patch.object(cli, "ClosePolicy", return_value=policy), \
                    patch.object(cli, "run", return_value={"final_state": {"date": DATES[-1]}}):
                cli.evaluate("growth", end=DATES[-1])
            self.assertEqual(bank.call_args[0][0], new_histories if corrected_view else old_histories)
            self.assertEqual(new_loader.call_count, int(corrected_view))
            self.assertEqual(old_loader.call_count, int(not corrected_view))
        self.assertEqual(scan.OUT, old_out)
        self.assertIs(scan.load_histories, old_scan_loader)

    def test_cli_corrected_profile_rejects_arbitrary_raw_or_qfq_directory_before_reading_it(self):
        with patch.object(cli, "load_profile", return_value=self.profile()), \
                patch.object(cli, "read_prices") as external, \
                patch.object(corrected, "load_histories") as internal, \
                patch.object(cli, "run") as execute:
            with self.assertRaisesRegex(ValueError, "arbitrary raw/QFQ"):
                cli.evaluate("growth", data_dir="untrusted-price-basis", qvix_file="does-not-override-the-block")
            external.assert_not_called()
            internal.assert_not_called()
            execute.assert_not_called()

    def test_corrected_signal_discloses_total_return_assumptions_and_no_next_open_support(self):
        output = io.StringIO()
        result = {"final_state": {"date": DATES[-1], "holding": "510300"}}
        with patch.object(cli, "evaluate", return_value=(self.profile(), result)), \
                patch("sys.argv", ["v10_h_close.cli", "signal", "growth", "--end", DATES[-1]]), \
                contextlib.redirect_stdout(output):
            cli.main()
        message = json.loads(output.getvalue())
        self.assertEqual(message["data_date"], DATES[-1])
        self.assertIn("reconstructed close total-return", message["data_view"])
        self.assertIn("Instant free close reinvestment", message["dividend_assumption"])
        self.assertFalse(message["next_open_supported"])
        self.assertFalse(message["order_submission"])
        self.assertTrue(message["not_deployed"])


if __name__ == "__main__":
    unittest.main()
