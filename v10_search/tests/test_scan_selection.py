"""Synthetic selection-boundary checks; never executes a market-data search."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from v10_next.frozen import strategy as frozen_strategy
from v10_search import scan
from v10_search.data import ASSET_ORDER, BENCHMARK, CASH, GOLD
from v10_search.policy import SearchPolicy
from v10_search.registry import PRESETS, build_registry, candidate_hash, controls


def row(name, development, early=None, late=None, full=-.99, mode="ma250"):
    return dict(id=name, development={"cagr": development},
                early={"cagr": development if early is None else early},
                late={"cagr": development if late is None else late},
                full={"cagr": full}, validation={"cagr": full},
                report_only_2026={"cagr": full}, family="synthetic", regime_mode=mode,
                return_path_hash="synthetic_" + name)


class DevelopmentOnly(dict):
    """Reading a non-development field fails the test immediately."""
    def __getitem__(self, key):
        if key not in ("id", "development", "early", "late"):
            raise AssertionError("Selection read a prohibited field: " + key)
        return super().__getitem__(key)


class ScanSelectionTests(unittest.TestCase):
    def test_choose_never_reads_full_validation_or_2026(self):
        benchmark = DevelopmentOnly(row("benchmark", .20, full=10000))
        rows = [DevelopmentOnly(row("winner", .35, full=-10000)),
                DevelopmentOnly(row("hindsight", .21, early=.10, full=1e9))]
        selected = scan.choose(rows, benchmark)
        self.assertEqual(selected["champion"], "winner")
        self.assertEqual(selected["top_ids"], ["winner"])
        self.assertFalse(selected["selection_uses_validation"])

    def test_top20_is_frozen_by_development_rank_and_input_order_does_not_matter(self):
        rows = [row("c%02d" % i, .30 + i * .001, full=100 - i) for i in range(25)]
        rows.append(row("failed_early", 99, early=.19, late=99, full=1e10))
        before = deepcopy(rows)
        selected = scan.choose(list(reversed(rows)), row("benchmark", .20))
        expected = ["c%02d" % i for i in range(24, 4, -1)]
        self.assertEqual(selected["top_ids"], expected)
        self.assertEqual(selected["champion"], "c24")
        self.assertEqual(selected["eligible_count"], 25)
        self.assertEqual(rows, before)
        for item in rows:
            item["full"]["cagr"] *= -100
            item["validation"]["cagr"] = 1e12 if item["id"] == "c00" else -1e12
            item["report_only_2026"]["cagr"] = 1e15
        self.assertEqual(scan.choose(rows, row("benchmark", .20))["top_ids"], expected)

    def test_all_three_strict_gates_apply_and_no_eligible_means_no_champion(self):
        rows = [row("equal", .20), row("early_equal", .4, early=.20),
                row("late_below", .4, late=.19), row("development_below", .19, early=.4, late=.4),
                row("below_numerical_tolerance", .20 + 5e-13)]
        selected = scan.choose(rows, row("benchmark", .20))
        self.assertIsNone(selected["champion"])
        self.assertEqual(selected["top_ids"], [])
        self.assertEqual(selected["eligible_count"], 0)
        tied = scan.choose([row("b", .3), row("a", .3)], row("benchmark", .2))
        self.assertEqual(tied["top_ids"], ["a", "b"])

    def test_evaluate_labels_hindsight_without_replacing_saved_champion(self):
        """Exercise the controller with four fake paths and temporary ledgers."""
        rows = [row("dev_winner", .35, full=.10),
                row("full_winner", .21, early=.10, full=.90, mode="always_bull"),
                row("open_stock", .19, full=.25, mode="open_stock"),
                row("all_assets", .18, full=.20, mode="all_assets")]
        configurations = [dict(id=r["id"], family=r["family"], regime_mode=r["regime_mode"], score_windows=[])
                          for r in rows]
        fake_controls = [dict(id=name) for name in ("c_v9", "c_v91", "c_v92")]
        fingerprints = {"synthetic_fixture": "fixed"}
        dates = ["2020-01-02", "2022-01-03"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "registry.json").write_text(json.dumps({"candidates": configurations}))
            (output / "development.json").write_text(json.dumps({"rows": rows}))
            np.save(output / "development_returns.npy", np.zeros((1, len(rows))))
            np.save(output / "full_returns.npy", np.zeros((2, len(rows))))
            registration = dict(fingerprints=fingerprints, registry_sha256=scan.sha(output / "registry.json"))
            (output / "registration.json").write_text(json.dumps(registration))
            selected = scan.choose(rows, row("benchmark", .20))
            selected.update(registration_sha256=scan.sha(output / "registration.json"),
                            development_sha256=scan.sha(output / "development.json"),
                            development_matrix_sha256=scan.sha(output / "development_returns.npy"))
            selection_path = output / "selection.json"
            selection_path.write_text(json.dumps(selected))
            original_selection = selection_path.read_bytes()
            with patch.object(scan, "OUT", output), \
                    patch.object(scan, "protect", return_value=126), \
                    patch.object(scan, "fingerprints", return_value=fingerprints), \
                    patch.object(scan, "setup"), \
                    patch.object(scan, "_FRAME", SimpleNamespace(dates=dates)), \
                    patch.object(scan, "controls", return_value=fake_controls), \
                    patch.object(scan, "simulate", side_effect=lambda config, *a, **kw: {"id": config["id"]}), \
                    patch.object(scan, "summaries", return_value=({"full": {"cagr": .2}}, np.zeros(2))), \
                    patch.object(scan, "scan_all", return_value=(deepcopy(rows), dates)), \
                    patch("v10_search.same_close.simulate_same_close", return_value={"synthetic": True}), \
                    patch("v10_next.metrics.summarize", return_value={"cagr": .2}), \
                    contextlib.redirect_stdout(io.StringIO()):
                scan.evaluate(workers=1)
            evaluated = json.loads((output / "evaluation.json").read_text())
            paths = json.loads((output / "selected_paths.json").read_text())
            self.assertEqual(selection_path.read_bytes(), original_selection)
            self.assertEqual(evaluated["selection"]["champion"], "dev_winner")
            self.assertEqual(evaluated["selection"]["top_ids"], ["dev_winner"])
            self.assertEqual(evaluated["hindsight_winner"]["id"], "full_winner")
            self.assertIn("not development selected", evaluated["hindsight_label"])
            self.assertEqual(set(paths["candidates"]), {"dev_winner", "full_winner"})

    def test_every_registered_config_is_accepted_by_the_bounded_policy_constructor(self):
        registered = build_registry()
        self.assertEqual(registered["unique_count"], 5156)
        presets = json.loads(PRESETS.read_text(encoding="utf-8"))

        class ConfigurationBank:
            def __init__(self):
                self.presets = presets

            def ranked(self, mode, windows):
                if mode == "wls":
                    if tuple(windows) not in ((20,), (25,), (30,), (40,)):
                        raise AssertionError("Unexpected WLS configuration")
                elif mode not in ("mom20_vol", "mom60_vol", "mom20") or windows:
                    raise AssertionError("Unexpected ranking configuration")
                return []  # No prices or returns are computed by construction.

        saved = {name: deepcopy(value) for name, value in vars(frozen_strategy).items()
                 if name.isupper() or name == "_state_bull"}
        bank = ConfigurationBank()
        try:
            for candidate in registered["candidates"] + controls():
                self.assertEqual(candidate_hash(candidate), candidate["candidate_hash"])
                self.assertEqual(candidate["risk_weight"], 1.0)
                self.assertEqual((candidate["gold"], candidate["cash"], candidate["benchmark_code"]),
                                 (GOLD, CASH, BENCHMARK))
                self.assertEqual(candidate["tie_break"], "fixed_input_asset_order")
                self.assertEqual(tuple(candidate["input_asset_order"]), tuple(ASSET_ORDER))
                policy = SearchPolicy(candidate, bank)
                self.assertEqual(policy.trade_pool, set(candidate["stock_pool"]) | set(candidate["global_pool"]) | {GOLD})
                self.assertEqual(policy.feature_pool, policy.trade_pool | {BENCHMARK})
        finally:
            for name, value in saved.items():
                setattr(frozen_strategy, name, value)


if __name__ == "__main__":
    unittest.main()
