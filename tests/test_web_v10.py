"""Frozen website curve fidelity and fail-closed provenance tests; read-only data."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import web_v10


class FrozenWebV10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = json.loads(web_v10.PROFILES.read_text())["variants"]
        cls.files = (web_v10.PROFILES, web_v10.SELECTED_PATHS, web_v10.RECEIPT,
                     web_v10.FEATURE_CACHE, web_v10.FEATURE_META, web_v10.TR_MANIFEST)
        cls.before = {path: (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None
                      for path in cls.files}
        with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden in web export")):
            cls.versions = web_v10.export_versions()

    def test_full_metrics_preserve_frozen_high_precision_results(self):
        expected = {"v10-h": (53.31044487823024, -23.883339122470115, 315, 27),
                    "v9.2-tr": (46.31748938609171, -24.164829045993052, 272, 26)}
        self.assertEqual(set(self.versions), set(expected))
        for vid, (annual, dd, switches, crashes) in expected.items():
            version = self.versions[vid]
            self.assertAlmostEqual(version["metrics"]["ann"], annual, places=11)
            self.assertAlmostEqual(version["metrics"]["max_dd"], dd, places=11)
            self.assertEqual(version["metrics"]["switches"], switches)
            self.assertEqual(len(version["crash_buys"]), crashes)
            self.assertEqual(version["metrics"]["days"], 3097)
            self.assertAlmostEqual(version["metrics"]["total_ret"],
                                   100 * (version["metrics"]["nav"] - 1), places=10)
            self.assertEqual(version["full_metrics"]["cagr"] * 100, version["metrics"]["ann"])

    def test_exact_unrounded_nav_returns_trade_and_crash_dates_are_compatible(self):
        for version in self.versions.values():
            daily, returns = version["daily"], version["daily_returns"]
            self.assertEqual(len(daily), len(returns))
            self.assertEqual(daily[0][1], 1.)
            self.assertEqual(returns[0], 0.)
            np.testing.assert_array_equal([row[1] for row in daily], np.cumprod(1 + np.asarray(returns)))
            self.assertTrue(any(row[1] != round(row[1], 4) for row in daily))
            self.assertEqual(daily[-1][1], version["metrics"]["nav"])
            by_date = {row[0]: (index, row) for index, row in enumerate(daily)}
            for date, previous, target, nav in version["trades"]:
                index, row = by_date[date]
                self.assertEqual(row[1:], [nav, target])
                self.assertEqual(previous, daily[index - 1][2] if index else None)
            self.assertEqual(len(version["trades"]) - 1, version["metrics"]["switches"])
            filled = {(row[0], row[2]) for row in version["trades"]}
            self.assertTrue(all(tuple(row) in filled for row in version["crash_buys"]))
            self.assertEqual(len(version["crash_buys"]), len({tuple(row) for row in version["crash_buys"]}))
            self.assertEqual(daily[-1][2], "513100")

    def test_metadata_keeps_research_clock_and_shadow_history_separate(self):
        for version in self.versions.values():
            meta = version["metadata"]
            self.assertEqual(meta["basis"], "corrected_tr_same_close")
            self.assertEqual(meta["kind"], "research")
            self.assertEqual(meta["period"], dict(start="2014-01-02", end="2026-09-24"))
            self.assertEqual(version["daily"][0][0], "2014-01-02")
            self.assertEqual(version["daily"][-1][0], "2026-09-24")
            self.assertTrue(meta["frozen"])
            self.assertTrue(meta["first_session_free"])
            self.assertFalse(meta["shadow_nav_included"])
            self.assertFalse(meta["execution_price_provided"])
            self.assertFalse(meta["clean_oos"])
            self.assertEqual(meta["annualization_sessions"], 244)
            self.assertIn("当天收益主要归此前持仓", meta["holding_convention"])
            self.assertTrue(version["metrics_convention"]["returns_aligned_with_daily"])
            self.assertTrue(version["metrics_convention"]["first_return_included"])
            self.assertTrue(meta["warnings"])
        # NumPy internals and nonfinite placeholders must not leak into JSON.
        json.dumps(self.versions, ensure_ascii=False, allow_nan=False)

    def test_benchmarks_are_from_corrected_seed_on_the_same_calendar(self):
        for vid, version in self.versions.items():
            for code in ("510300", "518880"):
                benchmark = version["benchmarks"][code]
                with (web_v10.TR_SNAPSHOTS / (code + ".csv")).open(newline="") as stream:
                    closes = {row[0]: float(row[2]) for row in csv.reader(stream) if row}
                self.assertEqual(benchmark["daily"][0], ["2014-01-02", 1.])
                self.assertEqual(benchmark["daily"][-1][1], closes["2026-09-24"] / closes["2014-01-02"])
                self.assertEqual([row[0] for row in benchmark["daily"]], [row[0] for row in version["daily"]])
                self.assertEqual(benchmark["metadata"]["basis"], version["metadata"]["basis"])
                self.assertEqual(benchmark["metadata"]["period"], version["metadata"]["period"])
                self.assertEqual(benchmark["metadata"]["fees_charged"], 0.)
                self.assertEqual(benchmark["daily_returns"][0], 0.)

    def test_export_did_not_rewrite_frozen_artifacts_or_caches(self):
        for path, previous in self.before.items():
            self.assertTrue(path.exists())
            if previous is not None:
                self.assertEqual(previous, (path.stat().st_size, path.stat().st_mtime_ns))
            else:
                self.assertIn(path, (web_v10.FEATURE_CACHE, web_v10.FEATURE_META))

    def test_tampered_selected_path_refuses_export_instead_of_reusing_or_recomputing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected_paths.json"
            path.write_text("{}")
            with patch.object(web_v10, "SELECTED_PATHS", path), \
                    patch.object(web_v10, "load_profile", side_effect=lambda name: deepcopy(self.profiles[name])), \
                    patch.object(web_v10.reference, "run_reference", side_effect=AssertionError("unverified curve")):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    web_v10.export_versions()

    def test_metadata_date_axis_and_reference_source_are_individually_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            meta_path = Path(directory) / "features.json"
            meta = json.loads(web_v10.FEATURE_META.read_text())
            meta["dates"][-1] = "2099-01-01"
            meta_path.write_text(json.dumps(meta))
            with patch.object(web_v10, "FEATURE_META", meta_path), \
                    patch.object(web_v10, "load_profile", side_effect=lambda name: deepcopy(self.profiles[name])):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    web_v10.export_versions()
            source_path = Path(directory) / "reference.py"
            source_path.write_text("# changed implementation\n")
            with patch.object(web_v10.reference, "__file__", str(source_path)), \
                    patch.object(web_v10, "load_profile", side_effect=lambda name: deepcopy(self.profiles[name])):
                with self.assertRaisesRegex(ValueError, "reference implementation changed"):
                    web_v10.export_versions()


class ColdDerivedCacheTests(unittest.TestCase):
    def test_absent_cache_rebuilds_once_and_checks_exact_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "derived"
            array, metadata = root / "features.npz", root / "features.json"
            contents = {array: b"fixed derived arrays", metadata: b"fixed metadata"}
            pins = dict(feature_cache_sha256=hashlib.sha256(contents[array]).hexdigest(),
                        feature_metadata_sha256=hashlib.sha256(contents[metadata]).hexdigest())
            def rebuild():
                for path, raw in contents.items():
                    path.write_bytes(raw)
            with patch.object(web_v10, "FEATURE_CACHE", array), patch.object(web_v10, "FEATURE_META", metadata), \
                    patch.object(web_v10, "build_frozen_features", side_effect=rebuild) as builder:
                web_v10._ensure_feature_cache(pins)
                web_v10._ensure_feature_cache(pins)
                builder.assert_called_once_with()
            self.assertEqual(array.read_bytes(), contents[array])
            self.assertEqual(metadata.read_bytes(), contents[metadata])

    def test_existing_wrong_file_is_not_overwritten_even_when_other_file_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            array, metadata = Path(directory) / "features.npz", Path(directory) / "features.json"
            metadata.write_bytes(b"changed metadata")
            pins = dict(feature_cache_sha256="0" * 64, feature_metadata_sha256="1" * 64)
            with patch.object(web_v10, "FEATURE_CACHE", array), patch.object(web_v10, "FEATURE_META", metadata), \
                    patch.object(web_v10, "build_frozen_features") as builder:
                with self.assertRaisesRegex(ValueError, "input changed"):
                    web_v10._ensure_feature_cache(pins)
                builder.assert_not_called()
            self.assertEqual(metadata.read_bytes(), b"changed metadata")
            self.assertFalse(array.exists())

    def test_rebuilt_different_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            array, metadata = Path(directory) / "features.npz", Path(directory) / "features.json"
            pins = dict(feature_cache_sha256="0" * 64, feature_metadata_sha256="1" * 64)
            def wrong_rebuild():
                array.write_bytes(b"different arrays")
                metadata.write_bytes(b"different metadata")
            with patch.object(web_v10, "FEATURE_CACHE", array), patch.object(web_v10, "FEATURE_META", metadata), \
                    patch.object(web_v10, "build_frozen_features", side_effect=wrong_rebuild):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    web_v10._ensure_feature_cache(pins)

if __name__ == "__main__":
    unittest.main()
