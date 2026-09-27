"""Cash scenarios: independent economic identities, causal prefixes, no writes."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from v10_deep.precision import SCENARIOS, build_scenario, scenario_histories


def fixture():
    dates = ["2025-12-19", "2025-12-22", "2025-12-23", "2025-12-24", "2025-12-26"]
    raw = {
        "510300": [(d, 100., 100., 100.) for d in dates],
        "512890": [(d, p, p, v) for d, p, v in zip(dates, [100., 110., 55., 56., 58.],
                                                    [100., 100., 200., 200., 200.])],
        "515100": [(d, p, p, 100.) for d, p in zip(dates, [1.5, 1.543, 1.437, 1.45, 1.46])],
    }
    index_515100 = [1., 1.543 / 1.5, (1.437 + .105) / 1.5,
                   (1.437 + .105) / 1.5 * 1.45 / 1.437,
                   (1.437 + .105) / 1.5 * 1.46 / 1.437]
    corrected = {
        "510300": [(d, 1., 1., 100.) for d in dates],
        "512890": [(d, v, v, 200.) for d, v in zip(dates, [1., 1.1, 1.1, 1.12, 1.16])],
        "515100": [(d, v, v, 100.) for d, v in zip(dates, index_515100)],
    }
    assets = {}
    for code in raw:
        assets[code] = dict(rows=5, first_date=dates[0], last_date=dates[-1],
                            split_events=[], cash_events=[])
    assets["512890"]["split_events"] = [dict(date=dates[2], ratio=2, ratio_exact="2")]
    assets["515100"]["cash_events"] = [dict(date=dates[2], cash_per_old_share=.105,
                                             lower=.104, upper=.106,
                                             provenance="synthetic_rounded_cash")]
    return dict(end=dates[-1], assets=assets), raw, corrected


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_fixture(directory, manifest, raw, corrected):
    root = Path(directory)
    manifest = deepcopy(manifest)
    for code in raw:
        raw_path = root / "results/price_audit" / (code + "_raw_2010-01-01_" + manifest["end"] + ".json")
        csv_path = root / "corrected_snapshots" / (code + ".csv")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Tencent order: date, open, close, high, low, volume.
        vendor = [[d, str(o), str(c), str(max(o, c)), str(min(o, c)), str(v)] for d, o, c, v in raw[code]]
        raw_path.write_text(json.dumps(dict(rows=vendor)))
        with csv_path.open("w", newline="") as f:
            csv.writer(f).writerows(corrected[code])
        manifest["assets"][code].update(raw_sha256=digest(raw_path), sha256=digest(csv_path))
    (root / "corrected_manifest.json").write_text(json.dumps(manifest))
    return root


class PrecisionScenarioTests(unittest.TestCase):
    def test_fixed_cash_scenarios_have_expected_event_wealth(self):
        manifest, raw, base = fixture()
        original = deepcopy((manifest, raw, base))
        expected_cash = dict(cash_lower=.104, cash_upper=.106, official_515100=.1053)
        for name in SCENARIOS:
            histories, meta = scenario_histories(manifest, raw, base, name)
            with self.subTest(scenario=name):
                # First two observations are bit-for-bit preserved.
                self.assertEqual(histories["515100"][:2], base["515100"][:2])
                gross = histories["515100"][2][2] / histories["515100"][1][2]
                self.assertAlmostEqual(gross, (1.437 + expected_cash[name]) / 1.543)
                self.assertAlmostEqual(histories["515100"][3][2] / histories["515100"][2][2], 1.45 / 1.437)
                self.assertEqual(meta["changed_codes"], ["515100"])
                self.assertEqual(meta["calendar"], [r[0] for r in base["510300"]])
                self.assertFalse(meta["usable_for_next_open"])
                self.assertEqual(meta["assets"]["515100"]["cash_events"][0]["scenario_cash_per_old_share"], expected_cash[name])
        self.assertEqual((manifest, raw, base), original)

    def test_split_only_assets_do_not_acquire_fake_cash_or_new_volume(self):
        manifest, raw, base = fixture()
        for name in SCENARIOS:
            histories, meta = scenario_histories(manifest, raw, base, name)
            self.assertEqual(histories["512890"], base["512890"])
            self.assertEqual(meta["assets"]["512890"]["cash_events"], [])
            self.assertEqual(histories["512890"][2][2], histories["512890"][1][2])
            for code in base:
                self.assertEqual([r[3] for r in histories[code]], [r[3] for r in base[code]])
                self.assertEqual([r[0] for r in histories[code]], [r[0] for r in base[code]])
                self.assertTrue(all(r[1] == r[2] for r in histories[code]))

    def test_same_day_split_and_cash_use_previous_share_units(self):
        manifest, raw, base = fixture()
        c = "512890"
        raw[c][2] = (raw[c][2][0], 53., 53., 200.)
        # 2*53 + 4 = prior close 110: event itself has zero economic return.
        manifest["assets"][c]["cash_events"] = [dict(date=raw[c][2][0], cash_per_old_share=4., lower=3., upper=5.)]
        base[c][3] = (base[c][3][0], 1.1 * 56 / 53, 1.1 * 56 / 53, 200.)
        base[c][4] = (base[c][4][0], 1.1 * 58 / 53, 1.1 * 58 / 53, 200.)
        for name, amount in (("cash_lower", 3.), ("cash_upper", 5.)):
            h, _ = scenario_histories(manifest, raw, base, name)
            self.assertAlmostEqual(h[c][2][2] / h[c][1][2], (2 * 53 + amount) / 110)

    def test_appending_future_cash_and_split_does_not_change_tr_prefix(self):
        manifest, raw, base = fixture()
        earlier, _ = scenario_histories(manifest, raw, base, "cash_upper")
        m2, r2, b2 = deepcopy((manifest, raw, base))
        future = "2025-12-29"
        m2["end"] = future
        for code in r2:
            last = r2[code][-1]
            r2[code].append((future, last[1], last[2], last[3]))
            previous_index = b2[code][-1][2]
            b2[code].append((future, previous_index, previous_index, b2[code][-1][3]))
            m2["assets"][code].update(rows=6, last_date=future)
        # Future cash does not rebase any earlier raw close or TR index.
        r2["515100"][-1] = (future, 1.36, 1.36, 100.)
        m2["assets"]["515100"]["cash_events"].append(dict(date=future, cash_per_old_share=.1, lower=.09, upper=.11))
        # Latest-share-unit volumes legitimately scale when the data vintage is
        # extended across another split. They remain identical across scenarios
        # within that new vintage; earlier economic returns must still match.
        r2["512890"][-1] = (future, 29., 29., 400.)
        m2["assets"]["512890"]["split_events"].append(dict(date=future, ratio=2, ratio_exact="2"))
        b2["512890"] = [(d, o, c, 400.) for d, o, c, v in b2["512890"]]
        extended, _ = scenario_histories(m2, r2, b2, "cash_upper")
        for code in earlier:
            for a, b in zip(earlier[code], extended[code]):
                self.assertAlmostEqual(a[2], b[2], places=13)
            self.assertEqual([r[3] for r in extended[code]], [r[3] for r in b2[code]])

    def test_sensitivity_bounds_are_not_treated_as_new_event_detection(self):
        manifest, raw, base = fixture()
        manifest["assets"]["515100"]["cash_events"][0]["lower"] = -.001
        with self.assertRaisesRegex(ValueError, "cash interval"):
            scenario_histories(manifest, raw, base, "cash_lower")
        manifest, raw, base = fixture()
        manifest["assets"]["515100"]["cash_events"][0]["date"] = "2025-12-25"
        with self.assertRaisesRegex(ValueError, "quoted date"):
            scenario_histories(manifest, raw, base, "cash_lower")

    def test_rejects_bad_coverage_volume_and_point_reconstruction(self):
        for mutation, message in (("gap", "date axes"), ("volume", "normalized volume"), ("close", "reconstruct frozen close")):
            manifest, raw, base = fixture()
            if mutation == "gap":
                raw["515100"].pop(1)
            elif mutation == "volume":
                d, o, c, v = base["512890"][0]
                base["512890"][0] = (d, o, c, v + 1)
            else:
                d, o, c, v = base["515100"][-1]
                base["515100"][-1] = (d, o + .1, c + .1, v)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, message):
                scenario_histories(manifest, raw, base, "cash_lower")

    def test_official_replacement_is_only_the_declared_event(self):
        manifest, raw, base = fixture()
        m2 = deepcopy(manifest)
        m2["assets"]["515100"]["cash_events"] = []
        # A missing event must not silently produce an apparently valid
        # official scenario, even if prices are adjusted to fit that omission.
        no_cash = [(d, p / 1.5, p / 1.5, v) for d, o, p, v in raw["515100"]]
        base["515100"] = no_cash
        with self.assertRaisesRegex(ValueError, "event is absent"):
            scenario_histories(m2, raw, base, "official_515100")
        with self.assertRaisesRegex(ValueError, "Unknown fixed"):
            scenario_histories(manifest, raw, base, "choose_better_cash")

    def test_loader_preserves_old_csv_hashes_and_emits_source_fingerprints(self):
        manifest, raw, base = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = write_fixture(directory, manifest, raw, base)
            before = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
            metas = []
            for name in SCENARIOS:
                _, meta = build_scenario(name, root)
                metas.append(meta)
                self.assertEqual(meta["source_file_sha256"], before)
                self.assertEqual(len(meta["constructor_source_sha256"]), 64)
            after = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(len(set(m["histories_content_sha256"] for m in metas)), 3)
            self.assertEqual(len(set(m["calendar_sha256"] for m in metas)), 1)

    def test_loader_refuses_mutated_raw_file_before_building(self):
        manifest, raw, base = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = write_fixture(directory, manifest, raw, base)
            path = next((root / "results/price_audit").glob("*.json"))
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "source hash changed"):
                build_scenario("cash_lower", root)


if __name__ == "__main__":
    unittest.main()
