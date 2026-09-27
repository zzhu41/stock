"""Synthetic selection math only; no real V12 candidate returns are loaded."""
from copy import deepcopy
import math
import unittest

from numpy import nextafter

from v12.selection import SCENARIOS, select


END = "2026-09-24"
CONTROLS = {name: "control_" + name for name in ("v9", "v91", "v92", "simple", "h")}


def record(cid, cagr=.56, dd=-.16, tail=.70, complexity=1, turnover=30., selectable=True, kind="single"):
    scenarios = {}
    for name in SCENARIOS:
        scenarios[name] = dict(
            full=dict(cagr=cagr if name == "close_1bp" else .45,
                      max_dd=dd if name == "close_1bp" else -.20),
            tail=dict(total_return=tail),
            blocks={block: dict(cagr=.40) for block in ("early", "middle", "recent")},
            turnover_equivalent=turnover)
    return dict(id=cid, scenarios=scenarios), dict(id=cid, kind=kind, selectable=selectable,
                                                   complexity=complexity, config={}, families=["synthetic"])


def fixture(*additions):
    rows, configs = [], []
    for cid in CONTROLS.values():
        row, config = record(cid, .50, -.20, .50, complexity=0, selectable=False)
        for name in SCENARIOS:
            if name != "close_1bp":
                row["scenarios"][name]["full"].update(cagr=.40, max_dd=-.25)
            for value in row["scenarios"][name]["blocks"].values():
                value["cagr"] = .35
        rows.append(row)
        configs.append(config)
    for row, config in additions:
        rows.append(row)
        configs.append(config)
    return rows, configs


class SelectionTests(unittest.TestCase):
    def test_thresholds_are_inclusive_and_one_representable_step_below_fails(self):
        candidate = record("edge", .5 + .01, -.2 + .005, .5)
        for name in ("close_11bp", "lag1_11bp"):
            candidate[0]["scenarios"][name]["full"].update(cagr=.4, max_dd=-.25)
        for metrics in candidate[0]["scenarios"]["close_1bp"]["blocks"].values():
            metrics["cagr"] = .35 - .05
        rows, configs = fixture(candidate)
        selected = select(rows, configs, CONTROLS, END)
        self.assertEqual(selected["primary"], "edge")
        self.assertEqual(selected["qualified_count"], 1)
        self.assertTrue(all(selected["checks"]["edge"].values()))
        fields = [("close_1bp", "full", "cagr", "main_cagr"),
                  ("close_1bp", "full", "max_dd", "main_drawdown"),
                  ("close_1bp", "tail", "total_return", "tail_return"),
                  ("close_11bp", "full", "cagr", "close_11bp_cagr"),
                  ("close_11bp", "full", "max_dd", "close_11bp_drawdown"),
                  ("lag1_11bp", "full", "cagr", "lag1_11bp_cagr"),
                  ("lag1_11bp", "full", "max_dd", "lag1_11bp_drawdown")]
        for scenario, period, metric, check in fields:
            changed = deepcopy(rows)
            value = changed[-1]["scenarios"][scenario][period]
            value[metric] = float(nextafter(value[metric], -math.inf))
            result = select(changed, configs, CONTROLS, END)
            self.assertIsNone(result["primary"], check)
            self.assertFalse(result["checks"]["edge"][check])
        for block in ("early", "middle", "recent"):
            changed = deepcopy(rows)
            value = changed[-1]["scenarios"]["close_1bp"]["blocks"][block]
            value["cagr"] = float(nextafter(value["cagr"], -math.inf))
            result = select(changed, configs, CONTROLS, END)
            self.assertFalse(result["checks"]["edge"][block + "_cagr"])
            self.assertIsNone(result["primary"])

    def test_best_signed_drawdown_and_returns_are_recomputed_per_scenario(self):
        rows, configs = fixture(record("candidate"))
        mapping = {r["id"]: r for r in rows}
        mapping[CONTROLS["simple"]]["scenarios"]["close_1bp"]["full"]["max_dd"] = -.15
        mapping[CONTROLS["v91"]]["scenarios"]["close_11bp"]["full"]["cagr"] = .6
        mapping[CONTROLS["v9"]]["scenarios"]["lag1_11bp"]["full"]["max_dd"] = -.1
        result = select(rows, configs, CONTROLS, END)
        checks = result["checks"]["candidate"]
        self.assertFalse(checks["main_drawdown"])
        self.assertFalse(checks["close_11bp_cagr"])
        self.assertFalse(checks["lag1_11bp_drawdown"])
        self.assertEqual(result["baselines"]["close_1bp"]["best_drawdown"], -.15)
        self.assertEqual(result["baselines"]["close_11bp"]["best_cagr"], .6)
        self.assertIsNone(result["primary"])

    def test_tail_uses_v92_not_h_or_the_best_control(self):
        rows, configs = fixture(record("candidate", tail=.55))
        mapping = {r["id"]: r for r in rows}
        mapping[CONTROLS["h"]]["scenarios"]["close_1bp"]["tail"]["total_return"] = .4
        mapping[CONTROLS["simple"]]["scenarios"]["close_1bp"]["tail"]["total_return"] = .9
        self.assertEqual(select(rows, configs, CONTROLS, END)["primary"], "candidate")
        mapping[CONTROLS["v92"]]["scenarios"]["close_1bp"]["tail"]["total_return"] = .6
        self.assertIsNone(select(rows, configs, CONTROLS, END)["primary"])

    def test_pareto_filter_precedes_stress_tie_and_keeps_equal_coordinates(self):
        weaker = record("weaker")
        for name in ("close_11bp", "lag1_11bp"):
            for value in weaker[0]["scenarios"][name]["blocks"].values():
                value["cagr"] = 9.
        stronger = record("stronger", .57, -.15, .71)
        equal = deepcopy(stronger)
        equal[0]["id"] = equal[1]["id"] = "stronger_tie"
        rows, configs = fixture(weaker, stronger, equal)
        selected = select(rows, configs, CONTROLS, END)
        self.assertEqual(selected["qualified_count"], 3)
        self.assertEqual(selected["pareto_ids"], ["stronger", "stronger_tie"])
        self.assertEqual(selected["primary"], "stronger")

    def test_minimum_standardized_advantage_beats_only_highest_cagr(self):
        rows, configs = fixture(record("highest_cagr", .60, -.185, .80), record("balanced"))
        selected = select(rows, configs, CONTROLS, END)
        self.assertEqual(set(selected["pareto_ids"]), {"highest_cagr", "balanced"})
        self.assertEqual(selected["primary"], "balanced")
        self.assertAlmostEqual(selected["scores"]["balanced"]["min_margin"], 2.)
        self.assertAlmostEqual(selected["scores"]["highest_cagr"]["min_margin"], .75)
        self.assertEqual(selected["top_return"], "highest_cagr")
        self.assertEqual(selected["top_tail"], "highest_cagr")
        self.assertEqual(selected["least_drawdown"], "balanced")

    def test_tie_break_order_stress_then_complexity_then_turnover_then_id(self):
        a, b = record("a", complexity=2, turnover=20.), record("b", complexity=3, turnover=30.)
        b[0]["scenarios"]["close_11bp"]["blocks"]["early"]["cagr"] = .41
        # Improving only one block leaves the minimum unchanged.
        rows, configs = fixture(a, b)
        self.assertEqual(select(rows, configs, CONTROLS, END)["primary"], "a")
        for name in ("close_11bp", "lag1_11bp"):
            for block in b[0]["scenarios"][name]["blocks"].values():
                block["cagr"] = .41
        rows, configs = fixture(a, b)
        self.assertEqual(select(rows, configs, CONTROLS, END)["primary"], "b")
        b = (deepcopy(a[0]), b[1]); b[0]["id"] = "b"
        rows, configs = fixture(a, b)
        self.assertEqual(select(rows, configs, CONTROLS, END)["primary"], "a")
        b[1]["complexity"] = 2
        b[0]["scenarios"]["close_1bp"]["turnover_equivalent"] = 10.
        rows, configs = fixture(a, b)
        self.assertEqual(select(rows, configs, CONTROLS, END)["primary"], "b")
        b[0]["scenarios"]["close_1bp"]["turnover_equivalent"] = 20.
        rows, configs = fixture(a, b)
        self.assertEqual(select(list(reversed(rows)), list(reversed(configs)), CONTROLS, END)["primary"], "a")

    def test_controls_benchmarks_and_unselectable_rows_cannot_replace_no_winner(self):
        rows, configs = fixture(record("fail", dd=-.3),
                                record("benchmark", .9, -.01, 9., kind="benchmark"),
                                record("diagnostic", .9, -.01, 9., selectable=False))
        for config in configs[:5]:
            config["selectable"] = True
        result = select(rows, configs, CONTROLS, END)
        self.assertIsNone(result["primary"])
        self.assertEqual(result["qualified_count"], 0)
        self.assertEqual(result["pareto_ids"], [])
        self.assertEqual(result["top_return"], "fail")
        self.assertEqual(result["selectable_count"], 1)
        empty_rows, empty_configs = fixture()
        empty = select(empty_rows, empty_configs, CONTROLS, END)
        self.assertIsNone(empty["top_return"])

    def test_each_historical_cutoff_uses_only_existing_blocks_and_its_tail(self):
        for year, blocks in ((2017, ["early"]), (2019, ["early", "middle"]),
                             (2021, ["early", "middle"]), (2023, ["early", "middle", "recent"])):
            rows, configs = fixture(record("candidate"))
            end = str(year) + "-12-31"
            for row in rows:
                row["period_end"] = end
                for name in SCENARIOS:
                    row["scenarios"][name]["tail"].update(start=str(year)+"-01-01", end=end)
            original = deepcopy((rows, configs))
            first = select(rows, configs, CONTROLS, end)
            for row in rows:
                row["report_2026"] = {"cagr": 9999.}
                for name in SCENARIOS:
                    row["scenarios"][name]["future_2026"] = {"cagr": -9999.}
                    for block in ("early", "middle", "recent", "future"):
                        if block not in blocks:
                            row["scenarios"][name]["blocks"][block] = {"cagr": float("nan")}
            polluted = deepcopy((rows, configs))
            second = select(rows, configs, CONTROLS, end)
            self.assertEqual(first, second)
            self.assertEqual(second["available_blocks"], blocks)
            self.assertEqual(second["tail_period"], [str(year)+"-01-01", end])
            # Mapping or list inputs are equivalent, and input is never changed.
            self.assertEqual(second, select({r["id"]:r for r in rows}, {c["id"]:c for c in configs}, CONTROLS, end))
            for index,row in enumerate(rows):
                self.assertEqual(row["scenarios"]["close_1bp"]["full"], polluted[0][index]["scenarios"]["close_1bp"]["full"])
            self.assertEqual(configs, original[1])
            self.assertEqual(len(second["scores"]["candidate"]["stress_block_excess"]), 2*len(blocks))

    def test_ignored_low_fee_lag_and_extra_periods_do_not_add_gates(self):
        rows, configs = fixture(record("candidate"))
        first = select(rows, configs, CONTROLS, END)
        rows[-1]["scenarios"]["lag1_1bp"]["full"].update(cagr=-.99, max_dd=-.99)
        rows[-1]["scenarios"]["lag1_1bp"]["tail"]["total_return"] = -.99
        rows[-1]["scenarios"]["close_1bp"]["future_extra"] = {"cagr": 100.}
        second = select(rows, configs, CONTROLS, END)
        self.assertEqual(first["primary"], second["primary"])
        self.assertEqual(first["checks"], second["checks"])
        self.assertEqual(first["scores"], second["scores"])

    def test_missing_nonfinite_positive_drawdown_or_future_metric_dates_stop_selection(self):
        rows, configs = fixture(record("candidate"))
        original = deepcopy((rows, configs))
        select(rows, configs, CONTROLS, END)
        self.assertEqual((rows, configs), original)
        mutators = (
            lambda r: r[-1]["scenarios"].pop("lag1_1bp"),
            lambda r: r[-1]["scenarios"]["close_1bp"]["blocks"].pop("recent"),
            lambda r: r[-1]["scenarios"]["close_1bp"]["full"].update(cagr=float("nan")),
            lambda r: r[-1]["scenarios"]["close_1bp"]["full"].update(max_dd=.1),
            lambda r: r[-1]["scenarios"]["close_1bp"]["full"].update(end="2026-09-25"),
            lambda r: r[-1]["scenarios"]["close_1bp"]["tail"].update(start="2025-01-01"),
            lambda r: r[-1]["scenarios"]["close_1bp"].update(turnover_equivalent=-1.),
            lambda r: r[-1].update(period_end="2021-12-31"),
        )
        for change in mutators:
            copy = deepcopy(rows); change(copy)
            with self.assertRaises(ValueError):
                select(copy, configs, CONTROLS, END)
        with self.assertRaises(ValueError):
            select(rows+[deepcopy(rows[-1])], configs, CONTROLS, END)
        with self.assertRaises(ValueError):
            select(rows, configs[:-1], CONTROLS, END)
        with self.assertRaises(ValueError):
            select(rows, configs, dict(CONTROLS, v9=CONTROLS["h"]), END)


if __name__ == "__main__":
    unittest.main()
