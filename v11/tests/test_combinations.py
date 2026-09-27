"""Synthetic-only checks of the explicitly adaptive V11-B component generator."""
from copy import deepcopy
import unittest

from v10_deep.schema import baseline, identifier
from v11.registry import canonical
from v11.combinations import generate, RISK_KEYS, TURN_KEYS
from v11.scan import select


def fixture():
    h = baseline()
    h.update(score="wls20_smooth3", ma="ma180", panic_mode="volatility", panic=1.5)
    candidates, controls = {}, {}
    def add(family, changes=None, base=None):
        value = deepcopy(base or h)
        value.update(changes or {})
        value = canonical(value)
        digest = identifier(value)
        cid = "v11_" + digest[:20]
        if cid in candidates:
            candidates[cid]["families"] = sorted(set(candidates[cid]["families"] + [family]))
        else:
            candidates[cid] = dict(value, id=cid, hash=digest, families=[family], parents=[], stage="mechanisms")
        return cid
    controls["h"] = add("control_h")
    controls["v92"] = add("control_v92", base=baseline())
    controls["simple"] = add("control_simple", {"min_hold": 2}, base=baseline())
    for i in range(5):
        add("robust_regression", {"score": "robust_%d" % i})
        add("multi_horizon_consensus", {"score": "consensus_%d" % i})
        add("prior_risk_scale", {"panic": 1. + i * .25, "risk_context": "prior20"})
        add("fixed_risk_scale", {"panic_mode": "fixed", "panic": .02 + .01 * i})
        add("healthy_momentum_retention", {"buffer": .01 + i * .005, "global_buffer": .01 + i * .005,
                                          "gold_buffer": .01 + i * .005, "min_hold": i})
        add("healthy_rank_retention", {"buffer_mode": "rank", "rank_keep": 1 + i % 3, "min_hold": i})
    registry = dict(candidates=sorted(candidates.values(), key=lambda c:c["id"]), controls=controls, score_specs=[])
    rows = []
    for i, candidate in enumerate(registry["candidates"]):
        delta = 0 if candidate["id"] in controls.values() else .001 * (i + 1)
        scenarios = {}
        for name, growth, dd in (("close_1bp", .4, -.2), ("close_11bp", .36, -.22),
                                 ("lag1_1bp", .3, -.24), ("lag1_11bp", .28, -.25)):
            scenarios[name] = dict(full=dict(cagr=growth + delta, max_dd=dd),
                                  blocks={"early":dict(cagr=growth + delta), "late":dict(cagr=growth + delta)},
                                  switches=30-i)
        rows.append(dict(id=candidate["id"], families=list(candidate["families"]), scenarios=scenarios))
    return registry, rows


class CombinationGenerationTests(unittest.TestCase):
    def test_component_caps_full_product_and_controls_are_separate_references(self):
        registry, rows = fixture()
        result = generate(registry, rows)
        self.assertEqual([len(result["components"][name]) for name in ("score","risk","turnover")], [6,5,5])
        self.assertEqual(result["raw_count"], 150)
        self.assertLessEqual(result["combination_unique_count"], 150)
        self.assertLessEqual(result["unique_count"], 153)
        self.assertEqual(set(result["controls"].values()), set(registry["controls"].values()))
        self.assertTrue(set(registry["controls"].values()).issubset({c["id"] for c in result["candidates"]}))
        self.assertIn("Adaptive", result["scope"])
        self.assertIn("not initially preregistered", result["scope"])

    def test_qualified_component_precedes_more_profitable_unqualified_component(self):
        registry, rows = fixture()
        selection = select(rows, registry["candidates"], registry["controls"])
        family = [c for c in registry["candidates"] if "robust_regression" in c["families"]]
        rejected = family[0]
        selection["checks"][rejected["id"]] = {"eligible": False}
        selection["scores"][rejected["id"]]["worst_block_excess"] = 99999
        result = generate(registry, rows, selection)
        sources = {item["source"] for item in result["components"]["score"]}
        self.assertNotIn(rejected["id"], sources)

    def test_top_two_per_family_follow_registered_dev_score_order(self):
        registry, rows = fixture()
        selection = select(rows, registry["candidates"], registry["controls"])
        result = generate(registry, rows, selection)
        for family, component, keys in (
            ("robust_regression","score",("score",)), ("multi_horizon_consensus","score",("score",)),
            ("prior_risk_scale","risk",RISK_KEYS), ("fixed_risk_scale","risk",RISK_KEYS),
            ("healthy_momentum_retention","turnover",TURN_KEYS), ("healthy_rank_retention","turnover",TURN_KEYS)):
            members = [c for c in registry["candidates"] if family in c["families"] and c["id"] not in registry["controls"].values()]
            def order(c):
                score = selection["scores"][c["id"]]
                return (not all(selection["checks"][c["id"]].values()), -score["worst_block_excess"],
                        -score["median_block_excess"], score["switches"], score["complexity"], c["id"])
            members.sort(key=order)
            selected = [x["source"] for x in result["components"][component] if x["source"] in {c["id"] for c in members}]
            self.assertEqual(selected, [c["id"] for c in members[:2]])

    def test_every_combination_is_canonical_and_changes_only_allowed_h_components(self):
        registry, rows = fixture()
        result = generate(registry, rows)
        source = {c["id"]:c for c in registry["candidates"]}
        h = canonical(source[registry["controls"]["h"]])
        allowed = set(RISK_KEYS) | set(TURN_KEYS) | {"score"}
        for candidate in result["candidates"]:
            self.assertEqual(identifier(candidate), candidate["hash"])
            self.assertEqual(candidate["id"], "v11_"+candidate["hash"][:20])
            if candidate["id"] not in result["membership"]:
                continue
            values = canonical(candidate)
            for key,value in h.items():
                if key not in allowed:
                    self.assertEqual(values[key],value, key)
            self.assertTrue(set(candidate["parents"]).issubset(source))
            self.assertEqual(candidate["parents"],sorted(set(candidate["parents"])))
            for name,keys in (("score",("score",)), ("risk",RISK_KEYS), ("turnover",TURN_KEYS)):
                values = {k:candidate[k] for k in keys}
                matching = [part for part in result["components"][name] if part["values"]==values]
                self.assertTrue(any(part["source"] in candidate["parents"] for part in matching),name)

    def test_duplicate_components_and_existing_a_combinations_are_explicit(self):
        registry, rows = fixture()
        # Duplicate A row/config, with no different mechanism, cannot create a
        # new component or a hidden weighting in the Cartesian product.
        original = generate(registry, rows)
        registry["candidates"].append(deepcopy(registry["candidates"][-1]))
        rows.append(deepcopy(rows[-1]))
        repeated = generate(registry, rows)
        self.assertEqual(original,repeated)
        self.assertEqual(len({c["id"] for c in repeated["candidates"]}),repeated["unique_count"])
        originals = {c["id"] for c in registry["candidates"]}
        for cid, flags in repeated["membership"].items():
            self.assertEqual(flags["previously_in_stage_a"],cid in originals)
        self.assertTrue(repeated["membership"][registry["controls"]["h"]]["previously_in_stage_a"])

    def test_future_reporting_fields_and_a_primary_do_not_drive_component_choice(self):
        registry, rows = fixture()
        original = generate(registry, rows)
        selection = select(rows,registry["candidates"],registry["controls"])
        changed_selection = deepcopy(selection)
        changed_selection.update(primary="fake_2026_winner", top_return="fake_confirmation_winner", confirmation={"best":"fake"})
        self.assertEqual(original,generate(registry,rows,changed_selection))
        changed_rows = deepcopy(rows)
        for i, row in enumerate(changed_rows):
            row["report_only_2026"] = {"cagr":1e12*i}
            row["confirmation"] = {"cagr":-1e12*i}
            row["scenarios"]["2026_diagnostic"] = {"cagr":1e20*i}
        self.assertEqual(original,generate(registry,changed_rows))

    def test_input_objects_remain_unchanged_and_b_selector_cannot_rename_controls(self):
        registry, rows = fixture()
        selection=select(rows,registry["candidates"],registry["controls"])
        before=deepcopy((registry,rows,selection))
        result=generate(registry,rows,selection)
        self.assertEqual(before,(registry,rows,selection))
        controls=set(result["controls"].values())
        base_row=next(row for row in rows if row["id"]==result["controls"]["h"])
        fake=[]
        for candidate in result["candidates"]:
            row=deepcopy(base_row);row.update(id=candidate["id"],families=candidate["families"])
            if candidate["id"] not in controls:
                row["scenarios"]["close_1bp"]["full"]["cagr"] = .10
            fake.append(row)
        b_selection=select(fake,result["candidates"],result["controls"])
        self.assertIsNone(b_selection["primary"])
        self.assertEqual(b_selection["parents"],[])
        self.assertEqual(selection,before[2])


if __name__=='__main__':
    unittest.main()
