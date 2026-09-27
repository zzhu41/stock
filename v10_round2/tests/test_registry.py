"""The second-round budget and simplifications are fixed before simulations."""
import unittest

from v10_next.candidates import get_candidate as first_round_candidate
from v10_round2.registry import CANDIDATES, HIGH_RETURN_IDS, STABILITY_IDS, get_candidate, registry


class RegistryTests(unittest.TestCase):
    def test_budget_controls_and_routes(self):
        self.assertEqual(len(CANDIDATES), 9)
        self.assertEqual(len({c["id"] for c in CANDIDATES}), 9)
        self.assertEqual(sum(c["family"] == "control" for c in CANDIDATES), 2)
        self.assertIn("c_v91", HIGH_RETURN_IDS)
        self.assertNotIn("c_v92", HIGH_RETURN_IDS)
        self.assertEqual(len(STABILITY_IDS), 4)

    def test_high_return_union_has_no_volume_channel(self):
        self.assertEqual(get_candidate("c_v92")["channels"], ("deep", "qvix", "volume"))
        for name in ("h_qvix_only", "h_qvix_score25_30"):
            self.assertEqual(get_candidate(name)["channels"], ("deep", "qvix"))
        self.assertEqual(get_candidate("h_qvix_score25_30")["score_windows"], (25, 30))
        self.assertEqual(get_candidate("s_v91_wls30_no_crash")["overrides"], {"crash_mom5": 0.0})

    def test_configs_are_independent_and_do_not_change_first_round(self):
        before = first_round_candidate("control_v91")
        value = get_candidate("s_accounts25_30")
        value["components"][0]["overrides"]["crash_mom5"] = 99
        self.assertEqual(get_candidate("s_accounts25_30")["components"][0]["overrides"], {})
        self.assertEqual(first_round_candidate("control_v91"), before)
        registered = registry()
        registered["candidates"][0]["risk_weight"] = 100
        self.assertEqual(get_candidate("c_v91")["risk_weight"], 1.0)

    def test_known_window_and_account_capital_are_explicit(self):
        self.assertTrue(get_candidate("s_v91_wls30")["previously_seen"])
        for name, windows in (("s_accounts25_30", (25, 30)), ("s_accounts20_25_30", (20, 25, 30))):
            candidate = get_candidate(name)
            self.assertEqual(tuple(c["score_windows"][0] for c in candidate["components"]), windows)
            self.assertTrue(all(w == 1 / len(windows) for w in candidate["initial_weights"]))
            self.assertEqual(candidate["rebalance"], "never")
            self.assertFalse(candidate["netting"])
            self.assertFalse(candidate["capital_transfers"])


if __name__ == "__main__":
    unittest.main()
