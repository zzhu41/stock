"""Past-prefix isolation checks for the separate research feature provider."""
import copy
import math
import unittest
from datetime import date, timedelta

from v10_next.data import Features, configure


def histories(n=380):
    days = [(date(2019, 1, 1) + timedelta(days=i)).isoformat() for i in range(n)]
    result = {}
    for k, code in enumerate(("510300", "518880", "511880")):
        result[code] = [(d, (10 + k) * (1 + .001 * i + .03 * math.sin(i / 7) + .01 * math.cos(i / 3)),
                        (10 + k) * (1 + .001 * i + .03 * math.sin(i / 7) + .01 * math.cos(i / 3)),
                        100 + i % 12) for i, d in enumerate(days)]
    return result, days


class FeaturesTests(unittest.TestCase):
    def setUp(self):
        self.histories, self.days = histories()
        self.features = Features(self.histories)
        configure(self.features.presets["v9.1"])

    def test_future_price_and_volume_perturbations_cannot_change_past_features(self):
        signal_date = self.days[300]
        changed = copy.deepcopy(self.histories)
        for code, rows in changed.items():
            changed[code] = [(d, o * 1000, c * 2000, v * 999) if d > signal_date else (d, o, c, v)
                             for d, o, c, v in rows]
        perturbed = Features(changed)
        for code in ("510300", "518880"):
            self.assertEqual(self.features(signal_date, code, (61, 122, 244)),
                             perturbed(signal_date, code, (61, 122, 244)))
        for windows in ((25,), (20, 25, 30)):
            self.assertEqual(self.features.table(signal_date, ("510300", "518880"), windows),
                             perturbed.table(signal_date, ("510300", "518880"), windows))

    def test_truncating_future_rows_preserves_past_table_and_momentum(self):
        signal_date = self.days[300]
        truncated = Features({code: [row for row in rows if row[0] <= signal_date]
                              for code, rows in self.histories.items()})
        self.assertEqual(self.features.table(signal_date, ("510300", "518880")),
                         truncated.table(signal_date, ("510300", "518880")))
        self.assertEqual(self.features(signal_date, "510300", (244,)),
                         truncated(signal_date, "510300", (244,)))

    def test_missing_current_bar_cannot_reuse_yesterday_as_today(self):
        signal_date = self.days[300]
        missing = Features({code: [row for row in rows if row[0] != signal_date]
                            for code, rows in self.histories.items()})
        self.assertIsNone(missing(signal_date, "510300", (244,)))
        self.assertEqual(missing.table(signal_date, ("510300", "518880")), [])
        self.assertFalse(missing.cash_available(signal_date))

    def test_exact_lookback_requires_one_more_price_than_return_lag(self):
        self.assertIsNone(self.features(self.days[243], "510300", (244,)))
        observation = self.features(self.days[244], "510300", (244,))
        expected = self.histories["510300"][244][2] / self.histories["510300"][0][2] - 1
        self.assertEqual(observation["returns"], {244: expected})

    def test_callers_cannot_modify_cached_table_indicators(self):
        date = self.days[300]
        before = self.features.table(date, ("510300", "518880"))
        original = copy.deepcopy(before)
        before[0][1]["score"] = -1e20
        self.assertEqual(self.features.table(date, ("510300", "518880")), original)


if __name__ == "__main__":
    unittest.main()
