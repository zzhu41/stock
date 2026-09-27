"""Economic identities and rejection behavior for the pure data-repair helper."""
import unittest

from v10_h_close.total_return_audit import affine_step, reconstruct_total_return


def rows(values, volume=100):
    return [("2020-01-%02d" % (i + 1), value, value, volume) for i, value in enumerate(values)]


class TotalReturnAuditTests(unittest.TestCase):
    def test_510880_additive_quote_example_restores_economic_returns(self):
        raw, q = rows([1.639, 1.581, 1.614]), rows([.227, .228, .261])
        result = reconstruct_total_return(raw, q, code="510880-example")
        self.assertEqual(len(result["cash_events"]), 1)
        self.assertAlmostEqual(result["cash_events"][0]["cash_per_old_share"], .059)
        first_gross = (1.581 + .059) / 1.639
        self.assertAlmostEqual(result["rows"][1][2], first_gross)
        self.assertAlmostEqual(result["rows"][2][2] / result["rows"][1][2], 1.614 / 1.581)
        self.assertGreater(.261 / .228 - 1, .14)
        self.assertLess(result["rows"][2][2] / result["rows"][1][2] - 1, .021)

    def test_future_split_scale_is_needed_even_on_non_event_days(self):
        raw, q = rows([100, 110, 55]), rows([50, 55, 55])
        raw[-1] = (raw[-1][0], 55, 55, 200)
        split = [dict(date=raw[-1][0], ratio=2, ratio_exact="2")]
        result = reconstruct_total_return(raw, q, split)
        self.assertEqual(result["cash_events"], [])
        self.assertAlmostEqual(result["rows"][1][2], 1.1)
        self.assertAlmostEqual(result["rows"][2][2], 1.1)
        self.assertEqual([r[3] for r in result["rows"]], [200, 200, 200])
        self.assertAlmostEqual(affine_step(100, 110, 50, 55)["gross"], 1.05)
        self.assertAlmostEqual(affine_step(100, 110, 50, 55, .5, .5)["gross"], 1.1)

    def test_combined_split_and_cash_preserve_total_wealth(self):
        raw, q = rows([100, 48, 50]), rows([48, 48, 50])
        split = [dict(date=raw[1][0], ratio=2, ratio_exact="2")]
        result = reconstruct_total_return(raw, q, split)
        self.assertAlmostEqual(result["cash_events"][0]["cash_per_old_share"], 4)
        self.assertAlmostEqual(result["rows"][1][2], 1.0)
        self.assertAlmostEqual(result["rows"][2][2], 50 / 48)
        algebra = affine_step(100, 48, 48, 48, .5, 1)
        self.assertAlmostEqual(algebra["cash_per_old_share"], 4)
        self.assertAlmostEqual(algebra["gross"], algebra["difference_formula_gross"])

    def test_reverse_split_changes_units_without_creating_a_return(self):
        raw, q = rows([10, 40]), rows([40, 40])
        raw[-1] = (raw[-1][0], 40, 40, 25)
        event = [dict(date=raw[-1][0], ratio=.25, ratio_exact="0.25")]
        result = reconstruct_total_return(raw, q, event)
        self.assertEqual(result["cash_events"], [])
        self.assertEqual([r[2] for r in result["rows"]], [1, 1])
        self.assertEqual([r[3] for r in result["rows"]], [25, 25])

    def test_split_rounding_noise_is_not_inferred_as_daily_dividends(self):
        from datetime import date, timedelta
        raw, q = [], []
        for i in range(80):
            d = (date(2020, 1, 1) + timedelta(days=i)).isoformat()
            old_price = 100 + i * .031
            p = old_price if i < 60 else old_price / 3
            a = 1 / 3 if i < 60 else 1
            raw.append((d, p, p, 100 if i < 60 else 300))
            q.append((d, round(a * p, 3), round(a * p, 3), 100))
        result = reconstruct_total_return(raw, q, [dict(date=raw[60][0], ratio=3)])
        self.assertEqual(result["cash_events"], [])
        self.assertGreater(result["diagnostics"]["suppressed_rounding_noise_days"], 0)
        self.assertAlmostEqual(result["rows"][-1][2], (100 + 79 * .031) / 100)

    def test_unknown_split_and_negative_adjustment_are_not_silently_accepted(self):
        raw = [("2020-01-01", 99, 100, 100), ("2020-01-02", 50, 51, 200)]
        q = [("2020-01-01", 49.5, 50, 100), ("2020-01-02", 50, 51, 200)]
        with self.assertRaisesRegex(ValueError, "split scale"):
            reconstruct_total_return(raw, q)
        with self.assertRaisesRegex(ValueError, "negative cash"):
            reconstruct_total_return(rows([100, 100]), rows([90, 89]))

    def test_explicit_official_cash_uses_raw_actions_not_vendor_noise(self):
        raw = rows([100, 90])
        q = rows([70, 83])  # Deliberately wrong vendor-adjusted quotes.
        result = reconstruct_total_return(raw, q, cash_events={raw[1][0]: 10})
        self.assertEqual(result["rows"][-1][2], 1)
        self.assertEqual(result["diagnostics"]["supplied_cash_sum"], 10)
        self.assertEqual(result["diagnostics"]["inferred_cash_event_count"], 0)
        self.assertFalse(result["diagnostics"]["vendor_model_validated_at_half_tick"])
        self.assertGreater(result["diagnostics"]["max_vendor_quote_residual"], .001)
        better_q = rows([90, 90])
        better = reconstruct_total_return(raw, better_q, cash_events={raw[1][0]: 10})
        self.assertEqual(result["rows"], better["rows"])

    def test_official_split_and_cash_have_old_share_units(self):
        raw, q = rows([100, 48]), rows([48, 48])
        split = [dict(date=raw[1][0], ratio=2)]
        result = reconstruct_total_return(raw, q, split,
                                           cash_events=[dict(date=raw[1][0], cash_per_old_share=4,
                                                             sources=["caller verified source"])])
        self.assertEqual(result["rows"][-1][2], 1)
        self.assertEqual(result["cash_events"][0]["cash_per_old_share"], 4)
        self.assertEqual(result["cash_events"][0]["sources"], ["caller verified source"])

    def test_future_adjustment_does_not_change_exact_earlier_total_return_prefix(self):
        old = reconstruct_total_return(rows([100, 110]), rows([100, 110]))
        with_dividend = reconstruct_total_return(rows([100, 110, 100]), rows([90, 100, 100]))
        with_split = reconstruct_total_return(rows([100, 110, 55]), rows([50, 55, 55]),
                                              [dict(date="2020-01-03", ratio=2)])
        self.assertEqual([r[2] for r in old["rows"]], [r[2] for r in with_dividend["rows"][:2]])
        self.assertEqual([r[2] for r in old["rows"]], [r[2] for r in with_split["rows"][:2]])

    def test_close_based_daily_scaling_is_not_a_causal_open_price(self):
        previous_close, opening, dividend = 100, 90, 10
        actual_open_wealth = opening + dividend
        scaled_open_90_close = opening * ((90 + dividend) / previous_close * 100) / 90
        scaled_open_99_close = opening * ((99 + dividend) / previous_close * 100) / 99
        self.assertEqual(scaled_open_90_close, actual_open_wealth)
        self.assertAlmostEqual(scaled_open_99_close, 99.0909090909091)
        self.assertNotEqual(scaled_open_99_close, actual_open_wealth)
        result = reconstruct_total_return(rows([100, 90]), rows([90, 90]))
        self.assertFalse(result["diagnostics"]["usable_for_next_open"])
        self.assertTrue(all(r[1] == r[2] for r in result["rows"]))

    def test_date_mismatch_and_unknown_event_date_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "date axes"):
            reconstruct_total_return(rows([100, 110]), rows([100]))
        with self.assertRaisesRegex(ValueError, "first-new-unit"):
            reconstruct_total_return(rows([100, 110]), rows([100, 110]),
                                       [dict(date="2020-01-09", ratio=2)])


if __name__ == "__main__":
    unittest.main()
