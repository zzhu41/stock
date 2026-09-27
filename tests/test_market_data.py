import copy
import csv
import os
import tempfile
import unittest
from unittest.mock import patch

import market_data as md


def quote(date="2026-09-02", price=2.194, volume=500.0):
    return {"date": date, "timestamp": date + " 14:50:00", "price": price,
            "open": 2.2, "volume": volume, "name": "ETF"}


def quote_payload(price="2.194", stamp="20260902145000", volume="500"):
    fields = [""] * 31
    fields[1:7] = ["ETF", "513100", price, "2.233", "2.2", volume]
    fields[30] = stamp
    return ('v_sh513100="' + "~".join(fields) + '";').encode("gbk")


class LiveHistoryTests(unittest.TestCase):
    def setUp(self):
        self.histories = {"513100": [("2026-08-31", 2.2, 2.223, 400.0),
                                     ("2026-09-01", 2.218, 2.233, 600.0)]}

    def test_appends_current_day_without_losing_yesterday(self):
        original = copy.deepcopy(self.histories)
        result = md.prepare_live_histories(self.histories, {"513100": quote()}, "2026-09-02")
        self.assertEqual(result["513100"][:-1], original["513100"])
        self.assertEqual(result["513100"][-1], ("2026-09-02", 2.2, 2.194, 500.0))
        self.assertEqual(self.histories, original)

    def test_replaces_existing_current_day_price_and_volume(self):
        self.histories["513100"].append(("2026-09-02", 2.2, 2.19, 300.0))
        result = md.prepare_live_histories(self.histories, {"513100": quote()}, "2026-09-02")
        self.assertEqual(len(result["513100"]), 3)
        self.assertEqual(result["513100"][-1][2:], (2.194, 500.0))
        self.assertEqual(self.histories["513100"][-1][2:], (2.19, 300.0))

    def test_rejects_stale_missing_or_inconsistent_quote(self):
        inconsistent = quote()
        inconsistent["timestamp"] = "2026-09-01 14:50:00"
        for quotes in ({}, {"513100": quote("2026-09-01")}, {"513100": inconsistent}):
            with self.subTest(quotes=quotes), self.assertRaises(ValueError):
                md.prepare_live_histories(self.histories, quotes, "2026-09-02")

    def test_rejects_future_history_and_invalid_quote_values(self):
        with self.assertRaises(ValueError):
            md.prepare_live_histories(self.histories, {"513100": quote()}, "2026-08-31")
        for field, value in (("price", 0), ("price", float("nan")), ("open", -1),
                             ("volume", -1), ("volume", None)):
            invalid = quote()
            invalid[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                md.prepare_live_histories(self.histories, {"513100": invalid}, "2026-09-02")

    def test_cash_without_quote_is_unchanged_but_supplied_quote_is_validated(self):
        histories = {md.CASH: [("2026-09-01", 100.0, 100.0, 1000.0)]}
        self.assertEqual(md.prepare_live_histories(histories, {}, "2026-09-02"), histories)
        with self.assertRaises(ValueError):
            md.prepare_live_histories(histories, {md.CASH: quote("2026-09-01")}, "2026-09-02")
        result = md.prepare_live_histories(histories, {md.CASH: quote(price=100)}, "2026-09-02")
        self.assertEqual(result[md.CASH][-1][0], "2026-09-02")


class RealtimeTests(unittest.TestCase):
    def test_detailed_has_quote_timestamp_and_current_cumulative_volume(self):
        with patch.object(md, "_get", return_value=quote_payload()):
            result = md.fetch_realtime(["513100"], detailed=True)
        self.assertEqual(result["513100"], quote())

    def test_legacy_tuple_api_is_preserved(self):
        with patch.object(md, "_get", return_value=quote_payload()):
            self.assertEqual(md.fetch_realtime(["513100"]), {"513100": ("ETF", 2.194)})

    def test_rejects_malformed_zero_price_and_missing_quote(self):
        for payload in (quote_payload(price="0"), quote_payload(price="nan"),
                        quote_payload(stamp="20260902"), quote_payload(volume=""), b""):
            with self.subTest(payload=payload), patch.object(md, "_get", return_value=payload):
                with self.assertRaises(ValueError):
                    md.fetch_realtime(["513100"], detailed=True)


class HistoryCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(md, "DATA_DIR", self.tmp.name).start()
        patch.object(md, "_today", return_value="2026-09-04").start()
        self.cache = os.path.join(self.tmp.name, "513100.csv")
        self.old = [("2026-09-01", 100.0, 100.0, 10.0),
                    ("2026-09-02", 101.0, 102.0, 20.0),
                    ("2026-09-03", 102.0, 103.0, 30.0)]
        with open(self.cache, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(self.old)
        self.original = self.read_bytes()

    def read_bytes(self):
        with open(self.cache, "rb") as f:
            return f.read()

    def test_incremental_checks_overlap_and_updates_partial_last_bar(self):
        new = self.old[:-1] + [("2026-09-03", 102.0, 104.0, 35.0),
                               ("2026-09-04", 104.0, 105.0, 15.0)]
        with patch.object(md, "_kline_page", return_value=new) as fetch:
            result = md.fetch_history("513100")
        self.assertEqual(result, new)
        fetch.assert_called_once_with("sh513100", "2026-09-01", "2026-09-04")

    def test_dividend_drift_rebases_entire_cache_before_publication(self):
        adjusted = [(d, o * 0.9, c * 0.9, v) for d, o, c, v in self.old]
        adjusted.append(("2026-09-04", 93.0, 94.0, 15.0))
        with patch.object(md, "_kline_page", side_effect=[adjusted, adjusted]) as fetch:
            result = md.fetch_history("513100")
        self.assertEqual(result, adjusted)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(fetch.call_args_list[1][0], ("sh513100", md.FULL_START, "2026-09-04"))
        with open(self.cache, newline="", encoding="utf-8") as f:
            self.assertEqual(float(next(csv.reader(f))[2]), 90.0)

    def test_failed_rebase_retains_old_cache(self):
        drifted = [(d, o * 0.9, c * 0.9, v) for d, o, c, v in self.old]
        for failure in ([], OSError("offline")):
            with self.subTest(failure=failure), patch.object(md, "_kline_page", side_effect=[drifted, failure]):
                with self.assertRaises((ValueError, OSError)):
                    md.fetch_history("513100")
                self.assertEqual(self.read_bytes(), self.original)

    def test_bad_or_missing_incremental_rows_never_overwrite_cache(self):
        for bad in ([], self.old[:1], [("2026-09-04", 1, 1, 10)],
                    self.old + [("2026-09-04", 0, 1, 10)],
                    self.old + [("2026-09-04", 1, float("nan"), 10)],
                    self.old + [("2026-09-05", 1, 1, 10)]):
            with self.subTest(bad=bad), patch.object(md, "_kline_page", return_value=bad):
                with self.assertRaises(ValueError):
                    md.fetch_history("513100")
                self.assertEqual(self.read_bytes(), self.original)

    def test_empty_or_regressed_full_refresh_preserves_cache(self):
        for bad in ([], self.old[:1], self.old[1:], [self.old[0], self.old[-1]]):
            with self.subTest(bad=bad), patch.object(md, "_kline_page", return_value=bad):
                with self.assertRaises(ValueError):
                    md.fetch_history("513100", full=True)
                self.assertEqual(self.read_bytes(), self.original)

    def test_atomic_replace_failure_keeps_old_file_and_cleans_temp(self):
        new = self.old + [("2026-09-04", 104.0, 105.0, 15.0)]

        def fail_replace(src, dst):
            self.assertEqual(dst, self.cache)
            self.assertEqual(self.read_bytes(), self.original)
            with open(src, newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.reader(f))), 4)
            raise OSError("rename failed")

        with patch.object(md, "_kline_page", return_value=new), patch.object(md.os, "replace", side_effect=fail_replace):
            with self.assertRaises(OSError):
                md.fetch_history("513100")
        self.assertEqual(self.read_bytes(), self.original)
        self.assertEqual(os.listdir(self.tmp.name), ["513100.csv"])


if __name__ == "__main__":
    unittest.main()
