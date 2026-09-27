"""Synthetic two-day forward smoke using real data/policy/ledger/runtime layers.

No real-time requests or production state writes. Frozen September 24 data is
copied into a temporary seed and extended with explicitly artificial September
28/29 observations, each using an injected matching clock.
"""
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import strategy
from v10_live import data, policy, runtime

ROOT = Path(__file__).resolve().parent.parent
FIRST, SECOND = "2026-09-28", "2026-09-29"


class LiveHEndToEndTests(unittest.TestCase):
    def test_artificial_two_day_pipeline_tracks_only_forward_signal_prices(self):
        before_strategy = {key: deepcopy(value) for key, value in vars(strategy).items()
                           if key.isupper() or key == "_state_bull"}
        with tempfile.TemporaryDirectory(prefix="v10_h_end_to_end_") as directory:
            base = Path(directory)
            seed, cache, state_path = base / "seed", base / "cache", base / "signals/H.json"
            (seed / "results/price_audit").mkdir(parents=True)
            (seed / "corrected_snapshots").mkdir()
            (seed / "snapshots").mkdir()
            source = ROOT / "v10_h_close"
            shutil.copy2(source / "corrected_manifest.json", seed / "corrected_manifest.json")
            qvix_path = base / "qvix50.csv"
            shutil.copy2(source / "snapshots/qvix50.csv", qvix_path)
            raw, qfq, quote_sets = {}, {}, {FIRST: {}, SECOND: {}}
            for code in data.CODES:
                filename = code + "_raw_2010-01-01_2026-09-24.json"
                shutil.copy2(source / "results/price_audit" / filename, seed / "results/price_audit" / filename)
                shutil.copy2(source / "corrected_snapshots" / (code + ".csv"), seed / "corrected_snapshots" / (code + ".csv"))
                shutil.copy2(source / "snapshots" / (code + ".csv"), seed / "snapshots" / (code + ".csv"))
                raw_payload = json.loads((seed / "results/price_audit" / filename).read_text())
                raw[code] = [(r[0], float(r[1]), float(r[2]), float(r[5])) for r in raw_payload["rows"]]
                with (seed / "snapshots" / (code + ".csv")).open(newline="") as stream:
                    qfq[code] = [(r[0], float(r[1]), float(r[2]), float(r[3])) for r in csv.reader(stream) if r]
                previous, volume = raw[code][-1][2], raw[code][-1][3]
                first_open, first_signal, first_close = (round(previous * multiplier, 3) for multiplier in (1.001, 1.005, 1.009))
                second_open, second_signal, second_close = (round(first_close * multiplier, 3) for multiplier in (1.001, 1.007, 1.011))
                artificial = [(FIRST, first_open, first_close, volume * 1.1),
                              (SECOND, second_open, second_close, volume * 1.2)]
                raw[code].extend(artificial)
                qfq[code].extend(artificial)  # same vintage, explicitly no new distributions/conversions
                for day, opening, price, prev in ((FIRST, first_open, first_signal, previous),
                                                  (SECOND, second_open, second_signal, first_close)):
                    quote_sets[day][code] = dict(date=day, timestamp=day + " 14:50:00",
                        open=opening, price=price, prev_close=prev, volume=volume * .8,
                        high=max(opening, price), low=min(opening, price))
            fetch_calls = []
            def fetch_pair(code, start, end):
                fetch_calls.append((code, start, end))
                return dict(raw=[r for r in raw[code] if start <= r[0] <= end],
                            qfq=[r for r in qfq[code] if start <= r[0] <= end],
                            sources={"raw": "temporary synthetic fixture", "qfq": "same-vintage temporary fixture"})
            views, decisions = [], []
            def build_view(quotes, signal_date, now=None):
                view = data.build_live_view(quotes, signal_date, now=now, cache_dir=cache, fetch_pair=fetch_pair)
                views.append(view)
                return view
            def decide(histories, calendar, signal_date, state=None):
                result = policy.decide(histories, calendar, signal_date, state=state, qvix_path=qvix_path)
                decisions.append(result)
                return result
            with patch.object(data, "SEED_DIR", seed), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden in end-to-end test")):
                first_lines = runtime.run(quote_sets[FIRST], FIRST, state_path=state_path,
                    now=datetime.fromisoformat(FIRST + "T14:50:30"), build_view=build_view, decide=decide)
                first_state = json.loads(state_path.read_text())
                self.assertEqual(first_state["candidate_id"], policy.CANDIDATE_ID)
                self.assertEqual(first_state["nav"], 1.)
                self.assertEqual(first_state["last_return"], 0.)
                self.assertEqual(first_state["start_date"], FIRST)
                self.assertEqual(first_state["entry_date"], FIRST)
                self.assertEqual(first_state["events"][0]["from"], None)
                self.assertEqual(first_state["events"][0]["fee_fraction"], 0.)
                held = first_state["holding"]
                self.assertEqual(first_state["mark_raw_price"], quote_sets[FIRST][held]["price"])
                self.assertAlmostEqual(first_state["units"], 1 / quote_sets[FIRST][held]["price"])
                self.assertTrue(decisions[0]["executable"])
                self.assertFalse(decisions[0]["diagnostics"]["qvix"]["available"])
                self.assertIn("停用", "\n".join(first_lines))
                self.assertIn("虚拟跟踪不下单", "\n".join(first_lines))
                first_cache = json.loads((cache / "completed.json").read_text())["state"]
                self.assertEqual(first_cache["last_completed"], "2026-09-24")
                self.assertTrue(all(not a["raw"] for a in first_cache["assets"].values()))
                state_bytes, cache_bytes = state_path.read_bytes(), (cache / "completed.json").read_bytes()
                with patch.object(data, "build_live_view", side_effect=AssertionError("Duplicate data request")), \
                        patch.object(policy, "decide", side_effect=AssertionError("Duplicate policy invocation")):
                    repeated = runtime.run({}, FIRST, state_path=state_path,
                        now=datetime.fromisoformat(FIRST + "T14:55:00"), build_view=build_view, decide=decide)
                self.assertEqual(repeated, first_lines)
                self.assertEqual(state_path.read_bytes(), state_bytes)
                self.assertEqual((cache / "completed.json").read_bytes(), cache_bytes)
                runtime.run(quote_sets[SECOND], SECOND, state_path=state_path,
                    now=datetime.fromisoformat(SECOND + "T14:50:30"), build_view=build_view, decide=decide)
                second_state = json.loads(state_path.read_text())
                expected = quote_sets[SECOND][held]["price"] / quote_sets[FIRST][held]["price"]
                if second_state["holding"] != held:
                    expected *= .9998
                self.assertAlmostEqual(second_state["nav"], expected, places=12)
                self.assertEqual(second_state["start_date"], FIRST)
                self.assertEqual(second_state["last_date"], SECOND)
                self.assertEqual(len(second_state["events"]), 2)
                self.assertEqual(second_state["events"][-1]["actions"], [])
                completed = json.loads((cache / "completed.json").read_text())["state"]
                self.assertEqual(completed["last_completed"], FIRST)
                self.assertEqual(completed["assets"][held]["raw"][-1][0], FIRST)
                self.assertNotEqual(completed["assets"][held]["raw"][-1][2], first_state["mark_raw_price"])
                self.assertNotIn(SECOND, completed["assets"][held]["actions"])
                self.assertEqual(views[-1]["calendar"][-2:], [FIRST, SECOND])
                self.assertTrue(all(len(rows[-1]) == 6 for rows in views[-1]["histories"].values()))
                self.assertEqual(len(fetch_calls), len(data.CODES) * 2)
                self.assertEqual(len(decisions), 2)
                # A mismatched clock cannot create or rewrite a forward account.
                saved = state_path.read_bytes()
                with self.assertRaisesRegex(ValueError, "historical quotes"):
                    runtime.run(quote_sets[FIRST], FIRST, state_path=state_path,
                        now=datetime.fromisoformat(SECOND + "T14:50:30"), build_view=build_view, decide=decide)
                self.assertEqual(state_path.read_bytes(), saved)
        self.assertEqual(before_strategy, {key: value for key, value in vars(strategy).items()
                                          if key.isupper() or key == "_state_bull"})


if __name__ == "__main__":
    unittest.main()
