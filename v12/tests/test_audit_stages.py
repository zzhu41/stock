"""Small independent-audit regression cases; no real candidate evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v12 import audit_stages as audit


class AuditTests(unittest.TestCase):
    def test_first_loss_sets_drawdown_and_first_return_is_never_dropped(self):
        result = audit.independent_metrics([-.1, .2, -.05])
        self.assertAlmostEqual(result["nav"], .9 * 1.2 * .95)
        self.assertAlmostEqual(result["max_dd"], -.1)
        self.assertEqual(result["sessions"], 3)
        self.assertAlmostEqual(result["total_return"], .026)
        later = audit.independent_metrics([-.05, .1])
        self.assertAlmostEqual(later["total_return"], .95 * 1.1 - 1)
        self.assertAlmostEqual(later["max_dd"], -.05)

    def test_invalid_daily_return_is_rejected(self):
        for values in ([], [-1.], [float("nan")], [float("inf")]):
            with self.assertRaises(AssertionError):
                audit.independent_metrics(values)

    def test_ten_gates_follow_signed_drawdown_and_v92_tail(self):
        controls = {key: key for key in audit.CONTROL_KEYS}
        rows = {}
        for cid in list(controls) + ["trial"]:
            scenarios = {}
            for name, unused_lag, unused_fee in audit.SCENARIOS:
                scenarios[name] = dict(full=dict(cagr=.5, max_dd=-.2), tail=dict(total_return=.5),
                                       blocks={b: dict(cagr=.4) for b, start, end in audit.BLOCKS})
            rows[cid] = dict(scenarios=scenarios)
        rows["trial"]["scenarios"]["close_1bp"]["full"].update(cagr=.51, max_dd=-.195)
        checks = audit.independent_checks(rows, controls)["trial"]
        self.assertEqual(len(checks), 10)
        self.assertTrue(all(checks.values()))
        changed = deepcopy(rows)
        changed["simple"]["scenarios"]["close_1bp"]["full"]["max_dd"] = -.19
        changed["v92"]["scenarios"]["close_1bp"]["tail"]["total_return"] = .6
        checks = audit.independent_checks(changed, controls)["trial"]
        self.assertFalse(checks["main_drawdown"])
        self.assertFalse(checks["tail_return"])
        self.assertTrue(checks["main_cagr"])

    def test_receipt_verifies_actual_bytes_before_reading_metrics(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            folder = base / "results/main"
            folder.mkdir(parents=True)
            def write(path, value):
                path.write_text(json.dumps(value))
            write(base / "registered_candidates.json", dict(candidates=[dict(id="a")], unique_count=1))
            write(folder / "registration.json", dict(design=dict(fingerprints={})))
            (folder / "paths.npz").write_bytes(b"synthetic saved path bytes")
            path_hash = audit.sha(folder / "paths.npz")
            reg_hash = audit.sha(folder / "registration.json")
            write(folder / "evaluation.json", dict(rows=[dict(id="a")], registration_sha256=reg_hash, paths_sha256=path_hash))
            write(folder / "path_metadata.json", dict(ids=["a"], dates=[audit.START, audit.END], sha256=path_hash))
            write(folder / "execution_metadata.json", dict(registration_sha256=reg_hash))
            proof = {key: audit.sha(folder / filename) for filename, key in (
                ("registration.json", "registration_sha256"), ("paths.npz", "paths_sha256"),
                ("evaluation.json", "evaluation_sha256"), ("path_metadata.json", "path_metadata_sha256"),
                ("execution_metadata.json", "execution_metadata_sha256"))}
            proof.update(registry_sha256=audit.sha(base / "registered_candidates.json"), fingerprints={})
            write(folder / "selection.json", dict(provenance=proof))
            with patch.object(audit, "BASE", base):
                self.assertEqual(audit.verify_receipts("main")[1]["unique_count"], 1)
                (folder / "paths.npz").write_bytes(b"modified data")
                with self.assertRaisesRegex(AssertionError, "Changed frozen artifact"):
                    audit.verify_receipts("main")


if __name__ == "__main__":
    unittest.main()
