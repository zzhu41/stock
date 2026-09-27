"""Continuous-account seams; new behavior uses only synthetic observations."""
from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v10_deep.reference import DatedValues,run_reference as frozen_reference
from v11.tests.test_walkforward import fixture,segments,ASSETS,A,B,C,F
from v12 import walkforward
from v12.reference import run_reference

CASH="511880"


def shock_case(n=9):
    arrays,meta,config=fixture(n)
    config.update(crash_mask=1,crash_lock=5,locked_panic_exit=1)
    arrays["features"][1,ASSETS.index(C),F["mom5"]]=-.1
    arrays["features"][1,ASSETS.index(C),F["ma250"]]=-.3
    p=[100.,100.,94.,90.,91.,92.,93.,94.,95.][:n]
    arrays["features"][:,ASSETS.index(C),F["close"]]=p
    arrays["features"][:,ASSETS.index(C),F["ret1"]]=[0.]+[p[i]/p[i-1]-1 for i in range(1,n)]
    return arrays,meta,config


class WalkForwardTests(unittest.TestCase):
    def test_same_flag1_across_segments_matches_continuous_reference_for_both_lags(self):
        arrays,meta,config=shock_case()
        for lag in (0,1):
            candidate=dict(config,lag=lag)
            schedule=segments(meta,[candidate]*3,[0,2,5,len(meta["dates"])])
            expected=run_reference(arrays,meta,candidate,fee=.0011)
            actual=walkforward.run(schedule,arrays,meta,fee=.0011)
            for field in ("returns","holdings","summary"):
                np.testing.assert_array_equal(actual[field],expected[field])
            self.assertTrue(all(b["policy_reused"] for b in actual["boundaries"][1:]))
            self.assertEqual([r["locked_exit_requested"] for r in actual["trace"]],
                             [r["locked_panic_exit_requested"] for r in expected["trace"]])

    def test_model_change_retains_live_lock_until_cash_fill_and_trace_records_fee(self):
        arrays,meta,second=shock_case()
        first=dict(second,locked_panic_exit=0)
        arrays["features"][2,ASSETS.index(CASH),F["close"]]=np.nan
        result=walkforward.run(segments(meta,[first,second],[0,2,9]),arrays,meta,fee=.0011)
        boundary=result["boundaries"][1]
        self.assertEqual(boundary["holding"],C)
        self.assertEqual(boundary["previous_mark"],100.)
        self.assertEqual(boundary["lock_until"],6)
        blocked=result["trace"][2]
        self.assertTrue(blocked["model_changed"])
        self.assertTrue(blocked["locked_exit_requested"])
        self.assertFalse(blocked["filled"])
        self.assertEqual(blocked["lock_until"],6)
        self.assertEqual(blocked["cost"],0.)
        filled=result["trace"][3]
        self.assertTrue(filled["locked_exit_requested"])
        self.assertTrue(filled["filled"])
        self.assertEqual(filled["holding"],CASH)
        self.assertEqual(filled["lock_until"],-1)
        self.assertAlmostEqual(filled["cost"],filled["marked_nav"]*.0022)
        self.assertTrue(result["trades"][-2]["locked_exit"])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"trace.csv"
            walkforward.write_trace_csv(path,result)
            with path.open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual(rows[2]["locked_exit_requested"],"True")
            self.assertEqual(rows[2]["filled"],"False")
            self.assertEqual(rows[3]["lock_until"],"-1")

    def test_switching_to_flag0_or_a_shorter_configured_lock_does_not_erase_actual_lock(self):
        arrays,meta,first=shock_case()
        first["locked_panic_exit"]=0
        second=dict(first,score="alternate",crash_mask=0,crash_lock=1)
        result=walkforward.run(segments(meta,[first,second],[0,2,9]),arrays,meta)
        self.assertTrue(all(r["holding"]==C and r["lock_until"]==6 for r in result["trace"][2:6]))
        self.assertEqual(result["daily"][6][2],B)
        self.assertFalse(any(r["locked_exit_requested"] for r in result["trace"]))

    def test_default_zero_is_same_model_and_confirmation_survives_only_identical_models(self):
        arrays,meta,first=fixture(6)
        first["switch_confirm"]=2
        arrays["scores"][0,1:,ASSETS.index(B)]=300.
        same=dict(first,locked_panic_exit=0,id="same-model-new-label")
        changed=dict(same,panic=.05)
        reused=walkforward.run(segments(meta,[first,same],[0,2,6]),arrays,meta)
        reset=walkforward.run(segments(meta,[first,changed],[0,2,6]),arrays,meta)
        self.assertTrue(reused["boundaries"][1]["policy_reused"])
        self.assertEqual(reused["daily"][2][2],B)
        self.assertEqual(reset["daily"][2][2],A)
        self.assertEqual(reset["daily"][3][2],B)
        self.assertTrue(reset["trace"][2]["pending_reset"])

    def test_age_peak_and_existing_cooldown_are_carried_at_model_changes(self):
        arrays,meta,first=fixture(9)
        first.update(panic_cooldown=5,locked_panic_exit=0)
        arrays["features"][1,ASSETS.index(A),F["ret1"]]=-.06
        arrays["scores"][0,1,ASSETS.index(B)]=200.
        arrays["scores"][2,2:,ASSETS.index(A)]=1000.
        arrays["features"][2:,ASSETS.index(A),F["mom20"]]=.5
        second=dict(first,score="alternate")
        result=walkforward.run(segments(meta,[first,second],[0,2,9]),arrays,meta)
        self.assertEqual(result["boundaries"][1]["holding"],B)
        self.assertEqual(result["boundaries"][1]["age"],0)
        self.assertGreater(result["boundaries"][1]["asset_peak"],0.)
        self.assertTrue(all(r[2]==B for r in result["daily"][2:7]))
        self.assertEqual(result["daily"][7][2],A)
        self.assertFalse(result["trace"][2]["filled"])
        self.assertEqual(result["trace"][2]["age"],1)

    def test_explicit_factory_masks_a_score_without_repricing_execution_or_mutating_inputs(self):
        arrays,meta,first=fixture(7)
        second=dict(first,score="alternate")
        arrays["scores"][2,:,ASSETS.index(B)]=-1e100
        original=deepcopy(arrays)
        seen=[]
        def factory(values,metadata,config,context):
            seen.append(config["score"])
            view=dict(values)
            view["features"]=values["features"].copy()
            score=values["scores"][metadata["score_names"].index(config["score"])]
            view["features"][:,:,F["valid"]]*=np.isfinite(score)&(score>-1e90)
            # Deliberately different signal-view marks must not become actual
            # settlement prices; execution always uses the original cube.
            view["features"][:,:,F["close"]]*=10.
            return DatedValues(view,metadata,config)
        result=walkforward.run(segments(meta,[first,second],[0,3,7]),arrays,meta,view_factory=factory)
        self.assertEqual(seen,["wls25_v20","alternate"])
        self.assertTrue(all(row[2]==A for row in result["daily"]))
        self.assertAlmostEqual(result["daily"][-1][1],arrays["features"][-1,0,F["close"]]/arrays["features"][0,0,F["close"]])
        for key in arrays:np.testing.assert_array_equal(arrays[key],original[key])

    def test_allocation_or_wrapped_models_fail_before_constructing_any_policy(self):
        arrays,meta,config=fixture(5)
        variants=[dict(kind="allocation",config=config),
                  dict(config=dict(mode="vol_target",base_id="h")),
                  dict(config=dict(kind="allocation",allocation={"mode":"progressive"})),
                  dict(config=dict(kind="single",config=config))]
        for bad in variants:
            schedule=segments(meta,[config,config],[0,2,5])
            schedule[1].update(bad)
            with patch.object(walkforward,"Policy",side_effect=AssertionError("must validate all models first")):
                with self.assertRaisesRegex(ValueError,"rejects allocation|wrapped registry"):
                    walkforward.run(schedule,arrays,meta)

    def test_future_perturbation_and_schedule_gaps_cannot_change_valid_prefix(self):
        arrays,meta,config=shock_case()
        schedule=segments(meta,[config]*2,[0,4,9])
        later=deepcopy(arrays)
        later["features"][6:,:,F["close"]]*=100.
        later["scores"][:,6:]*=-5
        before=walkforward.run(schedule,arrays,meta,end=meta["dates"][5])
        changed=walkforward.run(schedule,later,meta,end=meta["dates"][5])
        np.testing.assert_array_equal(before["returns"],changed["returns"])
        self.assertEqual(before["trace"],changed["trace"])
        schedule[0]["end"]=meta["dates"][2]
        with self.assertRaisesRegex(ValueError,"Every evaluated observation"):
            walkforward.run(schedule,arrays,meta)


class FrozenControlFidelityTests(unittest.TestCase):
    def test_fixed_flag0_controls_across_year_boundaries_match_continuous_old_reference(self):
        """Only already-known H/v92/simple, never new candidates or selection."""
        from v12.data import load_inputs
        arrays,meta,profiles=load_inputs()
        periods=(("2014-01-02","2017-12-31"),("2018-01-01","2019-12-31"),
                 ("2020-01-01","2021-12-31"),("2022-01-01","2023-12-31"),
                 ("2024-01-01","2026-09-24"))
        for name in ("h","v92","simple"):
            for lag in (0,1):
                config=dict(profiles[name],lag=lag,locked_panic_exit=0)
                schedule=[dict(start=start,end=end,config=deepcopy(config)) for start,end in periods]
                for fee in (.0001,.0011):
                    actual=walkforward.run(schedule,arrays,meta,fee=fee)
                    expected=frozen_reference(arrays,meta,config,start="2014-01-02",end="2026-09-24",fee=fee)
                    self.assertEqual(len(actual["dates"]),3097)
                    for field in ("returns","holdings","summary"):
                        np.testing.assert_array_equal(actual[field],expected[field])
                    self.assertTrue(all(b["policy_reused"] for b in actual["boundaries"][1:]))
                    self.assertFalse(any(r["locked_exit_requested"] for r in actual["trace"]))


if __name__=="__main__":
    unittest.main()
