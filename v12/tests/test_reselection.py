"""Synthetic historical selector orchestration; no new real candidate returns."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from v12 import reselection as r
from v12.reference import Policy, Portfolio, run_reference
from v12.walkforward import run as walk_run
from v10_deep.schema import identifier
from v11.tests.test_walkforward import fixture, segments, ASSETS, A, B, C, F


def candidate(config, selectable=True):
    config=deepcopy(config)
    return dict(id="v12_"+identifier(config)[:20],kind="single",config=config,selectable=selectable,
                complexity=1,families=["synthetic"])


class ReselectionTests(unittest.TestCase):
    def test_prefix_builder_never_receives_future_arrays_or_history(self):
        arrays,meta,config=fixture(6)
        arrays["orders"]=np.argsort(-arrays["scores"],axis=2).astype(np.int32)
        histories={A:[(day,1.,1.,1.) for day in meta["dates"]]}
        original=deepcopy(arrays);cutoff=meta["dates"][2];seen=[]
        def build(values,metadata,own):
            self.assertEqual(metadata["dates"],meta["dates"][:3])
            self.assertEqual(values["features"].shape[0],3)
            self.assertEqual(values["scores"].shape[1],3)
            self.assertEqual(values["orders"].shape[1],3)
            self.assertTrue(all(row[0]<=cutoff for row in own[A]))
            seen.append(True)
            values["features"][0,0,0]=999.
            return values,metadata,{}
        with patch.object(r,"build_exante",side_effect=build):
            r.prefix_features(arrays,meta,histories,cutoff)
        self.assertEqual(seen,[True])
        for name in arrays: np.testing.assert_array_equal(arrays[name],original[name])
        self.assertEqual(len(histories[A]),6)

    def test_union_preserves_control_status_and_context_and_does_not_mutate(self):
        arrays,meta,config=fixture(3)
        control=candidate(config,False);duplicate=deepcopy(control);duplicate["selectable"]=True
        duplicate["families"]=["other"]
        prior=candidate(dict(config,risk_context="prior20"))
        controls={"h":control["id"]}
        definitions=[dict(candidates=[control],controls=controls),
                     dict(candidates=[duplicate],controls=controls),
                     dict(candidates=[prior],controls=controls)]
        before=deepcopy(definitions);union=r.union_registry(definitions)
        self.assertEqual(union["unique_count"],2)
        saved={c["id"]:c for c in union["candidates"]}
        self.assertFalse(saved[control["id"]]["selectable"])
        self.assertEqual(saved[control["id"]]["source_stages"],["A","B"])
        self.assertEqual(saved[prior["id"]]["config"]["risk_context"],"prior20")
        self.assertEqual(definitions,before)
        definitions[1]["candidates"][0]["complexity"]=2
        with self.assertRaisesRegex(ValueError,"complexity"):
            r.union_registry(definitions)

    def test_grouping_keeps_allocation_and_exact_fixed_anchor_without_mutation(self):
        arrays,meta,config=fixture(3)
        h=candidate(config,False);prior=candidate(dict(config,risk_context="prior20"))
        allocation=dict(id="allocation",kind="allocation",allocation=dict(base_id=h["id"],mode="progressive"),
                        selectable=True,complexity=2,families=["allocation"])
        records=[h,prior,allocation];observed=[]
        def view(values,metadata,current):
            copied=dict(values,features=values["features"].copy())
            copied["features"][0,0,0]=20. if current.get("risk_context")=="prior20" else 0.
            return copied
        # Let the original baseline cube use the same fixed anchor marker.
        arrays["features"][0,0,0]=0.;before=deepcopy(arrays)
        def evaluator(subset,values,metadata,**kwargs):
            observed.append([c["id"] for c in subset])
            ids={c["id"] for c in subset}
            if "allocation" in ids:self.assertIn(h["id"],ids)
            n=len(subset);days=len(meta["dates"]);payload={}
            marker=values["features"][0,0,0]
            for name,lag,fee in r.SCENARIOS:
                daily=np.asarray([[marker]*days for c in subset],dtype=float)
                payload[name]=dict(returns=daily,holdings=np.zeros((n,days),dtype=np.int32),
                    summary=np.zeros((n,10)),turnover_equivalent=np.zeros(n),turnover_by_day=np.zeros((n,days)),
                    metadata={c["id"]:dict(kind=c["kind"]) for c in subset})
            return payload,list(meta["dates"])
        with patch.object(r,"view_for_config",side_effect=view):
            results,dates=r.run_union(records,arrays,meta,meta["dates"][-1],evaluator=evaluator)
        self.assertEqual(len(observed),3)
        self.assertEqual(results["close_1bp"]["returns"][1,0],20.)
        self.assertEqual(set(results["close_1bp"]["metadata"]),{c["id"] for c in records})
        for name in arrays:np.testing.assert_array_equal(arrays[name],before[name])

    def test_allocation_selection_is_preserved_and_execution_refuses_it(self):
        arrays,meta,config=fixture(3);h=candidate(config,False)
        allocation=dict(id="allocation",kind="allocation",allocation={},selectable=True)
        registry=dict(candidates=[h,allocation],controls={"h":h["id"]})
        trained=[dict(fold=r.FOLDS[0],selection=dict(chosen_id="allocation",chosen_kind="allocation"))]
        before=deepcopy(trained)
        with self.assertRaisesRegex(ValueError,"Allocation selected"):
            r.selected_schedules(registry,trained)
        self.assertEqual(trained,before)

    def test_grandfathered_out_of_pool_holding_keeps_risk_but_cannot_be_bought(self):
        arrays,meta,config=fixture(6)
        config.update(score="alternate",stock_pool=[B],global_pool=[C],min_hold=10,
                      locked_panic_exit=1,crash_guard="always",buffer_mode="momentum")
        arrays["scores"][2,:,ASSETS.index(A)]=-1e100
        arrays["features"][3,ASSETS.index(A),F["ret1"]]=-.06
        before=deepcopy(arrays)
        def strict(values,metadata,current):
            view=dict(values,features=values["features"].copy())
            view["features"][:,:,F["valid"]]*=values["scores"][2]>-1e90
            return view
        with patch.object(r,"view_for_config",side_effect=strict):
            values=r.grandfathered_factory(arrays,meta,config,"current20")
        self.assertTrue(values.informed(2,A))
        self.assertNotIn(A,values.ranked(2))
        self.assertEqual(values.score(2,A),0.)
        p=Policy(values,config)
        held=Portfolio(holding=A,age=3,asset_peak=100.)
        self.assertEqual(p.intent(2,held,True)[0],A)
        self.assertNotEqual(p.intent(2,Portfolio(),True)[0],A)
        held.lock_until=8
        target,panic,crash=p.intent(3,held,True)
        self.assertEqual(target,"511880")
        self.assertTrue(panic);self.assertFalse(crash)
        self.assertEqual(held.lock_until,8)
        for name in arrays:np.testing.assert_array_equal(arrays[name],before[name])

    def test_factory_does_not_change_constant_original_policy_path(self):
        arrays,meta,config=fixture(12)
        config.update(locked_panic_exit=0)
        for lag in (0,1):
            current=dict(config,lag=lag)
            schedule=segments(meta,[current,current],contexts=["current20","current20"])
            expected=run_reference(arrays,meta,current,fee=.0011)
            actual=walk_run(schedule,arrays,meta,fee=.0011,view_factory=r.grandfathered_factory)
            for field in ("returns","holdings","summary"):
                np.testing.assert_array_equal(actual[field],expected[field])

    def test_train_fold_uses_cutoff_year_tail_and_h_fallback(self):
        arrays,meta,config=fixture(4)
        meta["dates"]=["2014-01-02","2017-01-03","2017-12-29","2026-01-05"]
        configs=[candidate(dict(config,panic=.01+i*.01),False) for i in range(5)]
        controls=dict(zip(("v9","v91","v92","simple","h"),[c["id"] for c in configs]))
        registry=dict(candidates=configs,controls=controls)
        seen=[]
        def build(values,metadata,own):
            return values,metadata,{}
        def evaluator(records,values,metadata,end):
            self.assertEqual(end,"2017-12-31")
            self.assertEqual(metadata["dates"][-1],"2017-12-29")
            seen.append(len(metadata["dates"]));n=len(records);t=3
            result={name:dict(returns=np.zeros((n,t)),holdings=np.zeros((n,t),dtype=np.int32),
                     summary=np.zeros((n,10)),turnover_equivalent=np.zeros(n),turnover_by_day=np.zeros((n,t)),
                     metadata={c["id"]:{} for c in records}) for name,lag,fee in r.SCENARIOS}
            return result,list(metadata["dates"])
        with tempfile.TemporaryDirectory() as directory,patch.object(r,"build_exante",side_effect=build):
            result=r.train_fold(registry,arrays,meta,{},r.FOLDS[0],Path(directory),evaluator=evaluator)
            choice=result["selection"]
            self.assertEqual(choice["tail_period"],["2017-01-01","2017-12-31"])
            self.assertEqual(choice["available_blocks"],["early"])
            self.assertEqual(choice["chosen_id"],controls["h"])
            self.assertTrue(choice["used_h_fallback"])
            self.assertEqual(choice["qualified_count"],0)
        self.assertEqual(seen,[3])


if __name__=="__main__":unittest.main()
