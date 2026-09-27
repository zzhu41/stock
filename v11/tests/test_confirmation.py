"""Synthetic-only confirmation gates, continuous-path slicing and frozen roles."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v10_deep.schema import baseline, identifier
from v11.registry import canonical
from v11 import confirmation
from v11.scan import select


def comparison_rows():
    reference = dict(id="H", scenarios={})
    candidate = dict(id="A", scenarios={})
    for name, cagr, dd in (("close_1bp", .50, -.20), ("close_11bp", .40, -.23),
                            ("lag1_1bp", .35, -.26), ("lag1_11bp", .30, -.28)):
        reference["scenarios"][name] = dict(confirmation=dict(cagr=cagr,max_dd=dd))
        candidate["scenarios"][name] = dict(confirmation=dict(cagr=cagr,max_dd=dd))
    return candidate,reference


class ConfirmationGateTests(unittest.TestCase):
    def test_all_six_gates_match_protocol_boundary_and_negative_drawdown_direction(self):
        candidate, reference = comparison_rows()
        candidate["scenarios"]["close_1bp"]["confirmation"].update(cagr=.9*.50,max_dd=-.20-.02)
        for scenario in ("close_11bp","lag1_11bp"):
            candidate["scenarios"][scenario]["confirmation"]["max_dd"] = reference["scenarios"][scenario]["confirmation"]["max_dd"]-.02
        result=confirmation.gates(candidate,reference)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]),6)
        boundaries=(("close_1bp","cagr","return_floor"),("close_1bp","max_dd","main_drawdown"),
                    ("close_11bp","cagr","close_11bp_growth"),("close_11bp","max_dd","close_11bp_drawdown"),
                    ("lag1_11bp","cagr","lag1_11bp_growth"),("lag1_11bp","max_dd","lag1_11bp_drawdown"))
        for scenario,metric,gate in boundaries:
            changed=deepcopy(candidate)
            changed["scenarios"][scenario]["confirmation"][metric]-=.00001
            result=confirmation.gates(changed,reference)
            self.assertFalse(result["passed"])
            self.assertFalse(result["checks"][gate])

    def test_absolute_floor_30_is_not_lost_when_benchmark_ninety_percent_is_small(self):
        candidate,reference=comparison_rows()
        reference["scenarios"]["close_1bp"]["confirmation"]["cagr"] = .20
        candidate["scenarios"]["close_1bp"]["confirmation"]["cagr"] = .30
        self.assertTrue(confirmation.gates(candidate,reference)["checks"]["return_floor"])
        candidate["scenarios"]["close_1bp"]["confirmation"]["cagr"] = .299999
        self.assertFalse(confirmation.gates(candidate,reference)["checks"]["return_floor"])

    def test_other_periods_2026_and_other_scenarios_cannot_change_confirmation_verdict(self):
        candidate,reference=comparison_rows()
        before=confirmation.gates(candidate,reference)
        for row,number in ((candidate,1e9),(reference,-1e9)):
            row["full_cagr"]=number
            for payload in row["scenarios"].values():
                payload["development"]={"cagr":number,"max_dd":number}
                payload["report_only_2026"]={"cagr":number,"max_dd":number}
                payload["full"]={"cagr":number,"max_dd":number}
            row["scenarios"]["unknown_stress"]={"confirmation":{"cagr":number,"max_dd":number}}
        self.assertEqual(before,confirmation.gates(candidate,reference))
        del candidate["scenarios"]["close_1bp"]["confirmation"]
        with self.assertRaises(KeyError):
            confirmation.gates(candidate,reference)

    def test_confirmation_checks_are_identical_to_development_checks_for_same_metrics(self):
        candidate,reference=comparison_rows()
        candidate["scenarios"]["lag1_11bp"]["confirmation"]["max_dd"]=-.40
        def development(row):
            scenarios={}
            for name,payload in row["scenarios"].items():
                metrics=deepcopy(payload["confirmation"])
                scenarios[name]=dict(full=metrics,blocks={"early":metrics,"late":metrics},switches=10)
            return dict(id=row["id"],families=["synthetic"],scenarios=scenarios)
        configs=[dict(id=cid,risk_context="current20",score="same",families=["synthetic"]) for cid in ("H","A")]
        chosen=select([development(reference),development(candidate)],configs,{"h":"H"})
        self.assertEqual(chosen["checks"]["A"],confirmation.gates(candidate,reference)["checks"])


class ContinuousSlicingTests(unittest.TestCase):
    def test_confirmation_and_yearly_include_first_day_return_and_boundary_switch_fee(self):
        dates=["2021-12-30","2021-12-31","2022-01-04","2022-01-05","2025-12-31","2026-01-05"]
        fee=.0001
        returns=np.asarray([[0.,.02,1.03*(1-2*fee)-1,-.01,.02,1.04*(1-2*fee)-1]])
        holdings=np.asarray([[0,0,1,1,1,2]],dtype=np.int32)
        result=confirmation._rows([{"id":"A"}],{"close_1bp":{"returns":returns,"holdings":holdings}},dates,
              [("confirmation","2022-01-01","2025-12-31"),("report_only_2026","2026-01-01","2026-09-24")])[0]["scenarios"]["close_1bp"]
        expected=float(np.prod(1+returns[0,2:5])-1)
        self.assertAlmostEqual(result["confirmation"]["total_return"],expected,places=14)
        self.assertEqual(result["confirmation"]["sessions"],3)
        self.assertEqual(result["confirmation"]["switches"],1)
        self.assertAlmostEqual(result["yearly"]["2022"]["total_return"],1.03*(1-2*fee)*.99-1,places=14)
        self.assertAlmostEqual(result["yearly"]["2026"]["total_return"],returns[0,5],places=14)
        self.assertEqual(result["report_only_2026"]["switches"],1)
        # Omitting the boundary day's return or charging a second entry fee
        # would both disagree with the continuous saved daily path.
        self.assertNotAlmostEqual(result["confirmation"]["total_return"],float(np.prod(1+returns[0,3:5])-1),places=8)
        self.assertNotAlmostEqual(result["confirmation"]["total_return"],(1+expected)*(1-2*fee)-1,places=8)

    def test_subperiod_drawdown_includes_a_loss_on_its_first_observation(self):
        data={"close_1bp":dict(returns=np.asarray([[.5,-.10,.05]]),holdings=np.asarray([[0,1,1]]))}
        result=confirmation._rows([{"id":"A"}],data,["2021-12-31","2022-01-04","2022-01-05"],
                                  [("confirmation","2022-01-01","2025-12-31")])[0]["scenarios"]["close_1bp"]["confirmation"]
        self.assertAlmostEqual(result["max_dd"],-.10)
        self.assertAlmostEqual(result["total_return"],.9*1.05-1)
        self.assertEqual(result["switches"],1)


class FrozenRoleTests(unittest.TestCase):
    def test_a_b_primaries_remain_frozen_and_v9_controls_are_not_new_selections(self):
        def candidate(changes=None):
            value=baseline();value.update(changes or {});value=canonical(value)
            digest=identifier(value)
            return dict(value,id="v11_"+digest[:20],hash=digest,families=["synthetic"],parents=[],stage="synthetic")
        h=candidate(dict(score="wls20_smooth3",ma="ma180",panic_mode="volatility",panic=1.5))
        v92=candidate();simple=candidate(dict(min_hold=2))
        a_primary=candidate(dict(min_hold=3));b_primary=candidate(dict(min_hold=5))
        diagnostic=candidate(dict(buffer=.01))
        controls={"h":h["id"],"v92":v92["id"],"simple":simple["id"]}
        a=dict(candidates=[h,v92,simple,a_primary,diagnostic],controls=controls,score_specs=[])
        b=dict(candidates=[h,v92,simple,b_primary],controls=controls,score_specs=[])
        sa=dict(primary=a_primary["id"],top_return=diagnostic["id"],parents=[a_primary["id"],diagnostic["id"]],
                confirmation_best="never_select_this",report_2026_best="nor_this")
        sb=dict(primary=b_primary["id"],top_return=diagnostic["id"],parents=[b_primary["id"]])
        originals=deepcopy((a,b,sa,sb))
        with patch.object(confirmation,"_verified_stage",side_effect=[(a,sa),(b,sb)]), \
                patch.object(confirmation,"sha",return_value="0"*64):
            result=confirmation.frozen_finalists()
        self.assertEqual((a,b,sa,sb),originals)
        self.assertEqual(result["roles"]["v11_a"],a_primary["id"])
        self.assertEqual(result["roles"]["v11_b"],b_primary["id"])
        self.assertEqual(result["roles"]["a_highest_development_return"],diagnostic["id"])
        configs={c["id"]:c for c in result["candidates"]}
        v9=configs[result["roles"]["v9"]];v91=configs[result["roles"]["v91"]]
        self.assertEqual((v9["crash_mask"],v9["global_buffer"],v9["gold_buffer"]),(1,.02,.02))
        self.assertEqual((v91["crash_mask"],v91["global_buffer"],v91["gold_buffer"]),(1,.03,.03))
        self.assertEqual(len(configs),len(result["candidates"]))


class ConfirmationReceiptTests(unittest.TestCase):
    def setUp(self):
        cache=confirmation.BASE/'cache';cache.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='test-confirmation-',dir=str(cache))
        self.addCleanup(self.temporary.cleanup)
        self.base=Path(self.temporary.name)
        self.patch=patch.object(confirmation,'BASE',self.base);self.patch.start()
        self.addCleanup(self.patch.stop)
        self.dates=['2014-01-02','2021-12-31','2022-01-04','2025-12-31']
        self.finalists=dict(candidates=[{'id':'A'},{'id':'H'}],roles={'v11_a':'A','v10_h':'H'},score_specs=[],
                            a_selection_sha256='a'*64,b_selection_sha256='b'*64)
        self.meta=dict(dates=self.dates+['2026-01-05'],fingerprints={'fixture':True})
        self.folder=self.base/'results/confirmation'
        self.save_paths(self.folder,self.dates)
        source=self.base/'fixture_source.py';source.write_text('fixed synthetic source\n')
        sources={'fixture_source.py':confirmation.sha(source)}
        registration=dict(finalists=self.finalists,evaluation_end=confirmation.CONFIRM_END,
                          scenarios=[list(s) for s in confirmation.PRESSURES],sources=sources,
                          feature_fingerprints=self.meta['fingerprints'])
        self.write(self.folder/'registration.json',registration)
        evaluation=dict(registration_sha256=confirmation.sha(self.folder/'registration.json'),
                        paths_sha256=confirmation.sha(self.folder/'paths.npz'),roles=self.finalists['roles'],
                        no_reselection=True,sources=sources)
        self.write(self.folder/'evaluation.json',evaluation)

    @staticmethod
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def save_paths(self,folder,dates):
        folder.mkdir(parents=True,exist_ok=True)
        arrays={}
        for name,unused_lag,unused_fee in confirmation.PRESSURES:
            arrays[name+'__returns']=np.zeros((2,len(dates)))
            arrays[name+'__holdings']=np.zeros((2,len(dates)),dtype=np.int32)
            arrays[name+'__summary']=np.zeros((2,10))
        np.savez_compressed(folder/'paths.npz',**arrays)
        self.write(folder/'path_metadata.json',dict(ids=['A','H'],dates=dates,sha256=confirmation.sha(folder/'paths.npz')))

    def test_complete_confirmation_receipt_accepts_and_changed_roles_or_empty_ids_reject(self):
        verified=confirmation.validate_confirmation_before_2026(self.finalists,self.meta)
        self.assertEqual(verified['dates'],self.dates)
        changed=deepcopy(self.finalists);changed['roles']['v11_a']='H'
        with self.assertRaisesRegex(ValueError,'Finalists changed'):
            confirmation.validate_confirmation_before_2026(changed,self.meta)
        path=self.folder/'path_metadata.json';metadata=json.loads(path.read_text());metadata['ids']=[]
        self.write(path,metadata)
        with self.assertRaisesRegex(ValueError,'exact registered IDs/dates'):
            confirmation.validate_confirmation_before_2026(self.finalists,self.meta)

    def test_shape_and_registered_source_changes_cannot_pass_a_file_exists_gate(self):
        path=self.folder/'paths.npz'
        with np.load(path,allow_pickle=False) as saved:
            values={key:saved[key] for key in saved.files}
        values['lag1_21bp__returns']=np.zeros((1,len(self.dates)))
        np.savez_compressed(path,**values)
        digest=confirmation.sha(path)
        metadata=json.loads((self.folder/'path_metadata.json').read_text());metadata['sha256']=digest
        evaluation=json.loads((self.folder/'evaluation.json').read_text());evaluation['paths_sha256']=digest
        self.write(self.folder/'path_metadata.json',metadata);self.write(self.folder/'evaluation.json',evaluation)
        with self.assertRaisesRegex(ValueError,'shape'):
            confirmation.validate_confirmation_before_2026(self.finalists,self.meta)
        (self.base/'fixture_source.py').write_text('changed source\n')
        with self.assertRaisesRegex(ValueError,'source changed'):
            confirmation.validate_confirmation_before_2026(self.finalists,self.meta)

    def test_report_2026_rejects_partial_confirmation_before_running_any_model(self):
        for name in ('neighborhood','b_neighborhood','h_neighborhood'):
            self.write(self.base/'results'/name/'evaluation.json',{'synthetic':True})
        for name in ('development','combinations'):
            self.save_paths(self.base/'results'/name,self.dates[:2])
        def stage(folder,unused_registry_path):
            return ({'candidates':self.finalists['candidates']},
                    {'provenance':{'paths_sha256':confirmation.sha(folder/'paths.npz')}})
        self.write(self.folder/'evaluation.json',{})  # Exists, but not a completed receipt.
        with patch.object(confirmation,'frozen_finalists',return_value=self.finalists), \
                patch.object(confirmation,'build',return_value=({},self.meta)), \
                patch.object(confirmation,'validate_neighborhood_evidence',return_value={'synthetic':'verified'}), \
                patch.object(confirmation,'_verified_stage',side_effect=stage), \
                patch.object(confirmation,'run_candidates',side_effect=AssertionError('Must not unseal 2026')) as run:
            with self.assertRaises((KeyError,ValueError)):
                confirmation.run(report_2026=True)
            run.assert_not_called()
        self.assertFalse((self.base/'results/report_2026/registration.json').exists())


class NeighborhoodReceiptTests(unittest.TestCase):
    def setUp(self):
        cache=confirmation.BASE/'cache';cache.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='test-neighborhood-gate-',dir=str(cache))
        self.addCleanup(self.temporary.cleanup)
        self.base=Path(self.temporary.name)
        self.patch=patch.object(confirmation,'BASE',self.base);self.patch.start()
        self.addCleanup(self.patch.stop)
        self.finalists=dict(candidates=[{'id':c} for c in ('A','B','H')],score_specs=[],
                            roles={'v11_a':'A','v11_b':'B','v10_h':'H'},a_selection_sha256='a'*64)
        self.dates=[confirmation.START,confirmation.DEV_END]
        self.meta=dict(dates=self.dates+['2022-01-04'])
        for directory,parent in (('neighborhood','A'),('b_neighborhood','B'),('h_neighborhood','H')):
            folder=self.base/'results'/directory;folder.mkdir(parents=True)
            ids=[parent,parent+'_neighbor']
            record=dict(parents=[parent],parent_map={parent:{'center':parent}},candidates=[{'id':cid} for cid in ids])
            self.write(folder/'registered_candidates.json',record)
            self.write(folder/'matched_comparison_registration.json',{'synthetic_parent':parent})
            self.write(folder/'parent_override.json',{'synthetic_parent':parent})
            registration=dict(registry_sha256=confirmation.sha(folder/'registered_candidates.json'),parent=parent,
                matched_registration_sha256=confirmation.sha(folder/'matched_comparison_registration.json'),
                parent_override_sha256=confirmation.sha(folder/'parent_override.json'))
            self.write(folder/'registration.json',registration)
            self.write(folder/'feature_receipt.json',dict(registration_sha256=confirmation.sha(folder/'registration.json'),
                                                          execution_last_date=confirmation.DEV_END))
            arrays={}
            for name,unused_lag,unused_fee in confirmation.SCENARIOS:
                arrays[name+'__returns']=np.zeros((2,2))
                arrays[name+'__holdings']=np.zeros((2,2),dtype=np.int32)
                arrays[name+'__summary']=np.zeros((2,10))
            np.savez_compressed(folder/'paths.npz',**arrays)
            digest=confirmation.sha(folder/'paths.npz')
            self.write(folder/'path_metadata.json',dict(ids=ids,dates=self.dates,sha256=digest))
            evaluation=dict(period=self.dates,new_gate_added=False,
                registration_sha256=confirmation.sha(folder/'registration.json'),
                registry_sha256=confirmation.sha(folder/'registered_candidates.json'),
                feature_receipt_sha256=confirmation.sha(folder/'feature_receipt.json'),paths_sha256=digest,
                rows=[{'id':cid} for cid in ids],primary_unchanged='A',reselected=False,
                parent_selection_sha256='a'*64,diagnostic_parent=parent,A_primary_unchanged='A',B_primary_unchanged='B',
                selected_here=False,heldout_performance_accessed=False)
            self.write(folder/'evaluation.json',evaluation)

    @staticmethod
    def write(path,value):
        path.write_text(json.dumps(value))

    def test_three_complete_development_neighborhood_receipts_are_accepted(self):
        evidence=confirmation.validate_neighborhood_evidence(self.finalists,self.meta)
        self.assertEqual(set(evidence),{'neighborhood','b_neighborhood','h_neighborhood'})
        for name,digest in evidence.items():
            self.assertEqual(digest,confirmation.sha(self.base/'results'/name/'evaluation.json'))

    def test_parent_gate_receipt_and_id_date_mismatch_each_reject(self):
        folder=self.base/'results/b_neighborhood'
        cases=[('evaluation.json','diagnostic_parent','A'),('evaluation.json','new_gate_added',True),
               ('evaluation.json','heldout_performance_accessed',True),('evaluation.json','rows',[]),
               ('path_metadata.json','dates',[confirmation.START]),('path_metadata.json','ids',[])]
        for filename,key,value in cases:
            path=folder/filename;original=path.read_bytes();body=json.loads(original);body[key]=value
            self.write(path,body)
            with self.subTest(field=key),self.assertRaises(ValueError):
                confirmation.validate_neighborhood_evidence(self.finalists,self.meta)
            path.write_bytes(original)
        path=folder/'registration.json';path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'receipt hash differs'):
            confirmation.validate_neighborhood_evidence(self.finalists,self.meta)

    def test_bad_neighborhood_stops_confirmation_before_any_candidate_execution(self):
        path=self.base/'results/h_neighborhood/evaluation.json'
        body=json.loads(path.read_text());body['diagnostic_parent']='A';self.write(path,body)
        with patch.object(confirmation,'frozen_finalists',return_value=self.finalists), \
                patch.object(confirmation,'build',return_value=({},self.meta)), \
                patch.object(confirmation,'run_candidates',side_effect=AssertionError('Must not unseal confirmation')) as run:
            with self.assertRaisesRegex(ValueError,'B/H neighborhood parent'):
                confirmation.run(report_2026=False)
            run.assert_not_called()
        self.assertFalse((self.base/'results/confirmation/registration.json').exists())


if __name__=='__main__':
    unittest.main()
