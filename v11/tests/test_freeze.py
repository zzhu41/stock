"""Synthetic receipt-chain tests; no real profiles or research outputs are written."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v11 import freeze, artifacts


class FreezeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='v11-freeze-test-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.base=self.root/'v11';self.base.mkdir()
        change=patch.multiple(freeze,BASE=self.base,ROOT=self.root);change.start();self.addCleanup(change.stop)
        self.dates=[freeze.START,'2025-12-31',freeze.END]
        self.meta=dict(dates=self.dates,fingerprints={'synthetic':'fixed'})
        self.finalists=dict(candidates=[dict(id=cid) for cid in ('A','B','H','S')],
            roles={'v11_a':'A','v11_b':'B','v10_h':'H','simple_reference':'S'})
        self.write('fixture.py',{'fixed':True})
        folder='results/report_2026/'
        path=self.base/folder/'paths.npz';path.parent.mkdir(parents=True)
        arrays={name+'__'+field:np.zeros((4,width),dtype=np.int32 if field=='holdings' else np.float64)
                for name,unused,fee in freeze.PRESSURES for field,width in
                (('returns',len(self.dates)),('holdings',len(self.dates)),('summary',10))}
        np.savez_compressed(path,**arrays)
        self.registration=dict(finalists=self.finalists,evaluation_end=freeze.END,no_reselection=True,
            scenarios=[list(s) for s in freeze.PRESSURES],sources={'fixture.py':freeze.sha(self.base/'fixture.py')},
            feature_fingerprints=self.meta['fingerprints'])
        self.write(folder+'registration.json',self.registration)
        self.write(folder+'path_metadata.json',dict(ids=['A','B','H','S'],dates=self.dates,sha256=freeze.sha(path)))
        rows=[]
        for cid in ('A','B','H','S'):
            growth=.4 if cid in ('H','S') else .3
            rows.append(dict(id=cid,scenarios={name:dict(confirmation=dict(cagr=growth,max_dd=-.20))
                                              for name,unused,fee in freeze.PRESSURES}))
        expected={role:freeze.gates(rows[i],rows[2]) for i,role in enumerate(('v11_a','v11_b'))}
        self.report=dict(rows=rows,roles=self.finalists['roles'],no_reselection=True,
            registration_sha256=freeze.sha(self.base/folder/'registration.json'),paths_sha256=freeze.sha(path),
            sources=self.registration['sources'],confirmation_verdicts=expected)
        self.write(folder+'evaluation.json',self.report)

    def write(self,name,value):
        path=self.base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def test_valid_final_report_accepts_failed_candidates_but_recomputes_verdict(self):
        self.assertEqual(freeze.verify_final_report(self.finalists,self.meta),self.report)
        self.assertFalse(self.report['confirmation_verdicts']['v11_b']['passed'])
        changed=deepcopy(self.report);changed['confirmation_verdicts']['v11_b']['passed']=True
        self.write('results/report_2026/evaluation.json',changed)
        with self.assertRaisesRegex(ValueError,'verdict'):freeze.verify_final_report(self.finalists,self.meta)

    def test_final_report_requires_registration_hash_and_all_eight_scenarios(self):
        changed=deepcopy(self.registration);changed['evaluation_end']='2025-12-31'
        self.write('results/report_2026/registration.json',changed)
        with self.assertRaisesRegex(ValueError,'registration hash'):freeze.verify_final_report(self.finalists,self.meta)
        changed=deepcopy(self.registration);changed['scenarios']=changed['scenarios'][:-1]
        self.write('results/report_2026/registration.json',changed)
        report=deepcopy(self.report);report['registration_sha256']=freeze.sha(self.base/'results/report_2026/registration.json')
        self.write('results/report_2026/evaluation.json',report)
        with self.assertRaisesRegex(ValueError,'eight registered'):freeze.verify_final_report(self.finalists,self.meta)

    def test_final_report_cannot_accept_a_different_npz_shape_even_with_new_hash(self):
        path=self.base/'results/report_2026/paths.npz'
        with np.load(path) as source:arrays={key:source[key] for key in source.files}
        arrays['lag1_21bp__returns']=np.zeros((4,2));np.savez_compressed(path,**arrays)
        report=deepcopy(self.report);report['paths_sha256']=freeze.sha(path)
        self.write('results/report_2026/evaluation.json',report)
        self.write('results/report_2026/path_metadata.json',dict(ids=['A','B','H','S'],dates=self.dates,sha256=freeze.sha(path)))
        with self.assertRaisesRegex(ValueError,'path shape'):freeze.verify_final_report(self.finalists,self.meta)

    def audit_fixture(self):
        self.write('results/final_audit/trades.json',{'synthetic':True})
        audit=dict(passed=True,no_reselection=True,period=[freeze.START,freeze.END],
            input_sha256={'v11/fixture.py':freeze.sha(self.base/'fixture.py')},
            artifacts={'trades.json':freeze.sha(self.base/'results/final_audit/trades.json')},
            checks=dict(confirmation_prefix_exact=True,exact_reference_returns_holdings_summary=True,
                reference_days=len(self.dates),reference_roles=['v10_h','v11_a','v11_b','simple_reference'],
                quote_repriced_paths=32,recomputed_metric_values=1,maximum_daily_return_error=0.,maximum_metric_absolute_error=1e-15),
            confirmation_verdicts=self.report['confirmation_verdicts'])
        self.write('results/final_audit/audit.json',audit)
        return audit

    def test_audit_top_flag_does_not_override_failed_detail_or_numeric_errors(self):
        audit=self.audit_fixture();freeze.verify_final_audit(self.report,self.meta)
        for key,value in (('confirmation_prefix_exact',False),('reference_days',2),
                          ('maximum_daily_return_error',.01),('maximum_metric_absolute_error',float('nan'))):
            changed=deepcopy(audit);changed['checks'][key]=value;self.write('results/final_audit/audit.json',changed)
            with self.subTest(key=key),self.assertRaises(ValueError):freeze.verify_final_audit(self.report,self.meta)

    def test_audit_input_and_output_hashes_are_both_checked(self):
        self.audit_fixture()
        for name in ('fixture.py','results/final_audit/trades.json'):
            path=self.base/name;before=path.read_bytes();path.write_bytes(b'changed')
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'hash differs'):
                freeze.verify_final_audit(self.report,self.meta)
            path.write_bytes(before)

    def test_report_receipt_checks_outputs_not_just_presence(self):
        self.write('report.py',{'source':True})
        inputs=('results/report_2026/evaluation.json','results/report_2026/paths.npz','results/final_audit/audit.json',
                'results/leave_one_out/evaluation.json','results/reselection/evaluation.json',
                'results/combined_development_diagnostics/diagnostics.json','results/b_neighborhood/comparison.json')
        outputs=('REPORT.md','results/equity_drawdown.png','results/full_comparison.csv','results/annual_returns.csv','results/execution_stress.csv')
        for name in inputs+outputs:
            if not (self.base/name).exists():self.write(name,{'synthetic':name})
        receipt=dict(source_sha256=freeze.sha(self.base/'report.py'),candidate_selection_performed=False,
            inputs={n:freeze.sha(self.base/n) for n in inputs},outputs={n:freeze.sha(self.base/n) for n in outputs})
        self.write('results/report_receipt.json',receipt);freeze.verify_report_receipt()
        (self.base/'REPORT.md').write_text('stale output')
        with self.assertRaisesRegex(ValueError,'hash differs'):freeze.verify_report_receipt()


class ArchiveSealTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='v11-archive-seal-');self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);path=self.base/'results/x/paths.npz';path.parent.mkdir(parents=True)
        np.savez_compressed(path,returns=np.asarray([0.,.01]))
        for module in (freeze,artifacts):
            changed=patch.object(module,'BASE',self.base);changed.start();self.addCleanup(changed.stop)
        changed=patch.object(artifacts,'MANIFEST',self.base/'path_archives.json');changed.start();self.addCleanup(changed.stop)
        artifacts.pack()

    def test_exact_streaming_archive_verification_does_not_modify_raw(self):
        path=self.base/'results/x/paths.npz';before=path.read_bytes()
        self.assertEqual(freeze.verify_archives(),1)
        self.assertEqual(path.read_bytes(),before)

    def test_missing_archive_coverage_and_empty_manifest_are_rejected(self):
        path=self.base/'path_archives.json';original=json.loads(path.read_text())
        path.write_text(json.dumps(dict(original,archives=[])))
        with self.assertRaisesRegex(ValueError,'cover every raw'):freeze.verify_archives()
        path.write_text(json.dumps(original));np.savez_compressed(self.base/'results/unlisted.npz',x=np.asarray([1.]))
        with self.assertRaisesRegex(ValueError,'cover every raw'):freeze.verify_archives()

    def test_self_consistent_compressed_hash_still_must_reconstruct_original_bytes(self):
        path=self.base/'path_archives.json';manifest=json.loads(path.read_text());row=manifest['archives'][0]
        archive=self.base/row['archive']
        with gzip.open(archive,'wb') as out:out.write(b'different raw contents')
        row['archive_sha256']=freeze.sha(archive);path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'reconstruct'):freeze.verify_archives()


if __name__=='__main__':unittest.main()
