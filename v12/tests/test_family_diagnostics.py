"""Only synthetic completed stages; never read or bootstrap real candidate paths."""
from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v12 import family_diagnostics as fd


def record(label, kind='single'):
    payload={'synthetic_model':label}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(id='v12_'+digest[:20],hash=digest,kind=kind,
                **{{'single':'config','allocation':'allocation','benchmark':'benchmark'}[kind]:payload},
                selectable=label!='h',complexity=0,families=['synthetic'])


class StageFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v12-family-fixture-');self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        (self.base/'engine.py').write_text('fixed synthetic engine\n')
        fd.dump(self.base/'fixture_input.json',dict(synthetic=True))
        self.dates=[(date(2020,12,1)+timedelta(days=i)).isoformat() for i in range(70)]
        self.descriptors=[self.stage('A',['h','alpha','bad']),self.stage('B',['h','alpha','beta'])]

    def stage(self,name,labels):
        directory=self.base/'results'/name;directory.mkdir(parents=True)
        rows=sorted([record(label) for label in labels],key=lambda r:r['id'])
        ids=[r['id'] for r in rows];controls={'h':record('h')['id']}
        fd.dump(directory/'registered_candidates.json',dict(unique_count=len(rows),candidates=rows,controls=controls))
        values={}
        for scenario_index,scenario in enumerate(fd.SCENARIOS):
            daily=[]
            for row in rows:
                label=row['config']['synthetic_model'];t=np.arange(len(self.dates),dtype=float)
                if label=='h':r=np.zeros(len(t))
                elif label=='bad':r=np.full(len(t),-.001)
                else:r=.004*np.sin(t/5)+(.0004 if label=='alpha' else .0002)-scenario_index*.00001
                r[0]=0.;daily.append(r)
            matrix=np.asarray(daily,dtype=np.float64)
            summary=np.zeros((len(rows),10),dtype=np.float64)
            for i,r in enumerate(matrix):
                nav=np.cumprod(1+r);peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
                summary[i]=[nav[-1],np.expm1(np.log1p(r).sum()*244/len(r)),np.min(nav/peak-1),0,1,0,0,0,r.sum(),np.dot(r,r)]
            values[scenario+'__returns']=matrix
            values[scenario+'__holdings']=np.zeros(matrix.shape,dtype=np.int32)
            values[scenario+'__summary']=summary
        np.savez_compressed(directory/'paths.npz',**values)
        period=[self.dates[0],self.dates[-1]]
        fingerprint=dict(sources={'engine.py':fd.sha(self.base/'engine.py')},
                         inputs={'fixture_input.json':fd.sha(self.base/'fixture_input.json')})
        fd.dump(directory/'registration.json',dict(registered_at='2026-09-28T01:00:00+00:00',design=dict(
            candidate_count=len(rows),period=period,scenarios=fd.SCENARIO_DEFINITIONS,fingerprints=fingerprint)))
        fd.dump(directory/'path_metadata.json',dict(ids=ids,dates=self.dates,sha256=fd.sha(directory/'paths.npz')))
        fd.dump(directory/'evaluation.json',dict(rows=[{'id':cid} for cid in ids],period=period,observed_period=period,
            registration_sha256=fd.sha(directory/'registration.json'),paths_sha256=fd.sha(directory/'paths.npz')))
        fd.dump(directory/'execution_metadata.json',dict(registration_sha256=fd.sha(directory/'registration.json'),scenarios={}))
        selection=dict(primary=None,status='no_qualified_candidate',qualified_count=0,controls=controls,
            selection_period=period,frozen_at='2026-09-28T01:01:00+00:00',provenance={'fingerprints':fingerprint})
        fd.dump(directory/'selection.json',selection)
        self.resign(directory)
        return dict(name=name,directory=str(directory.relative_to(self.base)),
                    registry=str((directory/'registered_candidates.json').relative_to(self.base)))

    def resign(self,directory):
        selection=fd.read(directory/'selection.json')
        for key,filename in (('registry','registered_candidates.json'),('registration','registration.json'),
            ('evaluation','evaluation.json'),('paths','paths.npz'),('path_metadata','path_metadata.json'),
            ('execution_metadata','execution_metadata.json')):
            selection['provenance'][key+'_sha256']=fd.sha(directory/filename)
        fd.dump(directory/'selection.json',selection)

    def alter_matrix(self,field):
        directory=self.base/'results/B'
        ids=fd.read(directory/'path_metadata.json')['ids'];index=ids.index(record('alpha')['id'])
        with np.load(directory/'paths.npz') as source:arrays={k:source[k] for k in source.files}
        arrays['lag1_11bp__'+field][index,0]+=1 if field=='holdings' else 1e-10
        np.savez_compressed(directory/'paths.npz',**arrays)
        metadata=fd.read(directory/'path_metadata.json');metadata['sha256']=fd.sha(directory/'paths.npz')
        fd.dump(directory/'path_metadata.json',metadata)
        evaluation=fd.read(directory/'evaluation.json');evaluation['paths_sha256']=metadata['sha256']
        fd.dump(directory/'evaluation.json',evaluation);self.resign(directory)


class FamilyReceiptTests(StageFixture):
    def test_full_union_keeps_failed_trials_and_deduplicates_only_exact_semantics(self):
        result=fd.load_union(self.descriptors,self.base)
        self.assertEqual(result['scope']['stage_counts'],{'A':3,'B':3})
        self.assertEqual(result['scope']['union_candidates'],4)
        self.assertEqual(result['scope']['repeated_occurrences'],2)
        self.assertEqual(len(result['overlap_checks']),2)
        self.assertIn(record('bad')['id'],result['ids'])
        self.assertTrue(result['scope']['includes_all_controls_and_rejected_candidates'])
        for values in result['returns'].values():
            self.assertEqual(values.shape,(4,70));self.assertFalse(values.flags.writeable)

    def test_new_extension_files_do_not_invalidate_older_frozen_source_inventory(self):
        (self.base/'new_B_only_module.py').write_text('new extension, old engine unchanged\n')
        self.assertEqual(fd.load_union(self.descriptors,self.base)['scope']['union_candidates'],4)
        (self.base/'engine.py').write_text('changed registered source\n')
        with self.assertRaisesRegex(ValueError,'Registered sources changed'):fd.load_union(self.descriptors,self.base)

    def test_stage_without_redundant_observed_period_uses_hash_pinned_date_axis(self):
        directory=self.base/'results/B';evaluation=fd.read(directory/'evaluation.json')
        del evaluation['observed_period'];fd.dump(directory/'evaluation.json',evaluation);self.resign(directory)
        result=fd.load_union(self.descriptors,self.base)
        self.assertEqual(result['dates'],self.dates)
        self.assertEqual(result['scope']['union_candidates'],4)

    def test_each_duplicate_field_requires_exact_equality_even_with_valid_stage_receipts(self):
        for field in fd.FIELDS:
            with self.subTest(field=field):
                directory=self.base/'results/B'
                original={p.name:p.read_bytes() for p in directory.iterdir()}
                self.alter_matrix(field)
                with self.assertRaisesRegex(ValueError,'Repeated ID path differs exactly'):
                    fd.load_union(self.descriptors,self.base)
                for name,contents in original.items():(directory/name).write_bytes(contents)

    def test_unreconciled_matrix_tamper_is_rejected_by_hash_before_statistics(self):
        path=self.base/'results/B/paths.npz';path.write_bytes(path.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'selection receipt mismatch'):fd.load_union(self.descriptors,self.base)

    def test_source_change_while_matrix_is_loading_cannot_be_resealed(self):
        original_load=np.load
        def changed(path,*args,**kwargs):
            if Path(path).parent.name=='B':(self.base/'engine.py').write_text('changed during matrix read')
            return original_load(path,*args,**kwargs)
        with patch.object(fd.np,'load',side_effect=changed),self.assertRaisesRegex(ValueError,'changed during stage verification'):
            fd.load_union(self.descriptors,self.base)

    def test_changed_semantics_cannot_keep_the_old_candidate_id(self):
        directory=self.base/'results/B';registry=fd.read(directory/'registered_candidates.json')
        registry['candidates'][0]['config']['synthetic_model']='different'
        fd.dump(directory/'registered_candidates.json',registry);self.resign(directory)
        with self.assertRaisesRegex(ValueError,'Canonical candidate id/hash'):fd.load_union(self.descriptors,self.base)

    def test_calendar_mismatch_is_not_silently_intersected(self):
        directory=self.base/'results/B';metadata=fd.read(directory/'path_metadata.json')
        metadata['dates'][-1]='2021-02-10';fd.dump(directory/'path_metadata.json',metadata)
        period=[metadata['dates'][0],metadata['dates'][-1]]
        evaluation=fd.read(directory/'evaluation.json');evaluation.update(period=period,observed_period=period)
        registration=fd.read(directory/'registration.json');registration['design']['period']=period
        fd.dump(directory/'registration.json',registration)
        evaluation['registration_sha256']=fd.sha(directory/'registration.json');fd.dump(directory/'evaluation.json',evaluation)
        fd.dump(directory/'execution_metadata.json',dict(registration_sha256=fd.sha(directory/'registration.json')))
        selection=fd.read(directory/'selection.json');selection['selection_period']=period;fd.dump(directory/'selection.json',selection)
        self.resign(directory)
        with self.assertRaisesRegex(ValueError,'calendars/intervals'):fd.load_union(self.descriptors,self.base)

    def test_registration_precedes_complete_union_bootstrap_and_outputs_have_receipts(self):
        output=self.base/'diagnostics'
        watched={p:fd.sha(p) for p in self.base.rglob('*') if p.is_file()}
        real=fd.white_style_test;calls=[]
        def wrapped(values,benchmark,**kwargs):
            registration=fd.read(output/'registration.json')
            self.assertEqual(registration['design']['expected_candidate_count'],4)
            self.assertEqual(values.shape,(4,70))
            self.assertEqual(kwargs['expected_candidate_count'],4)
            self.assertEqual(kwargs['seed'],20260928);self.assertEqual(kwargs['draws'],2000)
            self.assertEqual(kwargs['method'],'stationary')
            calls.append(kwargs['block_length'])
            return real(values,benchmark,**kwargs)
        with patch.object(fd,'white_style_test',side_effect=wrapped):
            result=fd.run(self.descriptors,output=output,base=self.base)
        self.assertEqual(calls,[20,60]*4)
        for block in ('20','60'):
            samples={result['scenarios'][scenario][block]['resampling_count_sha256'] for scenario in fd.SCENARIOS}
            self.assertEqual(len(samples),1)
        concentration=fd.load_concentration(output,'close_1bp')
        self.assertEqual(set(concentration['candidates']),set(concentration['candidate_ids']))
        self.assertEqual(len(list(output.glob('concentration_*.json.gz'))),4)
        receipt=fd.read(output/'receipt.json')
        for name,digest in receipt['artifacts_sha256'].items():self.assertEqual(fd.sha(output/name),digest)
        for path,digest in watched.items():self.assertEqual(fd.sha(path),digest)
        with self.assertRaisesRegex(ValueError,'frozen'):fd.run(self.descriptors,output=output,base=self.base)
        archive=output/'concentration_close_1bp.json.gz';archive.write_bytes(archive.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'archive differs'):fd.load_concentration(output,'close_1bp')


class ConcentrationTests(unittest.TestCase):
    def test_cached_calendar_result_matches_frozen_numeric_helper_with_explicit_dates(self):
        from v11.diagnostics import advantage_concentration
        dates=['2020-12-30','2020-12-31','2021-01-04','2021-01-05']
        selected=np.asarray([.10,-.09,.07,-.02]);benchmark=np.asarray([.02,0.,-.01,.01])
        expected=advantage_concentration(selected,benchmark,dates=dates,top_k=(10,))
        actual=fd.concentration_by_year(selected,benchmark,dates)
        self.assertEqual({k:actual[k] for k in expected},expected)
    def test_yearly_net_and_positive_mass_are_different_and_sum_consistently(self):
        dates=['2020-12-30','2020-12-31','2021-01-04','2021-01-05']
        log_excess=np.asarray([.20,-.18,.10,-.05])
        result=fd.concentration_by_year(np.expm1(log_excess),np.zeros(4),dates)
        self.assertAlmostEqual(result['net_log_excess'],.07)
        self.assertAlmostEqual(result['positive_log_excess'],.30)
        self.assertAlmostEqual(result['top']['10']['share_of_net_log_excess'],.30/.07)
        self.assertAlmostEqual(result['top']['10']['share_of_positive_log_excess'],1.)
        self.assertAlmostEqual(result['years']['2020']['share_of_net_log_excess'],.02/.07)
        self.assertAlmostEqual(result['years']['2020']['share_of_positive_log_excess'],2/3)
        self.assertAlmostEqual(sum(v['net_log_excess'] for v in result['years'].values()),result['net_log_excess'])
        self.assertFalse(result['annualization_of_partial_year'])

    def test_nonpositive_net_does_not_report_meaningless_net_shares(self):
        result=fd.concentration_by_year(np.expm1([.1,-.2]),np.zeros(2),['2020-12-31','2021-01-04'])
        self.assertIsNone(result['top']['10']['share_of_net_log_excess'])
        self.assertTrue(all(value['share_of_net_log_excess'] is None for value in result['years'].values()))
        self.assertAlmostEqual(result['years']['2020']['share_of_positive_log_excess'],1.)


if __name__=='__main__':unittest.main()
