"""Registration/feature diagnostics only; no new performance in unit tests."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from v12 import sensitivity as s
from v12.tests.test_diagnostic_features import fixture,F

ROOT=Path(__file__).resolve().parents[2]


def definitions():
    records={}
    for name in ('registered_candidates.json','results/consensus/registered_candidates.json','results/exante/registered_candidates.json'):
        registry=json.loads((ROOT/'v12'/name).read_text())
        records.update({r['id']:r for r in registry['candidates']})
    original=json.loads((ROOT/'v12/registered_candidates.json').read_text())
    return records,original['controls']


class SensitivityTests(unittest.TestCase):
    def test_fixed_registration_is_nonselectable_and_retains_every_noop_and_invalid_proposal(self):
        records,controls=definitions();before=deepcopy(records)
        registry=s.generate(records,controls)
        self.assertEqual(records,before)
        self.assertEqual(set(registry['parents']),set(s.PARENTS)|{'h'})
        self.assertTrue(all(not r['selectable'] for r in registry['records']))
        self.assertIsNone(registry['official_primary'])
        self.assertEqual(registry['parents']['h']['center'],registry['controls']['h'])
        self.assertEqual(len(registry['parents']['h']['neighbors']),10)
        for name in s.PARENTS:self.assertEqual(len(registry['parents'][name]['neighbors']),12)
        for link in registry['parents'].values():
            off=next(p for p in link['proposals'] if p['label']=='disable_locked_panic_exit')
            self.assertTrue(off['no_op'])
            self.assertEqual(off['id'],link['center'])
        h=registry['parents']['h']['proposals']
        self.assertIn('invalid_reason',next(p for p in h if p['label']=='min_hold_-1'))
        self.assertTrue(next(p for p in h if p['label']=='min_hold_+1')['no_op'])
        c=registry['parents']['balanced_c']
        contexts=[p for p in c['proposals'] if p['label'] in ('restore_H_risk_context','restore_current20')]
        self.assertEqual(len(contexts),2)
        self.assertEqual(contexts[0]['id'],contexts[1]['id'])
        self.assertEqual(c['ablations'].count(contexts[0]['id']),1)

    def test_actual_feature_overrides_and_prior_context_are_in_identity(self):
        records,controls=definitions()
        c=records[s.PARENTS['balanced_c']]['config'];spec=s.feature_spec(c)
        self.assertNotEqual(s.identity(c,dict(spec,ma_window=144)),s.identity(c,dict(spec,ma_window=216)))
        self.assertNotEqual(s.identity(c,spec),s.identity(dict(c,risk_context='current20'),spec))
        h=records[controls['h']]['config'];fs=s.feature_spec(h)
        self.assertEqual(s.identity(dict(h,min_hold=0),fs),s.identity(dict(h,min_hold=1),fs))
        self.assertEqual(s.identity(h,fs),s.identity(dict(h,risk_context='current20'),fs))

    def test_leave_one_out_has_all_36_pairs_and_never_deletes_gold_cash_or_price_information(self):
        records,controls=definitions();registry=s.generate(records,controls)
        lookup={r['id']:r for r in registry['records']}
        self.assertEqual(len(registry['leave_one_out']),36)
        h=records[controls['h']]['config'];original=set(h['stock_pool']+h['global_pool'])
        for item in registry['leave_one_out']:
            self.assertNotIn(item['excluded'],('511880','518880'))
            for cid in (item['id'],item['paired_h']):
                c=lookup[cid]['config']
                self.assertEqual(set(c['stock_pool']+c['global_pool']),original-{item['excluded']})
            self.assertEqual(lookup[item['paired_h']]['feature_spec'],s.feature_spec(h))
        benchmark=[r for r in registry['leave_one_out'] if r['excluded']=='510300']
        self.assertEqual(len(benchmark),4)

    def test_transform_starts_from_base_then_applies_prior_once_without_changing_crash_MA250(self):
        arrays,meta,histories,unused=fixture(330)
        arrays['features'][:,:,F['vol20']]=np.arange(330)[:,None]*.001
        arrays['features'][:,:,F['vol60']]=np.arange(330)[:,None]*.002
        before=deepcopy(arrays)
        records,controls=definitions();parent=records[s.PARENTS['balanced_c']]['config']
        record=dict(config=parent,feature_spec=dict(window=25,smooth=3,ma_window=144))
        view,m,c=s.prepare_view(arrays,meta,histories,record)
        self.assertEqual(view['features'][300,0,F['vol20']],.598)
        np.testing.assert_array_equal(view['features'][:,:,F['ma250']],arrays['features'][:,:,F['ma250']])
        self.assertEqual(m['v12_diagnostic_features']['actual_ma_window'],144)
        self.assertEqual(c['ma'],'ma180')
        self.assertEqual(c['risk_context'],'prior_max20_60')
        for key in arrays:np.testing.assert_array_equal(arrays[key],before[key])

    def test_group_receipt_precedes_evaluator_and_merge_retains_registry_order(self):
        arrays,meta,histories,unused=fixture(330)
        definitions_,controls=definitions();registry=s.generate(definitions_,controls)
        # A center and C center share no transformed/risk view and must both
        # survive grouping without accidental double shifting or ID swapping.
        ids=[registry['parents'][label]['center'] for label in ('balanced_a','balanced_c')]
        registry=dict(registry,records=[r for r in registry['records'] if r['id'] in ids])
        constants={r['id']:i+.01 for i,r in enumerate(registry['records'])}
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            def evaluator(records,view,m,start,end,scenarios,workers):
                self.assertEqual(workers,1)
                receipts=[json.loads(p.read_text()) for p in (Path(directory)/'feature_receipts').glob('*.json')]
                self.assertTrue(any(set(x['ids'])=={r['id'] for r in records} for x in receipts))
                calls.extend(r['id'] for r in records)
                n,d=len(records),len(m['dates'])
                values=np.asarray([constants[r['id']] for r in records])
                data={name:dict(returns=np.repeat(values[:,None],d,axis=1),
                    holdings=np.zeros((n,d),dtype=np.int32),summary=np.repeat(values[:,None],10,axis=1),
                    turnover_equivalent=values.copy(),turnover_by_day=np.zeros((n,d)),
                    metadata={r['id']:{'synthetic':True} for r in records}) for name,_,_ in scenarios}
                return data,m['dates']
            result,dates=s.run_registered(registry,arrays,meta,histories,evaluator,output=directory)
            self.assertEqual(set(calls),set(ids))
            for i,record in enumerate(registry['records']):
                self.assertEqual(result['close_1bp']['returns'][i,0],constants[record['id']])
            self.assertEqual(dates,meta['dates'])


if __name__=='__main__':unittest.main()
