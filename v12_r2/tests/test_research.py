"""Finite-grid and selection checks; no new real candidate performance."""
from copy import deepcopy
import unittest
import numpy as np

from v12_r2 import research as r
from v12_r2.features import build,view,effective_config
from v12.tests.test_consensus_features import fixture,AI,A,F
from v12.native import Simulator
from v12.reference import run_reference
from tempfile import TemporaryDirectory


def fake_baselines():
    def scene(year21=.06,tail=.75,cagr=.5458,dd=-.205):
        return dict(full=dict(cagr=cagr,max_dd=dd),yearly={'2021':dict(total_return=year21),'2026':dict(total_return=tail)},
                    blocks={b:dict(cagr=.4) for b,_,_ in r.BLOCKS},switches=20)
    return {role:dict(scenarios={name:scene(year21=.21 if role=='h' else .15 if role=='c' else .06)
                               for name,_,_ in r.SCENARIOS}) for role in ('c4','c','h','simple')}


def candidate_row(cid,year21=.25,tail=.8,cagr=.57,dd=-.20):
    row=deepcopy(fake_baselines()['c4']);row['id']=cid
    for s in row['scenarios'].values():
        s['full'].update(cagr=cagr,max_dd=dd)
        s['yearly']['2021']['total_return']=year21;s['yearly']['2026']['total_return']=tail
        for block in s['blocks'].values():block['cagr']=.43
    return row


class ResearchTests(unittest.TestCase):
    def test_complete_finite_registry_has_exact_864_grid_and_four_nonselectable_controls(self):
        grid=r.generate()
        self.assertEqual(len(grid['records']),866)
        self.assertEqual(sum(x['selectable'] for x in grid['records']),862)
        records={x['id']:x for x in grid['records']}
        self.assertEqual(set(grid['controls']),{'c4','c','h','simple'})
        for cid in grid['controls'].values():self.assertFalse(records[cid]['selectable'])
        for row in grid['records']:
            self.assertEqual(r.identity(row['config'],row['feature_spec']),row['hash'])
            if row['selectable']:
                self.assertEqual(row['config']['ma'],'ma180')
                self.assertEqual(row['config']['crash_mask'],5)
                self.assertEqual(row['config']['locked_panic_exit'],0)
                self.assertIn(row['feature_spec']['window'],(20,25,30))
        center=records[grid['controls']['c4']]
        self.assertIn('v12d_5e020674285fc2d9db44',center['previous_v12_ids'])

    def test_prior_context_and_actual_score_definition_are_part_of_identity(self):
        g=r.generate();c=next(x for x in g['records'] if x['id']==g['controls']['c4'])
        self.assertNotEqual(r.identity(c['config'],c['feature_spec']),r.identity(dict(c['config'],risk_context='current20'),c['feature_spec']))
        self.assertNotEqual(r.identity(c['config'],c['feature_spec']),r.identity(c['config'],dict(c['feature_spec'],smooth=5)))

    def test_strict_and_public_compromise_are_distinct_without_post_failure_rescue(self):
        base=fake_baselines();registry=dict(records=[dict(id=x,selectable=True,complexity=1) for x in ('strict','balanced','fail')])
        rows=[candidate_row('strict'),candidate_row('balanced',year21=.16,tail=.67),candidate_row('fail',year21=.14,tail=.9)]
        out=r.select(rows,registry,base)
        self.assertEqual(out['primary'],'strict')
        self.assertIn('balanced',out['qualified_ids']['balanced'])
        self.assertNotIn('balanced',out['qualified_ids']['strict'])
        self.assertNotIn('fail',out['qualified_ids']['balanced'])
        none=r.select([rows[-1]],dict(records=[registry['records'][-1]]),base)
        self.assertIsNone(none['primary']);self.assertIsNone(none['balanced_reference'])

    def test_cost_delay_and_block_deterioration_cannot_be_rescued_by_2021_or_2026(self):
        base=fake_baselines();registry=dict(records=[dict(id='x',selectable=True,complexity=0)])
        row=candidate_row('x',year21=1.,tail=2.,cagr=.9)
        row['scenarios']['lag1_11bp']['blocks']['middle']['cagr']=.34
        out=r.select([row],registry,base)
        self.assertIsNone(out['balanced_reference'])
        self.assertFalse(out['checks']['x']['balanced']['lag1_11bp_middle'])
        row=candidate_row('x');row['scenarios']['lag1_11bp']['yearly']['2021']['total_return']=.01
        self.assertIsNone(r.select([row],registry,base)['balanced_reference'])

    def test_scores_trail_own_quotes_and_current_score_never_uses_prior_decision_volatility(self):
        arrays,meta,histories=fixture(310,missing=(280,))
        spec=dict(window=30,smooth=5,ma_window=180)
        out,m=build(arrays,meta,histories,[spec])
        g=r.generate();record=deepcopy(g['records'][0]);record['feature_spec']=spec
        record['config'].update(risk_context='prior60')
        before=out['scores'].copy();v=view(out,m,record)
        np.testing.assert_array_equal(before,v['scores'])
        self.assertEqual(out['scores'][-1,280,AI[A]],-1e100)
        changed=deepcopy(histories)
        for code,rows in changed.items():changed[code]=[x if x[0]<=meta['dates'][295] else (x[0],x[1],x[2]*9,x[3]) for x in rows]
        future,unused=build(arrays,meta,changed,[spec])
        np.testing.assert_array_equal(out['scores'][:,:296],future['scores'][:,:296])

    def test_synthetic_registered_mechanisms_match_independent_reference(self):
        arrays,meta,histories=fixture(310)
        specs=[dict(window=w,smooth=s,ma_window=180) for w,s in ((20,5),(25,4),(30,3))]
        expanded,m=build(arrays,meta,histories,specs)
        c4=r.controls()['c4']['config']
        with TemporaryDirectory() as directory:
            for spec,context in zip(specs,('current20','prior20','prior_max20_60')):
                record=dict(config=dict(c4,risk_context=context),feature_spec=spec)
                v=view(expanded,m,record);engine=Simulator(v,m,cache_dir=directory)
                for lag in (0,1):
                    config=effective_config(record,lag)
                    native=engine.run([config],start=m['dates'][270],end=m['dates'][-1],workers=1)
                    reference=run_reference(v,m,config,m['dates'][270],m['dates'][-1])
                    np.testing.assert_array_equal(native['returns'][0],reference['returns'])
                    np.testing.assert_array_equal(native['holdings'][0],reference['holdings'])


if __name__=='__main__':unittest.main()
