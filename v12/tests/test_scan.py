"""Synthetic multi-kind orchestration and immutable evaluation receipts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from v12 import scan
from v10_deep.tests.test_native import inputs, finish, prices as set_prices, config, AI, A, B, X, CASH, GOLD, BENCH


def single(cid, **changes):
    return dict(id=cid,kind='single',config=dict(config(**changes),locked_panic_exit=0),
                selectable=False,complexity=0,families=['synthetic_control'])


def allocation(cid, mode, base='h', **changes):
    return dict(id=cid,kind='allocation',allocation=dict(base_id=base,mode=mode,**changes),
                selectable=mode not in ('full_exposure','fixed_exposure'),complexity=1,families=['synthetic_allocation'])


def benchmark(cid, mode, assets):
    return dict(id=cid,kind='benchmark',benchmark=dict(mode=mode,assets=assets),
                selectable=False,complexity=0,families=['synthetic_benchmark'])


class CandidateRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='v12-scan-native-');cls.cache=Path(cls.temp.name)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def fixture(self):
        arrays,meta=inputs(8)
        set_prices(arrays,A,[100.,100.,100.,120.,np.nan,140.,150.,160.])
        arrays['scores'][:,6:,AI[B]]=80.
        return finish(arrays),meta

    def test_single_allocation_fixed_and_benchmark_rows_align_and_exclude_free_initial_turnover(self):
        arrays,meta=self.fixture();before={k:v.copy() for k,v in arrays.items()}
        records=[single('h'),single('simple',min_hold=2),allocation('full','full_exposure'),
                 allocation('gradual','progressive',base='simple',speed=.5),
                 allocation('vol','vol_target',target_vol=.2,risk_context='prior20'),
                 allocation('half','fixed_exposure',weight=.5),benchmark('buy','buy_hold',[X])]
        originals=deepcopy(records)
        result,days=scan.run_candidates(records,arrays,meta,start=meta['dates'][2],end=meta['dates'][-1],cache_dir=self.cache)
        self.assertEqual(days,meta['dates'][2:]);self.assertEqual(records,originals)
        for name,payload in result.items():
            self.assertEqual(payload['returns'].shape,(7,6));self.assertEqual(payload['summary'].shape,(7,10))
            self.assertTrue(np.all(payload['holdings'][2:]==-1))
            np.testing.assert_allclose(payload['returns'][0],payload['returns'][2],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(payload['turnover_by_day'][:,0],np.zeros(7))
            np.testing.assert_allclose(payload['turnover_equivalent'],payload['turnover_by_day'].sum(axis=1),rtol=0,atol=1e-14)
            self.assertTrue(payload['metadata']['full']['base_replay_fidelity']['compatible'])
            self.assertEqual(payload['metadata']['vol']['overlay']['target_alignment'],'execution_day')
        # Fixed 50% remains a REAL daily rebalance: A drifts to 6/11 after +20%.
        self.assertAlmostEqual(result['close_1bp']['turnover_by_day'][5,1],1/22)
        self.assertAlmostEqual(1+result['close_1bp']['returns'][5,1],1.09999)
        for name,value in arrays.items():np.testing.assert_array_equal(value,before[name])

    def test_missing_base_and_incompatible_start_fail_without_silently_dropping_trials(self):
        arrays,meta=self.fixture()
        with self.assertRaisesRegex(ValueError,'explicit single'):
            scan.run_candidates([allocation('a','full_exposure')],arrays,meta,start=meta['dates'][2],end=meta['dates'][-1],cache_dir=self.cache)
        with self.assertRaisesRegex(ValueError,'startup is incompatible'):
            scan.run_candidates([single('h'),allocation('a','full_exposure')],arrays,meta,
                start=meta['dates'][0],end=meta['dates'][-1],scenarios=(scan.SCENARIOS[2],),cache_dir=self.cache)

    def test_training_prefix_does_not_even_read_invalid_future_price_rows(self):
        arrays,meta=self.fixture();records=[single('h'),allocation('clone','full_exposure')]
        before,days=scan.run_candidates(records,arrays,meta,start=meta['dates'][2],end=meta['dates'][3],cache_dir=self.cache)
        changed={name:value.copy() for name,value in arrays.items()}
        changed['features'][4:,:,meta['feature_names'].index('close')]=np.inf
        after,again=scan.run_candidates(records,changed,meta,start=meta['dates'][2],end=meta['dates'][3],cache_dir=self.cache)
        self.assertEqual(days,again)
        for name in before:np.testing.assert_array_equal(before[name]['returns'],after[name]['returns'])

    def test_buy_hold_waits_for_maturity_then_keeps_missing_held_bars(self):
        arrays,meta=inputs(6);f={n:i for i,n in enumerate(meta['feature_names'])}
        set_prices(arrays,A,[100.,105.,110.,120.,np.nan,144.])
        set_prices(arrays,CASH,[100.,101.,102.,103.,104.,105.])
        arrays['features'][:3,AI[A],f['valid']]=0.
        arrays=finish(arrays)
        result,days=scan.run_candidates([benchmark('buy','buy_hold',[A])],arrays,meta,
            start=meta['dates'][0],end=meta['dates'][-1],scenarios=(('test',0,.001),))
        row=result['test'];self.assertAlmostEqual(row['summary'][0,0],1.03*.998*1.2)
        self.assertEqual(row['summary'][0,3],1.)
        self.assertEqual(row['summary'][0,6],1.)
        self.assertEqual(row['turnover_equivalent'][0],1.)
        self.assertEqual(row['metadata']['buy']['first_fill']['target'],{CASH:1.})
        self.assertEqual(row['returns'][0,4],0.)

    def test_monthly_equal_uses_signal_clock_and_preserves_drift_between_resets(self):
        arrays,meta=inputs(8)
        meta['dates']=['2020-01-02','2020-01-30','2020-01-31','2020-02-03',
                       '2020-02-04','2020-02-05','2020-03-02','2020-03-03']
        set_prices(arrays,A,[100.,100.,120.,120.,120.,120.,120.,120.])
        f={n:i for i,n in enumerate(meta['feature_names'])}
        arrays['features'][:4,AI[BENCH],f['valid']]=0.
        arrays=finish(arrays)
        record=benchmark('month','monthly_equal',[A,B,X,GOLD,BENCH])
        result,days=scan.run_candidates([record],arrays,meta,start=meta['dates'][1],end=meta['dates'][-1])
        for scenario in ('close_1bp','lag1_1bp'):
            target=result[scenario]['metadata']['month']['first_fill']['target']
            self.assertAlmostEqual(target[CASH],.2)
            self.assertNotIn(BENCH,target)
            self.assertEqual(result[scenario]['turnover_by_day'][0,1],0.)  # Jan31 drift is not reset.
        self.assertGreater(result['close_1bp']['turnover_by_day'][0,2],0.)  # Feb3 signal/fill.
        self.assertEqual(result['lag1_1bp']['turnover_by_day'][0,2],0.)
        self.assertGreater(result['lag1_1bp']['turnover_by_day'][0,3],0.)  # Feb3 signal -> Feb4 fill.
        self.assertGreater(result['close_1bp']['turnover_by_day'][0,5],0.)  # Newly mature slot enters March.
        self.assertGreater(result['lag1_1bp']['turnover_by_day'][0,6],0.)


class ContinuousSummaryTests(unittest.TestCase):
    def payload(self):
        dates=['2014-01-02','2017-12-29','2018-01-02','2021-12-31',
               '2022-01-04','2025-12-31','2026-01-05','2026-09-24']
        returns=np.asarray([[0.,.05,.10*.9978+.9978-1,.02,-.10,.03,.04,.01]])
        turnover=np.asarray([[0.,1.,.3,0.,1.,.2,1.,0.]])
        payload=dict(returns=returns,holdings=np.zeros((1,8),dtype=np.int32),summary=np.zeros((1,10)),
                     turnover_by_day=turnover,turnover_equivalent=turnover.sum(axis=1))
        return dates,payload

    def test_blocks_tail_and_yearly_include_boundary_return_and_actual_turnover(self):
        dates,payload=self.payload();records=[single('h')]
        row=scan.summary_rows(records,{'close_1bp':payload},dates)[0]['scenarios']['close_1bp']
        self.assertEqual(set(row['blocks']),{'early','middle','recent'})
        self.assertAlmostEqual(row['blocks']['middle']['total_return'],1.10*.9978*1.02-1)
        self.assertAlmostEqual(row['blocks']['middle']['turnover_equivalent'],.3)
        self.assertAlmostEqual(row['blocks']['recent']['max_dd'],-.1)
        self.assertAlmostEqual(row['tail']['total_return'],1.04*1.01-1)
        self.assertEqual(row['tail']['start'],'2026-01-05')
        self.assertEqual(row['tail']['turnover_equivalent'],1.)
        self.assertAlmostEqual(row['yearly']['2018']['total_return'],1.10*.9978-1)

    def test_partial_training_prefix_uses_its_own_tail_year_and_no_future_blocks(self):
        dates,payload=self.payload();n=3
        prefix={k:(v[:,:n] if k in ('returns','holdings','turnover_by_day') else v.copy()) for k,v in payload.items()}
        prefix['turnover_equivalent']=prefix['turnover_by_day'].sum(axis=1)
        row=scan.summary_rows([single('h')],{'close_1bp':prefix},dates[:n],period_end='2018-12-31')[0]
        self.assertEqual(row['period_end'],'2018-12-31')
        value=row['scenarios']['close_1bp'];self.assertEqual(set(value['blocks']),{'early','middle'})
        self.assertEqual(value['tail']['start'],'2018-01-02')
        self.assertEqual(set(value['yearly']),{'2014','2017','2018'})
        self.assertAlmostEqual(value['tail']['total_return'],1.10*.9978-1)


class EvaluationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='v12-evaluate-synthetic-');self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)/'v12';self.base.mkdir();self.output=self.base/'results/main'
        for name in ('registered_candidates.json','registration.json','PROTOCOL.md','protected_manifest.json',
                     'inputs/manifest.json','inputs/features.json','inputs/features.npz.gz','selection.py'):
            p=self.base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        patcher=patch.object(scan,'BASE',self.base);patcher.start();self.addCleanup(patcher.stop)
        self.records=[single('h')]
        self.registry=dict(unique_count=1,candidates=self.records,controls={'h':'h'})
        self.days=['2014-01-02','2018-01-02','2022-01-04','2026-09-24']
        self.payload={name:dict(returns=np.asarray([[0.,.01,.02,.03]]),holdings=np.zeros((1,4),dtype=np.int32),
            summary=np.zeros((1,10)),turnover_by_day=np.zeros((1,4)),turnover_equivalent=np.zeros(1),metadata={})
            for name,unused,fee in scan.SCENARIOS}
        self.choice=dict(primary=None,qualified_count=0,status='no_qualified_candidate',top_return=None,top_tail=None,least_drawdown=None)

    def exercise(self, mutate_during_run=False, mutate_during_select=False):
        def evaluate(*args,**kwargs):
            self.assertTrue((self.output/'registration.json').exists())
            self.assertFalse((self.output/'selection.json').exists())
            if mutate_during_run:(self.base/'selection.py').write_text('changed during evaluation')
            return self.payload,self.days
        def select(*args,**kwargs):
            self.assertTrue((self.output/'evaluation.json').exists())
            self.assertTrue((self.output/'paths.npz').exists())
            if mutate_during_select:(self.base/'selection.py').write_text('changed during selection')
            return deepcopy(self.choice)
        with patch('v12.registry.register',return_value=self.registry),patch('v12.selection.select',side_effect=select), \
             patch.object(scan,'protect',return_value=1),patch.object(scan,'load_inputs',return_value=({}, {}, {})), \
             patch.object(scan,'run_candidates',side_effect=evaluate):
            return scan.evaluate(output=self.output)

    def test_registration_precedes_paths_and_selection_seals_exact_output_hashes(self):
        result=self.exercise();proof=result['provenance']
        for name,key in (('registration.json','registration_sha256'),('evaluation.json','evaluation_sha256'),
                         ('paths.npz','paths_sha256'),('path_metadata.json','path_metadata_sha256')):
            self.assertEqual(scan.sha(self.output/name),proof[key])
        with np.load(self.output/'paths.npz',allow_pickle=False) as values:
            self.assertEqual(len(values.files),len(scan.SCENARIOS)*len(scan.NUMERIC_FIELDS))
        with self.assertRaisesRegex(ValueError,'Selection is frozen'):self.exercise()

    def test_source_drift_cannot_produce_frozen_selection(self):
        with self.assertRaisesRegex(ValueError,'changed during evaluation'):self.exercise(mutate_during_run=True)
        self.assertFalse((self.output/'selection.json').exists())

    def test_source_drift_while_selecting_cannot_produce_frozen_selection(self):
        with self.assertRaisesRegex(ValueError,'changed while selecting'):self.exercise(mutate_during_select=True)
        self.assertFalse((self.output/'selection.json').exists())


if __name__=='__main__':unittest.main()
