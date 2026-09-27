"""C registry and risk-context dispatch, with no new market performance."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from v12 import registry,consensus,exante
from v12.tests.test_registry import anchors


class ExanteTests(unittest.TestCase):
    def test_fixed_1152_candidates_plus_controls_and_actual_context_identity(self):
        with patch.object(registry,'profiles',return_value=anchors()),patch.object(registry,'prior_signatures',return_value=({},{})):
            a=registry.generate()
        mapping={'equal_score':'v12_b_equal_score','equal_rank':'v12_b_equal_rank',
                 'wls25_smooth3':'v12_b_wls25_smooth3','multi_10_20_40':'v12_c_wls_10_20_40_mean_s1'}
        with patch.object(consensus,'sha',return_value='0'*64):b=consensus.generate(a,{k:v for k,v in mapping.items() if k!='multi_10_20_40'})
        before=deepcopy((a,b));c=exante.generate(a,b,mapping)
        self.assertEqual((a,b),before)
        self.assertEqual(c['unique_count'],1157)
        self.assertEqual(sum(row['selectable'] for row in c['candidates']),1152)
        self.assertEqual(len(c['prior_stage_duplicate_ids']),77)
        records={r['id']:r for r in c['candidates']};old={r['id']:r for r in a['candidates']}
        for cid in c['controls'].values():self.assertEqual(records[cid],old[cid])
        for r in c['candidates']:
            if not r['selectable']:continue
            config=r['config'];ctx=config.get('risk_context','current20')
            self.assertEqual(r['id'],'v12_'+exante.identifier(config)[:20])
            self.assertIn(ctx,('current20','prior20','prior60','prior_max20_60'))
            if ctx=='current20':self.assertNotIn('risk_context',config)
            self.assertEqual(config['crash_mask'],5)
            self.assertEqual(config['panic_mode'],'volatility')

    def test_dispatch_crops_before_risk_view_and_preserves_registry_order(self):
        arrays=dict(features=np.zeros((4,2,1)),scores=np.zeros((1,4,2)),orders=np.zeros((1,4,2),dtype=np.int32),fear=np.zeros(4,dtype=np.int32))
        arrays['features'][3:]=np.inf
        meta=dict(dates=['2014-01-02','2015-01-05','2017-12-29','2026-09-24'],shape=[4,2,1])
        records=[dict(id='late',config={'score':'x','risk_context':'prior60'}),dict(id='current',config={'score':'x'})]
        seen=[]
        def view(a,m,c):
            self.assertEqual(len(m['dates']),3);self.assertTrue(np.isfinite(a['features']).all())
            seen.append(c.get('risk_context','current20'));return a
        def run(rs,a,m,**kwargs):
            value=1. if rs[0]['id']=='current' else 0.
            out={s:dict(returns=np.full((1,3),value),holdings=np.zeros((1,3),dtype=np.int32),summary=np.zeros((1,10)),
                 turnover_by_day=np.zeros((1,3)),turnover_equivalent=np.zeros(1),metadata={rs[0]['id']:{}}) for s,_,_ in exante.SCENARIOS}
            return out,m['dates']
        with patch.object(exante,'view_for_config',side_effect=view),patch.object(exante,'run_candidates',side_effect=run):
            result,dates=exante.run_c(records,arrays,meta,end='2017-12-31')
        self.assertEqual(seen,['prior60','current20'])
        for data in result.values():np.testing.assert_array_equal(data['returns'][:,0],[0.,1.])
        self.assertEqual(len(meta['dates']),4)
        self.assertTrue(np.isinf(arrays['features'][3:]).all())


if __name__=='__main__':unittest.main()
