"""B family construction and per-score eligibility; no market experiments."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from v12 import registry,consensus
from v12.tests.test_registry import anchors


class ConsensusRegistryTests(unittest.TestCase):
    def test_fixed_288_family_and_exact_old_control_roles(self):
        with patch.object(registry,'profiles',return_value=anchors()),patch.object(registry,'prior_signatures',return_value=({},{})):
            original=registry.generate()
        mapping={'equal_score':'v12_b_equal_score','equal_rank':'v12_b_equal_rank','wls25_smooth3':'v12_b_wls25_smooth3'}
        before=deepcopy(original)
        with patch.object(consensus,'sha',return_value='0'*64):r=consensus.generate(original,mapping)
        self.assertEqual(original,before)
        self.assertEqual(r['unique_count'],293)
        self.assertEqual(sum(c['selectable'] for c in r['candidates']),288)
        old={c['id']:c for c in original['candidates']};new={c['id']:c for c in r['candidates']}
        for cid in r['controls'].values():self.assertEqual(new[cid],old[cid])
        for c in r['candidates']:
            if c['selectable']:
                self.assertEqual(c['config']['crash_mask'],5)
                self.assertIn(c['config']['score'],mapping.values())
                self.assertIn(c['config']['locked_panic_exit'],(0,1))

    def test_group_masks_only_selected_lane_and_preserves_control_view(self):
        arrays={'features':np.ones((2,2,1)), 'scores':np.ones((3,2,2))}
        arrays['scores'][1,0,0]=-1e100;arrays['scores'][2,1,1]=np.nan
        meta={'feature_names':['valid'],'score_names':['wls25_v20','v12_b_a','v12_b_b']}
        records=[{'id':'b','config':{'score':'v12_b_b'}},{'id':'h','config':{'score':'wls25_v20'}},
                 {'id':'a','config':{'score':'v12_b_a'}}]
        captured={};before=arrays['features'].copy()
        def run(group,view,metadata,**kwargs):
            c=group[0];captured[c['id']]=view['features'].copy()
            scalar={'b':0.,'h':1.,'a':2.}[c['id']]
            values={name:dict(returns=np.full((1,2),scalar),holdings=np.full((1,2),int(scalar),dtype=np.int32),
                             summary=np.full((1,10),scalar),turnover_by_day=np.zeros((1,2)),
                             turnover_equivalent=np.zeros(1),metadata={c['id']:{}}) for name,_,_ in consensus.SCENARIOS}
            return values,['2020-01-02','2020-01-03']
        with patch.object(consensus,'run_candidates',side_effect=run):result,days=consensus.run_b(records,arrays,meta)
        np.testing.assert_array_equal(arrays['features'],before)
        np.testing.assert_array_equal(captured['h'],before)
        self.assertEqual(captured['a'][0,0,0],0)
        self.assertEqual(captured['a'][1,1,0],1)
        self.assertEqual(captured['b'][0,0,0],1)
        self.assertEqual(captured['b'][1,1,0],0)
        for values in result.values():np.testing.assert_array_equal(values['returns'][:,0],[0.,1.,2.])


if __name__=='__main__':unittest.main()
