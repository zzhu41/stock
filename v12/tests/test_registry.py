"""Bounded search definitions, control identities, and lookup receipts."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from v10_deep.schema import baseline
from v12 import registry


def anchors():
    b=baseline();h=dict(b,score='wls20_smooth3',ma='ma180',panic_mode='volatility',panic=1.5)
    return dict(h=h,v92=b,simple=dict(b,min_hold=2),v9=dict(b,crash_mask=1,global_buffer=.02,gold_buffer=.02),v91=dict(b,crash_mask=1))


class RegistryTests(unittest.TestCase):
    def generate(self):
        source=anchors();original=deepcopy(source)
        with patch.object(registry,'profiles',return_value=source),patch.object(registry,'prior_signatures',return_value=({},{})):
            record=registry.generate()
        self.assertEqual(source,original)
        return record

    def test_fixed_counts_controls_and_no_outcome_selection(self):
        r=self.generate();self.assertEqual(r['raw_proposals'],442);self.assertEqual(r['unique_count'],439)
        self.assertEqual(len({c['id'] for c in r['candidates']}),439)
        records={c['id']:c for c in r['candidates']}
        self.assertEqual(sum(c['selectable'] for c in records.values()),421)
        for cid in list(r['controls'].values())+list(r['pool_controls'].values())+list(r['allocation_controls'].values()):
            self.assertFalse(records[cid]['selectable'])
        self.assertFalse(r['new_performance_read'])
        self.assertEqual(records[r['controls']['simple']]['config']['min_hold'],2)
        self.assertEqual(records[r['controls']['h']]['config']['score'],'wls20_smooth3')

    def test_each_new_risk_switch_has_same_pool_same_parameters_off_ablation(self):
        r=self.generate();singles={c['id']:c['config'] for c in r['candidates'] if c['kind']=='single'}
        for c in singles.values():
            if c['locked_panic_exit']:
                old=dict(c,locked_panic_exit=0)
                self.assertIn('v12_'+registry.identifier(old)[:20],singles)

    def test_allocation_anchors_precede_search_and_no_vol_progressive_cross(self):
        r=self.generate()
        for c in r['candidates']:
            if c['kind']!='allocation':continue
            spec=c['allocation']
            self.assertIn(spec['base_id'],[r['controls']['h'],r['controls']['simple']])
            self.assertFalse('target_vol' in spec and 'speed' in spec)
            if c['selectable']:self.assertIn(spec['mode'],('vol_target','progressive'))


if __name__=='__main__':unittest.main()
