"""Actual saved-path smoke tests; no searches, simulations, network or writes."""
import json
import unittest
from bisect import bisect_right
from pathlib import Path

import numpy as np

from v12_r2.explore import evaluate,FIXED_IDS,FROZEN_RELEASE_SHA256
from v12_r2.data import BASE,sha
from v12_r2.release import verify


class ExploreTests(unittest.TestCase):
    def test_recovery_alias_reads_the_fixed_unqualified_full_path(self):
        result=evaluate('recovery_2021')
        self.assertEqual(result['candidate_id'],FIXED_IDS['main_three_gates_1'])
        self.assertEqual(result['source_role'],'main_three_gates_1')
        self.assertAlmostEqual(result['full']['cagr'],.5504471356427667,places=13)
        self.assertAlmostEqual(result['full']['max_dd'],-.2016446047948911,places=13)
        self.assertAlmostEqual(result['yearly']['2021']['total_return'],.18698299493558434,places=13)
        self.assertAlmostEqual(result['yearly']['2026']['total_return'],.46817975194027795,places=13)
        self.assertEqual(result['feature_spec'],dict(window=20,smooth=5,ma_window=180))
        self.assertEqual(result['effective_config']['panic'],1.4)
        self.assertFalse(result['strict_qualified']);self.assertFalse(result['balanced_qualified'])
        self.assertIsNone(result['official_primary']);self.assertFalse(result['deployed'])

    def test_2021_cutoff_and_delayed_high_fee_slice_preserve_continuous_returns(self):
        result=evaluate('recovery_2021',end='2021-12-31',lag=1,fee_bp=11)
        self.assertEqual(result['effective_config']['lag'],1)
        self.assertEqual(result['scenario'],'lag1_11bp')
        self.assertEqual(result['period'],['2014-01-02','2021-12-31'])
        self.assertNotIn('2022',result['yearly'])
        self.assertAlmostEqual(result['yearly']['2021']['total_return'],.07932399567885762,places=13)
        metadata=json.loads((BASE/'results/path_metadata.json').read_text())
        index=metadata['ids'].index(FIXED_IDS['main_three_gates_1'])
        stop=bisect_right(metadata['dates'],'2021-12-31')
        with np.load(BASE/'results/paths.npz',allow_pickle=False) as stored:
            expected=float(np.prod(1+stored['lag1_11bp__returns'][index,:stop]))
        self.assertEqual(result['full']['nav'],expected)
        self.assertFalse(result['balanced_qualified'])

    def test_only_saved_explanatory_ids_are_allowed_and_original_release_stays_valid(self):
        before=sha(BASE/'release_receipt.json')
        for role in ('highest_cagr','highest_2021'):
            self.assertEqual(evaluate(role)['candidate_id'],FIXED_IDS[role])
        for role in ('primary','balanced',FIXED_IDS['main_three_gates_1'],'another_winner'):
            with self.assertRaisesRegex(ValueError,'explicit fixed explanatory'):
                evaluate(role)
        with self.assertRaisesRegex(ValueError,'Only lag0'):
            evaluate('recovery_2021',fee_bp=5)
        with self.assertRaisesRegex(ValueError,'observed date'):
            evaluate('recovery_2021',end='2026-09-28')
        self.assertEqual(before,FROZEN_RELEASE_SHA256)
        self.assertEqual(sha(BASE/'release_receipt.json'),before)
        self.assertTrue(verify()['passed'])


if __name__=='__main__':unittest.main()
