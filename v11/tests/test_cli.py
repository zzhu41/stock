"""Temporary synthetic profiles only; never depend on or create real profiles."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

from v11 import cli
from v10_deep.schema import baseline, identifier


class FrozenCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='v11-cli-test-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.base=self.root/'v11';self.base.mkdir()
        changes=patch.multiple(cli,BASE=self.base,ROOT=self.root);changes.start();self.addCleanup(changes.stop)
        self.write('frozen.py',{'source':'fixed'});(self.root/'dependency.txt').write_text('fixed dependency')
        self.configs={}
        for variant,hold in (('a',3),('b',4),('h',0)):
            c=baseline();c.update(min_hold=hold,risk_context='current20');digest=identifier(c)
            self.configs[variant]=dict(c,id='v11_'+digest[:20],hash=digest,families=['synthetic'],parents=[],stage='test')
        controls={'h':self.configs['h']['id']}
        self.write('registered_candidates.json',dict(candidates=[self.configs['a'],self.configs['h']],controls=controls))
        self.write('results/development/selection.json',dict(primary=self.configs['a']['id']))
        self.write('results/combinations/registered_candidates.json',dict(candidates=[self.configs['b'],self.configs['h']],controls=controls))
        self.write('results/combinations/selection.json',dict(primary=self.configs['b']['id']))
        report=dict(roles={name:self.configs[variant]['id'] for variant,name in
                          (('a','v11_a'),('b','v11_b'),('h','v10_h'))},
                    confirmation_verdicts={name:dict(passed=False) for name in ('v11_a','v11_b')})
        self.write('results/report_2026/evaluation.json',report)
        artifacts=('registered_candidates.json','results/development/selection.json',
            'results/combinations/registered_candidates.json','results/combinations/selection.json',
            'results/report_2026/evaluation.json')
        self.profiles=dict(source_sha256={'frozen.py':cli.sha(self.base/'frozen.py')},
            dependency_sha256={'dependency.txt':cli.sha(self.root/'dependency.txt')},
            artifact_sha256={name:cli.sha(self.base/name) for name in artifacts},
            score_specs=[],feature_fingerprints={'synthetic':'fixed'},feature_array_sha256='f'*64,variants={})
        for variant in ('a','b','h'):
            self.profiles['variants'][variant]=dict(config=self.configs[variant],name=variant,
                confirmation_verdict=None if variant=='h' else dict(passed=False),
                status='existing_control' if variant=='h' else 'research_only_failed_confirmation')
        self.write('profiles.json',self.profiles)
        self.meta=dict(dates=[cli.START,cli.END],assets=['511880','510300'],
                       fingerprints=deepcopy(self.profiles['feature_fingerprints']),cache_array_sha256='f'*64)

    def write(self,name,value):
        path=self.base/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))

    def test_valid_frozen_roles_load_and_return_detached_config(self):
        for variant in ('a','b','h'):
            profile,unused=cli.load_profile(variant)
            self.assertEqual(profile['config'],self.configs[variant])
            profile['config']['min_hold']=999
            self.assertNotEqual(cli.load_profile(variant)[0]['config']['min_hold'],999)

    def test_source_dependency_and_artifact_tampering_refuse(self):
        for path in (self.base/'frozen.py',self.root/'dependency.txt',self.base/'results/development/selection.json'):
            original=path.read_bytes();path.write_bytes(b'changed')
            with self.subTest(path=path.name),self.assertRaisesRegex(ValueError,'evidence changed'):
                cli.load_profile('b')
            path.write_bytes(original)

    def test_a_valid_different_config_cannot_be_relabelled_as_b(self):
        self.profiles['variants']['b']['config']=deepcopy(self.configs['h'])
        self.write('profiles.json',self.profiles)
        with self.assertRaisesRegex(ValueError,'selected role'):cli.load_profile('b')

    def test_id_and_status_cannot_drift_from_pinned_evidence(self):
        self.profiles['variants']['b']['config']['id']='v11_fake'
        self.write('profiles.json',self.profiles)
        with self.assertRaisesRegex(ValueError,'definition changed'):cli.load_profile('b')
        self.profiles['variants']['b']['config']['id']='v11_'+self.profiles['variants']['b']['config']['hash'][:20]
        self.profiles['variants']['b']['status']='research_only_confirmation_passed_not_clean_oos'
        self.write('profiles.json',self.profiles)
        with self.assertRaisesRegex(ValueError,'status differs'):cli.load_profile('b')

    def test_missing_role_receipt_rejects_self_consistent_profile(self):
        del self.profiles['artifact_sha256']['results/combinations/selection.json']
        self.write('profiles.json',self.profiles)
        with self.assertRaisesRegex(ValueError,'receipts are missing'):cli.load_profile('b')

    def test_cache_fingerprint_and_frozen_bytes_are_separately_checked(self):
        for field,value,error in (('fingerprints',{'changed':True},'feature inputs'),
                                  ('cache_array_sha256','a'*64,'feature array bytes')):
            meta=deepcopy(self.meta);meta[field]=value
            with patch.object(cli,'build',return_value=({},meta)),patch.object(cli,'Simulator') as engine:
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,error):cli.evaluate('b')
                engine.assert_not_called()

    def test_invalid_clock_cost_or_date_does_not_execute_simulator(self):
        with patch.object(cli,'build',return_value=({},self.meta)),patch.object(cli,'Simulator') as engine:
            for kwargs in ({'lag':2},{'fee_bp':10},{'end':'2013-12-31'},
                           {'end':'2026-09-25'},{'end':'2014-01-03'}):
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):cli.evaluate('b',**kwargs)
            engine.assert_not_called()

    def test_registered_cost_lag_applied_without_mutating_frozen_config(self):
        model=Mock();model.run.return_value={'synthetic':'result'}
        with patch.object(cli,'build',return_value=({'input':'base'},self.meta)), \
                patch.object(cli,'risk_view',return_value={'input':'view'}) as view, \
                patch.object(cli,'Simulator',return_value=model):
            profile,result,meta=cli.evaluate('b',end=cli.END,lag=1,fee_bp=11)
        self.assertEqual(result,{'synthetic':'result'})
        args,kwargs=model.run.call_args
        self.assertEqual(args[0][0]['lag'],1)
        self.assertEqual(kwargs,dict(start=cli.START,end=cli.END,fee=.0011,workers=1))
        self.assertEqual(profile['config']['lag'],0)
        view.assert_called_once_with({'input':'base'},self.meta,'current20',score=self.configs['b']['score'])

    def test_signal_stdout_is_json_and_explicitly_historical_not_deployed(self):
        profile=self.profiles['variants']['b']
        result=dict(dates=[cli.END],holdings=np.asarray([[0]]),summary=np.zeros((1,10)))
        output=io.StringIO()
        with patch.object(cli,'evaluate',return_value=(profile,result,self.meta)), \
                patch('sys.argv',['v11.cli','signal','b','--lag','1','--fee-bp','11']),redirect_stdout(output):
            cli.main()
        value=json.loads(output.getvalue())
        self.assertEqual(value['historical_close_target'],'511880')
        self.assertEqual(value['status'],'research_only_failed_confirmation')
        self.assertFalse(value['deployed']);self.assertFalse(value['order_submission']);self.assertFalse(value['clean_oos'])
        self.assertIn('historical',value['warning'])


if __name__=='__main__':unittest.main()
