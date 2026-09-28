"""Four real adapters prepare main R2 and three comparison accounts.

All September 29/30 quotes are synthetic. Network, real state paths and actual
message sending are never used. A successful collect is still ONLY preparation;
the parent daily journal owns promotion into production account projections.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import daily_extras
import premium
import shadow_0906
import shadow_v92
import strategy
from v10_live import data, policy, runtime
from v92_plus_live import policy as plus_policy, runtime as plus_runtime
from v12_live import policy as r2_policy, runtime as r2_runtime

ROOT = Path(__file__).resolve().parent.parent
DAY, NEXT_DAY = '2026-09-29', '2026-09-30'
NOW = datetime(2026, 9, 29, 14, 50, 30)
ACCOUNT_NAMES = ('shadow_v12_r2.json', 'shadow_v92.json', 'shadow_v92_plus.json', 'shadow_v10.json')
REAL_COLLECT_INPUTS = daily_extras._collect_inputs


def strategy_parameters():
    # _state_bull/BULL_HYST_PENDING are the old strategy's legitimate transient
    # regime state. Parameter leakage means changing its configured rule values.
    return {name:deepcopy(value) for name,value in vars(strategy).items()
            if name.isupper() and name != 'BULL_HYST_PENDING'}


class DailyExtrasIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('urllib.request.urlopen', side_effect=AssertionError('Network forbidden')):
            seed=data._load_seed()
            pairs,quotes={},{}
            for code in data.CODES:
                with (data.SEED_DIR/'snapshots'/(code+'.csv')).open(newline='') as stream:
                    history={r[0]:(r[0],float(r[1]),float(r[2]),float(r[3])) for r in csv.reader(stream) if r}
                raw=seed['raw'][code][-22:]
                price,volume=float(raw[-1][2]),float(raw[-1][3])*.8
                row=(DAY,price,price,volume)
                pairs[code]=dict(raw=raw+[row],qfq=[history[r[0]] for r in raw]+[row],
                                 sources={'raw':'synthetic fixture','qfq':'same-vintage fixture'})
                quotes[code]=dict(date=DAY,timestamp=DAY+' 14:50:00',price=price,open=price,
                                  volume=volume,prev_close=price)
            normalized=data._quotes(quotes,DAY,NOW)
            cls.view,unused=data._build(seed,data._empty_state(seed),pairs,normalized,DAY,None)
            cls.quotes=quotes
            cls.qvix_bytes=(data.SEED_DIR/'snapshots/qvix50.csv').read_bytes()

    @contextmanager
    def isolated(self):
        with tempfile.TemporaryDirectory(prefix='three_strategy_stage_') as directory, ExitStack() as stack:
            root=Path(directory);stage=root/'stage';stage.mkdir();production=root/'production';production.mkdir()
            markers={}
            for name in ACCOUNT_NAMES:
                path=production/name;path.write_text(json.dumps({'production_marker':name}))
                markers[path]=path.read_bytes()
            legacy_csv=production/'shadow_v92_trades.csv';legacy_csv.write_text('production CSV marker\n')
            markers[legacy_csv]=legacy_csv.read_bytes()
            stack.enter_context(patch.object(shadow_v92,'STATE_FILE',str(production/'shadow_v92.json')))
            stack.enter_context(patch.object(shadow_v92,'TRADES_FILE',str(legacy_csv)))
            stack.enter_context(patch.object(shadow_v92,'PORTFOLIO',str(production/'unused_actual_portfolio.json')))
            stack.enter_context(patch.object(runtime,'STATE',production/'shadow_v10.json'))
            stack.enter_context(patch.object(plus_runtime,'STATE',production/'shadow_v92_plus.json'))
            stack.enter_context(patch.object(r2_runtime,'STATE',production/'shadow_v12_r2.json'))
            qvix_path=root/'qvix50.csv';qvix_path.write_bytes(self.qvix_bytes)
            stack.enter_context(patch.object(shadow_0906,'QVIX_CSV',str(qvix_path)))
            stack.enter_context(patch.object(policy,'QVIX_PATH',qvix_path))
            if hasattr(plus_policy,'QVIX_PATH'):
                stack.enter_context(patch.object(plus_policy,'QVIX_PATH',qvix_path))
            stack.enter_context(patch('urllib.request.urlopen',side_effect=AssertionError('Network/message forbidden')))
            removed=stack.enter_context(patch.object(shadow_0906,'block',side_effect=AssertionError('0906 account was removed from daily cards')))
            view_builder=stack.enter_context(patch.object(data,'build_live_view',return_value=self.view))
            qvix=stack.enter_context(patch.object(shadow_0906,'qvix_state',return_value=(None,None,'2026-09-24',False,'当日QVIX缺失，恐慌通道停用')))
            premium_fetch=stack.enter_context(patch.object(premium,'signal_block',return_value=['  513100 纳指ETF 溢价 +0.00%（离线样例）']))
            # Most adapter tests keep mocks in this process so call assertions
            # remain meaningful; a separate case exercises real JSON IPC.
            def local_inputs(jobs,stage,**kwargs):
                results={}
                for name,(function,args,options) in jobs.items():
                    try:results[name]=dict(ok=True,value=function(*args,**options))
                    except Exception as exc:
                        results[name]=dict(ok=False,error=type(exc).__name__,
                            diagnostic=daily_extras.exception_detail(exc,'input.'+name))
                return results
            stack.enter_context(patch.object(daily_extras,'_collect_inputs',side_effect=local_inputs))
            v92=stack.enter_context(patch.object(shadow_v92,'block',wraps=shadow_v92.block))
            real_h_run,real_plus_run,real_r2_run=runtime.run,plus_runtime.run,r2_runtime.run
            def guarded_h(*args,**kwargs):
                self.assertEqual(kwargs.get('state_path'),stage/'shadow_v10.json')
                return real_h_run(*args,**kwargs)
            def guarded_plus(*args,**kwargs):
                self.assertEqual(kwargs.get('state_path'),stage/'shadow_v92_plus.json')
                return real_plus_run(*args,**kwargs)
            def guarded_r2(*args,**kwargs):
                self.assertEqual(kwargs.get('state_path'),stage/'shadow_v12_r2.json')
                return real_r2_run(*args,**kwargs)
            h_run=stack.enter_context(patch.object(runtime,'run',side_effect=guarded_h))
            plus_run=stack.enter_context(patch.object(plus_runtime,'run',side_effect=guarded_plus))
            r2_run=stack.enter_context(patch.object(r2_runtime,'run',side_effect=guarded_r2))
            h_decide=stack.enter_context(patch.object(policy,'decide',wraps=policy.decide))
            plus_decide=stack.enter_context(patch.object(plus_policy,'decide',wraps=plus_policy.decide))
            r2_decide=stack.enter_context(patch.object(r2_policy,'decide',wraps=r2_policy.decide))
            payload=dict(quotes=deepcopy(self.quotes),date=DAY,state_dir=str(stage),now=NOW.isoformat())
            yield dict(root=root,stage=stage,production=production,markers=markers,payload=payload,
                       removed=removed,view_builder=view_builder,qvix=qvix,premium=premium_fetch,
                       v92=v92,h_run=h_run,plus_run=plus_run,r2_run=r2_run,real_r2_run=real_r2_run,
                       h_decide=h_decide,plus_decide=plus_decide,r2_decide=r2_decide,
                       parameters=strategy_parameters())

    def assert_production_unchanged(self,context):
        for path,content in context['markers'].items():self.assertEqual(path.read_bytes(),content)
        self.assertFalse((context['production']/'daily_state.json').exists())
        self.assertEqual(strategy_parameters(),context['parameters'])
        context['removed'].assert_not_called()

    def assert_first_preparations(self,context,result):
        self.assertEqual(result['successful_accounts'],list(ACCOUNT_NAMES))
        lines='\n'.join(result['lines'])
        for label in ('【影子 V12-R2】','【影子 v9.2】','【影子 V9.2+】','【影子 V10-H】'):self.assertIn(label,lines)
        self.assertNotIn('【影子 v9.1-0906】',lines)
        self.assertNotIn('计算失败',lines)
        self.assertEqual(lines.count('虚拟净值 1.0000'),4)
        self.assertIn('离线样例',lines)
        context['view_builder'].assert_called_once()
        self.assertEqual(context['view_builder'].call_args.args[:2],(context['payload']['quotes'],DAY))
        context['qvix'].assert_called_once_with(DAY)
        context['premium'].assert_called_once_with(quotes=context['payload']['quotes'])
        for key,name in (('r2_run','shadow_v12_r2.json'),('h_run','shadow_v10.json'),('plus_run','shadow_v92_plus.json')):
            context[key].assert_called_once()
            self.assertEqual(context[key].call_args.kwargs['state_path'],context['stage']/name)
        v92_call=context['v92'].call_args
        self.assertIs(v92_call.kwargs['action_view'],self.view)
        expected={c:[(r[0],r[2],r[2],r[5]) for r in rows] for c,rows in self.view['histories'].items()}
        self.assertEqual(v92_call.args[1],expected)
        for name in ACCOUNT_NAMES:
            state=json.loads((context['stage']/name).read_text())
            self.assertEqual(state['nav'],1.)
            self.assertEqual(state['last_date'],DAY)
            self.assertEqual(state['start_date'],DAY)
            self.assertEqual(len(state['events']),1)
            self.assertAlmostEqual(state['units'],1/context['payload']['quotes'][state['holding']]['price'])
        h=json.loads((context['stage']/'shadow_v10.json').read_text())
        plus=json.loads((context['stage']/'shadow_v92_plus.json').read_text())
        r2=json.loads((context['stage']/'shadow_v12_r2.json').read_text())
        self.assertEqual(r2['candidate_id'],r2_policy.CANDIDATE_ID)
        self.assertNotEqual(r2['candidate_id'],h['candidate_id'])
        self.assertEqual(h['candidate_id'],policy.CANDIDATE_ID)
        self.assertEqual(plus['candidate_id'],plus_policy.CANDIDATE_ID)
        self.assertNotEqual(plus['candidate_id'],h['candidate_id'])
        self.assert_production_unchanged(context)

    def test_real_three_adapters_prepare_shared_inputs_without_touching_production(self):
        view_before=deepcopy(self.view)
        with self.isolated() as context:
            payload_before=deepcopy(context['payload'])
            result=daily_extras.collect(context['payload'])
            self.assert_first_preparations(context,result)
            saved={path:path.read_bytes() for path in context['stage'].glob('*.json')}
            csv_saved={path:path.read_bytes() for path in context['stage'].glob('*.csv')}
            with patch.object(strategy,'decide',side_effect=AssertionError('Duplicate v9.2 decision')), \
                    patch.object(policy,'decide',side_effect=AssertionError('Duplicate H decision')), \
                    patch.object(plus_policy,'decide',side_effect=AssertionError('Duplicate plus decision')), \
                    patch.object(r2_policy,'decide',side_effect=AssertionError('Duplicate main decision')):
                repeated=daily_extras.collect(context['payload'])
            self.assertEqual(repeated,result)
            for path,content in saved.items():self.assertEqual(path.read_bytes(),content)
            for path,content in csv_saved.items():self.assertEqual(path.read_bytes(),content)
            self.assertEqual(context['view_builder'].call_count,2)
            self.assertEqual(context['payload'],payload_before)
            self.assert_production_unchanged(context)
        self.assertEqual(self.view,view_before)

    def test_new_day_bad_shared_view_yields_three_failures_and_no_account_advancement(self):
        with self.isolated() as context:
            first=daily_extras.collect(context['payload']);self.assert_first_preparations(context,first)
            saved={path:path.read_bytes() for path in context['stage'].glob('*.json')}
            payload=deepcopy(context['payload']);payload.update(date=NEXT_DAY,now=NEXT_DAY+'T14:50:30')
            for quote in payload['quotes'].values():quote.update(date=NEXT_DAY,timestamp=NEXT_DAY+' 14:50:00')
            context['view_builder'].side_effect=ValueError('unverified corporate action fixture')
            result=daily_extras.collect(payload)
            self.assertEqual(result['successful_accounts'],[])
            first=result['diagnostics'][0]
            self.assertEqual(first['phase'],'input.view')
            self.assertIn('unverified corporate action fixture',first['message'])
            text='\n'.join(result['lines'])
            self.assertEqual(text.count('本次无有效建议'),4)
            self.assertNotIn('影子持仓:',text)
            for path,content in saved.items():self.assertEqual(path.read_bytes(),content)
            self.assert_production_unchanged(context)

    def test_individual_policy_failure_is_visible_and_not_in_successful_account_list(self):
        with self.isolated() as context:
            context['plus_decide'].side_effect=ValueError('synthetic plus failure')
            result=daily_extras.collect(context['payload'])
            self.assertEqual(result['successful_accounts'],['shadow_v12_r2.json','shadow_v92.json','shadow_v10.json'])
            self.assertFalse((context['stage']/'shadow_v92_plus.json').exists())
            self.assertIn('【影子 V9.2+】','\n'.join(result['lines']))
            self.assertIn('synthetic plus failure','\n'.join(result['lines']))
            self.assertTrue(any(d['phase']=='strategy.V9.2+' and 'synthetic plus failure' in d['message']
                                for d in result['diagnostics']))
            self.assert_production_unchanged(context)

    def test_post_compute_guard_excludes_a_prepared_file_after_deadline(self):
        with self.isolated() as context:
            moment=[NOW]
            real_block=context['real_r2_run']
            def crosses_deadline(*args,**kwargs):
                result=real_block(*args,**kwargs)
                moment[0]=NOW.replace(minute=55)
                return result
            context['r2_run'].side_effect=crosses_deadline
            with patch.object(runtime,'runtime_clock',return_value=lambda:moment[0]):
                result=daily_extras.collect(context['payload'])
            # The main adapter wrote only its staging copy; a failed final
            # guard must exclude that file from the parent's commit proposal.
            self.assertTrue((context['stage']/'shadow_v12_r2.json').exists())
            self.assertEqual(result['successful_accounts'],[])
            self.assertEqual('\n'.join(result['lines']).count('本次无有效建议'),4)
            self.assertNotIn('影子持仓:','\n'.join(result['lines']))
            self.assert_production_unchanged(context)

    def test_worker_rejects_a_nonexistent_stage_before_fetching_data(self):
        with self.isolated() as context:
            context['payload']['state_dir']=str(context['root']/'missing-stage')
            with self.assertRaisesRegex(ValueError,'isolated staging'):
                daily_extras.collect(context['payload'])
            context['view_builder'].assert_not_called()
            self.assert_production_unchanged(context)

    def test_real_process_json_inputs_feed_all_three_adapters_without_live_writes(self):
        with self.isolated() as context, patch.object(daily_extras,'_collect_inputs',side_effect=REAL_COLLECT_INPUTS):
            result=daily_extras.collect(context['payload'])
            self.assertEqual(result['successful_accounts'],list(ACCOUNT_NAMES))
            self.assertEqual('\n'.join(result['lines']).count('虚拟净值 1.0000'),4)
            for name in ACCOUNT_NAMES:
                state=json.loads((context['stage']/name).read_text())
                self.assertEqual(state['last_date'],DAY)
                self.assertEqual(state['nav'],1.)
            self.assertFalse(list(context['stage'].glob('.network-inputs-*')))
            self.assert_production_unchanged(context)


if __name__=='__main__':unittest.main()
