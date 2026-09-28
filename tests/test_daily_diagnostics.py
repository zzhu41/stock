"""Safe, useful failure evidence without sending messages or changing accounts."""
from datetime import datetime
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch, Mock

import daily_diagnostics as diagnostics
import daily_job
import signal_daily
import signal_store


class DailyDiagnosticsTests(unittest.TestCase):
    def test_secret_urls_and_headers_are_redacted_but_market_error_survives(self):
        raw = ('510300 当前日配对raw/qfq缺失 https://example.invalid/a?access_token=urlprivate '
               'Authorization: Bearer shortsecret token=baresecret password="quoted secret" '
               "{'sessionWebhook': 'https://example.invalid/private'} SEC1234567890")
        text = diagnostics.safe_message(raw)
        self.assertIn('510300 当前日配对raw/qfq缺失',text)
        for secret in ('urlprivate','shortsecret','baresecret','quoted secret','example.invalid','SEC1234567890'):
            self.assertNotIn(secret,text)
        self.assertLessEqual(len(diagnostics.safe_message('x'*9000)),500)

    def test_nested_causes_and_local_trace_survive_boundary(self):
        try:
            raise ValueError('510300 配对日K缺失')
        except Exception as exc:
            cause = diagnostics.exception_detail(exc,'input.view')
        exc = diagnostics.GenerationInputError(dict(phase='prepare_versions',error_type='PreparationError',
            message='三个版本均无有效建议',causes=[cause]))
        stream = io.StringIO();diagnostics.emit(exc,'signal_generation',stream)
        value = diagnostics.child_failure('',stream.getvalue(),'signal_daily.py',1)
        self.assertEqual(value['causes'][0]['error_type'],'ValueError')
        self.assertIn('510300',value['causes'][0]['message'])
        self.assertTrue(value['causes'][0]['frames'])
        self.assertNotIn('source_line',str(value))

    def test_unstructured_child_output_never_copies_response_or_secret(self):
        value = diagnostics.child_failure('private-stdout','ValueError: short-sensitive-data','worker',1)
        self.assertEqual(value['error_type'],'ValueError')
        self.assertNotIn('sensitive',str(value));self.assertNotIn('private-stdout',str(value))

    def test_actual_failed_subprocess_exposes_safe_diagnostic(self):
        code = ("import sys;from daily_diagnostics import emit\n"
                "try: raise ValueError('510300 配对数据缺失 token=privatevalue')\n"
                "except Exception as e: emit(e,'input.view',sys.stderr);sys.exit(2)\n")
        with self.assertRaises(diagnostics.DiagnosticError) as caught:
            daily_job.run_child([sys.executable,'-B','-c',code],5)
        self.assertEqual(caught.exception.diagnostic['error_type'],'ValueError')
        self.assertIn('510300',str(caught.exception));self.assertNotIn('privatevalue',str(caught.exception))

    def test_cron_records_cause_without_creating_signal_or_account(self):
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026,9,28,14,50)
            def generate():
                raise diagnostics.GenerationInputError(dict(phase='prepare_versions',error_type='PreparationError',
                    message='三个版本均无有效建议',causes=[dict(phase='input.view',error_type='LiveDataError',
                        message='510300 当前日配对缺失 secret=privatevalue')]))
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(daily_job.run_job(directory,now,generate,lambda:self.fail('must not send')),1)
            state = signal_store.read_json(Path(directory)/'generation_status.json')
            self.assertEqual(state['status'],'failed');self.assertEqual(state['reason'],'PreparationError')
            self.assertIn('510300',state['diagnostic']['causes'][0]['message'])
            self.assertNotIn('privatevalue',json.dumps(state)+output.getvalue())
            self.assertFalse((Path(directory)/'daily_state.json').exists())

    def test_malformed_worker_protocol_is_a_diagnostic_failure(self):
        result = Mock(returncode=1,stdout='not-json secret=privatevalue',stderr='KeyError: privatevalue')
        with patch.object(signal_daily.subprocess,'run',return_value=result):
            with self.assertRaises(diagnostics.DiagnosticError) as caught:
                signal_daily.prepare_versions({},'2026-09-28','/tmp')
        self.assertNotIn('privatevalue',str(caught.exception))
        self.assertEqual(caught.exception.diagnostic['error_type'],'KeyError')


if __name__=='__main__':unittest.main()
