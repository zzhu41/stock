"""Pure auth and isolated callback/query regressions; no running app or messages.

Integration tests apply the reviewed patch in a temporary directory, or use the
already deployed matching source. They compile selected function definitions
only, so importing the real bot cannot start schedulers, threads or network work.
"""
import ast
import base64
from copy import deepcopy
import hashlib
import hmac
import io
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

from integrations import dingtalk_guard as guard
from integrations import dingtalk_momentum_transport as transport

ROOT = Path(__file__).resolve().parent.parent
BOT = Path('/root/dingtalk-stock-bot')


def signed(timestamp, secret):
    message = (str(timestamp) + '\n' + secret).encode('utf-8')
    return base64.b64encode(hmac.new(secret.encode('utf-8'), message, hashlib.sha256).digest()).decode('ascii')


class CallbackGuardTests(unittest.TestCase):
    def test_configured_legacy_token_exact_match_and_case_insensitive_header_name(self):
        self.assertEqual(guard.verify_request({'Token': 'synthetic-token'}, token='synthetic-token'), (True, 'verified_token'))
        for value in (None, '', 'wrong', 'synthetic-token,wrong', b'synthetic-token'):
            self.assertFalse(guard.verify_request({'token': value}, token='synthetic-token')[0])
        self.assertFalse(guard.verify_token(None, None))
        self.assertFalse(guard.verify_token('', ''))
        self.assertFalse(guard.verify_token('any', ''))

    def test_no_config_rejects_instead_of_open_access(self):
        self.assertEqual(guard.verify_request({}, token=''), (False, 'auth_not_configured'))
        self.assertEqual(guard.verify_request({}, mode='sign', app_secret=''), (False, 'auth_not_configured'))
        self.assertEqual(guard.verify_request({}, mode='auto', token='x'), (False, 'invalid_auth_mode'))

    def test_enterprise_signature_uses_appsecret_timestamp_and_raw_base64(self):
        timestamp, secret = 1790000000000, 'synthetic-app-secret'
        headers = {'Timestamp': str(timestamp), 'Sign': signed(timestamp, secret)}
        self.assertEqual(guard.verify_request(headers, mode='sign', app_secret=secret, now_ms=timestamp), (True, 'verified_sign'))
        for key, value in (('Sign', 'x' * 44), ('Timestamp', str(timestamp // 1000)), ('Timestamp', '-1')):
            bad = dict(headers, **{key: value})
            self.assertFalse(guard.verify_request(bad, mode='sign', app_secret=secret, now_ms=timestamp)[0])
        self.assertFalse(guard.verify_request(headers, mode='sign', app_secret='different-app', now_ms=timestamp)[0])

    def test_expired_and_future_signatures_reject_and_boundary_matches_protocol(self):
        timestamp, secret = 1790000000000, 'synthetic-app-secret'
        headers = {'timestamp': str(timestamp), 'sign': signed(timestamp, secret)}
        for offset in (-3600001, 3600001):
            self.assertEqual(guard.verify_request(headers, mode='sign', app_secret=secret, now_ms=timestamp + offset), (False, 'expired_timestamp'))
        for offset in (-3600000, 3600000):
            self.assertTrue(guard.verify_request(headers, mode='sign', app_secret=secret, now_ms=timestamp + offset)[0])

    def test_mode_never_downgrades_to_other_valid_credential(self):
        timestamp = 1790000000000
        self.assertFalse(guard.verify_request({'token': 'known'}, mode='sign', token='known', app_secret='secret', now_ms=timestamp)[0])
        self.assertFalse(guard.verify_request({'timestamp': str(timestamp), 'sign': signed(timestamp, 'secret')}, token='known', app_secret='secret', now_ms=timestamp)[0])

    def test_payload_body_limit_json_shape_and_sensitive_fields_do_not_escape(self):
        valid = {'text': {'content': '动量'}, 'senderId': 'synthetic', 'sessionWebhook': 'https://example.invalid/private-session'}
        self.assertEqual(guard.parse_payload(json.dumps(valid).encode('utf-8')), (valid, None))
        for data in (b'[]', b'null', b'{', b'\xff', b'{"text":null}', b'{"text":{"content":3}}', b'{"senderId":[]}'):
            self.assertEqual(guard.parse_payload(data), (None, 400))
        self.assertEqual(guard.parse_payload(b'x' * (guard.MAX_CALLBACK_BYTES + 1)), (None, 413))
        self.assertEqual(guard.parse_payload(b'{"msgtype":"picture"}'), ({'msgtype': 'picture'}, None))


def selected_functions(source, names, namespace):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<isolated_bot_functions>', 'exec'), namespace)
    return namespace


@unittest.skipUnless((BOT / 'app.py').is_file(), 'External bot checkout not installed')
class CallbackIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        receipt = json.loads((ROOT / 'integrations/dingtalk_security_review.json').read_text())
        presence = json.loads((ROOT / 'integrations/dingtalk_auth_presence_review.json').read_text())
        reply_review = json.loads((ROOT / 'integrations/dingtalk_momentum_reply_review.json').read_text())
        originals = {name: (BOT / name).read_text() for name in ('app.py', 'dingtalk.py')}
        hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in originals.items()}
        if hashes == reply_review['source_after_sha256']:
            cls.sources = originals
        elif hashes == presence['source_after_sha256']:
            cls.sources = originals
        elif hashes == receipt['source_after_sha256']:
            cls.sources = originals
        elif hashes == receipt['source_before_sha256']:
            with tempfile.TemporaryDirectory(prefix='bot_patch_test_') as directory:
                for name, source in originals.items():
                    (Path(directory) / name).write_text(source)
                subprocess.run(['patch', '--batch', '-p1', '-d', directory, '-i', str(ROOT / 'integrations/dingtalk_security.patch')],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cls.sources = {name: (Path(directory) / name).read_text() for name in originals}
            cls_hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in cls.sources.items()}
            if cls_hashes != receipt['source_after_sha256']:
                raise AssertionError('Patch output differs from the reviewed source')
        else:
            raise AssertionError('External bot source changed since patch review')
        cls_hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in cls.sources.items()}
        if cls_hashes == presence['source_before_sha256']:
            patch_path = ROOT / 'integrations/dingtalk_auth_presence.patch'
            if hashlib.sha256(patch_path.read_bytes()).hexdigest() != presence['patch_sha256']:
                raise AssertionError('Diagnostic patch changed')
            with tempfile.TemporaryDirectory(prefix='bot_presence_test_') as directory:
                for name, source in cls.sources.items():
                    (Path(directory) / name).write_text(source)
                subprocess.run(['patch', '--batch', '-p1', '-d', directory, '-i', str(patch_path)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cls.sources = {name: (Path(directory) / name).read_text() for name in originals}
            if {name: hashlib.sha256(source.encode()).hexdigest() for name, source in cls.sources.items()} != presence['source_after_sha256']:
                raise AssertionError('Diagnostic patch result differs')
        cls_hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in cls.sources.items()}
        if cls_hashes == reply_review['source_before_sha256']:
            patch_path = ROOT / 'integrations/dingtalk_momentum_reply.patch'
            if hashlib.sha256(patch_path.read_bytes()).hexdigest() != reply_review['patch_sha256']:
                raise AssertionError('Session reply patch changed')
            with tempfile.TemporaryDirectory(prefix='bot_reply_test_') as directory:
                for name, source in cls.sources.items():
                    (Path(directory) / name).write_text(source)
                subprocess.run(['patch', '--batch', '-p1', '-d', directory, '-i', str(patch_path)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                cls.sources = {name: (Path(directory) / name).read_text() for name in originals}
            if {name: hashlib.sha256(source.encode()).hexdigest() for name, source in cls.sources.items()} != reply_review['source_after_sha256']:
                raise AssertionError('Session reply patch result differs')

    def setUp(self):
        self.logs = []
        class Logger:
            def _record(inner, message, *args):
                self.logs.append(message % args if args else message)
            info = warning = error = _record
        self.logger = Logger()
        self.config = types.SimpleNamespace(DINGTALK_OUTGOING_TOKEN='synthetic-token', load_stocks=lambda: {})
        self.ding = selected_functions(self.sources['dingtalk.py'],
            {'_callback_guard', 'verify_outgoing_token', 'verify_callback_request', 'parse_callback_body'}, {'config': self.config})
        self.module = types.SimpleNamespace(**{name: self.ding[name] for name in ('verify_callback_request', 'parse_callback_body')})
        self.module.parse_command = lambda text: ('momentum', [])
        self.outgoing = []
        def fake_post(url, **kwargs):
            self.outgoing.append((url,kwargs))
            return types.SimpleNamespace(status_code=200,json=lambda:dict(errcode=0))
        self.module.requests = types.SimpleNamespace(post=fake_post)
        reply_cache = transport.ReplyCache()
        reply_module = types.SimpleNamespace(reply_momentum=lambda *args,**kwargs:
            transport.reply_momentum(*args,cache=reply_cache,dispatch=lambda work:work(),**kwargs))
        self.called = []
        def command(*args):
            self.called.append(args)
            return 'saved test signal'
        self.namespace = dict(logger=self.logger, dingtalk=self.module, os=os,
                              _momentum_reply_transport=lambda:reply_module,
                              router=types.SimpleNamespace(safety_check=lambda text: (False, '')),
                              jsonify=lambda value: value, handle_command=command)
        selected_functions(self.sources['app.py'], {'webhook'}, self.namespace)
        self.auth_env = patch.dict(os.environ, {'DINGTALK_CALLBACK_AUTH_MODE': 'token'}, clear=False)
        self.auth_env.start()
        self.addCleanup(self.auth_env.stop)

    def request(self, payload=None, headers=None, raw=None):
        payload = {'text': {'content': '动量'}, 'senderId': 'test'} if payload is None else payload
        self.namespace['request'] = types.SimpleNamespace(headers={} if headers is None else headers,
            stream=io.BytesIO(json.dumps(payload).encode('utf-8') if raw is None else raw))
        return self.namespace['webhook']()

    def test_no_auth_cannot_reach_command_or_parse_body(self):
        result = self.request(raw=b'not json')
        self.assertEqual(result[1], 401)
        self.assertEqual(self.called, [])
        self.config.DINGTALK_OUTGOING_TOKEN = ''
        self.assertEqual(self.request(headers={'token': 'any'})[1], 503)

    def test_rejected_callback_logs_header_presence_but_no_values_or_body(self):
        result = self.request(headers={'token':'wrong-private-token','sign':'private-signature','timestamp':'private-timestamp'},
                              raw=b'private-body-not-json')
        self.assertEqual(result[1],401)
        self.assertFalse(self.called)
        logged='\n'.join(self.logs)
        for value in ('wrong-private-token','private-signature','private-timestamp','private-body-not-json'):
            self.assertNotIn(value,logged)
        for value in ('token_present=True','sign_present=True','timestamp_present=True'):
            self.assertIn(value,logged)
        self.assertEqual(self.called, [])

    def test_valid_token_routes_but_never_logs_content_sender_or_session_webhook(self):
        payload = dict(text={'content': 'private-marker'}, senderNick='private-name', senderId='private-id',
                       sessionWebhook='https://example.invalid/secret-session', conversationType='2')
        result = self.request(payload, {'token': 'synthetic-token'})
        self.assertEqual(result, {})  # Unsafe supplied session URL is never called.
        self.assertEqual(self.outgoing, [])
        self.assertEqual(len(self.called), 1)
        logs = '\n'.join(self.logs)
        for private in ('private-marker', 'private-name', 'private-id', 'secret-session', 'synthetic-token'):
            self.assertNotIn(private, logs)
        self.assertIn('verified_token', logs)

    def test_authenticated_momentum_posts_to_its_session_once_and_only_acks_http(self):
        payload = dict(text={'content':'动量'},msgId='synthetic-message',senderId='private-sender',
                       sessionWebhook='https://oapi.dingtalk.com/robot/sendBySession?session=synthetic-session')
        self.assertEqual(self.request(payload,{'token':'synthetic-token'}),{})
        self.assertEqual(self.request(payload,{'token':'synthetic-token'}),{})
        self.assertEqual(len(self.outgoing),1)
        self.assertEqual(self.outgoing[0][1]['json']['markdown']['text'],'saved test signal')
        self.assertFalse(self.outgoing[0][1]['allow_redirects'])
        logs='\n'.join(self.logs)
        self.assertIn('status=sent',logs)
        self.assertIn('reason=duplicate',logs)
        for private in ('private-sender','synthetic-session','synthetic-message','saved test signal'):
            self.assertNotIn(private,logs)

    def test_unauthenticated_session_cannot_trigger_outbound_delivery(self):
        payload=dict(text={'content':'动量'},sessionWebhook='https://oapi.dingtalk.com/robot/sendBySession?session=synthetic')
        self.assertEqual(self.request(payload)[1],401)
        self.assertEqual(self.outgoing,[])

    def test_bad_or_oversized_json_returns_visible_http_failure_without_dispatch(self):
        for raw, status in ((b'{"text":null}', 400), (b'[' + b' ' * guard.MAX_CALLBACK_BYTES, 413)):
            self.assertEqual(self.request(headers={'token': 'synthetic-token'}, raw=raw)[1], status)
        self.assertEqual(self.called, [])

    def test_momentum_bypasses_unrelated_broken_stock_config_and_other_commands_fail_visibly(self):
        def invalid():
            raise ValueError('never print secret content')
        ns = dict(config=types.SimpleNamespace(load_stocks=invalid), logger=self.logger,
                  _handle_momentum_signal=lambda: 'read-only saved query')
        selected_functions(self.sources['app.py'], {'handle_command'}, ns)
        self.assertEqual(ns['handle_command']('momentum', [], 'test', 'test'), 'read-only saved query')
        result = ns['handle_command']('list', [], 'test', 'test')
        self.assertIn('配置暂时不可用', result)
        self.assertNotIn('secret content', '\n'.join(self.logs))

    def test_query_calls_shared_readonly_loader_without_opening_latest_itself(self):
        calls = []
        renderer = types.SimpleNamespace(render_saved_query=lambda **kwargs: calls.append(kwargs) or 'dated saved query')
        ns = dict(os=os, logger=self.logger, MOMENTUM_SIGNAL_FILE='/synthetic/signals/latest.txt')
        selected_functions(self.sources['app.py'], {'_handle_momentum_signal'}, ns)
        spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda module: None))
        with patch('importlib.util.spec_from_file_location', return_value=spec), \
                patch('importlib.util.module_from_spec', return_value=renderer), \
                patch('builtins.open', side_effect=AssertionError('Query must delegate reading to shared journal loader')):
            self.assertEqual(ns['_handle_momentum_signal'](), 'dated saved query')
        self.assertEqual(calls, [{'path': '/synthetic/signals/latest.txt'}])

    def test_all_sending_helpers_redact_exception_urls_and_remote_error_payloads(self):
        names = {'_sign_webhook', '_ensure_keyword', '_split_markdown', '_error_code',
                 'send_text', 'send_markdown', 'send_session_text', 'send_feedcard',
                 'send_session_feedcard', 'send_session_markdown'}
        config = types.SimpleNamespace(DINGTALK_WEBHOOK_URL='https://example.invalid/?access_token=private-token',
                                       DINGTALK_WEBHOOK_SECRET='', DINGTALK_KEYWORD='')
        calls = []
        response = types.SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {'errcode': 123, 'errmsg': 'private-response',
                          'sessionWebhook': 'https://example.invalid/?session=private-session'})
        def post(*args, **kwargs):
            calls.append((args, kwargs))
            if fail_exception:
                raise RuntimeError('request failed https://example.invalid/?session=private-exception')
            return response
        ns = dict(config=config, logger=self.logger, requests=types.SimpleNamespace(post=post),
                  time=types.SimpleNamespace(sleep=lambda seconds: None))
        selected_functions(self.sources['dingtalk.py'], names, ns)
        tasks = [('send_text', ('text',)), ('send_markdown', ('title', 'text')),
                 ('send_feedcard', ([],)), ('send_session_text', ('https://example.invalid/session', 'text')),
                 ('send_session_markdown', ('https://example.invalid/session', 'title', 'text')),
                 ('send_session_feedcard', ('https://example.invalid/session', []))]
        for fail_exception in (False, True):
            for name, args in tasks:
                self.assertFalse(ns[name](*args))
        self.assertEqual(len(calls), 12)
        joined = '\n'.join(self.logs)
        for marker in ('private-response', 'private-session', 'private-exception', 'private-token', 'https://'):
            self.assertNotIn(marker, joined)
        self.assertIn('123', joined)
        self.assertIn('RuntimeError', joined)


if __name__ == '__main__':
    unittest.main()
