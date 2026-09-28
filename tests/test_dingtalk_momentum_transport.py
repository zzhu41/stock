"""Fake HTTP only: no real session URLs or recipients are ever contacted."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from integrations import dingtalk_momentum_transport as transport

NOW = 1800000000000
URL = "https://oapi.dingtalk.com/robot/sendBySession?session=offline-placeholder"
TEXT = "private synthetic momentum body"
MESSAGE_ID = "private-synthetic-message-id"


def payload(**changes):
    value = dict(msgId=MESSAGE_ID, senderId="private-synthetic-user", sessionWebhook=URL,
                 sessionWebhookExpiredTime=NOW + 60000)
    value.update(changes)
    return value


def response(code=0, status=200):
    return SimpleNamespace(status_code=status, json=lambda: dict(errcode=code))


class MomentumTransportTests(unittest.TestCase):
    def setUp(self):
        self.cache = transport.ReplyCache()
        self.logger = Mock()
        self.requester = Mock(return_value=response())

    def reply(self, data=None, **changes):
        options = dict(authenticated=True, now_ms=NOW, cache=self.cache,
                       logger=self.logger, dispatch=lambda work:work())
        options.update(changes)
        return transport.reply_momentum(payload() if data is None else data, TEXT, self.requester, **options)

    def test_callback_diagnostics_contains_only_fixed_categories_and_presence(self):
        headers = {"Token":"private-token","SIGN":"private-sign","Timestamp":"private-time",
                   "Authorization":"private-user-credential"}
        record = transport.callback_diagnostics(headers,"token",False,"invalid_credentials")
        self.assertEqual(record,dict(mode="token",accepted=False,reason="invalid_credentials",
            token_present=True,sign_present=True,timestamp_present=True))
        self.assertNotIn("private",str(record))
        unknown = transport.callback_diagnostics({},"private-mode",False,"private-reason")
        self.assertEqual(unknown["mode"],"unknown")
        self.assertEqual(unknown["reason"],"authentication_unavailable")

    def test_authentication_is_mandatory_without_auto_fallback(self):
        for accepted in (False,None,"true",1):
            with self.assertRaises(PermissionError):
                self.reply(authenticated=accepted)
        self.requester.assert_not_called()
        self.assertEqual(self.cache.entries,{})

    def test_missing_session_keeps_inline_markdown_without_network(self):
        for data in ({},dict(sessionWebhook=None),dict(sessionWebhook="")):
            body,event = self.reply(data)
            self.assertEqual(body,dict(msgtype="markdown",markdown=dict(title="回复",text=TEXT)))
            self.assertEqual(event["transport"],"inline")
            self.assertEqual(event["reason"],"session_missing")
        self.requester.assert_not_called()

    def test_session_reply_ack_and_exact_request_options(self):
        original = payload();before=deepcopy(original)
        body,event=self.reply(original)
        self.assertEqual(body,{})
        self.assertEqual(event["status"],"queued")
        args,kwargs=self.requester.call_args
        self.assertEqual(args,(URL,))
        self.assertEqual(kwargs["timeout"],(2,3))
        self.assertIs(kwargs["allow_redirects"],False)
        self.assertEqual(kwargs["json"],dict(msgtype="markdown",markdown=dict(title="回复",text=TEXT)))
        self.assertEqual(original,before)
        self.assertEqual(next(iter(self.cache.entries.values()))["status"],"sent")

    def test_bad_urls_never_reach_requester(self):
        for url in (
            "http://oapi.dingtalk.com/robot/sendBySession?session=x",
            "https://oapi.dingtalk.com.evil.invalid/robot/sendBySession?session=x",
            "https://user:password@oapi.dingtalk.com/robot/sendBySession?session=x",
            "https://oapi.dingtalk.com:8443/robot/sendBySession?session=x",
            "https://oapi.dingtalk.com/robot/sendBySession?session=x#fragment",
            "https://oapi.dingtalk.com/robot/sendBySession?session=x#",
            "https://oapi.dingtalk.com/robot/send?access_token=x",
            "https://oapi.dingtalk.com/robot/sendBySession?session=",
            "https://oapi.dingtalk.com/robot/sendBySession?session=x&session=y",
            "https://oapi.dingtalk.com/robot/sendBySession?session=x&redirect=http://evil.invalid",
            " https://oapi.dingtalk.com/robot/sendBySession?session=x",
            "https://oapi.dingtalk.com/robot/sendBySession?session=x\n",
        ):
            body,event=self.reply(payload(sessionWebhook=url))
            self.assertEqual(body,{})
            self.assertEqual(event["reason"],"invalid_session_url")
        self.requester.assert_not_called()

    def test_expired_and_malformed_expiry_are_rejected(self):
        for expiry in (NOW,NOW-1,None,False,"1800000000001",float("nan"),-1):
            body,event=self.reply(payload(sessionWebhookExpiredTime=expiry))
            self.assertEqual(body,{})
            self.assertIn(event["reason"],("expired_session","invalid_session_expiry"))
        self.requester.assert_not_called()

    def test_missing_expiry_is_explicit_and_uses_short_process_ttl(self):
        data=payload();del data["sessionWebhookExpiredTime"]
        body,event=self.reply(data)
        self.assertEqual(body,{})
        self.assertFalse(event["expiry_verified"])
        self.assertEqual(event["reason"],"expiry_unverified")
        self.assertEqual(event["dedup_ttl_seconds"],600)
        self.assertEqual(event["dedup_scope"],"process")
        self.requester.assert_called_once()

    def test_only_2xx_and_integer_zero_confirm_sent_without_following_redirects(self):
        cases=[(response(False),"uncertain"),(response("0"),"uncertain"),
               (response(123),"failed"),(response(0,302),"failed"),(response(0,500),"failed")]
        for returned,expected in cases:
            self.cache=transport.ReplyCache()
            self.requester.return_value=returned
            self.reply()
            self.assertEqual(next(iter(self.cache.entries.values()))["status"],expected)
        self.assertTrue(all(call.kwargs["allow_redirects"] is False for call in self.requester.call_args_list))

    def test_timeout_is_uncertain_and_same_message_is_never_automatically_retried(self):
        self.requester.side_effect=TimeoutError("private URL "+URL)
        self.reply()
        body,event=self.reply()
        self.assertEqual(body,{})
        self.assertEqual(event["status"],"suppressed")
        self.assertEqual(event["prior_status"],"uncertain")
        self.requester.assert_called_once()

    def test_inflight_and_sent_duplicates_are_reserved_under_lock(self):
        jobs=[]
        self.reply(dispatch=jobs.append)
        body,duplicate=self.reply(dispatch=jobs.append)
        self.assertEqual(duplicate["prior_status"],"sending")
        self.assertEqual(len(jobs),1)
        self.requester.assert_not_called()
        jobs[0]()
        body,duplicate=self.reply()
        self.assertEqual(duplicate["prior_status"],"sent")
        self.requester.assert_called_once()
        self.assertNotIn(MESSAGE_ID,str(self.cache.entries))
        self.assertNotIn(URL,str(self.cache.entries))
        self.assertNotIn("private-synthetic-user",str(self.cache.entries))

    def test_full_cache_and_inflight_limit_reject_new_send_without_eviction(self):
        self.cache=transport.ReplyCache(capacity=1)
        self.reply()
        before=deepcopy(self.cache.entries)
        body,event=self.reply(payload(msgId="second-message"))
        self.assertEqual(event["reason"],"cache_full")
        self.assertEqual(before,self.cache.entries)
        self.requester.assert_called_once()
        self.cache=transport.ReplyCache(max_inflight=1)
        jobs=[];self.reply(dispatch=jobs.append)
        body,event=self.reply(payload(msgId="third-message"),dispatch=jobs.append)
        self.assertEqual(event["reason"],"busy")
        self.assertEqual(len(jobs),1)
        jobs[0]()

    def test_no_identifiers_payload_urls_or_exception_text_appear_in_logs(self):
        self.requester.side_effect=RuntimeError("private-synthetic-user "+URL+" "+TEXT)
        self.reply()
        logged=" ".join(str(call.args) for call in self.logger.info.call_args_list)
        for secret in (MESSAGE_ID,"private-synthetic-user",URL,TEXT):
            self.assertNotIn(secret,logged)
        self.assertIn("uncertain",logged)
        self.assertIn("transport_exception",logged)

    def test_ttl_is_bounded_and_missing_message_id_is_explicit(self):
        clock=[0.]
        self.cache=transport.ReplyCache(ttl_seconds=10,clock=lambda:clock[0])
        data=payload();del data["sessionWebhookExpiredTime"]
        self.reply(data)
        clock[0]=11.
        self.reply(data)
        self.assertEqual(self.requester.call_count,2)
        data=payload();del data["msgId"]
        body,event=self.reply(data)
        self.assertFalse(event["dedup_enabled"])

    def test_expiry_is_rechecked_when_background_work_starts(self):
        from unittest.mock import patch
        jobs=[];clock=[NOW]
        with patch.object(transport,"_now_ms",side_effect=lambda:clock[0]):
            self.reply(payload(sessionWebhookExpiredTime=NOW+100),now_ms=None,dispatch=jobs.append)
            clock[0]=NOW+101;jobs[0]()
        self.requester.assert_not_called()
        self.assertEqual(next(iter(self.cache.entries.values()))["status"],"expired")

    def test_default_background_dispatch_returns_ack_without_waiting_for_http(self):
        entered,release,done=threading.Event(),threading.Event(),threading.Event()
        def request(*args,**kwargs):
            entered.set()
            release.wait(3)
            done.set()
            return response()
        self.requester.side_effect=request
        started=time.monotonic()
        try:
            body,event=self.reply(dispatch=None)
            self.assertEqual(body,{})
            self.assertLess(time.monotonic()-started,1)
            self.assertTrue(entered.wait(1))
        finally:
            release.set()
            self.assertTrue(done.wait(1))

    def test_python36_absolute_import_and_mocked_session_dispatch(self):
        binary=shutil.which("python3.6")
        if not binary:self.skipTest("Python3.6 unavailable")
        code="\n".join([
            "import importlib.util",
            "from types import SimpleNamespace",
            "s=importlib.util.spec_from_file_location('transport',%r)" % str(Path(transport.__file__).resolve()),
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)",
            "calls=[]",
            "def fake(*a,**k):",
            " calls.append((a,k));return SimpleNamespace(status_code=200,json=lambda:{'errcode':0})",
            "body,event=m.reply_momentum(%r,'synthetic',fake,authenticated=True,now_ms=%d,dispatch=lambda f:f())" % (payload(),NOW),
            "assert body=={} and len(calls)==1 and calls[0][1]['allow_redirects'] is False",
        ])
        with tempfile.TemporaryDirectory() as directory:
            done=subprocess.run([binary,"-B","-c",code],cwd=directory,stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,universal_newlines=True,timeout=10)
        self.assertEqual(done.returncode,0,done.stderr)


if __name__=="__main__":unittest.main()
