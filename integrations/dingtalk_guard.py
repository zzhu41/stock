# -*- coding: utf-8 -*-
"""Python 3.6-compatible, side-effect-free DingTalk callback validation.

Legacy custom Outgoing: `token` header equals the configured outgoing token.
Official Alibaba API chatbot.install describes outgoing_token as a callback
header: https://developer.alibaba.com/docs/api.htm?apiId=47514

Enterprise HTTP robots: timestamp/sign headers, Base64(HMAC-SHA256 with
AppSecret, timestamp + '\\n' + AppSecret), and a one-hour timestamp window.
https://open.dingtalk.com/document/orgapp/receive-message

Modes are explicit. Never use a send-to-group Webhook signing secret as an
incoming AppSecret; never fall back to a different mode after a failed check.
No logging, config loading, network requests or message sending occurs here.
"""
import base64
import hashlib
import hmac
import json
import re
import time

MAX_CALLBACK_BYTES = 128 * 1024
MAX_TIMESTAMP_AGE_MS = 60 * 60 * 1000


def _headers(headers):
    return {str(key).lower(): value for key, value in headers.items()}


def _nonempty_string(value):
    return isinstance(value, str) and bool(value)


def verify_token(supplied, configured):
    """A missing token configuration is an error, never an allow-all mode."""
    return (_nonempty_string(configured) and _nonempty_string(supplied)
            and hmac.compare_digest(supplied.encode("utf-8"), configured.encode("utf-8")))


def verify_request(headers, mode="token", token=None, app_secret=None, now_ms=None):
    """Return (accepted, safe_reason); neither result contains credentials.

    Sign is the raw Base64 header value, NOT a URL-query-encoded Webhook sign.
    The published protocol signs the timestamp, not the request body; transport
    must therefore retain its HTTPS authentication boundary.
    """
    headers = _headers(headers)
    if mode == "token":
        if not _nonempty_string(token):
            return False, "auth_not_configured"
        return (True, "verified_token") if verify_token(headers.get("token"), token) else (False, "invalid_credentials")
    if mode != "sign":
        return False, "invalid_auth_mode"
    if not _nonempty_string(app_secret):
        return False, "auth_not_configured"
    timestamp, signature = headers.get("timestamp"), headers.get("sign")
    if (not isinstance(timestamp, str) or not re.fullmatch(r"[0-9]{13}", timestamp)
            or not isinstance(signature, str) or len(signature) != 44):
        return False, "invalid_credentials"
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if not isinstance(now_ms, (int, float)) or abs(int(timestamp) - now_ms) > MAX_TIMESTAMP_AGE_MS:
        return False, "expired_timestamp"
    message = (timestamp + "\n" + app_secret).encode("utf-8")
    expected = base64.b64encode(hmac.new(app_secret.encode("utf-8"), message, hashlib.sha256).digest()).decode("ascii")
    if not hmac.compare_digest(signature.encode("utf-8"), expected.encode("ascii")):
        return False, "invalid_credentials"
    return True, "verified_sign"


def parse_payload(raw):
    """Return (payload, HTTP_error_or_None), with no raw data in exceptions/logs.

    Nontext callbacks retain the existing empty-text behavior. Malformed JSON,
    non-object bodies and malformed routing/text fields return HTTP 400 rather
    than raising inside the command dispatcher.
    """
    if not isinstance(raw, bytes):
        return None, 400
    if len(raw) > MAX_CALLBACK_BYTES:
        return None, 413
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError):
        return None, 400
    if not isinstance(payload, dict):
        return None, 400
    text = payload.get("text", {})
    if not isinstance(text, dict) or not isinstance(text.get("content", ""), str):
        return None, 400
    for key in ("senderStaffId", "senderId", "senderNick", "conversationType",
                "conversationId", "sessionWebhook", "msgtype"):
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            return None, 400
    return payload, None
