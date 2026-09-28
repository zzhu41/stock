# -*- coding: utf-8 -*-
"""Authenticated momentum replies, separate from the HTTP callback ACK.

Python3.6 standard library only. Importing this module does not read config,
start threads, write files or send messages. Keep ONE imported module instance
per Gunicorn process so its bounded deduplication cache survives callbacks.
This is process-local best-effort deduplication, not a durable/cross-worker
exactly-once guarantee. No session URL, user id or message body is logged.
"""
import hashlib
import math
import threading
import time
from urllib.parse import parse_qs, urlsplit


REQUEST_TIMEOUT = (2, 3)
DEFAULT_CACHE_TTL_SECONDS = 600
MAX_CACHE_ENTRIES = 1024
MAX_INFLIGHT = 8
AUTH_REASONS = frozenset((
    "verified_token", "verified_sign", "invalid_credentials", "auth_not_configured",
    "invalid_auth_mode", "expired_timestamp", "authentication_unavailable",
))


def callback_diagnostics(headers, mode, accepted, reason):
    """Only header *presence*, an allowlisted mode and fixed error category."""
    keys = {str(key).lower() for key in headers.keys()}
    return dict(
        mode=mode if mode in ("token", "sign") else "unknown",
        accepted=accepted is True,
        token_present="token" in keys,
        timestamp_present="timestamp" in keys,
        sign_present="sign" in keys,
        reason=reason if reason in AUTH_REASONS else "authentication_unavailable",
    )


class ReplyCache:
    """Reserve under a lock before dispatch, and never evict a live receipt."""
    def __init__(self, ttl_seconds=DEFAULT_CACHE_TTL_SECONDS,
                 capacity=MAX_CACHE_ENTRIES, max_inflight=MAX_INFLIGHT, clock=None):
        if ttl_seconds <= 0 or capacity <= 0 or max_inflight <= 0:
            raise ValueError("Invalid reply cache bounds")
        self.ttl_seconds = float(ttl_seconds)
        self.capacity = int(capacity)
        self.max_inflight = int(max_inflight)
        self.clock = clock or time.monotonic
        self.lock = threading.Lock()
        self.entries = {}
        self.inflight = 0

    def reserve(self, message_id, ttl_seconds=None):
        key = hashlib.sha256(message_id.encode("utf-8")).hexdigest() if message_id else None
        ttl = self.ttl_seconds if ttl_seconds is None else max(1., float(ttl_seconds))
        with self.lock:
            now = self.clock()
            for old in list(self.entries):
                item = self.entries[old]
                # An outstanding network operation never becomes a free slot
                # merely because wall time passed.
                if item["status"] != "sending" and item["expires"] <= now:
                    del self.entries[old]
            if key in self.entries:
                return None, "duplicate", self.entries[key]["status"]
            if self.inflight >= self.max_inflight:
                return None, "busy", None
            if key is not None and len(self.entries) >= self.capacity:
                return None, "cache_full", None
            self.inflight += 1
            if key is not None:
                self.entries[key] = dict(status="sending", expires=now + ttl)
            return dict(key=key, active=True), None, None

    def finish(self, ticket, status):
        with self.lock:
            if not ticket["active"]:
                return
            ticket["active"] = False
            self.inflight -= 1
            if ticket["key"] is not None:
                self.entries[ticket["key"]]["status"] = status


_CACHE = ReplyCache()


def _now_ms():
    return int(time.time() * 1000)


def _endpoint(url):
    """Only the documented DingTalk session endpoint; no arbitrary webhook."""
    if (not isinstance(url, str) or not url or len(url) > 8192
            or any(ord(char) < 33 or ord(char) == 127 or char.isspace() for char in url)
            or "#" in url):
        raise ValueError("invalid_session_url")
    try:
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or parsed.hostname != "oapi.dingtalk.com"
                or parsed.username is not None or parsed.password is not None
                or parsed.port not in (None, 443) or parsed.fragment
                or parsed.path != "/robot/sendBySession"):
            raise ValueError("invalid_session_url")
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        if set(query) != {"session"} or len(query["session"]) != 1 or not query["session"][0]:
            raise ValueError("invalid_session_url")
    except (TypeError, ValueError):
        raise ValueError("invalid_session_url")
    return url


def _expiry(payload, now):
    if "sessionWebhookExpiredTime" not in payload:
        return None
    value = payload["sessionWebhookExpiredTime"]
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value != int(value) or value <= 0):
        raise ValueError("invalid_session_expiry")
    if value <= now:
        raise ValueError("expired_session")
    return int(value)


def _safe_log(logger, event):
    if logger is None:
        return
    # Explicit fields only: never log payload, response JSON, URL, identifiers,
    # arbitrary exceptions or the return value of reply_momentum as a whole.
    try:
        logger.info(
            "DingTalk momentum reply transport=%s status=%s reason=%s "
            "expiry_verified=%s dedup_enabled=%s prior_status=%s http_status=%s errcode=%s",
            event["transport"], event["status"], event["reason"],
            event["expiry_verified"], event["dedup_enabled"], event.get("prior_status"),
            event.get("http_status"), event.get("errcode"))
    except Exception:
        pass


def _thread_dispatch(work):
    thread = threading.Thread(target=work, name="dingtalk-momentum-reply")
    thread.daemon = True
    thread.start()


def reply_momentum(payload, markdown, requester, *, authenticated, logger=None,
                   now_ms=None, cache=None, dispatch=None, title="回复"):
    """Return (callback_body, safe_event); outbound delivery may finish later.

    Callers pass authenticated=True only after verifying credentials OR, for the
    read-only momentum command, when relying on the session id being an
    unguessable short-lived platform value and the content being public market
    data (anonymous fallback decided in the bot, not here). This module's own
    guarantees never depend on the caller's credential state: endpoint shape,
    expiry, dedup and bounded cache are enforced regardless. requester is
    requests.post or an injected fake; this module owns no HTTP client or
    credentials. With a session endpoint return an empty HTTP ACK, including
    failure/duplicate cases. Without a session retain the old inline Markdown
    response. A queued event is NOT delivery confirmation.

    dispatch(work) defaults to one bounded daemon thread; tests can capture or
    execute work synchronously. Missing expiry permits compatibility with an
    explicit expiry_verified=False and ten-minute process-local cache TTL.
    """
    if authenticated is not True:
        raise PermissionError("authentication_required")
    if not isinstance(payload, dict) or not isinstance(markdown, str) or not isinstance(title, str):
        raise ValueError("invalid_reply_input")
    session = payload.get("sessionWebhook")
    if session is None or session == "":
        event = dict(transport="inline", status="inline", reason="session_missing",
                     expiry_verified=False, dedup_enabled=False, dedup_scope="none")
        _safe_log(logger, event)
        return dict(msgtype="markdown", markdown=dict(title=title, text=markdown)), event
    event = dict(transport="session", status="rejected", reason="invalid_session_url",
                 expiry_verified=False, dedup_enabled=False, dedup_scope="process")
    moment = _now_ms() if now_ms is None else now_ms
    try:
        endpoint = _endpoint(session)
        expires = _expiry(payload, moment)
        event["expiry_verified"] = expires is not None
        message_id = payload.get("msgId")
        if message_id is not None and (not isinstance(message_id, str) or len(message_id) > 4096):
            raise ValueError("invalid_message_id")
        event["dedup_enabled"] = bool(message_id)
    except (TypeError, ValueError, OverflowError) as exc:
        # These validation failures are constructed only from fixed literals.
        category = str(exc)
        if category not in ("invalid_session_url", "invalid_session_expiry", "expired_session", "invalid_message_id"):
            category = "invalid_reply_input"
        event["reason"] = category
        _safe_log(logger, event)
        return {}, event
    cache = cache or _CACHE
    ttl = (expires - moment) / 1000. if expires is not None else cache.ttl_seconds
    event["dedup_ttl_seconds"] = ttl
    ticket, refused, previous = cache.reserve(message_id, ttl)
    if refused:
        event.update(status="suppressed" if refused == "duplicate" else "rejected",
                     reason=refused, prior_status=previous)
        _safe_log(logger, event)
        return {}, event
    event.update(status="queued", reason="expiry_verified" if expires is not None else "expiry_unverified")
    _safe_log(logger, event)
    body = dict(msgtype="markdown", markdown=dict(title=title, text=markdown))

    def send():
        final = dict(event, status="uncertain", reason="transport_exception")
        try:
            current = _now_ms() if now_ms is None else now_ms
            if expires is not None and expires <= current:
                final.update(status="expired", reason="expired_before_send")
                return
            response = requester(endpoint, json=body, timeout=REQUEST_TIMEOUT, allow_redirects=False)
            status = getattr(response, "status_code", None)
            final["http_status"] = status if type(status) is int else None
            if type(status) is not int or not 200 <= status < 300:
                final.update(status="failed", reason="http_status")
                return
            value = response.json()
            code = value.get("errcode") if isinstance(value, dict) else None
            final["errcode"] = code if type(code) is int else None
            if type(code) is not int:
                final.update(status="uncertain", reason="invalid_response")
            elif code != 0:
                final.update(status="failed", reason="api_rejected")
            else:
                final.update(status="sent", reason="confirmed")
        except Exception:
            # The server may already have accepted it. Do not guess/retry.
            final.update(status="uncertain", reason="transport_exception")
        finally:
            cache.finish(ticket, final["status"])
            _safe_log(logger, final)

    try:
        (dispatch or _thread_dispatch)(send)
    except Exception:
        cache.finish(ticket, "uncertain")
        event.update(status="uncertain", reason="dispatch_failed")
        _safe_log(logger, event)
    return {}, event
