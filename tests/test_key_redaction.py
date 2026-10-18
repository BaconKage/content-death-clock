"""The API key must never leave the client inside an error message.

On 2026-09-17 a 404 for one channel produced the error text
``404 Client Error: Not Found for url: ...&key=AIza...``. The runner wrote that
text into the cycle log, the cycle log was committed, and the repository is
public, so the live key sat in the open from then on. Its quota was drained
from outside on 2026-10-15, 10-17 and 10-18.

These tests rebuild that exact failure with a real ``requests.Response``, so
they exercise the message ``requests`` itself generates rather than a guess at
it.
"""
from __future__ import annotations

import pytest
import requests

from cdc.collect.youtube import (QuotaExceeded, QuotaMeter, YouTubeAPIError,
                                 YouTubeClient, redact)

KEY = "AIzaSyFAKE-test-key_0123456789abcdefghij"


class _Session:
    """Stands in for requests.Session; returns or raises what it is given."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        r = self.outcome
        r.url = requests.Request("GET", url, params=params).prepare().url
        return r


def _response(status: int, body: str = "{}") -> requests.Response:
    r = requests.Response()
    r.status_code = status
    r.reason = {404: "Not Found", 403: "Forbidden", 500: "Server Error"}.get(status, "")
    r._content = body.encode()
    return r


def _client(outcome) -> YouTubeClient:
    c = YouTubeClient(api_key=KEY, meter=QuotaMeter(budget=100))
    c.session = _Session(outcome)
    return c


def test_redact_strips_the_key_parameter_and_keeps_the_rest():
    url = ("https://www.googleapis.com/youtube/v3/playlistItems?part=snippet"
           f"&playlistId=UU123&maxResults=50&key={KEY}")
    out = redact(f"404 Client Error: Not Found for url: {url}")
    assert KEY not in out and "key=REDACTED" in out
    assert "playlistId=UU123" in out and "maxResults=50" in out


def test_redact_also_removes_the_literal_secret_wherever_it_appears():
    assert KEY not in redact(f"token {KEY} leaked", secret=KEY)


def test_the_2026_09_17_failure_no_longer_carries_the_key():
    """A 404 on playlistItems: the exact error that leaked the key."""
    c = _client(_response(404))
    with pytest.raises(YouTubeAPIError) as ei:
        c._get("playlistItems", {"playlistId": "UUVNmqNpLvAsJ1tgwY0AGGHQ"})
    msg = str(ei.value)
    assert "404" in msg and "Not Found" in msg      # still useful to a human
    assert KEY not in msg and "key=REDACTED" in msg
    # The original HTTPError, whose text holds the key, must not be chained
    # on, or a traceback in the public Actions log would print it.
    assert ei.value.__cause__ is None and ei.value.__suppress_context__


def test_a_network_error_with_the_url_in_it_is_redacted():
    err = requests.ConnectionError(
        "HTTPSConnectionPool(host='www.googleapis.com', port=443): Max retries "
        f"exceeded with url: /youtube/v3/videos?part=statistics&id=x&key={KEY}")
    c = _client(err)
    with pytest.raises(YouTubeAPIError) as ei:
        c._get("videos", {"id": "x"}, attempts=1)
    assert KEY not in str(ei.value)


def test_quota_message_is_redacted_too():
    c = _client(_response(403, '{"error": {"message": "quota exceeded"}}'))
    with pytest.raises(QuotaExceeded) as ei:
        c._get("videos", {"id": "x"})
    assert KEY not in str(ei.value)
