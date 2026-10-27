"""The Streamlit page, driven headless with a recorded video in place of YouTube."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest                       # noqa: E402

from cdc.tool import fetch as Fe                               # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "cdc" / "tool" / "app.py")
RAW = json.loads((Path(__file__).parent / "fixtures" / "tool_parity.json")
                 .read_text(encoding="utf-8"))["posts"][0]["raw"]


def _fake(age_h: float, subs=133_000, fail: str | None = None):
    item = copy.deepcopy(RAW)
    item["snippet"]["publishedAt"] = (datetime.now(timezone.utc) - timedelta(hours=age_h)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def lookup(video_id, api_key, budget, client=None):
        if fail:
            raise Fe.ToolError(fail)
        return Fe.Lookup(item=item, subscribers=subs, fetched_at=datetime.now(timezone.utc))
    return lookup


def _submit(monkeypatch, url: str, lookup) -> AppTest:
    import streamlit as st
    st.cache_data.clear()                  # lookups are cached by video id across runs
    monkeypatch.setattr(Fe, "lookup", lookup)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    at.text_input[0].input(url)
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at: AppTest) -> str:
    """The visible words of every HTML block the page drew (the page is built from st.html)."""
    import html
    import re
    body = " ".join(e.proto.body for e in at.get("html") if "<style>" not in e.proto.body)
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)))


def test_a_fresh_video_gets_the_full_report(monkeypatch):
    at = _submit(monkeypatch, "https://youtu.be/abcdefghijk?si=x", _fake(4.0))
    t = _text(at)
    for heading in ("The chance it fades within two days of being posted", "still getting views",
                    "Compared with similar videos", "What you can do", "How much can you trust this?",
                    "Technical details"):
        assert heading in t
    assert "matches the tested build" in t and "built slightly differently" not in t


def test_an_older_video_gets_comparisons_only(monkeypatch):
    t = _text(_submit(monkeypatch, "https://youtu.be/abcdefghijk", _fake(40.0)))
    assert "too old for a forecast" in t and "Compared with similar videos" in t
    assert "fades within two days of being posted" not in t


def test_bad_links_and_youtube_failures_show_plain_messages(monkeypatch):
    t = _text(_submit(monkeypatch, "https://vimeo.com/123", _fake(4.0)))
    assert "look like a link to a YouTube video" in t
    t = _text(_submit(monkeypatch, "https://youtu.be/abcdefghijk", _fake(4.0, fail="daily_cap")))
    assert "daily limit" in t


def test_before_any_check_the_page_explains_itself(monkeypatch):
    import streamlit as st
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    t = _text(at)
    assert "Will this video keep getting views?" in t and "What you’ll get" in t
