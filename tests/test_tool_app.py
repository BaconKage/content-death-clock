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
    parts = [m.value for m in at.markdown] + [m.value for m in at.info] + [m.value for m in at.error]
    parts += [m.value for m in at.warning] + [m.value for m in at.subheader]
    return "\n".join(parts)


def test_a_fresh_video_gets_the_full_report(monkeypatch):
    at = _submit(monkeypatch, "https://youtu.be/abcdefghijk?si=x", _fake(4.0))
    t = _text(at)
    for heading in ("Attention outlook", "How it compares", "What to do now",
                    "What tends to make videos last"):
        assert heading in t
    assert "chance it stops getting attention within two days" in t
    assert "built differently" not in t                       # the fingerprint matched


def test_an_older_video_gets_comparisons_only(monkeypatch):
    t = _text(_submit(monkeypatch, "https://youtu.be/abcdefghijk", _fake(40.0)))
    assert "Too old for a reliable outlook" in t and "How it compares" in t
    assert "Attention outlook" not in t


def test_bad_links_and_youtube_failures_show_plain_messages(monkeypatch):
    t = _text(_submit(monkeypatch, "https://vimeo.com/123", _fake(4.0)))
    assert "doesn't look like a link to a YouTube video" in t
    t = _text(_submit(monkeypatch, "https://youtu.be/abcdefghijk", _fake(4.0, fail="daily_cap")))
    assert "daily limit" in t
