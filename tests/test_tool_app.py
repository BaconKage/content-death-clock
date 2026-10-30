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


def test_verdict_shows_exact_one_decimal_probabilities_and_typical_value():
    from cdc.tool import app as A

    rendered = A.verdict({"fade_within_2_days": 0.224, "typical": 0.341, "risk": "typical"})

    assert "22.4%" in rendered
    assert "34.1%" in rendered
    assert "about 2 in 10" not in rendered
    assert "Exact model predictions: this video 22.4%; typical video 34.1%" in rendered
    assert "ten squares are a rounded visual guide" in rendered


def test_distinct_probabilities_no_longer_collapse_to_the_same_text():
    from cdc.tool import app as A

    first = 0.21
    second = 0.24
    assert A.in_ten(first)[1] == A.in_ten(second)[1] == "about 2 in 10"

    first_rendered = A.verdict({"fade_within_2_days": first, "typical": 0.3, "risk": "typical"})
    second_rendered = A.verdict({"fade_within_2_days": second, "typical": 0.3, "risk": "typical"})
    assert "21.0%" in first_rendered
    assert "24.0%" in second_rendered
    assert first_rendered != second_rendered


def test_probability_squares_remain_ten_and_round_visually():
    from cdc.tool import app as A

    rendered = A.verdict({"fade_within_2_days": 0.224, "typical": 0.341, "risk": "typical"})

    assert rendered.count("<i class=") == 20
    assert rendered.count('<i class="on"></i>') == 5  # .224 rounds to 2 in 10; .341 to 3 in 10
    assert 'class="cdc-sq "' in rendered
    assert 'class="cdc-sq typ"' in rendered


def _page_end(at: AppTest) -> list[str]:
    """What actually sits at the bottom of the page now (st.get can include replaced elements)."""
    return [getattr(getattr(c, "proto", None), "body", "") for c in at.main.children.values()][-2:]


def test_a_failed_check_never_shows_the_previous_videos_report(monkeypatch):
    good = _fake(4.0)

    def lookup(video_id, api_key, budget, client=None):
        if video_id == "zzzzzzzzzzz":
            raise Fe.ToolError("not_found")
        return good(video_id, api_key, budget, client)

    at = _submit(monkeypatch, "https://youtu.be/abcdefghijk", lookup)
    assert "The chance it fades within two days" in _text(at)
    at.text_input[0].input("https://youtu.be/zzzzzzzzzzz")
    at.button[0].click()
    at.run()
    assert not at.exception
    end = " ".join(_page_end(at))
    assert "no public video" in end
    assert "The chance it fades" not in end and "cdc-video" not in end


def test_a_failed_first_check_takes_down_the_loading_placeholder(monkeypatch):
    at = _submit(monkeypatch, "https://youtu.be/abcdefghijk", _fake(4.0, fail="daily_cap"))
    end = _page_end(at)
    assert "daily limit" in end[0]
    assert not any("cdc-skel" in b for b in end)
