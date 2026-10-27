"""Content Death Clock: paste a YouTube link, get the video's attention outlook.

Run locally::

    streamlit run src/cdc/tool/app.py

with the tool's own key in the environment as ``YOUTUBE_TOOL_API_KEY`` (or in
``.streamlit/secrets.toml``). On Streamlit Community Cloud the key goes in the
app's Secrets settings under the same name, and nowhere else.

The model is rebuilt from ``assets/training_table.csv`` when the app starts,
then checked against the fingerprint recorded when it was frozen. A saved model
file would break on the first library upgrade; a table and a recipe do not.
"""
from __future__ import annotations

import json
import os
import sys
import time
from html import escape
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st                                              # noqa: E402

import cdc.config  # noqa: E402,F401  loads .env.local, where a local run keeps YOUTUBE_TOOL_API_KEY
from cdc.tool import table as T                                     # noqa: E402
from cdc.tool.fetch import DailyBudget, ToolError, lookup, parse_video_id   # noqa: E402
from cdc.tool.model import InstantModel, fingerprint                # noqa: E402
from cdc.tool.report import build_report                            # noqa: E402

SITE = "https://baconkage.github.io/content-death-clock/"
CARD = T.ASSETS / "model_card.json"
TIPS = T.ASSETS / "tips.json"
SESSION_LIMIT, SESSION_WINDOW_S = 30, 3600

st.set_page_config(page_title="Content Death Clock", page_icon="⏳", layout="centered")

st.markdown("""
<style>
.cdc-badge{display:inline-block;padding:.2rem .7rem;border-radius:999px;color:#fff;font-weight:600;font-size:.9rem}
.cdc-high{background:#c0392b}.cdc-typical{background:#5f6b7a}.cdc-low{background:#13795b}
.cdc-row{display:grid;grid-template-columns:5.5rem 1fr 3.2rem;gap:.6rem;align-items:center;margin:.3rem 0}
.cdc-track{height:.75rem;border-radius:.4rem;background:rgba(128,128,128,.18);position:relative;overflow:hidden}
.cdc-fill{height:100%;border-radius:.4rem;background:#2a78d6}
.cdc-tick{position:absolute;top:-2px;bottom:-2px;width:2px;background:currentColor;opacity:.55}
.cdc-num{text-align:right;font-variant-numeric:tabular-nums}
.cdc-big{font-size:1.25rem;line-height:1.4;margin:.4rem 0 .2rem}
.cdc-muted{opacity:.72;font-size:.9rem}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Building the model from the study's data (once, under a minute)...")
def load():
    card = json.loads(CARD.read_text(encoding="utf-8"))
    tips = json.loads(TIPS.read_text(encoding="utf-8"))
    table = T.load_table()
    model = InstantModel(b=card["b"]).fit(table)
    return model, table, card, tips, fingerprint(model) == card["fingerprint"]


@st.cache_resource
def budget() -> DailyBudget:
    return DailyBudget()


def api_key() -> str | None:
    try:
        key = st.secrets.get("YOUTUBE_TOOL_API_KEY")
    except Exception:                     # no secrets file at all, e.g. a local run
        key = None
    return key or os.environ.get("YOUTUBE_TOOL_API_KEY")


@st.cache_data(ttl=600, max_entries=500, show_spinner=False)
def cached_lookup(video_id: str):
    # The key is read inside, never passed in, so it is not part of the cache key.
    lk = lookup(video_id, api_key(), budget())
    return lk.item, lk.subscribers, lk.fetched_at


def session_allows() -> bool:
    now = time.time()
    seen = [t for t in st.session_state.get("lookups", []) if now - t < SESSION_WINDOW_S]
    st.session_state["lookups"] = seen
    if len(seen) >= SESSION_LIMIT:
        return False
    seen.append(now)
    return True


# -------------------------------------------------------------------- render
def _n(x) -> str:
    return "hidden" if x is None else f"{int(x):,}"


def _age(h: float) -> str:
    if h < 1:
        return f"{int(h * 60)} minutes"
    if h < 48:
        return f"{h:.1f} hours"
    return f"{h / 24:.1f} days"


def bars(rows: list[dict], typical: list[dict]) -> str:
    out = []
    for r, t in zip(rows, typical):
        out.append(
            f'<div class="cdc-row"><span>{escape(r["after"])}</span>'
            f'<div class="cdc-track" role="img" aria-label="{r["chance"]:.0%}, typical video {t["chance"]:.0%}">'
            f'<div class="cdc-fill" style="width:{100 * r["chance"]:.1f}%"></div>'
            f'<div class="cdc-tick" style="left:calc({100 * t["chance"]:.1f}% - 1px)"></div></div>'
            f'<span class="cdc-num">{r["chance"]:.0%}</span></div>')
    return "".join(out)


def render(rep: dict) -> None:
    f = rep["facts"]
    st.subheader(f["title"] or "Untitled video")
    st.caption(f"{f['channel']} · posted {_age(f['age_h'])} ago")
    c = st.columns(4)
    c[0].metric("Views", _n(f["views"]))
    c[1].metric("Likes", _n(f["likes"]))
    c[2].metric("Comments", _n(f["comments"]))
    c[3].metric("Subscribers", _n(f["subscribers"]))

    status = rep["status"]
    if status == "too_new":
        st.info(f"**Too new for an outlook.** The outlook starts 15 minutes after posting. "
                f"Check back in about {rep['minutes_until_ready']} minutes.")
    elif status == "too_old_for_odds":
        st.info("**Too old for a reliable outlook.** The outlook was built and tested on videos up to "
                "13 hours old. The comparisons below still apply.")
    elif status in ("live", "beyond_window", "no_numbers"):
        for n in rep["notes"]:
            st.info(n)
        return

    o = rep["outlook"]
    if o:
        st.markdown("### Attention outlook")
        label = {"high": "High risk of fading fast", "typical": "About average",
                 "low": "Likely to keep going"}[o["risk"]]
        st.markdown(f'<span class="cdc-badge cdc-{o["risk"]}">{label}</span>', unsafe_allow_html=True)
        st.markdown(f'<p class="cdc-big"><b>{o["fade_within_2_days"]:.0%}</b> chance it stops getting '
                    f'attention within two days of posting <span class="cdc-muted">(a typical video: '
                    f'{o["typical"]:.0%})</span></p>', unsafe_allow_html=True)
        st.markdown("**Chance it's still getting attention after...**", unsafe_allow_html=True)
        st.markdown(bars(o["still_going"], o["typical_still_going"]), unsafe_allow_html=True)
        st.caption("Blue bar: this video. Dark tick: a typical video. These are odds from one reading, "
                   "not a countdown, and they include the chance it has already faded.")

    if rep["comparisons"]:
        st.markdown("### How it compares")
        for cmp in rep["comparisons"]:
            lo, hi = cmp["at_h"]
            when = f"{lo:g}h" if lo == hi else f"{lo:g}-{hi:g}h"
            pct = cmp["percentile"]
            where = (f"higher than **{pct:.0f}%**" if pct >= 50 else f"lower than **{100 - pct:.0f}%**")
            group = ("channels of any size" if cmp["group"] == "all channel sizes"
                     else f"channels with {cmp['group']}")
            st.markdown(f"- **{cmp['metric'].capitalize()}:** {where} of "
                        f"videos at the same age ({when}) from {group} "
                        f"<span class='cdc-muted'>({cmp['videos']:,} videos, {cmp['channels']} channels)</span>",
                        unsafe_allow_html=True)

    if rep["todo"]:
        st.markdown("### What to do now")
        for t in rep["todo"]:
            st.markdown(f"- {t}")

    for n in rep["notes"]:
        st.warning(n)

    tips = rep["tips"]
    st.markdown("### What tends to make videos last")
    if tips.get("tips"):
        for t in tips["tips"]:
            st.markdown(f"- {t['text']}")
        st.caption("Patterns in the videos we followed, held in both groups we studied. "
                   "They are associations: we can't show that changing these would make a video last longer.")
    else:
        names = ", ".join(c["name"] for c in tips["candidates"].values())
        st.markdown(f"We checked {len(tips['candidates'])} patterns within channels ({names}). "
                    "**None held up in both groups of videos we studied**, so we don't give advice on them. "
                    "Likes per view and comments per view leaned towards longer life in both groups, "
                    "but each was clear in only one.")

    with st.expander("How accurate is this?"):
        a = rep["accuracy"]
        src = ("Tested once on new videos the model had never seen, from channels left out of its training."
               if a["source"] == "test" else
               "Rehearsal: trained on videos posted before 16 September, then checked on videos posted "
               "16 September to 3 October, each one by a model that had never seen its channel. "
               "A single test on fresh videos is fixed in the study's analysis plan.")
        st.markdown(f"{src}\n\n- Of videos it marked **high risk**, **{a['flag_precision']:.0%}** really "
                    f"stopped getting attention within two days, against **{a['base_rate']:.0%}** of all videos.\n"
                    f"- Its odds were closer to what happened than quoting the average rate "
                    f"(Brier skill score {a['brier_skill']:+.2f}; 0 would mean no better than average).\n"
                    + (f"- It picks which of two videos fades first correctly about "
                       f"{round(10 * a['ranking'])} times in 10. " if a.get("ranking") else "- ")
                    + "It cannot give an exact time: the study found that can't be done from public numbers.\n"
                    f"- {a['videos']:,} videos from {a['channels']} channels.")
    with st.expander('What does "stops getting attention" mean?'):
        st.markdown("A video's views per hour are measured over its life. It **stops getting attention** at "
                    "the first point where its views per hour fall below 5% of its own best rate and stay "
                    "there. A small video and a huge one are judged against their own peak, not each other. "
                    "This is the study's definition, fixed before any data was collected.")


# ---------------------------------------------------------------------- page
def main() -> None:
    st.title("⏳ Content Death Clock")
    model, table, card, tips, fp_ok = load()
    n = card["trained_on"]
    st.markdown(f"Paste a YouTube video link to see how likely it is to keep getting attention, "
                f"based on {n['videos']:,} videos from {n['channels']} channels we followed for ten days.")

    with st.form("check"):
        url = st.text_input("YouTube link", placeholder="https://www.youtube.com/watch?v=...",
                            max_chars=500, label_visibility="collapsed")
        go = st.form_submit_button("Check this video", type="primary")
    if go:
        vid = parse_video_id(url)
        if not vid:
            st.error(ToolError("bad_link").message)
        elif not session_allows():
            st.error("You've checked a lot of videos in the last hour. Please wait a while and try again.")
        else:
            try:
                with st.spinner("Asking YouTube..."):
                    item, subs, fetched_at = cached_lookup(vid)
                render(build_report(item, subs, fetched_at, model, table, card, tips))
            except ToolError as e:
                st.error(e.message)

    st.divider()
    st.caption(f"RV University study · [how the study works]({SITE}) · links you paste are not stored. "
               f"Model {card['fingerprint']}" + ("" if fp_ok else " (built differently on this server)"))
    if not fp_ok:
        st.warning("This server's model doesn't exactly match the tested one (a library difference). "
                   "The numbers should be close, but treat them with extra caution.")


main()
