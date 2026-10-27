"""Content Death Clock: paste a YouTube link, see whether the video will keep getting views.

Run locally::

    streamlit run src/cdc/tool/app.py

with the tool's own key in the environment as ``YOUTUBE_TOOL_API_KEY`` (or in
``.env.local``). On Streamlit Community Cloud the key goes in the app's Secrets
settings under the same name, and nowhere else.

The model is rebuilt from ``assets/training_table.csv`` when the app starts,
then checked against the fingerprint recorded when it was frozen. A saved model
file would break on the first library upgrade; a table and a recipe do not.

**Written for people who are not statisticians.** Chances are given as "2 in
10" next to a picture of ten squares, percentiles as "ahead of 69 in 100
similar videos", and every number the study itself would quote (Brier skill,
C-index, the model fingerprint) is kept in the technical details at the bottom.
The look follows the public website: same fonts, colours and corners, set in
``.streamlit/config.toml``.
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
TEAM = "Amogha V Prasad, S Anannya, Sanidhya Tiwari and Shubhang Srinivas Varda"

st.set_page_config(page_title="Content Death Clock: will your video keep getting views?",
                   page_icon=":material/hourglass_top:", layout="centered")

CSS = """
<style>
:root{
  --paper:#f4f5f2;--panel:#ffffff;--ink:#14181b;--ink2:#48525b;--muted:#5f6870;--rule:#dcdfdb;--chip:#eceee9;
  --accent:#2a78d6;--high:#b85c00;--ok:#0a7f3f;--typ:#48525b;--empty:#c9ced2;--bad:#b3261e;
  --display:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif;
  --body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
}
@media (prefers-color-scheme: dark){:root{
  --paper:#0f1315;--panel:#161c1f;--ink:#edf0f1;--ink2:#b5bdc3;--muted:#9aa4ab;--rule:#273136;--chip:#1f272b;
  --accent:#5b9bea;--high:#e8954a;--ok:#2fb36b;--typ:#b5bdc3;--empty:#3a454b;--bad:#f08b84;}}

[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="InputInstructions"]{display:none!important}
[data-testid="stMainBlockContainer"],.block-container{max-width:760px;padding-top:0.5rem;padding-bottom:3rem}
[data-testid="stTextInput"] label p{font-weight:600;font-size:0.95rem}
[data-testid="stTextInput"] input{height:3rem;font-size:1rem}
[data-testid="stFormSubmitButton"] button{height:3rem;font-weight:600;font-size:1rem;white-space:nowrap}
[data-testid="stFormSubmitButton"] button:active{transform:translateY(1px)}

.cdc{font-family:var(--body);color:var(--ink)}
.cdc a{color:var(--accent);text-decoration:none}
.cdc a:hover{text-decoration:underline;text-underline-offset:3px}
.cdc h1,.cdc h2,.cdc h3{font-family:var(--display);letter-spacing:-0.015em;color:var(--ink);margin:0;padding:0}
.cdc p{margin:0}
.cdc dl{padding:0}.cdc dt,.cdc dd{margin:0}

.cdc-nav{display:flex;justify-content:space-between;align-items:center;height:60px;
  border-bottom:1px solid var(--rule);margin-bottom:2.5rem;gap:1rem}
.cdc-brand{font-family:var(--display);font-weight:700;font-size:1.1rem;color:var(--ink)!important}
.cdc-nav a.cdc-brand:hover{text-decoration:none}
.cdc-navlink{font-size:0.95rem;white-space:nowrap}

.cdc-hero h1{font-weight:700;font-size:clamp(2rem,5.4vw,2.85rem);line-height:1.08;margin-bottom:0.85rem}
.cdc-hero p{font-size:1.1rem;line-height:1.55;color:var(--ink2);max-width:50ch;margin-bottom:0.5rem}
.cdc-help{font-size:0.9rem;color:var(--muted);line-height:1.5}
.cdc-error{color:var(--bad);font-size:0.95rem;line-height:1.5}

.cdc-intro{margin-top:2.75rem}
.cdc-intro h2{font-size:1.2rem;margin-bottom:1.1rem}
.cdc-intro dl{display:grid;grid-template-columns:11rem 1fr;gap:0.9rem 1.5rem;margin:0}
.cdc-intro dt{font-weight:600}
.cdc-intro dd{margin:0;color:var(--ink2);line-height:1.5}

.cdc-video{display:grid;grid-template-columns:176px 1fr;gap:1.1rem;align-items:center;margin:2.5rem 0 1.25rem}
.cdc-video img{width:176px;aspect-ratio:16/9;object-fit:cover;border-radius:10px;background:var(--chip);display:block}
.cdc-vtitle{font-family:var(--body)!important;font-weight:600;font-size:1.15rem!important;line-height:1.35;
  letter-spacing:0!important;margin-bottom:0.3rem!important}
.cdc-vmeta{color:var(--ink2);font-size:0.95rem}
.cdc-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;padding:1rem 0;
  border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);margin-bottom:1.75rem}
.cdc-stats b{display:block;font-size:1.3rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.25}
.cdc-stats span{color:var(--ink2);font-size:0.875rem}

.cdc-notice{background:var(--chip);border-radius:10px;padding:0.8rem 1rem;font-size:0.95rem;
  color:var(--ink2);line-height:1.5;margin-bottom:0.75rem}

.cdc-verdict{--state:var(--typ);background:var(--panel);border:1px solid var(--rule);
  border-left:5px solid var(--state);border-radius:10px;padding:1.5rem 1.5rem 1.3rem;margin-bottom:0.5rem}
.cdc-verdict.high{--state:var(--high)}.cdc-verdict.low{--state:var(--ok)}
.cdc-verdict h2{font-size:1.55rem;line-height:1.2;margin-bottom:0.65rem}
.cdc-lead{font-size:1.08rem;line-height:1.55;margin-bottom:1.2rem!important}
.cdc-lead strong{font-weight:600}
.cdc-array{display:grid;grid-template-columns:7.5rem auto;gap:0.6rem 1rem;align-items:center;margin-bottom:1rem}
.cdc-array .who{font-size:0.95rem;color:var(--ink2)}
.cdc-sq{display:inline-grid;grid-template-columns:repeat(10,20px);gap:5px}
.cdc-sq i{display:block;width:20px;height:20px;border-radius:5px;border:2px solid var(--empty);box-sizing:border-box}
.cdc-sq i.on{background:var(--state);border-color:var(--state)}
.cdc-sq.typ i.on{background:var(--typ);border-color:var(--typ)}
.cdc-key{font-size:0.875rem;color:var(--muted);line-height:1.5}

.cdc-sec{margin-top:2.5rem}
.cdc-sec h3{font-size:1.25rem;margin-bottom:0.35rem}
.cdc-sub{color:var(--ink2);font-size:0.95rem;line-height:1.5;margin-bottom:1.1rem!important}
.cdc-week{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem}
.cdc-week .when{font-size:0.9rem;color:var(--ink2);display:block;margin-bottom:0.15rem}
.cdc-week b{font-family:var(--display);font-size:2rem;font-weight:700;line-height:1.1;display:block;
  font-variant-numeric:tabular-nums}
.cdc-week .vs{font-size:0.85rem;color:var(--muted)}
.cdc-cmp{display:grid;grid-template-columns:1fr auto;gap:0.15rem 1rem;padding:0.85rem 0;border-top:1px solid var(--rule)}
.cdc-cmp .what{font-weight:500}
.cdc-cmp .lvl{font-weight:600;text-align:right}
.cdc-cmp .det{grid-column:1/-1;color:var(--muted);font-size:0.875rem}
.cdc-sec ul{margin:0;padding-left:1.15rem;line-height:1.6}
.cdc-sec li{margin-bottom:0.35rem}
.cdc-body{line-height:1.6;color:var(--ink);margin-bottom:0.75rem!important;max-width:65ch}

.cdc-details{margin-top:2.25rem;border-top:1px solid var(--rule);padding-top:1rem}
.cdc-details summary{cursor:pointer;font-weight:600;color:var(--accent);list-style:none;display:inline-flex;
  align-items:center;gap:0.45rem}
.cdc-details summary::-webkit-details-marker{display:none}
.cdc-details summary::before{content:"";width:0.45rem;height:0.45rem;border-right:2px solid currentColor;
  border-bottom:2px solid currentColor;transform:rotate(-45deg);transition:transform .15s ease}
.cdc-details[open] summary::before{transform:rotate(45deg)}
@media (prefers-reduced-motion: reduce){.cdc-details summary::before{transition:none}}
.cdc-details dl{display:grid;grid-template-columns:12rem 1fr;gap:0.55rem 1.25rem;margin:1rem 0 0;font-size:0.925rem}
.cdc-details dt{color:var(--muted)}
.cdc-details dd{margin:0;line-height:1.5;font-variant-numeric:tabular-nums}

.cdc-foot{margin-top:3.5rem;padding-top:1.25rem;border-top:1px solid var(--rule);color:var(--muted);
  font-size:0.875rem;line-height:1.6}

.cdc-skel{margin-top:2.5rem}
.cdc-skel div{background:var(--chip);border-radius:10px}
@media (prefers-reduced-motion: no-preference){.cdc-skel div{animation:cdc-pulse 1.2s ease-in-out infinite alternate}}
@keyframes cdc-pulse{from{opacity:1}to{opacity:0.55}}

@media (max-width:600px){
  .cdc-stats,.cdc-week{grid-template-columns:repeat(2,1fr);row-gap:1.1rem}
  .cdc-video{grid-template-columns:1fr}.cdc-video img{width:100%}
  .cdc-array{grid-template-columns:1fr;gap:0.3rem}.cdc-array .who{margin-top:0.4rem}
  .cdc-intro dl,.cdc-details dl{grid-template-columns:1fr;gap:0.2rem}
  .cdc-intro dd,.cdc-details dd{margin-bottom:0.7rem}
  .cdc-verdict{padding:1.2rem 1.1rem 1.1rem}
  .cdc-navlink{font-size:0.9rem}
}
</style>
"""


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Getting ready. The first visit takes a few seconds.")
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


# --------------------------------------------------------------- plain words
def count(x) -> str:
    """A count the way people say it: 110,195 or 21.4 million."""
    if x is None:
        return "Hidden"
    x = int(x)
    if x >= 1_000_000:
        v = f"{x / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{v} million"
    return f"{x:,}"


def ago(h: float) -> str:
    if h < 1:
        m = max(1, int(round(h * 60)))
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if h < 48:
        whole = int(h)
        mins = int(round((h - whole) * 60))
        if mins == 60:
            whole, mins = whole + 1, 0
        hrs = f"{whole} hour{'s' if whole != 1 else ''}"
        return f"{hrs} {mins} min ago" if whole < 6 and mins else f"{hrs} ago"
    return f"{int(h // 24)} days ago"


def in_ten(p: float) -> tuple[int, str]:
    n = int(round(p * 10))
    if p < 0.05:
        return 0, "less than 1 in 10"
    if p > 0.95:
        return 10, "almost 10 in 10"
    return n, f"about {n} in 10"


def level(pct: float) -> str:
    if pct < 20:
        return "Much lower than usual"
    if pct < 40:
        return "Lower than usual"
    if pct <= 60:
        return "About usual"
    if pct <= 80:
        return "Higher than usual"
    return "Much higher than usual"


# -------------------------------------------------------------------- blocks
def html(body: str) -> None:
    st.html(f'<div class="cdc">{body}</div>')


def squares(n: int, cls: str = "") -> str:
    return f'<span class="cdc-sq {cls}">' + "".join(
        f'<i class="{"on" if i < n else ""}"></i>' for i in range(10)) + "</span>"


def video_header(rep: dict) -> str:
    f = rep["facts"]
    vid = escape(str(f.get("video_id") or ""))
    thumb = (((rep.get("item_snippet") or {}).get("thumbnails") or {}).get("medium") or {}).get("url") \
        or f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg"
    return (
        f'<div class="cdc-video"><img src="{escape(thumb)}" alt="" loading="lazy">'
        f'<div><h2 class="cdc-vtitle">{escape(f["title"] or "Untitled video")}</h2>'
        f'<p class="cdc-vmeta">{escape(f["channel"])}, posted {ago(f["age_h"])}</p></div></div>'
        f'<div class="cdc-stats">'
        f'<div><b>{count(f["views"])}</b><span>views</span></div>'
        f'<div><b>{count(f["likes"])}</b><span>likes</span></div>'
        f'<div><b>{count(f["comments"])}</b><span>comments</span></div>'
        f'<div><b>{count(f["subscribers"])}</b><span>subscribers</span></div></div>')


def notices(rep: dict) -> str:
    return "".join(f'<p class="cdc-notice">{escape(n)}</p>' for n in rep["notes"])


def panel(title: str, text: str, cls: str = "") -> str:
    return f'<div class="cdc-verdict {cls}"><h2>{escape(title)}</h2><p class="cdc-lead">{escape(text)}</p></div>'


HEADLINE = {"low": "This video looks likely to keep getting views for a while.",
            "typical": "This video is on a normal path.",
            "high": "This video may stop getting views soon."}


def verdict(o: dict) -> str:
    n, said = in_ten(o["fade_within_2_days"])
    t, tsaid = in_ten(o["typical"])
    return (
        f'<div class="cdc-verdict {o["risk"]}"><h2>{HEADLINE[o["risk"]]}</h2>'
        f'<p class="cdc-lead">The chance it fades within two days of being posted is <strong>{said}</strong>. '
        f'For a typical video, it’s {tsaid}.</p>'
        f'<div class="cdc-array" role="img" aria-label="This video {n} in 10, typical video {t} in 10">'
        f'<span class="who">This video</span>{squares(n)}'
        f'<span class="who">Typical video</span>{squares(t, "typ")}</div>'
        f'<p class="cdc-key">Each filled square is a 1 in 10 chance of fading within two days. '
        f'Fading means its views slow to a trickle (under 5% of its busiest pace) and stay that low.</p></div>')


def week(o: dict) -> str:
    cells = "".join(
        f'<div><span class="when">{escape(s["after"])}</span><b>{s["chance"]:.0%}</b>'
        f'<span class="vs">Typical video: {t["chance"]:.0%}</span></div>'
        for s, t in zip(o["still_going"], o["typical_still_going"]))
    return (f'<section class="cdc-sec"><h3>Chance it’s still getting views</h3>'
            f'<p class="cdc-sub">Counted from when it was posted.</p><div class="cdc-week">{cells}</div></section>')


def compare(rep: dict) -> str:
    cs = rep["comparisons"]
    if not cs:
        return ""
    lo, hi = cs[0]["at_h"]
    age = f"{lo:g} hours old" if lo == hi else f"{lo:g} to {hi:g} hours old"
    group = ("channels of every size" if cs[0]["group"] == "all channel sizes"
             else f"channels with {cs[0]['group']}")
    rows = []
    for c in cs:
        p = c["percentile"]
        det = (f"Ahead of {round(p)} in 100 similar videos" if p >= 50
               else f"Behind {round(100 - p)} in 100 similar videos")
        rows.append(f'<div class="cdc-cmp"><span class="what">{escape(c["label"])}</span>'
                    f'<span class="lvl">{level(p)}</span><span class="det">{det}</span></div>')
    return (f'<section class="cdc-sec"><h3>Compared with similar videos</h3>'
            f'<p class="cdc-sub">Videos {age}, from {group}. Based on {cs[0]["videos"]:,} videos '
            f'from {cs[0]["channels"]} channels.</p>{"".join(rows)}</section>')


def todo(rep: dict) -> str:
    if not rep["todo"]:
        return ""
    items = "".join(f"<li>{escape(t)}</li>" for t in rep["todo"])
    return f'<section class="cdc-sec"><h3>What you can do</h3><ul>{items}</ul></section>'


def trust(rep: dict) -> str:
    a = rep["accuracy"]
    right, _ = in_ten(a["flag_precision"])
    base, _ = in_ten(a["base_rate"])
    if a["source"] == "test":
        tested = (f"We tested the forecast once, on {a['videos']:,} new videos from channels it had "
                  f"never seen. When it warned that a video might fade within two days, it was right "
                  f"about {right} times in 10. Across all those videos, about {base} in 10 faded that fast.")
    else:
        tested = (f"In a trial run on {a['videos']:,} videos from channels it had never seen, when it "
                  f"warned that a video might fade within two days, it was right about {right} times in 10. "
                  f"Across all those videos, about {base} in 10 faded that fast.")
    return (f'<section class="cdc-sec"><h3>How much can you trust this?</h3>'
            f'<p class="cdc-body">{tested}</p>'
            f'<p class="cdc-body">It gives chances, not an exact time. Our study found that the exact '
            f'hour a video fades can’t be predicted from public numbers.</p>'
            f'<p class="cdc-body">We also looked for habits that make videos last longer, such as more '
            f'likes or comments, video length and titles. None held up clearly in both groups of videos we '
            f'studied, so we don’t give tips we can’t back up.</p></section>')


def details(rep: dict, card: dict, fp_ok: bool) -> str:
    a = rep["accuracy"]
    c = card.get("cohort_c") or {}
    u = c.get("unseen_channels") or {}
    n = card["trained_on"]
    rows = [("Method", f"Random survival forest, pulled part of the way toward the average curve "
                       f"(weight {card['b']:.2f})"),
            ("Trained on", f"{n['videos']:,} YouTube videos from {n['channels']} channels, each followed for ten days")]
    if u:
        rows += [("Tested on", f"{u['n_videos']:,} new videos from {u['n_channels']} channels, each predicted by "
                               f"a model that had not seen its channel"),
                 ("High-risk label", f"{u['flag_precision']:.1%} faded within two days, against "
                                     f"{u['base_rate']:.1%} of all videos"),
                 ("Brier skill score", f"{u['brier_skill']:+.3f} against quoting the average rate"),
                 ("Ranking (C-index)", f"{c.get('ranking_c_index', float('nan')):.3f}")]
    elif a.get("ranking"):
        rows += [("Ranking (C-index)", f"{a['ranking']:.3f} in the trial run")]
    rows += [("Model version", f"{card['fingerprint']} "
                               + ("(matches the tested build)" if fp_ok else "(differs from the tested build)")),
             ("Where it comes from", f'The RV University study described on the <a href="{SITE}" '
                                     f'target="_blank" rel="noopener">project website</a>')]
    dl = "".join(f"<dt>{escape(k)}</dt><dd>{v if k == 'Where it comes from' else escape(v)}</dd>" for k, v in rows)
    return f'<details class="cdc-details"><summary>Technical details</summary><dl>{dl}</dl></details>'


def report_html(rep: dict, card: dict, fp_ok: bool) -> str:
    s = rep["status"]
    parts = [video_header(rep), notices(rep) if s not in ("live", "beyond_window", "no_numbers") else ""]
    if s == "ok" and rep["outlook"]:
        o = rep["outlook"]
        parts += [verdict(o), week(o), compare(rep), todo(rep), trust(rep)]
    elif s == "ok":
        parts += [panel("Forecasts are switched off.", "The forecast did not pass its test on new videos, "
                        "so this page only shows comparisons."), compare(rep)]
    elif s == "too_new":
        m = rep["minutes_until_ready"]
        parts += [panel(f"Check back in about {m} minute{'s' if m != 1 else ''}.",
                        "The forecast starts 15 minutes after a video goes up, once it has a few views to go on."),
                  compare(rep)]
    elif s == "too_old_for_odds":
        parts += [panel("This video is too old for a forecast.",
                        "Forecasts work for videos up to 13 hours old. You can still see how it compares "
                        "with videos of the same age."), compare(rep)]
    elif s == "live":
        parts += [panel("Live streams aren’t covered.", rep["notes"][0])]
    elif s == "beyond_window":
        parts += [panel("This video is more than 10 days old.", rep["notes"][0])]
    else:
        parts += [panel("There is nothing to measure yet.", rep["notes"][0])]
    if not fp_ok:
        parts.append('<p class="cdc-notice">This copy of the tool was built slightly differently from the '
                     'tested one, so its numbers may differ a little.</p>')
    parts.append(details(rep, card, fp_ok))
    return "".join(parts)


SKELETON = ('<div class="cdc-skel"><div style="height:99px;margin-bottom:1.25rem"></div>'
            '<div style="height:64px;margin-bottom:1.75rem"></div><div style="height:210px"></div></div>')


# ---------------------------------------------------------------------- page
def main() -> None:
    st.html(CSS)
    model, table, card, tips, fp_ok = load()
    n = card["trained_on"]
    html(f'<nav class="cdc-nav"><a class="cdc-brand" href="{SITE}" target="_blank" rel="noopener">'
         f'Content Death Clock</a><a class="cdc-navlink" href="{SITE}" target="_blank" rel="noopener">'
         f'How the study works</a></nav>'
         f'<header class="cdc-hero"><h1>Will this video keep getting views?</h1>'
         f'<p>Paste a YouTube link. We’ll estimate how long it keeps getting attention, using '
         f'{n["videos"]:,} videos we tracked for ten days.</p></header>')

    with st.form("check", border=False):
        left, right = st.columns([5, 1.45], vertical_alignment="bottom")
        url = left.text_input("YouTube video link", placeholder="https://www.youtube.com/watch?v=...")
        go = right.form_submit_button("Check video", type="primary", width="stretch")
    msg = st.empty()
    msg.html('<div class="cdc"><p class="cdc-help">Works on any public video or Short. Forecasts are for '
             'videos posted in the last 13 hours.</p></div>')

    out = st.empty()
    if not go:
        out.html('<div class="cdc"><section class="cdc-intro"><h2>What you’ll get</h2><dl>'
                 '<dt>The chances</dt><dd>How likely the video is to still be getting views 1, 2, 3 and 7 '
                 'days after it was posted.</dd>'
                 '<dt>A comparison</dt><dd>Its views, likes and comments next to videos of the same age '
                 'from channels of a similar size.</dd>'
                 '<dt>A next step</dt><dd>A plain suggestion, based on what happened to the videos we '
                 'tracked.</dd></dl></section></div>')
    else:
        vid = parse_video_id(url)
        error = None
        if not vid:
            error = ToolError("bad_link").message
        elif not session_allows():
            error = "You’ve checked a lot of videos in the last hour. Please wait a while and try again."
        else:
            out.html(f'<div class="cdc">{SKELETON}</div>')
            try:
                item, subs, fetched_at = cached_lookup(vid)
                rep = build_report(item, subs, fetched_at, model, table, card, tips)
                rep["item_snippet"] = item.get("snippet") or {}
                out.html(f'<div class="cdc">{report_html(rep, card, fp_ok)}</div>')
            except ToolError as e:
                error = e.message
        if error:
            out.empty()
            msg.html(f'<div class="cdc"><p class="cdc-error" role="alert">{escape(error)}</p></div>')

    html(f'<footer class="cdc-foot">A student research project at RV University, for Research Methodology '
         f'and Big Data Analytics, by {TEAM}. We don’t keep the links you check.</footer>')


main()
