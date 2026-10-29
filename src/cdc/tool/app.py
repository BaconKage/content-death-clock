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
                   page_icon=":material/hourglass_top:", layout="wide")

CSS = """
<style>
:root{
  --paper:#f5f0e6;--panel:#fffdf8;--ink:#25232b;--ink2:#514d58;--muted:#68636d;--rule:#e5ded2;--chip:#eee8dc;
  --accent:#7138c8;--accent-dark:#5925ad;--high:#bd5a22;--ok:#258052;--typ:#77717e;--empty:#d8d0c5;--bad:#a9322c;
  --display:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif;
  --body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="InputInstructions"]{display:none!important}
[data-testid="stMainBlockContainer"],.block-container{max-width:1600px;padding:1.2rem clamp(1rem,4vw,4rem) 3rem}
[data-testid="stSidebar"]{background:#292631;border-right:0}
[data-testid="stSidebar"] [data-testid="stMarkdown"]{color:#f7f3ec}
[data-testid="stSidebar"] a{color:#e8d5ff!important;text-decoration:none!important}
[data-testid="stSidebar"] a:hover{color:#fff!important}
[data-testid="stTextInput"] label p{font-weight:600;font-size:0.95rem;color:var(--ink)}
[data-testid="stTextInput"] input{height:3.25rem;font-size:1rem;background:var(--panel);border-color:var(--rule);color:var(--ink)}
[data-testid="stFormSubmitButton"] button{height:3.25rem;font-weight:700;font-size:1rem;white-space:nowrap;background:var(--accent);border-color:var(--accent);color:#fff}
[data-testid="stFormSubmitButton"] button:hover{background:var(--accent-dark);border-color:var(--accent-dark);color:#fff}
[data-testid="stFormSubmitButton"] button:active{transform:translateY(1px)}
[data-testid="stAppViewContainer"]{background:var(--paper)}
div[data-testid="stForm"]{background:var(--panel);border:1px solid var(--rule);border-radius:18px;padding:1.1rem 1.25rem;box-shadow:0 12px 36px #3627190d}
.cdc-side{display:flex;flex-direction:column;min-height:calc(100vh - 2rem);padding:.6rem 1rem;color:#f7f3ec;font-family:var(--body)}
.cdc-side-brand{font-family:var(--display);font-weight:700;font-size:1.05rem;color:#fff!important;margin:.35rem 0 2.6rem}
.cdc-side-label{font-size:.7rem;letter-spacing:.14em;font-weight:700;color:#aaa4b5;margin-bottom:.7rem!important}
.cdc-side-link{display:block;border-radius:9px;padding:.7rem .65rem;color:#e9e4ed!important;font-size:.92rem}
.cdc-side-link:hover{background:#ffffff16;text-decoration:none!important}
.cdc-side-note{margin-top:auto;border-top:1px solid #ffffff26;padding-top:1rem;color:#c6c1cd;font-size:.8rem;line-height:1.55}
.cdc-eyebrow{font-size:.75rem;font-weight:700;letter-spacing:.14em;color:var(--accent)}
.cdc-hero-copy h1{font-family:var(--display);font-weight:700;font-size:clamp(2.3rem,5vw,4rem);line-height:1.03;letter-spacing:-.02em;margin:0 0 .85rem;max-width:13ch}
.cdc-hero-copy p{font-size:1.08rem;line-height:1.6;color:var(--ink2);max-width:55ch;margin:0 0 1rem}

.cdc{font-family:var(--body);color:var(--ink)}
.cdc a{color:var(--accent);text-decoration:none}
.cdc a:hover{text-decoration:underline;text-underline-offset:3px}
.cdc h1,.cdc h2,.cdc h3{font-family:var(--display);letter-spacing:-0.015em;color:var(--ink);margin:0;padding:0}
.cdc p{margin:0}
.cdc dl{padding:0}.cdc dt,.cdc dd{margin:0}

.cdc-nav{display:flex;justify-content:space-between;align-items:center;height:48px;
  border-bottom:1px solid var(--rule);margin-bottom:1.6rem;gap:1rem}
.cdc-brand{font-family:var(--display);font-weight:700;font-size:1.1rem;color:var(--ink)!important}
.cdc-nav a.cdc-brand:hover{text-decoration:none}
.cdc-navlink{font-size:0.95rem;white-space:nowrap}

.cdc-hero h1{font-weight:700;font-size:clamp(2.3rem,5vw,4rem);line-height:1.03;margin-bottom:0.85rem;max-width:13ch}
.cdc-hero p{font-size:1.08rem;line-height:1.6;color:var(--ink2);max-width:55ch;margin-bottom:0.5rem}
.cdc-form{background:var(--panel);border:1px solid var(--rule);border-radius:18px;padding:1.2rem 1.35rem;box-shadow:0 12px 36px #3627190d}
.cdc-form .cdc-help{margin-top:.5rem}
.cdc-visual{min-height:205px;background:#eee6f6;border:1px solid #dfd1ee;border-radius:18px;padding:1.35rem;position:relative;overflow:hidden}
.cdc-visual h2{font-size:1.15rem;margin-bottom:.25rem}.cdc-visual p{font-size:.9rem;color:#5e5667;margin:0}
.cdc-chart{height:82px;margin-top:1.25rem;position:relative;border-bottom:2px solid #d8cde5}
.cdc-chart i{position:absolute;display:block;height:4px;background:#7138c8;border-radius:8px;transform-origin:left center}
.cdc-chart i:first-child{left:0;top:8px;width:22%;transform:rotate(7deg);background:#bd5a22}
.cdc-chart i:nth-child(2){left:22%;top:11px;width:20%;transform:rotate(15deg)}
.cdc-chart i:nth-child(3){left:42%;top:17px;width:20%;transform:rotate(4deg)}
.cdc-chart i:nth-child(4){left:62%;top:19px;width:19%;transform:rotate(13deg)}
.cdc-chart i:nth-child(5){left:81%;top:24px;width:19%;transform:rotate(5deg)}
.cdc-hero-grid{display:flex;flex-direction:column;gap:1rem}
.cdc-help{font-size:0.9rem;color:var(--muted);line-height:1.5}
.cdc-error{color:var(--bad);font-size:0.95rem;line-height:1.5}

.cdc-intro{margin-top:2rem}
.cdc-intro h2{font-size:1.35rem;margin-bottom:1rem}
.cdc-intro dl{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin:0}
.cdc-intro dl>div{background:var(--panel);border:1px solid var(--rule);border-radius:14px;padding:1rem 1.1rem;box-shadow:0 5px 18px #36271908}
.cdc-intro dt{font-weight:700;margin-bottom:.4rem}
.cdc-intro dd{margin:0;color:var(--ink2);line-height:1.5;font-size:.93rem}

.cdc-results{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,.85fr);gap:1rem;align-items:start;margin-top:1rem}
.cdc-results-main,.cdc-results-side{display:flex;flex-direction:column;gap:1rem;min-width:0}
.cdc-card{background:var(--panel);border:1px solid var(--rule);border-radius:16px;padding:1.2rem 1.35rem;box-shadow:0 5px 18px #36271908;margin:0!important}
.cdc-card .cdc-sec{margin-top:0}
.cdc-card .cdc-sec h3{font-size:1.15rem}
.cdc-card .cdc-details{margin:0;border:0;padding:0}
.cdc-video{display:grid;grid-template-columns:minmax(130px,220px) 1fr;gap:1.2rem;align-items:center;margin:0 0 1rem}
.cdc-video img{width:176px;aspect-ratio:16/9;object-fit:cover;border-radius:10px;background:var(--chip);display:block}
.cdc-vtitle{font-family:var(--body)!important;font-weight:600;font-size:1.15rem!important;line-height:1.35;
  letter-spacing:0!important;margin-bottom:0.3rem!important}
.cdc-vmeta{color:var(--ink2);font-size:0.95rem}
.cdc-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;padding:1rem 0 0;
  border-top:1px solid var(--rule);margin:0}
.cdc-stats b{display:block;font-size:1.3rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.25}
.cdc-stats span{color:var(--ink2);font-size:0.875rem}

.cdc-notice{background:#f8eadf;border:1px solid #f0d2bd;border-radius:12px;padding:0.8rem 1rem;font-size:0.95rem;
  color:var(--ink2);line-height:1.5;margin-bottom:0.75rem}

.cdc-verdict{--state:var(--typ);background:var(--panel);border:1px solid var(--rule);
  border-left:5px solid var(--state);border-radius:16px;padding:1.35rem;margin:0!important;box-shadow:0 5px 18px #36271908}
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

.cdc-sec{margin-top:1rem}
.cdc-sec h3{font-size:1.25rem;margin-bottom:0.35rem}
.cdc-sub{color:var(--ink2);font-size:0.95rem;line-height:1.5;margin-bottom:1.1rem!important}
.cdc-week{display:grid;grid-template-columns:repeat(2,1fr);gap:.85rem}
.cdc-week>div{background:var(--paper);border:1px solid var(--rule);border-radius:12px;padding:.8rem}
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

.cdc-foot{margin-top:3.5rem;color:var(--muted);font-size:0.875rem;line-height:1.6;text-align:center}

.cdc-skel{margin-top:2.5rem}
.cdc-skel div{background:var(--chip);border-radius:10px}
@media (prefers-reduced-motion: no-preference){.cdc-skel div{animation:cdc-pulse 1.2s ease-in-out infinite alternate}}
@keyframes cdc-pulse{from{opacity:1}to{opacity:0.55}}
@keyframes cdc-enter{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.cdc-report-enter{animation:cdc-enter .28s ease-out both}
@media (prefers-reduced-motion: reduce){.cdc-report-enter{animation:none}}

@media (max-width:600px){
  [data-testid="stMainBlockContainer"],.block-container{padding:1rem 1rem 2rem}
  .cdc-hero,.cdc-results{grid-template-columns:1fr}
  .cdc-hero h1{font-size:2.45rem}
  .cdc-hero-copy h1{font-size:2.45rem}
  .cdc-side{min-height:0}.cdc-side-brand{margin-bottom:1.2rem}.cdc-side-note{margin-top:1rem}
  .cdc-stats,.cdc-week{grid-template-columns:repeat(2,1fr);row-gap:.8rem}
  .cdc-video{grid-template-columns:1fr}.cdc-video img{width:100%}
  .cdc-array{grid-template-columns:1fr;gap:0.3rem}.cdc-array .who{margin-top:0.4rem}
  .cdc-intro dl,.cdc-details dl{grid-template-columns:1fr;gap:.65rem}
  .cdc-intro dd,.cdc-details dd{margin-bottom:0.7rem}
  .cdc-verdict{padding:1.2rem 1.1rem 1.1rem}
  .cdc-navlink{font-size:0.9rem}
}
@media (min-width:601px) and (max-width:950px){.cdc-hero{grid-template-columns:1fr}.cdc-results{grid-template-columns:1fr}}
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


def begin_report() -> None:
    """Switch to the report view before Streamlit reruns after form submission."""
    st.session_state["cdc_report_view"] = True
    st.session_state["cdc_lookup_pending"] = True
    st.session_state["cdc_report_error"] = None


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
    head = video_header(rep)
    notice = notices(rep) if s not in ("live", "beyond_window", "no_numbers") else ""
    main, side = [], []
    if s == "ok" and rep["outlook"]:
        o = rep["outlook"]
        main += [verdict(o), week(o)]
        side += [compare(rep), todo(rep), trust(rep)]
    elif s == "ok":
        main += [panel("Forecasts are switched off.", "The forecast did not pass its test on new videos, "
                       "so this page only shows comparisons.")]
        side += [compare(rep)]
    elif s == "too_new":
        m = rep["minutes_until_ready"]
        main += [panel(f"Check back in about {m} minute{'s' if m != 1 else ''}.",
                       "The forecast starts 15 minutes after a video goes up, once it has a few views to go on.")]
        side += [compare(rep)]
    elif s == "too_old_for_odds":
        main += [panel("This video is too old for a forecast.",
                       "Forecasts work for videos up to 13 hours old. You can still see how it compares "
                       "with videos of the same age.")]
        side += [compare(rep)]
    elif s == "live":
        main += [panel("Live streams aren’t covered.", rep["notes"][0])]
    elif s == "beyond_window":
        main += [panel("This video is more than 10 days old.", rep["notes"][0])]
    else:
        main += [panel("There is nothing to measure yet.", rep["notes"][0])]
    if not fp_ok:
        side.append('<p class="cdc-notice">This copy of the tool was built slightly differently from the '
                    'tested one, so its numbers may differ a little.</p>')
    side.append(details(rep, card, fp_ok))
    wrap = lambda values: "".join(f'<div class="cdc-card">{v}</div>' for v in values if v)
    return (f'<div id="results" class="cdc-results">'
            f'<div class="cdc-results-main"><div class="cdc-card">{head}</div>{notice}{wrap(main)}</div>'
            f'<div class="cdc-results-side">{wrap(side)}</div></div>')


SKELETON = ('<div class="cdc-skel"><div style="height:99px;margin-bottom:1.25rem"></div>'
            '<div style="height:64px;margin-bottom:1.75rem"></div><div style="height:210px"></div></div>')


# ---------------------------------------------------------------------- page
def main() -> None:
    st.html(CSS)
    model, table, card, tips, fp_ok = load()
    report_view = st.session_state.get("cdc_report_view", False)
    if report_view:
        st.html('<style>[data-testid="stSidebar"]{display:none!important}'
                '[data-testid="stMainBlockContainer"],.block-container{max-width:1600px;margin-inline:auto}'
                'div[data-testid="stForm"]{max-width:100%;padding:.55rem .8rem;animation:cdc-enter .28s ease-out both}'
                '</style>')
        edge_left, center, edge_right = st.columns([1, 8, 1])
        with center:
            with st.form("check", border=False):
                input_col, button_col = st.columns([5, 1.8], vertical_alignment="bottom")
                input_col.text_input("YouTube video link", placeholder="https://www.youtube.com/watch?v=...",
                                     label_visibility="collapsed", key="cdc_url")
                button_col.form_submit_button("Check video", type="primary", width="stretch",
                                              on_click=begin_report)
        url = st.session_state.get("cdc_url", "")
        msg = st.empty()
        out = st.empty()
    else:
        st.sidebar.html(f'<div class="cdc-side"><a class="cdc-side-brand" href="{SITE}" target="_blank" rel="noopener">'
                        '◷ &nbsp; Content Death Clock</a><p class="cdc-side-label">WORKSPACE</p>'
                        '<a class="cdc-side-link" href="#check-video">⌕ &nbsp; Check a video</a>'
                        '<a class="cdc-side-link" href="#what-you-get">▦ &nbsp; What you’ll get</a>'
                        f'<a class="cdc-side-link" href="{SITE}" target="_blank" rel="noopener">↗ &nbsp; Study website</a>'
                        '<div class="cdc-side-note">A research tool for understanding how video attention changes over time.</div></div>')
        html('<nav class="cdc-nav"><span class="cdc-eyebrow">VIDEO ATTENTION FORECAST</span>'
             f'<a class="cdc-navlink" href="{SITE}" target="_blank" rel="noopener">About the study ↗</a></nav>')
        left, right = st.columns([1.1, .9], gap="large", vertical_alignment="center")
        with left:
            html('<header class="cdc-hero-copy" id="check-video"><h1>Will this video keep getting views?</h1></header>')
            with st.form("check", border=False):
                url = st.text_input("YouTube video link", placeholder="https://www.youtube.com/watch?v=...",
                                    key="cdc_url")
                st.form_submit_button("Check video", type="primary", width="stretch", on_click=begin_report)
            html('<p class="cdc-help">Works with public YouTube videos and Shorts. Forecasts are available for videos posted in the last 13 hours.</p>')
        with right:
            html('<div class="cdc-visual"><h2>Attention changes over time</h2>'
                 '<p>Every video has its own pace. We compare it with patterns seen across the study.</p>'
                 '<div class="cdc-chart" role="img" aria-label="Illustrative attention curve gradually fading over time">'
                 '<i></i><i></i><i></i><i></i><i></i></div></div>')
        out = st.empty()
        out.html('<div class="cdc"><section class="cdc-intro" id="what-you-get"><h2>What you’ll get</h2><dl>'
                 '<div><dt>The chances</dt><dd>How likely the video is to still be getting views 1, 2, 3 and 7 '
                 'days after it was posted.</dd></div>'
                 '<div><dt>A comparison</dt><dd>Its views, likes and comments next to videos of the same age '
                 'from channels of a similar size.</dd></div>'
                 '<div><dt>A next step</dt><dd>A plain suggestion, based on what happened to the videos we '
                 'tracked.</dd></div></dl></section></div>')
        html(f'<footer class="cdc-foot">A project by {TEAM}</footer>')
    if st.session_state.get("cdc_report_view", False):
        pending = st.session_state.pop("cdc_lookup_pending", False)
        error = st.session_state.get("cdc_report_error")
        if pending:
            error = None
            vid = parse_video_id(url)
            if not vid:
                error = ToolError("bad_link").message
            elif not session_allows():
                error = "You’ve checked a lot of videos in the last hour. Please wait a while and try again."
            else:
                out.html(f'<div class="cdc-report-enter"><div class="cdc">{SKELETON}</div></div>')
                try:
                    item, subs, fetched_at = cached_lookup(vid)
                    rep = build_report(item, subs, fetched_at, model, table, card, tips)
                    rep["item_snippet"] = item.get("snippet") or {}
                    st.session_state["cdc_report_html"] = report_html(rep, card, fp_ok)
                except ToolError as e:
                    error = e.message
            st.session_state["cdc_report_error"] = error
        if error:
            msg.html(f'<div class="cdc"><p class="cdc-error" role="alert">{escape(error)}</p></div>')
        if st.session_state.get("cdc_report_html"):
            out.html(f'<div class="cdc-report-enter cdc">{st.session_state["cdc_report_html"]}</div>')


main()
