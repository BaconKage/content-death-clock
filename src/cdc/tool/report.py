"""One lookup in, the report the page shows out. No I/O, so every branch is testable.

What the report says depends on the video's age, because that is what the data
supports:

* under ``MIN_AGE_H``: too new; come back at the first checkpoint
* ``MIN_AGE_H`` to ``MAX_AGE_H``: the outlook (odds), comparisons, what to do
* up to the last mark (240h): comparisons only. The outlook is not given for
  older videos; it was built and tested on the first half-day only
* older: the plain numbers, nothing modelled

The odds are hidden altogether if the model card records a failed Cohort C test.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from cdc.tool.features import FEATURES, MARKS, MAX_AGE_H, MIN_AGE_H, from_api
from cdc.tool.model import FAST_H, InstantModel, col

BANDS = ((None, 4.0, "under 10,000 subscribers"), (4.0, 6.0, "10,000 to 1 million subscribers"),
         (6.0, None, "over 1 million subscribers"))
MIN_REF_ROWS, MIN_REF_CHANNELS = 100, 5
LAST_MARK = MARKS[-1]
DAYS = ((24.0, "1 day"), (48.0, "2 days"), (72.0, "3 days"), (168.0, "7 days"))


def band_of(log_subs: float) -> tuple[float | None, float | None, str] | None:
    if not (isinstance(log_subs, float) and math.isfinite(log_subs)):
        return None
    for lo, hi, name in BANDS:
        if (lo is None or log_subs >= lo) and (hi is None or log_subs < hi):
            return lo, hi, name
    return None


def _bracket(age_h: float) -> tuple[float, float]:
    a = min(max(age_h, MARKS[0]), LAST_MARK)
    lo = max(m for m in MARKS if m <= a)
    hi = min(m for m in MARKS if m >= a)
    return lo, hi


def percentile(table: pd.DataFrame, feature: str, value: float, age_h: float,
               log_subs: float) -> dict[str, Any] | None:
    """Where a value sits among videos of a similar age and channel size, 0-100.

    Taken at the two scheduled marks either side of the video's age and
    interpolated on a log-age scale, so a 9-hour-old video is compared with
    6- and 12-hour readings rather than snapped to one of them.
    """
    if not (isinstance(value, float) and math.isfinite(value)):
        return None
    band = band_of(log_subs)
    lo, hi = _bracket(age_h)
    pcts, n_rows, chans, used_band = [], 0, set(), band[2] if band else "all channel sizes"
    for m in (lo, hi):
        ref = table[(table["mark"] == m) & table[feature].notna()]
        if band:
            b = ref[(ref["log_subscribers"] >= (band[0] if band[0] is not None else -np.inf))
                    & (ref["log_subscribers"] < (band[1] if band[1] is not None else np.inf))]
            if len(b) >= MIN_REF_ROWS and b["channel"].nunique() >= MIN_REF_CHANNELS:
                ref = b
            else:
                used_band = "all channel sizes"
        x = ref[feature].to_numpy(float)
        pcts.append(100.0 * (np.mean(x < value) + 0.5 * np.mean(x == value)))
        n_rows = max(n_rows, len(ref))
        chans |= set(ref["channel"])
    if lo == hi:
        pct = pcts[0]
    else:
        w = (math.log(min(max(age_h, lo), hi)) - math.log(lo)) / (math.log(hi) - math.log(lo))
        pct = (1 - w) * pcts[0] + w * pcts[1]
    return {"percentile": float(pct), "videos": int(n_rows), "channels": int(len(chans)),
            "group": used_band, "at_h": (lo, hi)}


def _odds_enabled(card: dict[str, Any]) -> bool:
    c = card.get("cohort_c")
    return c is None or bool(c.get("passes"))


def accuracy_text(card: dict[str, Any]) -> dict[str, Any]:
    c = card.get("cohort_c")
    src = c if c else (card.get("rehearsal_A_to_B") or {})
    p = c["unseen_channels"] if c else src["primary"]
    return {"source": "test" if c else "rehearsal", "videos": p["n_videos"], "channels": p["n_channels"],
            "flag_precision": p["flag_precision"], "base_rate": p["base_rate"],
            "brier_skill": p["brier_skill"], "ranking": src.get("ranking_c_index"),
            "passed": bool(c.get("passes")) if c else None}


def build_report(item: dict[str, Any], subscribers: Any, now: datetime, model: InstantModel,
                 table: pd.DataFrame, card: dict[str, Any], tips: dict[str, Any]) -> dict[str, Any]:
    feats, facts = from_api(item, subscribers, now)
    sn = item.get("snippet") or {}
    age = facts["age_h"]
    rep: dict[str, Any] = {"facts": facts, "notes": [], "outlook": None, "comparisons": [],
                           "todo": [], "accuracy": accuracy_text(card), "tips": tips}

    live = sn.get("liveBroadcastContent")
    if live in ("live", "upcoming") or age <= 0:
        rep["status"] = "live"
        rep["notes"].append("This is a live stream or a premiere that hasn't started. The study "
                            "followed ordinary uploads, so the tool can't say anything about it yet.")
        return rep
    if not feats:
        rep["status"] = "no_numbers"
        rep["notes"].append("YouTube didn't report a view count for this video, so there is nothing to measure.")
        return rep
    if age > LAST_MARK + 1.0:
        rep["status"] = "beyond_window"
        rep["notes"].append(f"This video is {age / 24:.0f} days old. The study followed videos for "
                            "their first 10 days, so beyond that there is nothing to compare it with.")
        return rep

    if subscribers is None:
        rep["notes"].append("This channel hides its subscriber count. The model used a typical "
                            "channel size in its place, and comparisons are against all channel sizes.")
    if facts["likes"] is None:
        rep["notes"].append("Likes are hidden on this video, so the model used a typical value.")
    if facts["comments"] is None:
        rep["notes"].append("Comments are turned off on this video, so the model used a typical value.")
    ls = feats.get("log_subscribers", float("nan"))
    tl = table.groupby("channel")["log_subscribers"].first()
    if math.isfinite(ls) and (ls < tl.min() or ls > tl.max()):
        rep["notes"].append("Your channel is outside the size range of the channels we studied, "
                            "so treat these numbers with extra caution.")

    size_feature = "log_views_per_sub" if math.isfinite(feats.get("log_views_per_sub", float("nan"))) else "log_views"
    for f, label in ((size_feature, "views per subscriber" if size_feature == "log_views_per_sub" else "views"),
                     ("log_like_rate", "likes per view"), ("log_comment_rate", "comments per view")):
        pc = percentile(table, f, feats.get(f, float("nan")), age, ls)
        if pc:
            rep["comparisons"].append({"metric": label, **pc})

    if age < MIN_AGE_H:
        rep["status"] = "too_new"
        rep["minutes_until_ready"] = int(math.ceil((MIN_AGE_H - age) * 60))
        return rep
    if age > MAX_AGE_H:
        rep["status"] = "too_old_for_odds"
        return rep

    rep["status"] = "ok"
    if not _odds_enabled(card):
        rep["notes"].append("The outlook is switched off: it did not pass its test on new videos.")
        return rep
    s = model.survival(pd.DataFrame([feats], columns=FEATURES))[0]
    p48 = float(1 - s[col(FAST_H)])
    typical = float(1 - model.average_[col(FAST_H)])
    th = card["thresholds_p48"]
    risk = "high" if p48 >= th["high"] else "low" if p48 <= th["low"] else "typical"
    rep["outlook"] = {
        "fade_within_2_days": p48, "typical": typical, "risk": risk,
        "still_going": [{"after": label, "h": h, "chance": float(s[col(h)])} for h, label in DAYS],
        "typical_still_going": [{"after": label, "h": h, "chance": float(model.average_[col(h)])}
                                for h, label in DAYS],
    }
    fx = tips.get("facts", {})
    early = (f"In our data, {fx.get('share_fastest_in_first_6h', 0):.0%} of videos had their fastest "
             f"views within six hours of posting, and a typical video got about "
             f"{fx.get('median_share_of_10_day_views_by_24h', 0):.0%} of its first ten days' views in its first day.")
    if risk == "high":
        rep["todo"] = ["This video looks more likely than most to stop getting attention within two days.",
                       early, "If you plan to share or promote it, doing it soon reaches it while it still has momentum."]
    elif risk == "low":
        rep["todo"] = ["This video looks more likely than most to keep getting views past its second day.",
                       early]
    else:
        rep["todo"] = ["This video's outlook is about average for its age.", early,
                       "If you plan to share or promote it, earlier is when it still has momentum."]
    return rep
