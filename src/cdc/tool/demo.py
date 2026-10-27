"""Fresh videos to try the tool on, from the latest collected data.

The outlook only works on videos 15 minutes to 13 hours old, so demo links go
stale within hours. This lists the freshest videos the collector has seen, with
a preview of what the tool will say about each, computed from the collector's
latest reading (the live page re-reads YouTube, so its numbers will have moved
a little).

These channels are the study's own, so the deployed model has seen them: its
calls on them are sharper than on a new channel. Try one video from a channel
outside the study too.

Usage (pull the latest collection first)::

    git pull
    python -m cdc.transform.silver
    python -m cdc.tool.demo
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from cdc.tool import table as T
from cdc.tool.features import FEATURES, MAX_AGE_H, MIN_AGE_H, from_silver
from cdc.tool.model import FAST_H, InstantModel, col


def candidates(now: pd.Timestamp | None = None) -> pd.DataFrame:
    from cdc.config import path_for
    from cdc.storage.bronze import iter_bronze

    now = now or pd.Timestamp.now(tz="UTC")
    sd = path_for("silver_dir")
    posts = pd.read_parquet(sd / "posts.parquet")
    posts = posts[(posts["platform"] == "youtube") & ~posts["over_creator_daily_cap"].astype(bool)]
    age_now = (now - posts["published_at"]).dt.total_seconds() / 3600
    posts = posts[(age_now >= MIN_AGE_H) & (age_now <= MAX_AGE_H - 1.0)].assign(age_now=age_now)
    if posts.empty:
        return pd.DataFrame()
    s = pd.read_parquet(sd / "snapshots.parquet")
    last = (s[s["post_id"].isin(posts["post_id"]) & s["views"].notna()]
            .sort_values("age_hours").groupby("post_id").last())
    # A preview from a reading taken minutes after upload says nothing; the
    # model starts at MIN_AGE_H, so the latest reading must be at least that old.
    last = last[last["age_hours"] >= MIN_AGE_H]
    posts = posts[posts["post_id"].isin(last.index)]
    titles = {r["post_id"]: r.get("title") for r in iter_bronze("posts", platform="youtube")
              if r.get("post_id") in set(posts["post_id"])}

    card = json.loads((T.ASSETS / "model_card.json").read_text(encoding="utf-8"))
    model = InstantModel(b=card["b"]).fit(T.load_table())
    feats = pd.DataFrame([from_silver(last.loc[p, "age_hours"], last.loc[p, "views"], last.loc[p, "likes"],
                                      last.loc[p, "comments"], r._asdict())
                          for p, r in zip(posts["post_id"], posts.itertuples(index=False))], columns=FEATURES)
    p48 = 1 - model.survival(feats)[:, col(FAST_H)]
    th = card["thresholds_p48"]
    return pd.DataFrame({
        "link": "https://youtu.be/" + posts["post_id"].to_numpy(),
        "channel": posts["creator_handle"].to_numpy(),
        "title": [str(titles.get(p) or "")[:60] for p in posts["post_id"]],
        "age_now_h": posts["age_now"].round(1).to_numpy(),
        "views_at_last_reading": last.loc[posts["post_id"], "views"].astype(int).to_numpy(),
        "fade_within_2_days": np.round(p48, 2),
        "preview": np.select([p48 >= th["high"], p48 <= th["low"]], ["HIGH RISK", "likely to keep going"], "about average"),
    }).sort_values("fade_within_2_days", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    c = candidates()
    if c.empty:
        print("No video in the collected data is between 15 minutes and 12 hours old right now. "
              "Pull the latest collection and rebuild silver, or try a channel you know just posted.")
        return 0
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(c.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
