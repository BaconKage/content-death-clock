"""A small synthetic training table for the tool's tests: fast, and with a known signal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cdc.tool.features import FEATURES, MARKS, fade_mark, instant_features


def synthetic_rows(n_videos: int = 240, n_channels: int = 24, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subs = 10 ** rng.uniform(2.5, 7.5, n_channels)
    rows = []
    for v in range(n_videos):
        c = v % n_channels
        speed = 10 ** rng.normal(np.log10(subs[c]) - 2.5, 0.5)       # views/hour early on
        # stronger starters fade later: the signal the model should find
        t = float(np.clip(rng.lognormal(np.log(60) + 0.6 * (np.log10(speed) - np.log10(subs[c]) + 2.5), 0.6), 1, 300))
        faded = t <= 168
        fade = fade_mark(t if faded else 240.0)
        for m in MARKS:
            age = m + rng.uniform(0.0, 0.5)
            views = speed * age ** 0.6
            f = instant_features(age, int(views), int(views * 0.04), int(views * 0.004), int(subs[c]),
                                 rng.integers(30, 1200), int(rng.integers(0, 24)), int(rng.integers(20, 90)))
            rows.append({"post_id": f"v{v:04d}", "creator_id": f"c{c:03d}", "cohort": "A",
                         "video_key": f"{v:06d}", "mark": m, "age_hours": age, **f,
                         "fade_h": fade, "faded": bool(faded)})
    return pd.DataFrame(rows)


def as_table(rows: pd.DataFrame) -> pd.DataFrame:
    t = rows.copy()
    t["video"] = t["video_key"].astype(int)
    t["channel"] = t["creator_id"].str[1:].astype(int)
    return t[["cohort", "video", "channel", "mark", *FEATURES, "fade_h", "faded"]]
