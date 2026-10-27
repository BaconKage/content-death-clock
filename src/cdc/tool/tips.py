"""What tended to make videos last, kept only if it held in Cohort A *and* Cohort B.

**Within a channel, not across channels.** Across channels, every pattern is
tangled up with channel size and audience: big channels' videos differ from
small channels' in everything at once. A creator wants to know what
distinguishes *their own* longer-lived videos. So each candidate is tested
within channels: for every pair of videos from the same channel whose fade
order is known, did the one with more of the factor last longer? The share of
pairs where it did is 0.5 if the factor is unrelated to lifespan.

**Fixed list, both cohorts, both directions reported.** The candidates are
declared below and in amendment 8 before any was computed. A pattern becomes a
tip only if its 95% creator-bootstrap interval excludes 0.5, on the same side,
in Cohort A and in Cohort B separately. Every candidate's numbers are written to
``assets/tips.json`` whether it passed or not. A single-cohort pattern is
exactly what the holdout caught once already (channel size), so one cohort is
not enough.

**What a tip is not.** These are associations in observed videos. Nothing here
shows that changing the factor would make a given video last longer, and the
wording in the app says so.

Usage::

    python -m cdc.tool.tips
"""
from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pandas as pd

from cdc.tool import table as T
from cdc.tool.model import SEED

TIPS_PATH = T.ASSETS / "tips.json"
AT_MARK = 6.0
N_BOOT = 2000

# factor -> (plain name, sentence if more goes with longer life, sentence if less does)
CANDIDATES: dict[str, tuple[str, str, str]] = {
    "log_like_rate": ("likes per view",
                      "Videos with more likes per view kept getting views for longer than the same channel's other videos.",
                      "Videos with fewer likes per view kept getting views for longer than the same channel's other videos."),
    "log_comment_rate": ("comments per view",
                         "Videos with more comments per view kept getting views for longer than the same channel's other videos.",
                         "Videos with fewer comments per view kept getting views for longer than the same channel's other videos."),
    "log_views_per_sub": ("views per subscriber in the first six hours",
                          "Videos that drew more views per subscriber in their first six hours kept going longer than the same channel's other videos.",
                          "Videos that drew fewer views per subscriber in their first six hours kept going longer than the same channel's other videos."),
    "log_duration_sec": ("video length",
                         "Longer videos kept getting views for longer than the same channel's shorter ones.",
                         "Shorter videos kept getting views for longer than the same channel's longer ones."),
    "title_len": ("title length",
                  "Videos with longer titles kept getting views for longer than the same channel's other videos.",
                  "Videos with shorter titles kept getting views for longer than the same channel's other videos."),
}


def pair_counts(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Per channel: summed pair scores and number of comparable pairs."""
    out = []
    for cid, g in df.groupby("creator_id"):
        g = g[g[factor].notna()]
        x, t, e = (g[factor].to_numpy(float), g["fade_h"].to_numpy(float), g["faded"].to_numpy(bool))
        score, n = 0.0, 0
        for i in range(len(g)):
            for j in range(len(g)):
                # i faded strictly first, and its fade was observed: the order is known.
                if t[i] < t[j] and e[i]:
                    n += 1
                    score += 1.0 if x[j] > x[i] else 0.5 if x[j] == x[i] else 0.0
        if n:
            out.append({"creator_id": cid, "score": score, "pairs": n})
    return pd.DataFrame(out, columns=["creator_id", "score", "pairs"])


def within_channel(df: pd.DataFrame, factor: str, n_boot: int = N_BOOT) -> dict[str, Any]:
    pc = pair_counts(df, factor)
    s, n = pc["score"].to_numpy(), pc["pairs"].to_numpy()
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(pc), len(pc)), minlength=len(pc))
        draws.append((w @ s) / (w @ n) if (w @ n) else np.nan)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"share_more_lasted_longer": float(s.sum() / n.sum()), "ci": [float(lo), float(hi)],
            "pairs": int(n.sum()), "channels": int(len(pc))}


def facts(rows: pd.DataFrame) -> dict[str, Any]:
    """Plain descriptive facts the report quotes, from Cohorts A and B."""
    from cdc.config import path_for
    s = pd.read_parquet(path_for("silver_dir") / "snapshots.parquet")
    s = s[(s["platform"] == "youtube") & s["post_id"].isin(set(rows["post_id"])) & s["views"].notna()]
    peak_early, by_day1 = [], []
    for _, g in s.sort_values("age_hours").groupby("post_id"):
        a = g["age_hours"].to_numpy(float)
        v = np.maximum.accumulate(g["views"].to_numpy(float))
        if len(a) < 3 or v[-1] <= 0 or a[-1] < 200:
            continue
        dt = np.diff(a)
        vel = np.where(dt > 0, np.diff(v) / np.where(dt > 0, dt, 1), 0)
        peak_early.append(a[1:][int(np.argmax(vel))] <= 6.0 + T.LATE_TOL_H)
        by_day1.append(np.interp(24.0, a, v) / v[-1])
    return {"videos": len(peak_early),
            "share_fastest_in_first_6h": float(np.mean(peak_early)),
            "median_share_of_10_day_views_by_24h": float(np.median(by_day1))}


def build(n_boot: int = N_BOOT, write: bool = True) -> dict[str, Any]:
    rows = T.build_rows(("A", "B"))
    at = rows[rows["mark"] == AT_MARK]
    results, kept = {}, []
    for f, (name, more, less) in CANDIDATES.items():
        r = {c: within_channel(at[at["cohort"] == c], f, n_boot) for c in ("A", "B")}
        sides = [("more" if x["ci"][0] > 0.5 else "less" if x["ci"][1] < 0.5 else None) for x in r.values()]
        keep = sides[0] is not None and sides[0] == sides[1]
        results[f] = {"name": name, "A": r["A"], "B": r["B"], "kept": keep,
                      "direction": sides[0] if keep else None}
        if keep:
            kept.append({"factor": f, "name": name, "direction": sides[0],
                         "text": more if sides[0] == "more" else less})
    out = {"rule": "kept if the 95% creator-bootstrap interval of the within-channel pair share "
                   "excludes 0.5 on the same side in Cohort A and in Cohort B",
           "measured_at_h": AT_MARK, "candidates": results, "tips": kept,
           "facts": facts(rows)}
    if write:
        TIPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TIPS_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    out = build()
    for f, r in out["candidates"].items():
        a, b = r["A"], r["B"]
        print(f"  {r['name']:<45} A {a['share_more_lasted_longer']:.3f} [{a['ci'][0]:.3f}, {a['ci'][1]:.3f}]"
              f"  B {b['share_more_lasted_longer']:.3f} [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}]"
              f"  {'KEPT (' + r['direction'] + ')' if r['kept'] else 'dropped'}")
    print(f"  facts: {out['facts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
