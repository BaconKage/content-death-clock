"""Build ``site/data.json``, every number the public website shows.

The website (``site/index.html``) is static and reads nothing else, so a figure
on it can always be traced to this file and from here to the repository's own
artifacts:

* silver tables (``python -m cdc.transform.silver``),
* the gold evaluation files and out-of-fold predictions
  (``python -m cdc.eval.report`` for Cohort A, its sensitivity runs, and the
  single Cohort B evaluation),
* the bronze cycle logs and the git history, for collection volume.

**Nothing identifies a creator or a post.** The site shows real videos, but
never a title, an id or a channel name, and channel size only as a broad band:
an exact subscriber count is enough to name some of these channels. A test pins
that no identifier reaches the output.

Usage::

    python -m cdc.site.build_data
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cdc.config import ROOT, path_for, settings

OUT = ROOT / "site" / "data.json"
LANDMARK_H = 7.0

# Measured on a 4-core GitHub Actions runner. Run 1 is BDA report section 7.4
# (2026-08-31, base 1,047 rows); run 2 is Actions run 39123036090 (2026-10-18,
# base 30,229 rows). Kept here because data/bench/ is not committed.
BENCHMARK = {
    "Aug 31 (1,047-row base)": [(1047, 0.007, 0.271, 0.201), (10470, 0.015, 0.197, 0.155),
                                (104700, 0.114, 0.188, 0.172), (1047000, 1.740, 0.415, 0.318)],
    "Oct 18 (30,229-row base)": [(30229, 0.011, 0.195, 0.151), (302290, 0.131, 0.147, 0.135),
                                 (3022900, 2.132, 0.361, 0.357)],
}

TIER_NAME = {"micro": "Small channel", "mid": "Mid-size channel", "large": "Large channel"}
PRETTY = {"random_survival_forest": "Random Survival Forest", "weibull_aft": "Weibull AFT",
          "subscriber_only": "Channel size only", "peak_velocity": "Peak velocity",
          "km_median": "Average lifetime", "constant_48h": "Everything dies at 48h",
          "loglinear_uncensored": "Log-linear (deaths only)",
          "gbm_uncensored": "Gradient boosting (deaths only)"}


# ----------------------------------------------------------------- helpers
def subscriber_band(n: float | None) -> str:
    """A band wide enough that it cannot name a channel."""
    if n is None or not np.isfinite(n):
        return "unknown size"
    for limit, label in ((1e4, "under 10k subscribers"), (1e5, "10k–100k subscribers"),
                         (5e5, "100k–500k subscribers"), (5e6, "500k–5M subscribers")):
        if n < limit:
            return label
    return "over 5M subscribers"


def round_views(x: float | None) -> int | None:
    """Two significant figures: enough to compare, not enough to look up."""
    if x is None or not np.isfinite(x):
        return None
    x = float(max(x, 0))
    if x < 100:
        return int(round(x))
    mag = 10 ** (int(np.floor(np.log10(x))) - 1)
    return int(round(x / mag) * mag)


def pick_pairs(frame: pd.DataFrame, preds: dict[str, pd.Series], n: int = 30,
               min_gap_h: float = 24.0, seed: int = 20260830) -> list[dict[str, Any]]:
    """Pairs of videos where it is known which stopped getting attention first.

    Both must be observed to die, at least ``min_gap_h`` apart, so the answer is
    a fact and not a tie on the snapshot grid. Each side carries only what a
    person could see at hour 7, plus the reveal: when it really died, when the
    forest expected it to, and when channel size alone expected it to.
    """
    d = frame[frame["event_observed"].astype(bool)].copy()
    d = d[d["post_id"].isin(preds["model"].index) & d["post_id"].isin(preds["size"].index)]
    rng = np.random.default_rng(seed)
    ids = d["post_id"].to_numpy()
    by_id = d.set_index("post_id")
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        a, b = rng.choice(ids, 2, replace=False)
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        ra, rb = by_id.loc[a], by_id.loc[b]
        if abs(ra["t_death_from_publish"] - rb["t_death_from_publish"]) < min_gap_h:
            continue
        seen.add(key)
        out.append({"a": _side(ra, preds, a), "b": _side(rb, preds, b),
                    "first": "a" if ra["t_death_from_publish"] < rb["t_death_from_publish"] else "b"})
    for p in out:
        p["model_pick"] = "a" if p["a"]["model_h"] < p["b"]["model_h"] else "b"
        p["size_pick"] = "a" if p["a"]["size_h"] < p["b"]["size_h"] else "b"
    return out


def _side(r: pd.Series, preds: dict[str, pd.Series], pid: str) -> dict[str, Any]:
    return {
        "size": TIER_NAME.get(r["stratum_tier"], "Channel"),
        "band": subscriber_band(r.get("follower_count")),
        "topic": str(r.get("stratum_category") or "other"),
        "views_1h": round_views(r.get("value_at_1h")),
        "views_3h": round_views(r.get("value_at_3h")),
        "views_6h": round_views(r.get("value_at_6h")),
        "died_h": round(float(r["t_death_from_publish"]), 1),
        "model_h": round(float(preds["model"][pid]) + LANDMARK_H, 1),
        "size_h": round(float(preds["size"][pid]) + LANDMARK_H, 1),
    }


def strip_ids(obj: Any) -> Any:
    """Defence in depth: drop any key that names an identifier."""
    if isinstance(obj, dict):
        return {k: strip_ids(v) for k, v in obj.items()
                if k not in ("post_id", "creator_id", "video_id", "channel_id", "handle", "title")}
    if isinstance(obj, list):
        return [strip_ids(v) for v in obj]
    return obj


# ------------------------------------------------------------------- build
def build() -> dict[str, Any]:
    from lifelines import KaplanMeierFitter
    from cdc.labels.death import Observation, _prepare, label_post, velocities
    from cdc.models import dataset

    warnings.filterwarnings("ignore")
    silver, gold = path_for("silver_dir"), path_for("gold_dir")
    snaps = pd.read_parquet(silver / "snapshots.parquet")
    posts = pd.read_parquet(silver / "posts.parquet")
    yt_s, yt_p = snaps[snaps.platform == "youtube"], posts[posts.platform == "youtube"]
    A = dataset.build(platform="youtube", cohort="A").frame
    B = dataset.build(platform="youtube", cohort="B").frame
    ev = {"A": json.loads((gold / "evaluation_youtube.json").read_text()),
          "B": json.loads((gold / "evaluation_youtube_cohortB.json").read_text())}

    # --- collection volume -------------------------------------------------
    ts = pd.to_datetime(yt_s.snapshot_ts, utc=True, format="mixed").dt.strftime("%Y-%m-%d")
    per_day = ts.value_counts()
    pub = pd.to_datetime(yt_p.published_at, utc=True, format="mixed")
    new_per_day = pub[pub >= "2026-08-30"].dt.strftime("%Y-%m-%d").value_counts()
    log = subprocess.run(["git", "log", "--format=%s"], cwd=ROOT, capture_output=True,
                         text=True).stdout.splitlines()
    runs = Counter(l.split("cycle ")[1][:10] for l in log if "collection cycle" in l)
    blocked = Counter()
    for f in glob.glob(str(ROOT / "data/bronze/_cycles/20*.json")):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("quota_exceeded"):
            blocked[d["started_at"][:10]] += 1
    last_day = max(per_day.index)
    days = pd.date_range("2026-08-30", last_day, freq="D").strftime("%Y-%m-%d")
    daily = [{"day": x, "snapshots": int(per_day.get(x, 0)), "new_videos": int(new_per_day.get(x, 0)),
              "runs": int(runs.get(x, 0)), "blocked": int(blocked.get(x, 0) > 0)} for x in days]

    # --- coverage by scheduled mark, Cohort A ---------------------------------
    cfg = settings()["collection"]
    sched = [float(m) for m in cfg["snapshot_schedule_hours"]]
    tol = float(cfg["snapshot_tolerance_hours"])
    cap = yt_p.get("over_creator_daily_cap", pd.Series(False, index=yt_p.index)).astype(bool)
    a_ids = set(yt_p.post_id[(pub < "2026-09-16") & ~cap])
    ages = yt_s[yt_s.post_id.isin(a_ids)].groupby("post_id").age_hours.apply(lambda a: np.sort(a.to_numpy()))
    coverage = [{"mark_h": int(m), "on_time": round(float(np.mean([np.any(np.abs(a - m) <= tol) for a in ages])), 3)}
                for m in sched]

    bronze = ROOT / "data" / "bronze"
    files = [p for p in bronze.rglob("*") if p.is_file()]
    summary = {
        "videos_tracked": int(len(yt_p)), "snapshots": int(len(yt_s)),
        "instagram_posts": int((posts.platform == "instagram").sum()),
        "channels": 63, "runs": int(sum(runs.values())), "days": len(days),
        "bronze_files": len(files), "bronze_mb": round(sum(p.stat().st_size for p in files) / 1e6, 1),
        "on_time": round(float(np.mean([c["on_time"] for c in coverage[:-1]])), 3),
        "a_videos": int(len(A)), "a_deaths": int(A.event_observed.sum()), "a_channels": int(A.creator_id.nunique()),
        "b_videos": int(len(B)), "b_deaths": int(B.event_observed.sum()), "b_channels": int(B.creator_id.nunique()),
        "units_per_run": 65, "runs_per_day": 52, "allowance": 10000, "discovery_share": 0.972,
        "landmark_h": LANDMARK_H, "threshold": 0.05, "watched_h": 240,
    }
    pipeline = [
        {"stage": "Collector", "count": summary["runs"], "unit": "runs, one every 30 minutes"},
        {"stage": "Raw files", "count": summary["bronze_files"], "unit": f"files · {summary['bronze_mb']} MB"},
        {"stage": "Clean tables", "count": int(len(snaps)), "unit": "snapshot rows"},
        {"stage": "Analysis set", "count": int(len(A) + len(B)), "unit": "videos"},
        {"stage": "Models", "count": int(len(ev["A"]["results"])), "unit": "models compared"},
    ]
    # Quota on the days the leaked key drained it: our runs before the refusal,
    # counted from the git history, at the measured 65 units a run.
    leak = []
    for day, start, stop in (("2026-10-15", "2026-10-15T07:00:00Z", "2026-10-15T16:31:00Z"),
                             ("2026-10-17", "2026-10-17T07:00:00Z", "2026-10-17T20:31:00Z"),
                             ("2026-10-18", "2026-10-18T07:00:00Z", "2026-10-18T09:31:00Z")):
        n = int(subprocess.run(["git", "log", f"--since={start}", f"--until={stop}", "--format=%s"],
                               cwd=ROOT, capture_output=True, text=True).stdout.count("collection cycle"))
        leak.append({"day": day, "ours": n * 65, "refused_at": stop[11:16] + " UTC"})

    # --- outcomes ------------------------------------------------------------------
    survival, tiers = {}, []
    for coh, d in (("A", A), ("B", B)):
        survival[coh] = {}
        for grp, g in [("all", d)] + [(t, d[d.stratum_tier == t]) for t in ("micro", "mid", "large")]:
            sf = KaplanMeierFitter().fit(g.t_death, g.event_observed.astype(int)).survival_function_.iloc[:, 0]
            thin = sf.groupby((sf.index.to_numpy() * 2).round() / 2).last()
            survival[coh][grp] = [[float(t), round(float(s), 4)] for t, s in thin.items()]
        for t in ("micro", "mid", "large"):
            g = d[d.stratum_tier == t]
            km = KaplanMeierFitter().fit(g.t_death, g.event_observed.astype(int))
            tiers.append({"cohort": coh, "tier": t, "name": TIER_NAME[t], "videos": int(len(g)),
                          "channels": int(g.creator_id.nunique()),
                          "median_h": round(float(km.median_survival_time_), 1)})
    models = []
    for coh, e in ev.items():
        for m, r in e["results"].items():
            const = m in ("constant_48h", "km_median")
            models.append({"cohort": coh, "model": PRETTY[m], "role": r.get("role"),
                           "c": 0.5 if const else round(r["c_index"], 3),
                           "lo": None if const else round(r["ci_low"], 3),
                           "hi": None if const else round(r["ci_high"], 3),
                           "error_x": round(10 ** r["mae_log10"], 2)})
    h2 = []
    runs_h2 = [("Main analysis", "A", ev["A"]), ("Sealed holdout", "B", ev["B"])] + [
        (lab, "A", json.loads((gold / f"evaluation_youtube_{f}.json").read_text()))
        for f, lab in (("lm3p0", "Predict from hour 3"), ("lm12p0", "Predict from hour 12"),
                       ("th0p02", "Stricter death line (2%)"), ("th0p1", "Looser death line (10%)"),
                       ("saturation", "Different death definition"),
                       ("exploratory", "All planned inputs (after the fact)"))]
    for lab, coh, e in runs_h2:
        for m in ("random_survival_forest", "weibull_aft"):
            k = f"{m} vs subscriber_only"
            d, w = e["paired_bootstrap"][k], e["wilcoxon"][k]
            h2.append({"analysis": lab, "cohort": coh, "model": PRETTY[m],
                       "dc": round(d["delta_c_index"], 3), "lo": round(d["delta_c_ci_low"], 3),
                       "hi": round(d["delta_c_ci_high"], 3), "p": float(f"{w['p_value']:.3g}"),
                       "both": bool(d["delta_c_ci_low"] > 0 and w["p_value"] < 0.05 and w["median_diff"] < 0)})

    # --- videos to watch die (Cohort A, anonymised) ---------------------------------------
    def series(pid):
        v = yt_s[yt_s.post_id == pid].sort_values("age_hours")
        obs, _ = _prepare([Observation(a, x) for a, x in zip(v.age_hours, v.primary_value) if pd.notna(x)])
        vel = velocities(obs)
        peak = max(x for _, x in vel)
        speed = {round(t, 2): round(max(x, 0) / peak, 4) for t, x in vel}
        lab = label_post(pid, obs)
        pts = [{"h": round(o.age_hours, 2), "views": round_views(o.value),
                "speed": speed.get(round(o.age_hours, 2))} for o in obs]
        return pts, (round(lab.t_death, 1) if lab.event_observed else None)

    def steady(pid):
        v = yt_s[yt_s.post_id == pid].sort_values("age_hours").primary_value.to_numpy(float)
        return len(v) >= 12 and v[1] > 0 and np.all(np.diff(v[:6]) > 0)

    rng = np.random.default_rng(7)
    examples = []
    for label, cond in (
            ("Small channel, fades in two days", (A.stratum_tier == "micro") & A.event_observed & A.t_death_from_publish.between(40, 50)),
            ("Mid-size channel, gone in a day", (A.stratum_tier == "mid") & A.event_observed & A.t_death_from_publish.between(20, 30)),
            ("Mid-size channel, lasts four days", (A.stratum_tier == "mid") & A.event_observed & A.t_death_from_publish.between(90, 100)),
            ("Large channel, lasts five days", (A.stratum_tier == "large") & A.event_observed & A.t_death_from_publish.between(110, 130)),
            ("Large channel, lasts a week", (A.stratum_tier == "large") & A.event_observed & A.t_death_from_publish.between(160, 175)),
            ("Still alive at day ten", ~A.event_observed & (A.t_death_from_publish > 235))):
        cand = A[cond & (A.n_intervals >= 10)]
        cand = cand[[steady(x) for x in cand.post_id]]
        if cand.empty:
            continue
        r = cand.iloc[int(rng.integers(len(cand)))]
        pts, died = series(r.post_id)
        examples.append({"label": label, "size": TIER_NAME[r.stratum_tier],
                         "band": subscriber_band(r.follower_count), "topic": str(r.stratum_category),
                         "died_h": died, "points": pts})

    # --- the guessing game: Cohort A out-of-fold predictions --------------------------------
    pr = pd.read_parquet(gold / "predictions_youtube.parquet")
    preds = {"model": pr[pr.model == "random_survival_forest"].set_index("post_id")["predicted"],
             "size": pr[pr.model == "subscriber_only"].set_index("post_id")["predicted"]}
    game = pick_pairs(A, preds, n=40)

    data = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "summary": summary, "daily": daily, "coverage": coverage, "pipeline": pipeline,
        "leak": leak, "benchmark": [{"run": run, "rows": n, "engine": e, "seconds": s}
                                    for run, rows in BENCHMARK.items() for n, *t in rows
                                    for e, s in zip(("pandas", "Spark, 1 core", "Spark, 4 cores"), t)],
        "survival": survival, "tiers": tiers, "models": models, "h2": h2,
        "examples": examples, "game": game,
    }
    return strip_ids(data)


def main(argv: list[str] | None = None) -> int:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB): "
          f"{len(data['examples'])} example videos, {len(data['game'])} game pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
