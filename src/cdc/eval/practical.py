"""The practical early-warning model, tested once on Cohort C.

**Exploratory extension, not part of H1 or H2.** Declared in the analysis plan's
amendment log on 2026-10-18, after the main study's results were known and
before any Cohort C outcome had been examined. Everything that decides the
result is fixed in this file and in that amendment; the evaluation refuses to
run before ``NOT_BEFORE`` and records every run in a committed ledger, the same
way the Cohort B holdout does.

What it asks. The main study tested whether predictions transfer to channels a
model has never seen, and found a modest ranking signal. A practical tool would
be used on channels already being tracked, and could use two signals the study
never did: likes and comments in the first seven hours. The question here is
narrower and more useful than "when will it die": **at hour 7, can the model
flag the videos that will stop getting attention within two days?**

Design, fixed in advance:

* Train on Cohort A + Cohort B (every video analysed so far). Test on Cohort C,
  videos published 2026-10-04 to 2026-10-15, which no analysis has touched.
  The channels overlap, deliberately: that is the setting a tracking tool is
  used in, and why this is not a test of H2.
* Model: the registered random survival forest, same settings, given the eleven
  confirmatory features plus likes/views and comments/views at the landmark.
* Primary landmark 7h. Secondary landmark 25h (the 24-hour reading arrives a
  little late, so the prediction is made an hour after it, as 7h is for 6h),
  which adds total views and current speed at the landmark.
* Nothing observed after the landmark is ever read.

Usage::

    python -m cdc.eval.practical --status      # Cohort C size and maturity, no labels
    python -m cdc.eval.practical --rehearsal   # smoke test: train A, test B (already examined)
    python -m cdc.eval.practical               # the single evaluation, on/after NOT_BEFORE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from cdc.config import ROOT, path_for, settings

COHORT_A_END = pd.Timestamp("2026-09-16T00:00:00Z")
COHORT_B_END = pd.Timestamp("2026-10-04T00:00:00Z")
COHORT_C_END = pd.Timestamp("2026-10-15T00:00:00Z")
# Every Cohort C post has its last scheduled reading (240h, amendment 6) by
# 2026-10-25T00:00Z; two hours' grace for a late cycle.
NOT_BEFORE = pd.Timestamp("2026-10-25T02:00:00Z")
WATCH_H = 240.0
LANDMARKS = (7.0, 25.0)
FLAG_SHARE = 0.20          # flag the 20% predicted to fade soonest
FAST_H = 48.0              # "fades fast" = attention death within 48h of posting
SEED = 20260830
LEDGER_NAME = "practical_evaluations.jsonl"

ENGAGEMENT = ["log_like_rate", "log_comment_rate"]
TRAJECTORY = ["log_views_at_landmark", "speed_share_at_landmark"]


# ------------------------------------------------------------------ inputs
def engagement_at(snaps: pd.DataFrame, landmark: float) -> pd.DataFrame:
    """Likes/views and comments/views at the last reading at or before the landmark.

    Strictly ``<= landmark``: the prediction is made at the landmark, so a
    reading taken even minutes later would be information from the future.
    """
    s = snaps[snaps["age_hours"] <= landmark].sort_values("age_hours")
    last = s.groupby("post_id").last()
    views = last["views"].astype(float).where(last["views"] > 0)
    like = (last["likes"].astype(float) / views).clip(lower=1e-5)
    com = (last["comments"].astype(float) / views).clip(lower=1e-6)
    return pd.DataFrame({"log_like_rate": np.log10(like), "log_comment_rate": np.log10(com)})


def trajectory_at(snaps: pd.DataFrame, landmark: float) -> pd.DataFrame:
    """Views at the landmark, and the latest speed as a share of the best so far."""
    out = {}
    for pid, g in snaps[snaps["age_hours"] <= landmark].sort_values("age_hours").groupby("post_id"):
        a = g["age_hours"].to_numpy(float)
        v = np.maximum.accumulate(g["primary_value"].to_numpy(float))
        if len(a) < 2:
            out[pid] = (np.log10(v[-1] + 1) if len(v) else np.nan, np.nan)
            continue
        dt = np.diff(a)
        vel = np.where(dt > 0, np.diff(v) / np.where(dt > 0, dt, 1.0), 0.0)
        peak = vel.max()
        out[pid] = (np.log10(v[-1] + 1), max(vel[-1], 0.0) / peak if peak > 0 else np.nan)
    return pd.DataFrame.from_dict(out, orient="index", columns=TRAJECTORY)


def cohort_of(published_at: pd.Series) -> pd.Series:
    pub = pd.to_datetime(published_at, utc=True, format="mixed")
    return pd.Series(np.select([pub < COHORT_A_END, pub < COHORT_B_END, pub < COHORT_C_END],
                               ["A", "B", "C"], "later"), index=published_at.index)


def frame_at(landmark: float, cohorts: tuple[str, ...]) -> pd.DataFrame:
    """Labelled YouTube posts still alive at the landmark, for the named cohorts only.

    Only the cohorts asked for are labelled. A rehearsal therefore never computes
    an outcome for a Cohort C post, even in memory; Cohort C is labelled only by
    the single evaluation.
    """
    from cdc.models import dataset
    parts = [dataset.build(platform="youtube", cohort=c, landmark=landmark).frame
             for c in cohorts if c in ("A", "B")]
    if "C" in cohorts:
        allf = dataset.build(platform="youtube", cohort="all", landmark=landmark).frame
        parts.append(allf[cohort_of(allf["published_at"]).to_numpy() == "C"])
    d = pd.concat(parts, ignore_index=True)
    snaps = pd.read_parquet(path_for("silver_dir") / "snapshots.parquet")
    snaps = snaps[(snaps["platform"] == "youtube") & snaps["post_id"].isin(d["post_id"])]
    d = d.merge(engagement_at(snaps, landmark), left_on="post_id", right_index=True, how="left")
    if landmark > settings()["modelling"]["feature_cutoff_hours"]:
        d = d.merge(trajectory_at(snaps, landmark), left_on="post_id", right_index=True, how="left")
    d["cohort"] = cohort_of(d["published_at"]).to_numpy()
    return d


def inputs_for(landmark: float) -> list[str]:
    from cdc.models.synthetic import FEATURE_COLUMNS
    cols = list(FEATURE_COLUMNS) + ENGAGEMENT
    if landmark > settings()["modelling"]["feature_cutoff_hours"]:
        cols += TRAJECTORY
    return cols


# ------------------------------------------------------------------ metrics
def flag_metrics(pred: np.ndarray, fast: np.ndarray, creators: np.ndarray,
                 n_boot: int = 2000, seed: int = SEED) -> dict[str, Any]:
    """The early-warning claim: flag the FLAG_SHARE predicted shortest-lived.

    The flagged set is fixed once, on the whole test set. The bootstrap then
    resamples creators and recomputes, on each draw, the share of flagged videos
    that really faded fast (precision), the share of all videos that did (base
    rate), and the share of fast faders that were flagged (recall).
    """
    pred, fast, creators = np.asarray(pred, float), np.asarray(fast, bool), np.asarray(creators)
    flagged = pred <= np.quantile(pred, FLAG_SHARE)

    def stats(idx):
        f, y = flagged[idx], fast[idx]
        prec = y[f].mean() if f.any() else np.nan
        base = y.mean()
        rec = (f & y).sum() / y.sum() if y.any() else np.nan
        return prec, base, rec, prec - base

    point = stats(np.arange(len(pred)))
    rows = {c: np.flatnonzero(creators == c) for c in sorted(set(creators))}
    names = np.array(list(rows))
    rng = np.random.default_rng(seed)
    draws = np.array([stats(np.concatenate([rows[c] for c in rng.choice(names, len(names))]))
                      for _ in range(n_boot)], dtype=float)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    keys = ("precision", "base_rate", "recall", "precision_minus_base")
    out = {k: float(point[i]) for i, k in enumerate(keys)}
    out.update({f"{k}_ci": [float(lo[i]), float(hi[i])] for i, k in enumerate(keys)})
    out["lift"] = out["precision"] / out["base_rate"] if out["base_rate"] else float("nan")
    out["n_flagged"] = int(flagged.sum())
    out["n_fast"] = int(fast.sum())
    out["passes"] = bool(lo[3] > 0)        # declared rule: lower bound of (precision - base) above 0
    return out


def evaluate(train: pd.DataFrame, test: pd.DataFrame, landmark: float,
             n_boot: int = 2000) -> dict[str, Any]:
    from cdc.eval.validate import concordance, mae_log10, paired_delta_bootstrap
    from cdc.models.baselines import KaplanMeierMedian, SubscriberOnly
    from cdc.models.survival import RandomSurvivalForest

    cols = [c for c in inputs_for(landmark) if c in train
            and pd.to_numeric(train[c], errors="coerce").notna().mean() >= 0.5
            and pd.to_numeric(train[c], errors="coerce").nunique() > 1]
    dur_tr, ev_tr = train["t_death"].to_numpy(float), train["event_observed"].to_numpy(bool)
    model = RandomSurvivalForest(cols).fit(train, dur_tr, ev_tr)
    size = SubscriberOnly().fit(train, dur_tr, ev_tr)
    km = KaplanMeierMedian().fit(train, dur_tr, ev_tr)

    d, e = test["t_death"].to_numpy(float), test["event_observed"].to_numpy(bool)
    p_model, p_size, p_km = model.predict(test), size.predict(test), km.predict(test)
    fast = e & (test["t_death_from_publish"].to_numpy(float) <= FAST_H)
    stacked = {n: pd.DataFrame({"post_id": test["post_id"].to_numpy(), "creator_id": test["creator_id"].to_numpy(),
                                "predicted": p, "duration": d, "event": e})
               for n, p in (("model", p_model), ("size", p_size))}
    delta = paired_delta_bootstrap(stacked, [("model", "size")], n_boot=n_boot, seed=SEED)["model vs size"]
    return {
        "landmark_h": landmark, "inputs": cols,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_test_deaths": int(e.sum()), "n_test_creators": int(test["creator_id"].nunique()),
        "c_index": float(concordance(d, e, p_model)),
        "c_index_size_only": float(concordance(d, e, p_size)),
        "delta_c_vs_size": delta,
        "ranking_passes": bool(delta["delta_c_ci_low"] > 0),
        "timing_error_factor": float(10 ** mae_log10(d, e, p_model)),
        "timing_error_factor_median_guess": float(10 ** mae_log10(d, e, p_km)),
        "fast_flag": flag_metrics(p_model, fast, test["creator_id"].to_numpy(), n_boot=n_boot),
    }


# ------------------------------------------------------------------ guard rails
def ledger_path():
    return path_for("gold_dir") / LEDGER_NAME


def prior_runs() -> list[dict]:
    p = ledger_path()
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_allowed(now: pd.Timestamp | None = None, again: bool = False) -> None:
    now = now or pd.Timestamp.now(tz="UTC")
    if now < NOT_BEFORE:
        raise SystemExit(f"\nREFUSED - Cohort C is evaluated once, on or after {NOT_BEFORE.isoformat()}.\n"
                         f"  Its last videos have not finished their {WATCH_H:.0f}-hour window yet.\n"
                         "  Use --status to see where it stands, or --rehearsal for a smoke test on Cohort B.\n")
    if prior_runs() and not again:
        first = prior_runs()[0]
        raise SystemExit(f"\nREFUSED - Cohort C was already evaluated at {first.get('evaluated_at')}.\n"
                         "  A second run is a second look at the test set and must be reported as one.\n"
                         "  Pass --again only if you mean that.\n")


def status(now: pd.Timestamp | None = None) -> dict[str, Any]:
    """Cohort C's size and maturity from publication dates only. Computes no labels."""
    now = now or pd.Timestamp.now(tz="UTC")
    posts = pd.read_parquet(path_for("silver_dir") / "posts.parquet")
    posts = posts[posts["platform"] == "youtube"]
    cap = posts.get("over_creator_daily_cap", pd.Series(False, index=posts.index)).astype(bool)
    pub = pd.to_datetime(posts["published_at"], utc=True, format="mixed")
    c = (cohort_of(posts["published_at"]) == "C") & ~cap
    done = (pub + pd.Timedelta(hours=WATCH_H)) <= now
    return {"cohort_c_posts": int(c.sum()), "finished_watching": int((c & done).sum()),
            "run_allowed_from": NOT_BEFORE.isoformat(), "allowed_now": bool(now >= NOT_BEFORE)}


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------ run
def run(rehearsal: bool = False, again: bool = False, n_boot: int = 2000, write: bool = True) -> dict:
    if not rehearsal:
        check_allowed(again=again)
    results = {}
    for L in LANDMARKS:
        f = frame_at(L, ("A", "B") if rehearsal else ("A", "B", "C"))
        if rehearsal:
            train, test = f[f["cohort"] == "A"], f[f["cohort"] == "B"]
        else:
            train, test = f[f["cohort"].isin(["A", "B"])], f[f["cohort"] == "C"]
        results[f"{L:g}h"] = evaluate(train, test, L, n_boot=n_boot)
    out = {"evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "kind": "rehearsal (train A, test B, already examined)" if rehearsal else "Cohort C, single evaluation",
           "git_commit": _git_commit(), "results": results}
    _print(out)
    if write:
        gold = path_for("gold_dir")
        name = "practical_rehearsal_B.json" if rehearsal else "practical_cohortC.json"
        (gold / name).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        if not rehearsal:
            digest = hashlib.sha256(json.dumps(results, sort_keys=True, default=str).encode()).hexdigest()[:12]
            head = results[f"{LANDMARKS[0]:g}h"]
            entry = {"evaluated_at": out["evaluated_at"], "git_commit": out["git_commit"],
                     "n_test": head["n_test"], "n_test_deaths": head["n_test_deaths"],
                     "results_digest": digest, "fast_flag_passes": head["fast_flag"]["passes"],
                     "ranking_passes": head["ranking_passes"]}
            with ledger_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
            print(f"\n  recorded in data/gold/{LEDGER_NAME} (digest {digest}). Commit it.")
    return out


def _print(out: dict) -> None:
    print("=" * 72)
    print(f"  PRACTICAL EARLY-WARNING MODEL — {out['kind']}")
    print("=" * 72)
    for k, r in out["results"].items():
        ff, dc = r["fast_flag"], r["delta_c_vs_size"]
        print(f"\n  predicting at {k}: train {r['n_train']}, test {r['n_test']} videos "
              f"({r['n_test_deaths']} deaths, {r['n_test_creators']} channels)")
        print(f"    ranking (C-index)        {r['c_index']:.3f}   channel size only {r['c_index_size_only']:.3f}"
              f"   gain {dc['delta_c_index']:+.3f} [{dc['delta_c_ci_low']:+.3f}, {dc['delta_c_ci_high']:+.3f}]"
              f"   {'PASS' if r['ranking_passes'] else 'no'}")
        print(f"    fast-fade flag (top {FLAG_SHARE:.0%}) precision {ff['precision']:.1%} "
              f"[{ff['precision_ci'][0]:.1%}, {ff['precision_ci'][1]:.1%}] vs base rate {ff['base_rate']:.1%}"
              f" = {ff['lift']:.1f}x; recall {ff['recall']:.0%}   {'PASS' if ff['passes'] else 'no'}")
        print(f"    timing error             {r['timing_error_factor']:.2f}x   (guessing the median: "
              f"{r['timing_error_factor_median_guess']:.2f}x)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Practical early-warning model, Cohort C (exploratory).")
    ap.add_argument("--status", action="store_true", help="Cohort C size and maturity; computes no labels")
    ap.add_argument("--rehearsal", action="store_true", help="smoke test: train A, test B (already examined)")
    ap.add_argument("--again", action="store_true", help="permit a second Cohort C run (recorded as such)")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.status:
        print(json.dumps(status(), indent=2))
        return 0
    run(rehearsal=args.rehearsal, again=args.again, n_boot=args.boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
