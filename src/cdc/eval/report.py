"""Run the pre-registered evaluation and print the results table.

Everything here follows the frozen analysis plan: creator-grouped
cross-validation, the four baselines, C-index primary, bootstrap intervals
resampled by creator, paired Wilcoxon against each baseline.

**Power is reported first, and loudly.** A C-index computed on nineteen deaths
is arithmetic, not evidence, and the single easiest way for this project to
embarrass itself is to quote an early number as a finding. The header states the
sample size, the death count and the creator count before any result appears,
and the footer refuses to interpret an underpowered run.

Usage::

    python -m cdc.eval.report
    python -m cdc.eval.report --platform instagram
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cdc.config import ROOT, path_for, settings
from cdc.eval.validate import (
    bootstrap_ci_by_creator,
    concordance,
    grouped_cv,
    mae_log10,
    paired_wilcoxon,
)
from cdc.eval import holdout
from cdc.eval.robustness import label_agreement, print_agreement, saturation_frame
from cdc.models import dataset
from cdc.models.baselines import (
    ConstantLifetime,
    KaplanMeierMedian,
    PeakVelocityHeuristic,
    SubscriberOnly,
)
from cdc.models.survival import RandomSurvivalForest, WeibullAFT

# Below these, the run is a rehearsal rather than a result. The thresholds are
# deliberately generous: even at 50 deaths a C-index is wobbly.
MIN_DEATHS = 50
MIN_CREATORS = 10


def _json_safe(o):
    """Replace non-finite floats with null, recursively.

    A Wilcoxon test on too few pairs returns NaN, which is a legitimate result
    and is printed as "n/a". But `json.dumps` writes it as a bare `NaN` literal,
    which is not JSON: Python reads it back, and JSON.parse, R's jsonlite and
    most other tooling reject the file outright. This is the machine-readable
    output of the whole study, so it has to be readable by machines that are not
    Python.
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.floating):
        return _json_safe(float(o))
    if isinstance(o, np.integer):
        return int(o)
    return o


def dumps_strict(obj) -> str:
    """Serialise to strict RFC 8259 JSON.

    `allow_nan=False` is the guard, not the mechanism: `_json_safe` has already
    removed the non-finite values, so if this still raises then something new
    has appeared that we have not thought about, and failing loudly beats
    writing another file that only Python can read.
    """
    return json.dumps(_json_safe(obj), indent=2, default=str, allow_nan=False)


def run(platform: str = "youtube", n_splits: int | None = None,
        n_boot: int = 500, write: bool = True, outcome: str = "death",
        cohort: str = "A", unlock_holdout: bool = False) -> dict:
    prior_holdout: list = []
    if cohort.upper() == "B":
        # Refuses unless deliberately unlocked; see cdc.eval.holdout.
        prior_holdout = holdout.check_unlocked(unlock_holdout, platform)

    af = dataset.build(platform=platform, cohort=cohort)
    df = af.frame

    print("=" * 68)
    print(f"  EVALUATION — {platform}  [cohort {cohort.upper()}]"
          + ("" if outcome == "death" else f"  [outcome: {outcome}]"))
    print(f"  freeze instant: {dataset.freeze_instant().isoformat()}")
    print("=" * 68)
    if cohort.upper() == "B":
        holdout.warn_if_repeated(prior_holdout)
    print("  sample attrition:")
    for stage, n in af.attrition.items():
        arrow = "  " if n >= 0 else "  -"
        print(f"    {stage:<38} {arrow}{abs(n)}")
    print("-" * 68)

    if df.empty:
        if cohort.upper() == "B":
            print("  Cohort B is empty — the freeze instant has not passed, or")
            print("  no post published after it has enough observations yet.")
            print("  Nothing was recorded in the holdout ledger.")
        else:
            print("  no analysable posts yet.")
        return {"platform": platform, "cohort": cohort, "status": "no data"}

    # Robustness label: reported alongside on every run, per the frozen plan.
    agreement = label_agreement(df)
    print_agreement(agreement)

    # A robustness check cannot be better powered than the study it is checking.
    # Without this, the saturation run reports 54 "deaths" - every successful
    # curve fit counts as an event - sails past the power guard, and prints
    # "these are reportable results" while the primary analysis still has 7.
    primary_deaths = int(pd.Series(af.frame["event_observed"]).astype(bool).sum())

    if outcome == "saturation":
        df = saturation_frame(df)
        print(f"\n  switched outcome to t_saturation: {len(df)} posts retained")
        if df.empty:
            print("  no post has a usable saturation outcome yet.")
            return {"platform": platform, "outcome": outcome,
                    "status": "no saturation outcomes", "agreement": agreement}

    # Counted from the frame in play, not from the AnalysisFrame, so the
    # saturation run reports its own sample rather than the primary one's.
    deaths = int(pd.Series(df["event_observed"]).astype(bool).sum())
    creators = int(df["creator_id"].nunique())
    print("\n" + "-" * 68)
    print(f"  posts               {len(df)}")
    print(f"  observed deaths     {deaths}")
    cens = (1.0 - deaths / len(df)) if len(df) else float("nan")
    print(f"  censored (alive)    {len(df) - deaths}   ({cens:.0%})")
    print(f"  creators            {creators}")

    underpowered = (deaths < MIN_DEATHS or creators < MIN_CREATORS
                    or primary_deaths < MIN_DEATHS)
    if underpowered:
        print()
        print("  *** UNDERPOWERED — this is a DRESS REHEARSAL, not a result. ***")
        print(f"      {deaths} deaths (want >={MIN_DEATHS}), "
              f"{creators} creators (want >={MIN_CREATORS}).")
        if outcome != "death" and deaths >= MIN_DEATHS:
            print(f"      Gated on the PRIMARY analysis, which has "
                  f"{primary_deaths} observed deaths.")
        print("      Numbers below verify the pipeline runs end to end.")
        print("      They must not be reported as findings.")

    if outcome != "death" and deaths and cens == 0.0:
        print()
        print("  NOTE: 0% censoring. Every post with a successful curve fit is")
        print("  counted as an event, so this outcome has no survivors by")
        print("  construction. Censoring-aware models cannot show their")
        print("  advantage here, and the comparison is correspondingly weaker.")

    feats = dataset.usable_features(df)
    print(f"\n  features used ({len(feats)}): {', '.join(feats) if feats else 'NONE'}")
    if not feats:
        print("\n  no feature has enough coverage yet — cannot model.")
        return {"platform": platform, "status": "no usable features",
                "attrition": af.attrition}

    splits = n_splits or int(settings()["modelling"]["cv"]["n_splits"])
    splits = min(splits, creators)
    if splits < 2:
        print(f"\n  only {creators} creator(s) — cannot group-split. Need >=2.")
        return {"platform": platform, "status": "too few creators",
                "attrition": af.attrition}
    if splits < (n_splits or 5):
        print(f"  NOTE: folds reduced to {splits} (one per creator available)")

    # The plan names two primary models, both censoring-aware, and four
    # baselines. Keeping the two groups separate here is what lets the
    # hypothesis tests below pair every model against every baseline without
    # accidentally testing a baseline against another baseline.
    primary = {
        "weibull_aft": lambda: WeibullAFT(feats),
        "random_survival_forest": lambda: RandomSurvivalForest(feats),
    }
    baselines = {
        "constant_48h": lambda: ConstantLifetime(48.0),
        "km_median": lambda: KaplanMeierMedian(),
        "subscriber_only": lambda: SubscriberOnly(),
        "peak_velocity": lambda: PeakVelocityHeuristic(),
    }
    models = {**primary, **baselines}

    stacked, results = {}, {}
    print("\n" + "-" * 68)
    print(f"  {'model':<24} {'C-index':>9} {'95% CI':>18} {'MAE log10':>11}")
    print("-" * 68)
    for name, factory in models.items():
        try:
            s = grouped_cv(factory, df, n_splits=splits).stacked()
        except Exception as exc:
            print(f"  {name:<24}  failed: {exc}")
            continue
        stacked[name] = s
        c = concordance(s["duration"], s["event"], s["predicted"])
        mae = mae_log10(s["duration"], s["event"], s["predicted"])
        _, lo, hi = bootstrap_ci_by_creator(s, "concordance", n_boot=n_boot)
        results[name] = {"c_index": c, "ci_low": lo, "ci_high": hi, "mae_log10": mae}
        print(f"  {name:<24} {c:>9.3f} {f'[{lo:.3f}, {hi:.3f}]':>18} {mae:>11.3f}")

    # --- hypothesis tests, every primary model vs every baseline
    tests: dict[str, dict] = {}
    fitted_primary = [m for m in primary if m in stacked]
    if fitted_primary:
        print("\n" + "-" * 68)
        print("  paired Wilcoxon — model vs baseline (negative favours model)")
        print("-" * 68)
        for mname in fitted_primary:
            for bname in baselines:
                if bname not in stacked:
                    continue
                t = paired_wilcoxon(stacked[mname], stacked[bname])
                tests[f"{mname} vs {bname}"] = t
                p, md = t["p_value"], t["median_diff"]
                ps = "n/a" if np.isnan(p) else f"{p:.4f}"
                mds = "n/a" if np.isnan(md) else f"{md:+.4f}"
                print(f"  {mname:<24} vs {bname:<16} "
                      f"n={t['n_pairs']:<4} med {mds:<10} p={ps}")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform, "underpowered": underpowered,
        "n_posts": len(df), "n_deaths": deaths, "n_creators": creators,
        "outcome": outcome, "cohort": cohort.upper(),
        "freeze_instant": dataset.freeze_instant().isoformat(),
        "censoring_rate": cens, "n_splits": splits,
        "agreement": agreement,
        "features": feats, "attrition": af.attrition,
        "results": results, "wilcoxon": tests,
    }

    print("\n" + "=" * 68)
    if underpowered:
        print("  STATUS: rehearsal only. Do not quote these numbers.")
    else:
        print("  STATUS: powered. These are reportable results.")
    print("=" * 68)

    if cohort.upper() == "B":
        entry = holdout.record(platform, len(df), deaths, creators, results)
        print()
        print(f"  recorded in the holdout ledger "
              f"(digest {entry['results_digest']}). Commit it.")

    if write:
        suffix = "" if outcome == "death" else f"_{outcome}"
        suffix += "" if cohort.upper() == "A" else f"_cohort{cohort.upper()}"
        p = path_for("gold_dir") / f"evaluation_{platform}{suffix}.json"
        p.write_text(dumps_strict(out), encoding="utf-8")
        print(f"  wrote {p.relative_to(ROOT)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the pre-registered evaluation.")
    ap.add_argument("--platform", default="youtube")
    ap.add_argument("--splits", type=int, default=None)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--cohort", default="A", choices=["A", "B", "all"],
                    help="A = the analysis set (default). B = the temporal "
                         "holdout, evaluated exactly once and only with "
                         "--unlock-holdout.")
    ap.add_argument("--unlock-holdout", action="store_true",
                    help="required to evaluate Cohort B; the run is recorded")
    ap.add_argument("--outcome", default="death", choices=["death", "saturation"],
                    help="which pre-registered outcome to model "
                         "(saturation is the plan's robustness label)")
    args = ap.parse_args(argv)
    run(args.platform, args.splits, args.boot, write=not args.no_write,
        outcome=args.outcome, cohort=args.cohort,
        unlock_holdout=args.unlock_holdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
