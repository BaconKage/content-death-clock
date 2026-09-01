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


def run(platform: str = "youtube", n_splits: int | None = None,
        n_boot: int = 500, write: bool = True) -> dict:
    af = dataset.build(platform=platform)
    df = af.frame

    print("=" * 68)
    print(f"  EVALUATION — {platform}")
    print("=" * 68)
    print("  sample attrition:")
    for stage, n in af.attrition.items():
        arrow = "  " if n >= 0 else "  -"
        print(f"    {stage:<38} {arrow}{abs(n)}")
    print("-" * 68)

    if df.empty:
        print("  no analysable posts yet.")
        return {"platform": platform, "status": "no data"}

    deaths, creators = af.n_deaths, af.n_creators
    print(f"  posts               {len(df)}")
    print(f"  observed deaths     {deaths}")
    print(f"  censored (alive)    {len(df) - deaths}   ({af.censoring_rate:.0%})")
    print(f"  creators            {creators}")

    underpowered = deaths < MIN_DEATHS or creators < MIN_CREATORS
    if underpowered:
        print()
        print("  *** UNDERPOWERED — this is a DRESS REHEARSAL, not a result. ***")
        print(f"      {deaths} deaths (want >={MIN_DEATHS}), "
              f"{creators} creators (want >={MIN_CREATORS}).")
        print("      Numbers below verify the pipeline runs end to end.")
        print("      They must not be reported as findings.")

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
        "censoring_rate": af.censoring_rate, "n_splits": splits,
        "features": feats, "attrition": af.attrition,
        "results": results, "wilcoxon": tests,
    }

    print("\n" + "=" * 68)
    if underpowered:
        print("  STATUS: rehearsal only. Do not quote these numbers.")
    else:
        print("  STATUS: powered. These are reportable results.")
    print("=" * 68)

    if write:
        p = path_for("gold_dir") / f"evaluation_{platform}.json"
        p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"  wrote {p.relative_to(ROOT)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the pre-registered evaluation.")
    ap.add_argument("--platform", default="youtube")
    ap.add_argument("--splits", type=int, default=None)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    run(args.platform, args.splits, args.boot, write=not args.no_write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
