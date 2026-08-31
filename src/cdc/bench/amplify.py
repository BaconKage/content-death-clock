"""Synthetically amplify the real dataset to benchmark scale.

The honest framing, which the BDA report states plainly: **~25,000 real snapshot
rows is not big data.** Claiming otherwise invites an easy challenge. What we can
do instead is measure how the pipeline behaves as volume grows, and report where
a distributed engine starts to pay for itself. A measured scaling curve with a
crossover point is a stronger result than an inflated row count.

Amplification must preserve the properties the feature job actually depends on,
or the benchmark measures something other than our workload:

* **Within-post time structure.** Each replicated post keeps its own ordered
  sequence of observations, because the job is a per-post windowed computation
  over time. Shuffling rows would turn a realistic workload into a trivial one.
* **Creator grouping.** Replicated posts stay attached to a creator, so
  group-by cardinality grows the way it really would.
* **Value distributions.** Counts are jittered multiplicatively rather than
  copied exactly, so the amplified data has realistic spread rather than
  thousands of identical rows a query engine could optimise away.

What amplification does **not** do is create new information. It is a load
generator for measuring throughput, and no modelling result is ever computed
from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def amplify(snapshots: pd.DataFrame, factor: int,
            jitter: float = 0.15, seed: int = 20260830) -> pd.DataFrame:
    """Return `factor` copies of the snapshot table with distinct post ids.

    ``factor=1`` returns the input unchanged, so the 1x row of the benchmark is
    the genuine dataset rather than a reconstruction of it.
    """
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if factor == 1:
        return snapshots.copy()

    rng = np.random.default_rng(seed)
    out = []
    for rep in range(factor):
        chunk = snapshots.copy()
        if rep > 0:
            suffix = f"_r{rep}"
            chunk["post_id"] = chunk["post_id"].astype(str) + suffix
            # Creators are replicated too, so group cardinality scales with the
            # data rather than staying fixed at 63 and making the shuffle
            # unrealistically cheap.
            chunk["creator_id"] = chunk["creator_id"].astype(str) + suffix
            for col in ("views", "likes", "comments", "primary_value"):
                if col in chunk.columns:
                    v = chunk[col].to_numpy(dtype="float64")
                    mult = rng.normal(1.0, jitter, size=len(v))
                    chunk[col] = np.where(np.isnan(v), np.nan,
                                          np.maximum(0.0, v * mult))
            # Nudge observation ages so the sort inside each post is not a
            # repeat of an already-sorted run.
            if "age_hours" in chunk.columns:
                a = chunk["age_hours"].to_numpy(dtype="float64")
                chunk["age_hours"] = np.maximum(
                    0.0, a * rng.normal(1.0, 0.02, size=len(a)))
        out.append(chunk)

    return pd.concat(out, ignore_index=True)


def amplified_row_counts(n_rows: int, factors: list[int]) -> pd.DataFrame:
    """What each factor will produce — printed before a run so nobody is
    surprised by a 40-million-row job."""
    return pd.DataFrame({
        "factor": factors,
        "rows": [n_rows * f for f in factors],
    })
