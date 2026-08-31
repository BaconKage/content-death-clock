"""pandas vs Spark on the identical workload, measured across scales.

This produces the BDA report's headline result: **the crossover point** — the
data volume above which distributing the work starts to pay for itself. Below
it, Spark's scheduling and serialisation overhead makes it slower than a single
pandas process, and saying so is more useful than pretending otherwise.

**The workload** is the real one: a per-post windowed time-series computation.
For each post, order its observations by age, difference consecutive values, and
divide by elapsed time to get engagement velocity, then aggregate per post. It
is a group-by plus an ordered window plus an aggregation — the shape that
actually stresses a distributed engine, unlike a row-wise map which would
parallelise trivially and prove nothing.

**Equivalence is verified, not assumed.** Before timing anything, both engines
run on the same input and their outputs are compared. A benchmark of two
implementations that compute different things is worthless, and it is an easy
mistake to make when one is written in pandas and the other in Spark SQL.

Windows note: PySpark needs ``winutils.exe`` and ``HADOOP_HOME`` locally, which
is a half-day of yak-shaving. Run this on the GitHub Actions Ubuntu runner
instead — see ``.github/workflows/benchmark.yml``. Without PySpark installed the
pandas path still runs and Spark rows are reported as skipped.

Usage::

    python -m cdc.bench.spark_bench --factors 1 10 100
    python -m cdc.bench.spark_bench --factors 1 10 100 1000 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cdc.bench.amplify import amplify
from cdc.config import ROOT, path_for

RESULTS_PATH = ROOT / "data" / "bench" / "scaling.json"


# --------------------------------------------------------------- the workload
def feature_job_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Per-post velocity features. The reference implementation."""
    d = df.sort_values(["post_id", "age_hours"])
    g = d.groupby("post_id", sort=False)
    d = d.assign(
        prev_value=g["primary_value"].shift(1),
        prev_age=g["age_hours"].shift(1),
    )
    dt = d["age_hours"] - d["prev_age"]
    dv = d["primary_value"] - d["prev_value"]
    d = d.assign(velocity=np.where(dt > 0, dv / dt, np.nan))

    out = d.groupby("post_id", sort=False).agg(
        n_obs=("age_hours", "size"),
        max_age=("age_hours", "max"),
        peak_velocity=("velocity", "max"),
        mean_velocity=("velocity", "mean"),
        final_value=("primary_value", "last"),
    ).reset_index()
    return out.sort_values("post_id").reset_index(drop=True)


def feature_job_spark(sdf, spark):
    """The same computation in Spark. Must agree with the pandas version."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy("post_id").orderBy("age_hours")
    d = (sdf
         .withColumn("prev_value", F.lag("primary_value").over(w))
         .withColumn("prev_age", F.lag("age_hours").over(w)))
    dt = F.col("age_hours") - F.col("prev_age")
    dv = F.col("primary_value") - F.col("prev_value")
    d = d.withColumn("velocity", F.when(dt > 0, dv / dt).otherwise(F.lit(None)))

    last_w = Window.partitionBy("post_id").orderBy("age_hours") \
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    d = d.withColumn("final_value", F.last("primary_value", True).over(last_w))

    return (d.groupBy("post_id")
            .agg(F.count("age_hours").alias("n_obs"),
                 F.max("age_hours").alias("max_age"),
                 F.max("velocity").alias("peak_velocity"),
                 F.mean("velocity").alias("mean_velocity"),
                 F.first("final_value").alias("final_value")))


# ------------------------------------------------------------------ harness
@dataclass
class Timing:
    engine: str
    factor: int
    rows: int
    seconds: float
    rows_per_second: float
    note: str = ""


def _time(fn, repeats: int) -> float:
    """Median of `repeats` runs. Median, not mean: one GC pause or one JIT warm-up
    should not define the result."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        # Spark is lazy; force materialisation inside the timed region or we
        # would be timing the construction of a query plan.
        if hasattr(out, "count"):
            out.count()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def make_spark(cores: int):
    from pyspark.sql import SparkSession
    return (SparkSession.builder
            .master(f"local[{cores}]")
            .appName("cdc-bench")
            .config("spark.sql.shuffle.partitions", str(max(cores * 2, 8)))
            .config("spark.ui.enabled", "false")
            .config("spark.driver.memory", "3g")
            .getOrCreate())


def verify_equivalence(df: pd.DataFrame, spark) -> dict:
    """Both engines must compute the same thing before either is timed."""
    p = feature_job_pandas(df)
    s = (feature_job_spark(spark.createDataFrame(df), spark)
         .toPandas().sort_values("post_id").reset_index(drop=True))

    checks = {"rows_match": bool(len(p) == len(s)),
              "post_ids_match": bool(set(p["post_id"]) == set(s["post_id"]))}
    merged = p.merge(s, on="post_id", suffixes=("_p", "_s"))
    for col in ("n_obs", "max_age", "peak_velocity", "mean_velocity", "final_value"):
        a = merged[f"{col}_p"].to_numpy(float)
        b = merged[f"{col}_s"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
        checks[f"{col}_matches"] = bool(np.all(close | both_nan))
    checks["all_match"] = bool(all(checks.values()))
    return checks


def load_snapshots() -> pd.DataFrame:
    p = path_for("silver_dir") / "snapshots.parquet"
    if not p.exists():
        raise SystemExit("silver missing — run: python -m cdc.transform.silver")
    df = pd.read_parquet(p)
    keep = [c for c in ("post_id", "creator_id", "age_hours", "primary_value",
                        "views", "likes", "comments") if c in df.columns]
    return df[keep].dropna(subset=["post_id", "age_hours"])


def run(factors: list[int], repeats: int = 3, spark_cores: list[int] | None = None,
        write: bool = True) -> dict:
    spark_cores = spark_cores or [1, 4]
    base = load_snapshots()
    print(f"base dataset: {len(base):,} rows, {base.post_id.nunique():,} posts\n")

    try:
        import pyspark  # noqa: F401
        spark_available = True
    except ImportError:
        spark_available = False
        print("PySpark not installed — pandas only. See requirements-bench.txt\n")

    spark = None
    equivalence = None
    if spark_available:
        spark = make_spark(max(spark_cores))
        spark.sparkContext.setLogLevel("ERROR")
        equivalence = verify_equivalence(base, spark)
        print(f"equivalence check: {'PASS' if equivalence['all_match'] else 'FAIL'}")
        if not equivalence["all_match"]:
            print(f"  {equivalence}")
            raise SystemExit("engines disagree — benchmark would be meaningless")
        print()

    timings: list[Timing] = []
    print(f"{'engine':<16} {'factor':>7} {'rows':>12} {'seconds':>9} {'rows/s':>12}")
    print("-" * 60)
    for f in factors:
        df = amplify(base, f)
        n = len(df)

        t = _time(lambda: feature_job_pandas(df), repeats)
        timings.append(Timing("pandas", f, n, t, n / t))
        print(f"{'pandas':<16} {f:>7} {n:>12,} {t:>9.3f} {n/t:>12,.0f}")

        if spark_available:
            for cores in spark_cores:
                sp = make_spark(cores)
                sp.sparkContext.setLogLevel("ERROR")
                sdf = sp.createDataFrame(df).cache()
                sdf.count()                      # warm the cache OUTSIDE timing
                t = _time(lambda: feature_job_spark(sdf, sp), repeats)
                timings.append(Timing(f"spark[{cores}]", f, n, t, n / t))
                print(f"{f'spark[{cores}]':<16} {f:>7} {n:>12,} {t:>9.3f} {n/t:>12,.0f}")
                sdf.unpersist()

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
        "base_rows": len(base),
        "repeats": repeats,
        "equivalence": equivalence,
        "timings": [asdict(t) for t in timings],
        "crossover": _crossover(timings),
    }
    if write:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {RESULTS_PATH.relative_to(ROOT)}")
    if spark is not None:
        spark.stop()
    return result


def _crossover(timings: list[Timing]) -> dict:
    """Smallest row count at which some Spark configuration beats pandas.

    Reported as None when pandas wins everywhere tested — which is a legitimate
    and publishable result, not a failed benchmark.
    """
    by_rows: dict[int, dict[str, float]] = {}
    for t in timings:
        by_rows.setdefault(t.rows, {})[t.engine] = t.seconds
    for rows in sorted(by_rows):
        row = by_rows[rows]
        pandas_t = row.get("pandas")
        spark_ts = {k: v for k, v in row.items() if k.startswith("spark")}
        if pandas_t and spark_ts and min(spark_ts.values()) < pandas_t:
            best = min(spark_ts, key=spark_ts.get)
            return {"rows": rows, "engine": best,
                    "spark_seconds": spark_ts[best], "pandas_seconds": pandas_t}
    return {"rows": None,
            "note": "pandas faster at every scale tested — a real result, "
                    "not a failure. Report it and state the range tested."}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pandas vs Spark scaling benchmark.")
    ap.add_argument("--factors", type=int, nargs="+", default=[1, 10, 100])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--spark-cores", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    r = run(args.factors, args.repeats, args.spark_cores, write=not args.no_write)
    print("\n" + "=" * 60)
    c = r["crossover"]
    if c.get("rows"):
        print(f"  CROSSOVER at {c['rows']:,} rows: {c['engine']} "
              f"({c['spark_seconds']:.2f}s) overtakes pandas ({c['pandas_seconds']:.2f}s)")
    else:
        print(f"  NO CROSSOVER in the range tested — {c['note']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
