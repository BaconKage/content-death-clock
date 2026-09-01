# Pre-registered Analysis Plan

**Content Death Clock** — RV University, 7th Semester
Research Methodology + Big Data Analytics

> **Commit this file, unchanged, before examining any outcome data.** Its value comes
> entirely from being timestamped in git *before* we know what the results look like. It
> is the cheapest credibility we can buy, and it is the difference between "we tested a
> hypothesis" and "we found something after trying several things".
>
> Amendments are allowed and expected — but they must be made as *new commits with stated
> reasons*, never by editing history. The paper reports the amendments.

| | |
|---|---|
| **Status** | ✅ **FROZEN** |
| **Frozen on** | **2026-08-30** |
| **Frozen at commit** | the commit that set this status — `git log --diff-filter=A --follow ANALYSIS_PLAN.md` |
| **Authors** | Amogha V Prasad · S Anannya · Sanidhya Tiwari · Shubhang Srinivas Varda |

**State of the data at freezing**, so the claim is auditable rather than asserted:
179 posts collected (124 YouTube, 55 Instagram), 182 snapshots, **0 posts with a computed
death label**, 0 models fitted, 0 results examined. Collection began 2026-08-30 ~15:45 UTC,
roughly two hours before this freeze.

---

## 1. Research question

Can the time at which a social media post stops receiving meaningful attention be
predicted from signals observable within the first six hours after publication?

## 2. Hypotheses

**H1 (primary).** A model using early-engagement and metadata features observed at
t ≤ 6h predicts time-to-attention-death more accurately than a naive constant-lifetime
baseline.

- *H1₀*: model performance ≤ best naive baseline performance.
- *H1₁*: model performance > best naive baseline performance.

**H2 (secondary).** Early engagement *velocity* carries predictive information beyond
creator size alone — i.e. the model beats a follower-count-only baseline.

> H2 is the hypothesis that actually matters. Beating "everything dies at 48 hours" is
> trivially easy and proves almost nothing; beating creator size proves the *dynamics*
> carry signal.

**H3 (exploratory, not confirmatory).** Time-to-death differs across creator-size strata
and content categories. Reported as exploratory; not used to support the main claim.

## 3. Design

Observational, prospective panel. Non-experimental — we predict decay, we do not
manipulate anything, and no causal claim is made.

## 4. Sample

- **Frame**: 63 public YouTube channels in `config/channels.resolved.yaml`, stratified by
  subscriber tier (micro <10k / mid 10k–500k / large >500k) and content category.
  As frozen: **30 large, 17 mid, 16 micro** across 8 categories.
- **How the frame was built** (2026-08-30). An initial convenience sample of channels the
  authors follow produced 30 large / 2 mid / 1 micro, which could not support H3. The
  frame was therefore rebuilt by API search: for each query in
  `config/discovery_queries.yaml` we searched *recent videos ordered by date*, resolved
  the channels behind them, and kept those in the under-represented tiers with ≥1
  upload/week. Recent-video search was used rather than channel search because channel
  search ranks by relevance — a proxy for popularity — and so cannot surface sub-10k
  channels, the stratum we lacked.
- **Sampling claim, stated plainly.** This is **not** a random sample of YouTube. It is
  the set of channels reachable by a fixed, committed query list, filtered by size and
  upload frequency. Generalisation beyond that frame is not claimed. The queries are
  committed so the frame is reproducible by a third party — a weaker claim than random
  sampling, a stronger one than an unrecorded convenience sample.
- **Known frame biases**: the query list leans toward Indian and long-tail creators;
  channels uploading less than weekly are excluded by construction, so the frame
  over-represents frequent uploaders; micro-tier picks span 1,240–7,250 subscribers and
  mid-tier 10,800–373,000, so neither tier covers its full nominal range.
- **Target**: ~800–1500 YouTube posts across the 63 channels.
- **Admission**: any post published by a frame channel during the collection window and
  discovered within `discovery_lookback_hours` of publication.
- **Observation**: snapshots at t+1, 3, 6, 12, 24, 36, 48, 72, 96, 120, 168, 240, 336h.

**Instagram is a feasibility demonstration, not a second test set.** The Scrape Creators
credit budget (100 credits per key) cannot fund a continuous panel at a useful size, so
Instagram is collected as bounded 48h cohorts of 5–6 accounts. Its purpose is to
demonstrate that the pipeline generalises across platforms and to supply one
cross-platform figure. **n will be too small to support inference, and no hypothesis
above is tested on Instagram data.** Instagram accounts were selected on a measured
criterion: the profile endpoint returns only ~12 recent posts, so accounts posting more
than ~6 times/day push their own posts off the observable grid before decay completes,
and accounts posting less than ~once/day yield no fresh posts inside a 48h window.

**Exclusions**, all pre-specified, all counted and reported in a sample-attrition table:

| Rule | Reason code |
|---|---|
| Fewer than 4 usable inter-snapshot intervals | `insufficient_observations` |
| Never achieved positive view velocity | `no_positive_velocity` |
| Post deleted or made private during observation | recorded as censored at last observation |
| Beyond 5 posts from one creator on one UTC day (YouTube only) | `creator_daily_cap` — **added by amendment 2026-08-31, see log** |

## 5. Variables

**Dependent.** `t_death` — first time engagement velocity falls below 5% of that post's
own peak velocity and remains below for 2 consecutive intervals. Right-censored at last
observation if never reached. Velocity is computed on **actual observed timestamps**,
never an assumed schedule.

**The primary metric differs by platform, and this is a substantive choice.**
YouTube uses **views**; Instagram uses **likes**. Instagram reports no view count at all
for image posts (`video_view_count` is null when `is_video` is false), so views cannot
serve there. Likes saturate faster than views, so **death times are not directly
comparable across platforms**: per-platform results are reported separately and never
pooled. Posts whose creator has hidden like/view counts are excluded — the API reports 0,
which is indistinguishable from genuine non-engagement and would register as instant
death.

**Robustness DV.** `t_saturation` — time to reach 90% of the asymptote A from a fitted
C(t) = A(1 − e^(−kt)). Reported alongside. Systematic disagreement between the two is a
finding to report, not a problem to hide.

**Independent — hard constraint: nothing observed after t = 6h.** Early velocity at 1h,
3h, 6h; log growth ratios between consecutive snapshots; acceleration; video duration;
title length; tag count; publish hour and weekday; category; creator subscriber count;
creator historical median velocity.

## 6. Analysis

**Models.** Weibull AFT and Random Survival Forest (primary, censoring-aware);
log-time linear regression and gradient boosting (comparators, uncensored subset).

**Baselines**, all four reported:
1. Constant 48h ("all content dies at 48 hours")
2. Training-set median lifetime
3. Subscriber-count-only model
4. Peak-velocity heuristic

**Validation.** `GroupKFold(n_splits=5)` **grouped by creator**. Random k-fold would place
the same creator in train and test, leaking creator identity and inflating scores; this is
pre-specified precisely so it cannot be quietly relaxed later.

**Temporal holdout.** Posts collected after the Cohort A freeze date form Cohort B and are
evaluated **exactly once**, at the end, after all model selection is complete.

**Metrics.** Harrell's C-index (primary, censoring-aware); MAE and RMSE on log₁₀
time-to-death (uncensored subset); calibration plot of predicted vs observed.

**Inference.** Paired Wilcoxon signed-rank test on per-post absolute errors, model vs each
baseline. Bootstrap (2,000 resamples, resampled by creator) for confidence intervals on
metric differences. α = 0.05. Effect sizes reported alongside p-values; a p-value alone
will not be treated as a result.

**Multiplicity.** H1 and H2 are the only confirmatory tests. Everything else is labelled
exploratory in the paper.

## 7. Pre-specified stopping rule

**Cohort A closes 2026-09-16T00:00:00Z**, fixed in `settings.yaml`
(`modelling.cohort_a_freeze_utc`) on 2026-08-30, before any outcome data was examined —
**not** when results look good. Posts published before that instant, with ≥4 usable
intervals, constitute the analysis set.

Collection continues after that date. Posts published between 2026-09-16 and the end of
collection form **Cohort B**, the temporal holdout, which is evaluated **exactly once**
after all model selection is complete.

## 8. What we will report regardless of outcome

- If the model does **not** beat the baselines, that is the reported result. A negative
  finding, honestly reported with an accompanying power discussion, is a legitimate RM
  paper. Searching for a specification that wins is not.
- Censoring rate, sample attrition by reason code, collection completeness and outages,
  and the count of monotonicity repairs (retracted view counts) — all reported.
- Any deviation from this plan, with its reason and its date.

## 9. Known limitations, stated in advance

Observational design; no causal claim. Convenience sample of channels — not a random
sample of YouTube, so generalisation beyond the frame is not claimed. Views are a
platform-reported metric we cannot independently audit. Scrape Creators is an unofficial
third-party API. The observation window truncates genuinely long-lived content, which is
what censoring-aware modelling is there to handle and what the retrospective cohort is
there to bound.

---

## Amendment log

**2026-09-01 — landmark design; outcome becomes remaining lifetime.**

*Change.* Predictions are made **at a landmark of t = 7 hours**. Only posts still
alive and observed at the landmark are eligible, and the outcome is the time
**remaining** from the landmark until attention death, not the time from
publication. Both quantities are retained in the analysis frame so a sensitivity
analysis needs no re-derivation.

*Reason.* With the outcome measured from publication, **53% of observed deaths
occurred inside the 6-hour feature window**. A post dying at 3.1h has its death
determined by the very observations its features are built from, so predicting
it is circular rather than predictive. Landmarking is the standard survival
remedy: a subject must be at risk at the moment prediction is made. Ten posts
were excluded on this ground, and they were never genuinely predictable.

*Why 7 hours and not 6.* The collector runs every 30 minutes, so the nominal
6-hour measurement lands at a median of **6.08h**. A strict 6.00h cutoff
discarded it for 72 of 86 posts, which is why every 6-hour-derived feature was
0% available. The snapshot schedule contains nothing between 6h and 12h, so any
observation at or before 7h *is* one of the scheduled 1/3/6-hour measurements
arriving late. Feature coverage went from 4 usable columns to 8.

*Related defect fixed the same day.* `follower_count` was never recorded for
YouTube posts — `videos.list` does not return channel subscriber counts — so it
was 0% populated and the subscriber-only baseline, which H2 is defined against,
had degenerated into a constant. It is now joined from the frozen resolved frame.

*Status of the outcome data at amendment.* Seven observed deaths in the
landmarked set, no model selection performed, no result reported or interpreted.
The change was made because the previous design was incoherent, not because a
result was unwelcome.

**2026-08-31 — per-creator daily admission cap (YouTube only).**

*Change.* At most 5 posts from one creator per UTC day enter the YouTube
analysis sample; the earliest published are kept, and the remainder are excluded
under reason code `creator_daily_cap` and counted in the attrition table. No cap
is applied to Instagram.

*Reason.* One channel published **100 videos within 25 seconds** and came to
supply 59% of the YouTube panel. Those posts carried a median of 4 views against
1,152 for every other post in the panel. They are not 100 independent
observations — one creator, one moment, one action — and leaving them in would
have made the creator-size stratification meaningless and dominated every fold
of the cross-validation. The contamination reached us through a measurement bug:
our upload-rate check paged only the 50 most recent videos, so any channel
uploading faster than ~7/day reported exactly 11.67/week regardless of its true
rate, and the "at least weekly" filter admitted firehoses along with active
creators. That bug is fixed separately; the cap protects the design from the
general case rather than this one channel.

*Instagram exempt.* Its accounts were selected deliberately for frequent posting,
no hypothesis is tested on Instagram data (section 4), and its collection cost
non-renewable credits. Capping it would discard paid-for data to solve a problem
it does not have.

*Status of the outcome data at amendment.* No death label had been computed and
no model had been fitted. The amendment is a sampling decision made before any
result existed, not after seeing one.

| Date | Commit | Change | Reason |
|---|---|---|---|
| 2026-08-31 | *this commit* | Per-creator daily admission cap, YouTube only | Bulk uploader supplied 59% of the panel; see above |
| 2026-09-01 | *this commit* | Landmark design: predict remaining lifetime from t=7h | 53% of deaths fell inside the feature window, making prediction circular; see below |
