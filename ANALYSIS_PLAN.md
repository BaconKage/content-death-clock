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

**2026-10-18 — An exploratory extension, declared before its test data is examined:
a practical early-warning model, evaluated once on a new Cohort C.**

*Status.* This is not a change to H1, H2 or H3, and it is not evidence for or against any of
them. It was conceived after the main study's results, the Cohort B holdout included, were
known. That is why it gets a test set of its own and rules of its own, both fixed here
before any Cohort C outcome has been computed.

*Why.* The main study asks whether predictions transfer to channels a model has never seen.
A practical tool would be used on channels that are already being tracked, and could use two
signals the study recorded but never modelled: likes and comments in the first hours.
Exploration on 2026-10-18 (trained on Cohort A, tested on Cohort B, both already examined)
suggested that such a model ranks videos better and, more usefully, flags fast-fading videos
at about three times the base rate, while its timing error stays where the study found it.
That suggestion came from data that had already been examined, so it is tested here on data
that has not.

*Test set: Cohort C.* YouTube posts published 2026-10-04T00:00Z ≤ t < 2026-10-15T00:00Z,
creator daily cap applied, labelled exactly as in the main study (5% of peak, two intervals,
7-hour landmark). By publication record on 2026-10-18 there are 533 such posts, 215 of them
already past their 240-hour window. No label, feature or prediction has been examined for
any Cohort C post. One disclosure: the first rehearsal below built its data with a call
that labelled every post in memory, Cohort C included, before keeping only Cohorts A and B.
Nothing about Cohort C was printed, stored or looked at, and the code was changed the same
day so that a rehearsal labels only the cohorts it uses. Every Cohort C post completes its
effective 240-hour window (amendment of 2026-10-16) by 2026-10-25T00:00Z.

*Known damage, stated in advance.* The quota outages caused by the leaked key
(2026-10-15T16:31Z–2026-10-16T07:00Z, 2026-10-17T20:31Z–2026-10-18T07:00Z, and from
2026-10-18T09:31Z to the next reset) cost some Cohort C posts one or more scheduled
readings. Nothing is excluded or imputed for that reason. Velocity is computed on actual
timestamps as always, posts left with too few intervals fall under the existing exclusion
rules and are counted, and the number of Cohort C posts missing any scheduled reading from 1h
to 240h is reported with the result.

*Model, fixed.* `src/cdc/eval/practical.py` at this commit. It is trained on Cohort A plus
Cohort B. It is the random survival forest with the registered settings (300 trees, minimum
leaf 5, square-root feature sampling, seed 20260830), summarised by restricted mean survival
time. Its inputs are the eleven confirmatory features plus log10(likes/views) and
log10(comments/views) at the last reading at or before the landmark, under the 50% coverage
rule on the training set. The primary landmark is 7h. The secondary landmark is 25h, one hour
after the scheduled 24-hour reading, as 7h is for 6h; it adds log10 views and the latest speed
as a share of the best speed so far, both at or before 25h. Nothing observed after a landmark
is read. The same channels appear in training and test, deliberately: that is how a tracking
tool is used, and it is the reason this is not a test of H2.

*Claims and decision rules, fixed.*

1. **Primary: the 7-hour fast-fade flag.** The 20% of Cohort C posts with the shortest
   predicted remaining time are flagged. A post "fades fast" if its attention death is
   observed within 48 hours of posting. The claim **passes** if the lower bound of the 95%
   creator-resampled bootstrap interval (2,000 draws) for *precision minus base rate* is
   above zero. Precision, base rate, lift and recall are reported with their intervals.
2. **Secondary: ranking at 7 hours.** Harrell's C-index, and the paired creator-bootstrap
   ΔC-index against the creator-size-only baseline trained on the same data. Passes if that
   interval excludes zero.
3. **Descriptive: the 25-hour model** (the same two quantities), and the timing error against
   guessing the median, at both landmarks. No pass/fail.

*Rehearsal, already run, not a result.* The frozen code was run in rehearsal mode (trained on
Cohort A, tested on Cohort B, both already examined) as a smoke test. At 7h: ranking 0.663
against 0.555 for creator size (gain +0.108, [+0.059, +0.158]); flag precision 41.9%
[29.5%, 56.8%] against a base rate of 13.5%, 3.1×, recall 62%; timing error 1.73× against
1.77× for guessing the median. At 25h: ranking 0.704; flag precision 19.6% against 6.5%,
3.0×; timing error 1.78× against 1.87×. These numbers are the reference for Cohort C, and
Cohort C should be expected to come out lower, not higher, because nothing was chosen on it.

*Command and timing.* Run **once**, on or after **2026-10-25T02:00Z**, from this commit (checked
out separately if later commits have touched `src/cdc/eval/`, `src/cdc/models/`,
`src/cdc/labels/` or `config/settings.yaml`):

```
python -m cdc.transform.silver
python -m cdc.eval.practical
```

The harness refuses to run earlier and records the run in
`data/gold/practical_evaluations.jsonl`, which is committed whatever the result. A second run
requires `--again` and is reported as a second look. The result goes into the RM paper as a
labelled exploratory extension, and into the technical report, whatever it says. It changes
no verdict on H1 or H2.

*Status of the outcome data at amendment.* Cohorts A and B fully examined. Cohort C:
publication counts only.

**2026-10-16 — The confirmatory feature set is the eleven that ran; the Cohort B
evaluation is declared before it is opened; the 336-hour observation was never
taken.**

Three parts. Parts 1 and 3 record things that had already happened; part 2 fixes
the holdout procedure in writing two days before the holdout can be opened.

*Part 1 — a deviation from section 5, found after the results.*

*What the plan says and what ran.* Section 5 lists the independent variables.
Five of them never entered any model: **tag count, publish weekday, content
category, creator historical median velocity, and acceleration as a raw
difference of velocities.** Acceleration did enter in ratio form
(`decay_ratio_3_6_over_1_3`, velocity over 3–6h divided by velocity over 1–3h).
The feature whitelist, `FEATURE_COLUMNS` in `src/cdc/models/synthetic.py`, was
written on **2026-08-31** (commit `aef3cd7`) with eleven columns, before any
death label existed, and every model reported so far has used exactly those
eleven: value at 1h, 3h and 6h; velocity to 6h; log growth 1→3h and 3→6h; the
3–6h/1–3h decay ratio; log subscriber count; log duration; publish hour; title
length. Three of the five omitted variables were computed by the feature layer
and never whitelisted; creator historical median velocity was never built.

*When it was found.* 2026-10-16, **after** the Cohort A results, including the
H2 null, had been examined and written up. The omission was not declared at the
time and should have been.

*Decision.* The confirmatory specification for H1 and H2, on Cohort A **and on
Cohort B**, is the eleven-column set as run. The five omitted variables were
built on 2026-10-16 (commit `8910e05`, with leakage tests) and may appear only in
a labelled exploratory analysis, `--feature-set exploratory`, which writes to its
own file and which the harness **refuses to run on Cohort B**. Its result is
reported in the paper whatever it says, and is not evidence for or against H1 or
H2.

*Reason.* The eleven-column whitelist is the specification that was in force
before any outcome was seen. Adopting the fuller list now, with the H2 answer in
view, would make the choice of feature set a decision taken after the result,
which is the kind of decision this plan exists to prevent. Building variables
the plan named is legitimate. Promoting them into the confirmatory test once
the test has disappointed is not, and the difference between the two is only
the timing.

*Part 2 — the Cohort B evaluation, declared before it is opened.*

*State of the holdout.* The ledger `data/gold/holdout_evaluations.jsonl` does
not exist: no model has been fitted to, or scored on, Cohort B. **One
disclosure.** On 2026-10-16, during a status check, death labels were computed
for Cohort B in order to count its analysable posts, observed deaths and
creators (831, 655 and 50). No model was fitted, no metric computed and no
prediction made. A death count is outcome data, so the look is recorded here.

*Procedure.* The registered evaluation, applied unchanged to Cohort B: Cohort B
posts only (published 2026-09-16T00:00Z ≤ t < 2026-10-04T00:00Z, creator daily cap
applied), the 7-hour landmark, the 5% threshold sustained for 2 intervals, the
eleven confirmatory features under the 50% coverage rule, `GroupKFold(5)` grouped
by creator **within Cohort B**, both primary models, all four baselines and both
comparators, 2,000 creator-resampled bootstrap draws, seed 20260830.

*Why Cohort B is cross-validated within itself rather than scored by a model
trained on Cohort A.* The frame's channels publish in both cohorts: **43 of
Cohort B's 51 creators also appear in Cohort A** (counted from publication
records only). A model trained on Cohort A and scored on Cohort B would
therefore have seen almost every test creator during training. That is the
leakage section 6 pre-specifies grouping by creator to prevent. The within-cohort
path has also been the implemented one since 2026-09-02 (commit `b002da9`),
before any Cohort A result existed. What the holdout tests is therefore whether
the registered procedure's result **replicates on an independent, later sample
of posts**, not how a model frozen on Cohort A performs going forward in time,
and the paper describes it that way.

*Command and timing.* Run **once**, on or after 2026-10-18 (every Cohort B post
completes its 336-hour window at 2026-10-17T23:00Z):

```
python -m cdc.eval.report --platform youtube --cohort B --unlock-holdout
```

from the commit containing this amendment, or from a later commit that changes
nothing under `src/cdc/models/`, `src/cdc/labels/`, `src/cdc/eval/` or
`config/settings.yaml`. The ledger entry is committed whatever the result.

*Interpretation, fixed now.* Exactly as Cohort A was read. H1 is supported on
discrimination if a primary model's C-index interval excludes 0.5, and on
absolute error only if that model beats the better of the two naive baselines
(Wilcoxon, α = 0.05). H2 is counted as supported only if a primary model beats
the creator-size-only baseline on **both** pre-specified tests: the paired
Wilcoxon at α = 0.05 favouring the model, **and** a paired bootstrap interval on
ΔC-index that excludes zero. Holm-adjusted Wilcoxon p-values across the two
primary models are reported alongside. A disagreement between Cohort A and
Cohort B is reported as it falls, in either direction.

*Pipeline additions in this commit, all outcome-independent.* The evaluation
harness now produces quantities section 6 names that it had never produced. The
first is the **paired creator bootstrap on metric differences**, ΔC-index and
ΔMAE, with both members of each pair drawn on the same creators. Until now the
paper's interval on a difference was computed outside the harness, and for
Cohort B it could not have been computed at all without a second run. The others
are **RMSE** on log₁₀ time, the two **comparators** (log-time linear regression
and gradient boosting, uncensored subset, outside every hypothesis test), the
bootstrap count read from `settings.yaml` and recorded in the artifact (the CLI
had defaulted to 500), and the **out-of-fold predictions** written to
`data/gold/`, so that the plan's calibration plot needs no second look. Re-running
Cohort A on this commit reproduces the matured Cohort A run of
2026-10-16T05:28Z to the last digit (every C-index, interval, MAE and Wilcoxon
statistic); the additions only add rows.

*Part 3 — the 2026-10-15 quota outage, and the scheduling defect it exposed.*

*What happened.* From **2026-10-15T16:31Z** every YouTube Data API call returned
HTTP 403 `quotaExceeded`. The last complete cycle ran at 16:02Z. The collector
kept firing every 30 minutes and recorded no observations. The quota resets at
midnight Pacific time, **2026-10-16T07:00Z**, after this amendment was written.

*The cause is not established at the time of writing.* The collector's own
meter recorded 64–65 units per cycle at about 52 cycles per day, roughly 3,400
units per day against a 10,000-unit allowance. The quota day resets at midnight
Pacific (07:00Z), so by 16:31Z the collector had spent about 1,200 units of that
day's allowance. Its own usage cannot account for the exhaustion, which points
to other consumption of the same Google Cloud project's quota. This is to be
confirmed from the Cloud console and reported in the technical report. The
runner guards spending per cycle (`max_quota_per_cycle`) but not per day, so the
first warning was the 403 itself.

*Effect on the data: none, on either cohort.* No post in either cohort was owed
an observation inside the outage. The last observation any post receives is its
240-hour one (see below), and the latest Cohort B post, published 2026-10-03,
received it before 2026-10-14T00:00Z.

*The defect found while checking that.* **The 336-hour observation the plan
schedules (section 4) has never been taken, for any post.** No YouTube post in
the dataset has an observation older than 241 hours. Since the first commit
(2026-08-30, `95e18e8`), `Panel.due` owes a post a scheduled mark only once its
age has passed that mark, but drops the post from the panel as soon as its age
exceeds `max_track_hours`, which is 336, the last mark itself. The 336-hour mark
could therefore be met only by a cycle landing at exactly 336.000 hours, and
none ever did. The collection monitor did not show it: its per-mark coverage
table lists only marks with at least one hit, so a mark that was missed every
time had no row at all.

*What it changes, stated as it is.*

1. **The effective observation window is 240 hours (10 days), not 14 days.**
   Of Cohort A's 123 censored posts, 121 are censored at their 240-hour
   observation, 233 hours after the landmark; the other two stopped being
   observable earlier.
2. **The latest confirmable death is at the 168-hour observation.** Death needs
   two consecutive sub-threshold intervals and 168→240h is the last interval
   there is, so a post still above threshold at 168 hours is right-censored at
   240. The latest death in Cohort A is at 168.5 hours.
3. **It is the same for every post in both cohorts and does not depend on any
   outcome.** It narrows the estimand, to death within the first week, rather
   than biasing any comparison between models, baselines or cohorts.
4. The 336-hour maturity rule of 2026-09-20 was therefore conservative:
   outcomes stop changing at 240 hours. The declared Cohort B date of
   2026-10-18 stands regardless.

*Handling.* No change to the analysis. The papers report the window as 240
hours and death as confirmable up to 168 hours. The collector is **not** fixed
now. Both cohorts were collected under the same instrument, and no hypothesis
uses posts published after the Cohort B close, so a fix would buy no analysable
observation and would make the record harder to read.

*Status of the outcome data at amendment.* Cohort A has fully matured (686
posts, 563 observed deaths) and its results have been examined. Cohort B has had
only the counts disclosed in Part 2 computed. The ledger is empty.

**2026-09-20 — Cohort B closes 2026-10-04, and both cohorts are analysed only
once every member has completed its observation window.**

*Change.* Two scheduling decisions, neither of which alters the design.

1. **Cohort B closes to new admissions at 2026-10-04T00:00:00Z**, fixed in
   `settings.yaml` as `modelling.cohort_b_close_utc` and enforced in
   `cdc.models.dataset`. Section 7 bounded the holdout below, at the Cohort A
   freeze, but left it open above, so it grew every time the collector ran and
   "evaluated exactly once" had no fixed referent.
2. **Neither cohort is analysed for the record until every post in it has
   completed the 336-hour tracking window** set in `collection.max_track_hours`.
   Cohort A's final analysis therefore falls on or after **2026-09-30**; the
   Cohort B holdout evaluation falls on or after **2026-10-18**.

*Reason for the close date.* A holdout that can be extended is not a holdout.
Running it, disliking the number, and collecting another fortnight would convert
it into a second validation set — the same failure the unlock flag and the
ledger exist to prevent, arrived at by a slower route. The date was chosen on
2026-09-20, after the Cohort A analysis was complete and written up, and
**before the holdout was opened or inspected in any way**. The ledger at
`data/gold/holdout_evaluations.jsonl` was empty at that moment and remains so,
which is the record that the choice was made blind.

*Reason for the maturation rule.* The Cohort A analysis reported on 2026-09-20
was computed while **416 of its 852 posts were still inside their 14-day
observation window**. Cohort A's membership was frozen by publication date on
2026-09-16, but its *outcomes* continue to mature: a post currently recorded as
right-censored may yet be observed to die. The effect is already measurable —
between two runs hours apart on the same frozen membership, observed deaths rose
from 498 to 500. Reporting a half-matured analysis understates deaths,
overstates the censoring rate, and would leave every figure in section 4 liable
to drift for another nine days.

It would also confound the comparison the holdout exists to make. If Cohort A
were analysed at partial maturity and Cohort B at full maturity, any difference
between them would mix temporal generalisation — the thing being tested — with
how long each cohort happened to be watched. Holding both to the same completed
window is what makes the comparison interpretable.

*Status of the outcome data at amendment.* Cohort A results have been examined
and are reported in the paper draft; they are explicitly provisional under this
amendment and will be superseded by the matured run. **Cohort B has not been
opened, inspected, or evaluated.** No model selection decision was informed by
anything in this entry: the close date is a calendar choice made blind, and the
maturation rule is mechanical and outcome-independent.

**2026-09-05 — Instagram Cohort B ran with an 18-hour hole, and the collector
paid for it.**

*Change.* None to the design. This entry records a collection defect and its
effect on Cohort B's observation record, so the gap is reported rather than
discovered later in the data.

*What happened.* Between 2026-09-04T00:00Z and 2026-09-04T17:00Z the Scrape
Creators profile endpoint returned HTTP 200 with an empty payload for all three
Cohort B accounts. Eighteen consecutive cycle-hours recorded `posts_seen: 0`.
The endpoint recovered at 2026-09-04T18:00Z and the remaining rounds completed
normally; the cohort ran its full 24 rounds and closed on schedule at
2026-09-05T11:00Z.

*Why it cost more than the data.* The cadence gate decided whether a round was
due by looking at when data was last written to bronze, not at when a round was
last attempted. A round returning no posts writes no snapshots, so the gate's
reference timestamp stopped advancing and every 30-minute CI invocation
qualified as "due". Credits remaining fell from 161 to 41 across the window:
120 spent against the 18 the schedule called for, so 102 were wasted outright,
on a budget that cannot be topped up. The failure was invisible in the cycle
reports because an empty payload is not an exception, so `errors` stayed empty
and each round recorded itself as clean.

*Effect on the data.* Cohort B has no Instagram observations in that 18-hour
window. Posts published inside it were not seen until the endpoint recovered,
and posts already being tracked carry a gap of up to 18 hours in their series,
which for some will fall below the four usable intervals a death label
requires. The affected posts are identifiable from the snapshot timestamps and
are counted in the attrition table under `insufficient_observations` like any
other. Velocity is computed on actual observed timestamps, so the gap is
recorded honestly rather than absorbed into an assumed grid.

*Status of the outcome data at amendment.* No hypothesis is tested on Instagram
data (section 4), so no confirmatory result is affected and the YouTube
analysis is untouched. Cohort B's data is retained and reported, including this
gap.

*Fixed* (commit `6731e9c`). The gate now counts rounds attempted rather than
rounds that returned data; an empty profile is recorded as an error; and a
circuit breaker stops spending after three consecutive empty rounds. The
existing cohort tests all used a client that always returns a post, so the gate
had only ever been exercised on the happy path.

*A second defect found the same day* (commit `c36041e`). The collection monitor
could not detect an outage that was still in progress: its gap scan walks only
between the first and last hours that have data, so a gap with no closing edge
is outside the loop by construction, and the 12-hour alarm window CI passes
empties the cycle-hour set entirely once an outage outlasts it. Simulated
outages of 2 to 96 hours all reported `STATUS: healthy` and exited 0. Since the
README identifies a silent outage as this project's most expensive failure, the
completeness figures reported in the papers rest on that alarm working; it now
measures time since last data from the full unwindowed history and fails the
build past three hours.

*One further correction, same date* (commit `dd2bc7a`). The screening artifact
`config/ig_screening.json` cited by the 2026-09-02 amendment did not contain the
Cohort B measurements. `ig_screen.run()` wrote to that path unconditionally and
the test suite calls it with placeholder handles, so running the tests
overwrote the real measurements with mocked failures, and that state was
committed. The genuine artifact has been recovered from commit `a8b044a` and
does back the account table: spotify 1.5 posts/day, bonappetitmag 1.3, 9gag
6.1, all matching. Two caveats in the recovered artifact are now recorded in
`config/channels.yaml` — it was measured for a 48h window, and `bonappetitmag`
carries `verdict: REJECT` at that duration, which extending the cohort to 72h
is what overturned. The screening tool now takes an output path and the tests
write elsewhere.

**2026-09-02 — Instagram Cohort B: three accounts over 72h, not five over 48h.**

*Change.* Section 4 describes Instagram as "bounded 48h cohorts of 5–6
accounts". Cohort B instead runs **three accounts at 3h intervals for 72
hours** (24 rounds, 72 credits). The account-selection criterion is
restated as a measured one: posting rate estimated from the profile grid
**excluding pinned posts**.

*Reason.* Cohort A executed the plan as written and produced almost
nothing usable: 326 posts at a median of **one** snapshot each, 43
analysable, **one** observed death. The cause is structural rather than
bad luck. The profile endpoint returns only ~12 posts, so a post is
observable only until twelve newer ones displace it; five fast-posting
accounts therefore generate a wide, shallow panel in which almost no post
accumulates the four intervals a death label requires. Spending the same
credits on fewer accounts over a longer window buys depth instead of
width, which is the quantity the label actually needs.

*The selection defect.* Cohort A's accounts **were** screened, and the
screening was wrong. Posting rate was measured from the span of the
returned grid without excluding pinned posts, and a pinned post sits at
the top of the grid indefinitely. `@sarcastic_us` screened at 0.5
posts/day; excluding its single pinned post from eight days earlier, the
true rate was **37.6 posts/day**. It went on to contribute 79 posts at 1.9
snapshots each. The screening tool now excludes pinned posts and retains
its raw measurements so the same class of error is detectable without
spending credits again.

*Status of the outcome data at amendment.* No hypothesis is tested on
Instagram data (section 4), so no confirmatory result is affected. The
YouTube analysis is untouched. Cohort A's data is retained and reported,
including its failure, and its account list is preserved in
`config/channels.yaml` so the earlier records stay interpretable.

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
| 2026-09-02 | *this commit* | Instagram Cohort B: 3 accounts x 72h, pinned-aware screening | Cohort A yielded 1 death; its screening was distorted by pinned posts; see above |
| 2026-09-05 | *this commit* | No design change; records an 18h Instagram collection gap, the gate defect that tripled spend during it, a blind outage alarm, and the recovery of the screening artifact | API returned empty payloads 2026-09-04T00:00-17:00Z; 102 credits wasted; see above |
| 2026-09-20 | *this commit* | Cohort B closes 2026-10-04; both cohorts analysed only after every member completes its 336h window | An unbounded holdout grows on every collection run; and Cohort A was analysed with 416 of 852 posts still maturing, deaths already drifting 498→500; see above |
| 2026-10-16 | *this commit* | Confirmatory feature set fixed as the 11 that ran, exploratory set barred from Cohort B; Cohort B procedure, command and reading rule declared before opening; harness adds the plan's paired-difference intervals, RMSE, comparators and saved predictions; records a quota outage and that the 336h observation was never taken | Section 5 named 5 variables no model used, found only after the results; a holdout must be specified before it is opened; checking the 2026-10-15 outage's effect showed no post was ever observed past 241h, so the effective window is 240h; see above |
| 2026-10-18 | *this commit* | Exploratory extension declared: a practical early-warning model (registered forest plus likes/comments rates, trained on Cohorts A+B), tested once on a new Cohort C (published 2026-10-04 to 2026-10-15) on or after 2026-10-25, with its pass/fail rules fixed | Built after the main results, so it needs an untouched test set and rules written before that set is examined; not a test of H1 or H2; see above |
