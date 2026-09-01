# Content Death Clock: Predicting Time-to-Attention-Death for YouTube Videos from First-Session Engagement Signals

**Amogha V Prasad · S Anannya · Sanidhya Tiwari · Shubhang Srinivas Varda**
RV University, Bengaluru — 7th Semester, Research Methodology

> **Draft status.** Method (§3) is final and follows `ANALYSIS_PLAN.md`, frozen
> 2026-08-30, plus two dated amendments reproduced in Appendix A. Results (§4) are
> scaffolded: every table exists in its final shape with `[PENDING]` in place of numbers
> that do not yet exist. Cohort A closes 2026-09-16T00:00:00Z.

---

## Abstract

**Background.** Research on social media content overwhelmingly predicts *growth* — which
posts will go viral, how large a cascade will become, what a video's view count will be at
some future horizon. The complementary question is barely asked: given that attention is
finite and decays, *when does a post stop receiving meaningful attention?*

**Objective.** We test whether the time at which a post's engagement velocity collapses can
be predicted from signals observable in its first seven hours, and specifically whether
early *dynamics* carry information beyond the creator's audience size.

**Method.** Prospective observational panel. A pre-registered, frozen analysis plan
specifies hypotheses, exclusion rules, models, baselines and inference before any outcome
data was examined. Sixty-three public YouTube channels, stratified by subscriber tier and
content category, were tracked by an automated collector taking snapshots at thirteen
scheduled ages between 1 and 336 hours. Attention death is defined per post as the first
time engagement velocity falls below 5% of that post's own peak velocity and remains below
for two consecutive intervals; posts that never reach it are right-censored, not discarded.
Predictions are made at a **landmark of t = 7h** and the outcome is the time *remaining*
from the landmark. Models are Weibull AFT and Random Survival Forest, evaluated with
`GroupKFold` grouped by creator against four baselines including a creator-size-only model.

**Results.** `[PENDING — Cohort A freeze 2026-09-16]`

**Conclusion.** `[PENDING]`

**Keywords:** attention decay, survival analysis, right-censoring, landmark analysis,
social media, pre-registration, YouTube

---

## 1. Introduction

Attention is the scarce resource in online media, and it is spent, not stored. A post
accumulates engagement quickly, plateaus, and then stops mattering — but the literature has
concentrated almost entirely on the first half of that arc. Popularity prediction, cascade
prediction and virality detection all ask a growth question: *how big will this get?* The
decay half of the curve is treated as a nuisance to be modelled away rather than as the
quantity of interest.

We invert the question. Given a post and its first few hours of behaviour, **how long until
its attention dies?**

The inversion is not cosmetic; it changes what has to be measured and how the problem must
be posed statistically.

1. **The outcome is a time, not a count.** "How many views at 30 days" is a regression
   target. "When does velocity collapse" is a *duration*, and durations observed inside a
   finite window are systematically incomplete — some posts have not died yet when
   observation stops. Discarding them biases every estimate toward short-lived content,
   which is exactly the wrong direction. This makes survival analysis with right-censoring
   the correct frame rather than a stylistic preference.
2. **The outcome must be scale-free.** A channel with twenty million subscribers and a
   channel with two thousand differ by four orders of magnitude in raw velocity. An
   absolute threshold ("fewer than ten views per hour") would measure channel size, not
   decay. We therefore normalise each post against **its own** peak velocity, so a large
   video and a small video that decay identically receive identical labels.
3. **The prediction must be made while the post is still alive.** If a post dies at 3.1
   hours and its features are built from its first 6 hours, the "prediction" is a
   restatement of the observations that defined the outcome. We found empirically that this
   affected 53% of the deaths in our first analysis, and adopted a landmark design in
   response (§3.5, Appendix A).

### 1.1 Why this is worth predicting

A decay estimate is directly actionable in ways a popularity estimate is not. It tells a
creator when a post has stopped repaying promotion; it tells a platform when a piece of
content should leave a recommendation surface; it tells an advertiser when incremental spend
against a placement stops buying incremental attention. All three are decisions about *when
to stop*, and none of them is answered by a forecast of eventual total views.

### 1.2 Contributions

- **An operational, per-post-normalised definition of attention death** that is invariant to
  creator scale, computed on actually-observed timestamps rather than an assumed schedule,
  and paired with a second, mechanistically different robustness label (time to 90% of a
  fitted saturation asymptote) so the conclusions can be checked against a definition that
  does not share the first one's failure modes.
- **A prospectively collected, pre-registered panel.** The analysis plan was committed to
  version control on 2026-08-30, when the dataset contained 179 posts, **zero** computed
  death labels and zero fitted models. Both subsequent deviations are logged as dated
  amendments with stated reasons, and both were made before any result was interpreted.
- **A landmark survival formulation** that makes the prediction task non-circular, together
  with the measurement that forced it.
- **Baselines chosen to be hard.** Beating "everything dies at 48 hours" proves nothing. Our
  second confirmatory hypothesis is stated against a **creator-size-only** model, so the
  claim under test is that early dynamics add information beyond knowing who posted.
- **A negative-result-tolerant reporting commitment** (§3.9) made in advance.

---

## 2. Related work and positioning

### 2.1 Popularity prediction from early signals

The foundational result is that early views predict later views remarkably well on a log
scale. Szabo and Huberman (2010) showed that the logarithm of a video's views at an early
time is strongly linearly related to its logarithm at a later time, which set the template
for the field: regress a future count on an early count. Pinto, Almeida and Gonçalves (2013)
improved on this by using the *shape* of the early view series rather than a single scalar.
Figueiredo and colleagues' TrendLearner classified content by its popularity *trend* — the
qualitative form of the curve — before predicting magnitude.

All of these predict a **count at a horizon**. None predicts a **time**.

### 2.2 Cascade prediction

Cheng, Adamic, Dow, Kleinberg and Leskovec (2014) reframed cascade prediction as a well-posed
binary problem ("will this cascade double in size?") and showed that temporal features
dominate content features. More recent deep-learning work on cascades — hierarchical and
graph-temporal architectures such as HierCas — continues in that direction with far greater
model capacity. The unit of analysis there is a diffusion structure over a social graph, and
the target remains growth.

Our unit of analysis is a single post's own engagement time series, which is what is
observable through public platform APIs for the overwhelming majority of content, and our
target is its cessation.

### 2.3 Collective attention and decay

Wu and Huberman (2007) established that collective attention to novel items decays, and
modelled it with a stretched-exponential form. This is the closest ancestor of our question,
but it is a *descriptive*, population-level model of how attention fades in aggregate. It
does not produce a per-item, ahead-of-time prediction of when a specific item's attention
will end, and it is not evaluated as a predictive task.

### 2.4 Survival methods applied to content

**We must be careful not to overclaim here.** Censoring-aware modelling of online content
lifetime is *not* unprecedented. Lee, Moon and Salamatian (2012) applied a Cox proportional
hazards model to the popularity dynamics of online content, treating item lifetime as a
survival outcome with explanatory covariates. That work is the nearest methodological
precedent to ours and is cited as such.

What we add relative to it: (i) an outcome defined by *velocity collapse normalised to the
post's own peak*, rather than by thread inactivity or a fixed absolute threshold; (ii) a
**landmark** formulation that guarantees the subject is at risk at the moment of prediction;
(iii) pre-registration with a frozen plan and dated amendments; and (iv) a creator-size-only
baseline as the primary comparator, which turns the study into a test of whether dynamics
add information rather than a demonstration that a model fits.

### 2.5 Positioning summary

| Work | Target | Outcome type | Censoring | Prediction time |
|---|---|---|---|---|
| Szabo & Huberman (2010) | Views at horizon | Count | Not handled | Fixed early time |
| Pinto et al. (2013) | Views at horizon | Count | Not handled | Fixed early time |
| TrendLearner (Figueiredo et al.) | Popularity trend class + magnitude | Class / count | Not handled | Early window |
| Cheng et al. (2014) | Cascade doubling | Binary | N/A | Cascade size k |
| HierCas and successors | Cascade size | Count | Not handled | Observation window |
| Wu & Huberman (2007) | Aggregate attention decay | Descriptive fit | N/A | Retrospective |
| Lee, Moon & Salamatian (2012) | Content lifetime | **Duration (Cox PH)** | **Handled** | Covariates at entry |
| **This work** | **Time to velocity collapse** | **Duration (AFT / RSF)** | **Handled** | **Landmark t = 7h** |

> **Citation verification required before submission.** Full bibliographic details for
> TrendLearner, HierCas and SMTPD must be checked against the source PDFs — in particular
> HierCas's venue and year, and SMTPD's sampling grid density, which determines whether that
> dataset could in principle support a decay label at all. Do not submit with these
> unverified.

---

## 3. Method

### 3.1 Design

Observational, prospective panel. Non-experimental: nothing is manipulated, no causal claim
is made, and the language throughout is predictive rather than causal.

### 3.2 Sampling frame

The frame is **63 public YouTube channels**, recorded in `config/channels.resolved.yaml`
with their subscriber counts as measured at frame-resolution time (2026-08-30T17:21 UTC),
stratified by subscriber tier and content category:

| Tier | Definition | Channels |
|---|---|---|
| micro | < 10,000 subscribers | 16 |
| mid | 10,000 – 500,000 | 17 |
| large | > 500,000 | 30 |
| **Total** | | **63** across 8 content categories |

**How the frame was built, and why it was rebuilt.** An initial convenience sample of
channels the authors follow produced 30 large / 2 mid / 1 micro. That distribution cannot
support H3 and would have made any claim about creator size vacuous. The frame was therefore
rebuilt by API search over a **committed query list** (`config/discovery_queries.yaml`): for
each query we searched *recent videos ordered by date*, resolved the channels behind them,
and admitted those falling in the under-represented tiers with an upload rate of at least
one per week.

Recent-video search was used rather than channel search deliberately. Channel search ranks
by relevance, which is a proxy for popularity, and therefore structurally cannot surface
sub-10,000-subscriber channels — precisely the stratum we lacked.

**The sampling claim, stated plainly.** This is **not** a random sample of YouTube. It is the
set of channels reachable by a fixed, version-controlled query list, filtered by size and
upload frequency. We do not claim generalisation beyond that frame. Because the queries are
committed, the frame is reproducible by a third party — a weaker claim than random sampling
and a considerably stronger one than an unrecorded convenience sample.

**Known frame biases**, stated in advance:

- the query list leans toward Indian and long-tail creators;
- channels uploading less than weekly are excluded by construction, so the frame
  over-represents frequent uploaders;
- micro-tier picks span 1,240–7,250 subscribers and mid-tier 10,800–373,000, so neither tier
  covers its full nominal range.

### 3.3 Data collection instrument

An automated collector runs every 30 minutes. Each cycle: (i) discovers posts published by
frame channels within the last `discovery_lookback_hours` = 6h, (ii) admits them subject to
the exclusion rules in §3.6, and (iii) re-measures every admitted post whose age has crossed
one of thirteen scheduled marks.

**Snapshot schedule (hours after publication):** 1, 3, 6, 12, 24, 36, 48, 72, 96, 120, 168,
240, 336. Tolerance 0.75h.

Three properties of the instrument matter for validity and are engineered rather than
assumed:

- **Ages come from observed timestamps.** A snapshot that arrives late is recorded as late.
  No velocity in this study is computed against an assumed grid.
- **Zero is not missing.** A metric the creator has hidden arrives as null and stays null.
  Coercing it to 0 would read as a genuine observation of no engagement, which the death
  label would score as instant death.
- **Writes are idempotent.** Records are keyed on `(post_id, snapshot_ts)` and merged, so a
  re-run cannot duplicate an observation. A duplicated snapshot silently halves a velocity.

The scheduler itself was measured rather than trusted; see §3.10 and the technical report.

### 3.4 Measures

#### Dependent variable — attention death

`t_death` is the first time at which engagement velocity falls below **5% of that post's own
peak velocity** and remains below for **2 consecutive intervals**. Velocity between
consecutive observations *i−1* and *i* is

> v_i = (C_i − C_(i−1)) / (a_i − a_(i−1))

where C is the cumulative primary metric and a is age in hours, both as actually observed. A
post that never satisfies the condition by its last observation is **right-censored at that
observation**, not dropped.

Cumulative counts occasionally decrease (platforms retract inflated views). Such
non-monotonic points are clamped, and **the number of repairs is counted and reported**
rather than silently absorbed.

#### The primary metric differs by platform, and this is substantive

YouTube uses **views**; Instagram uses **likes**. Instagram reports no view count at all for
image posts, so views cannot serve there. Likes saturate faster than views, which means
**death times are not comparable across the two platforms**. Per-platform results are
reported separately and are never pooled.

#### Robustness dependent variable

`t_saturation` — time to reach 90% of the asymptote A from a fitted C(t) = A(1 − e^(−kt)).
Reported alongside the primary label. Systematic disagreement between the two is a finding to
report, not a problem to conceal.

#### Independent variables — hard constraint: nothing observed after the landmark

Early value and velocity at 1h, 3h and 6h; log growth ratios between consecutive snapshots;
acceleration; video duration; title length; tag count; publish hour and weekday; content
category; **creator subscriber count**; creator historical median velocity.

The leakage boundary is enforced in code, not by discipline: features are computed by a
function that receives only observations at or before the cutoff, and a test asserts that a
deliberately leaky split scores differently from a grouped one on pure noise (§3.10).

One detail worth stating because it is easy to get wrong: log growth ratios are computed
**without epsilon smoothing**. Adding a constant to the denominator distorts small quantities
far more than large ones, which would have biased the micro tier specifically and
contaminated H3. Where the baseline is zero the feature is missing, which is the honest
encoding.

### 3.5 Landmark design

Predictions are made at a **landmark of t = 7 hours**. A post is eligible only if it was
**observed and still alive** at the landmark, and the outcome modelled is the time
**remaining** from the landmark to attention death. Both the landmarked outcome and the raw
time-from-publication are retained in the analysis frame, so a sensitivity analysis requires
no re-derivation.

**Why.** With the outcome measured from publication, 53% of observed deaths fell inside the
6-hour feature window. A post that dies at 3.1h has its death determined by the very
observations its features are built from; predicting it is circular. Landmarking is the
standard survival remedy — a subject must be at risk at the moment the prediction is made.

**Why 7 hours and not 6.** The collector runs every 30 minutes, so the nominal 6-hour
measurement lands at a median age of 6.08h. A strict 6.00h cutoff discarded it for 72 of 86
posts, which is why every 6-hour-derived feature was 0% available in the first run. The
snapshot schedule contains nothing between 6h and 12h, so **any** observation at or before 7h
is one of the scheduled 1/3/6-hour measurements arriving slightly late. Usable feature
columns went from 4 to 8.

This is Amendment 2 (Appendix A), made when the outcome data contained seven observed deaths,
with no model selection performed and no result interpreted.

### 3.6 Exclusions

All exclusions are pre-specified, applied in one place, and **counted**. The attrition table
in §4.1 is generated directly by the analysis code.

| Rule | Reason code |
|---|---|
| Fewer than 4 usable inter-snapshot intervals | `insufficient_observations` |
| Never achieved positive velocity | `no_positive_velocity` |
| Creator has hidden like/view counts | `counts_hidden` |
| Died at or before the landmark | `died before the landmark` |
| Not yet observed at the landmark | `not yet observed at the landmark` |
| Beyond 5 posts from one creator on one UTC day (YouTube only) | `creator_daily_cap` |
| Deleted or made private during observation | recorded as **censored**, not excluded |

The creator daily cap is Amendment 1 (Appendix A). It exists because one channel published
**100 videos within 25 seconds** and came to supply 59% of the YouTube panel, at a median of
4 views against 1,152 for every other post. Those are not 100 independent observations — one
creator, one moment, one action.

### 3.7 Models and baselines

**Models.** Weibull accelerated failure time and Random Survival Forest (primary, both
censoring-aware); log-time linear regression and gradient boosting as comparators on the
uncensored subset. The two primary models are chosen to fail differently: AFT buys
interpretable coefficients at the price of a rigid parametric shape, the forest buys
flexibility and interactions at the price of interpretability.

**How the forest's prediction is summarised, and one consequence for the metrics.** A
forest predicts a survival *curve* per post, but the evaluation requires a scalar time. The
conventional summary — the median, i.e. where the curve crosses 0.5 — is undefined for most
of our sample, because with high censoring most predicted curves never reach 0.5 inside the
observation window. We therefore summarise by **restricted mean survival time**: the area
under the predicted curve up to the last time observed in the training fold.

RMST is always defined and orders posts correctly, so the C-index — the plan's primary
metric — is unaffected. But RMST is *restricted*: it is truncated at the training horizon
and so systematically under-states long lifetimes. **The forest's MAE is therefore not
directly comparable with the AFT model's, which extrapolates freely, and the paper reports
the two MAEs with that caveat attached rather than ranking them against each other.** We
could remove the truncation by fitting a tail beyond the horizon; we deliberately do not,
because extrapolating past the data here would contradict the no-extrapolation rule the
feature layer already enforces (§3.4).

**Baselines, all four reported:**

1. **Constant 48h** — "all content dies at two days."
2. **Training-set median lifetime** (Kaplan–Meier median).
3. **Creator-size-only** — subscriber count alone.
4. **Peak-velocity heuristic** — extrapolate from the largest observed velocity.

Baselines 3 and 4 are the ones that matter. Baseline 3 is what H2 is defined against.

### 3.8 Hypotheses, validation and inference

**H1 (confirmatory).** A model using features observed at t ≤ 7h predicts
time-to-attention-death more accurately than the best naive baseline (1 or 2).
*H1₀*: model ≤ best naive baseline. *H1₁*: model > best naive baseline.

**H2 (confirmatory — the one that matters).** Early engagement dynamics carry predictive
information beyond creator size: the model beats the creator-size-only baseline (3).

**H3 (exploratory, explicitly not confirmatory).** Time-to-death differs across creator-size
strata and content categories.

**Validation.** `GroupKFold(n_splits=5)` **grouped by creator**. Random k-fold would place
the same creator in train and test, leaking creator identity and inflating every score. This
is pre-specified precisely so it cannot be quietly relaxed later.

**Temporal holdout.** Posts published on or after 2026-09-16T00:00:00Z form **Cohort B** and
are evaluated **exactly once**, after all model selection is complete. The instant was fixed in
`settings.yaml` on 2026-08-30, before any outcome data was examined.

**The "exactly once" commitment is enforced, not merely stated.** A holdout that can be
inspected casually stops being one: each look leaks into the choices made next, and after
several looks it has quietly become a second validation set. Cohort B therefore cannot be
evaluated by the ordinary command — it requires an explicit unlock flag, and every evaluation
is appended to a committed ledger recording the timestamp, the git commit, the sample size and
a digest of the results. A repeat evaluation prints a warning naming the date of the first.

The ledger is a record, not a lock; anyone determined can run it twice. That is the intent. The
purpose is to make a second look **visible**, including to ourselves, which is the same
reasoning that motivates pre-registration in the first place. If Cohort B is evaluated more
than once, the paper reports every evaluation, not the preferred one.

**Metrics.** Harrell's C-index (primary, censoring-aware); MAE and RMSE on log₁₀
time-to-death on the uncensored subset; a calibration plot of predicted against observed.

**Inference.** Paired Wilcoxon signed-rank test on per-post absolute errors, model against
each baseline. Bootstrap confidence intervals with 2,000 resamples, **resampled by creator**
rather than by post, because posts within a creator are not independent. α = 0.05. Effect
sizes are reported alongside p-values, and a p-value alone will not be treated as a result.

**Multiplicity.** H1 and H2 are the only confirmatory tests. Everything else is labelled
exploratory in this paper.

**Power.** The evaluation harness refuses to interpret a run with fewer than 50 observed
deaths or fewer than 10 creators, and prints sample size, death count and creator count
*before* any metric. The threshold was set in advance and is deliberately generous: even at
50 deaths a C-index is unstable.

### 3.9 Pre-registration, and what we will report regardless of outcome

`ANALYSIS_PLAN.md` was committed on **2026-08-30**, when the repository contained 179
collected posts, **0 computed death labels**, 0 fitted models and 0 examined results. That
state is recorded inside the plan itself, so the claim is auditable rather than asserted.

Committed in advance:

- **If the model does not beat the baselines, that is the reported result.** A negative
  finding honestly reported with a power discussion is a legitimate outcome. Searching for a
  specification that wins is not.
- Censoring rate, attrition by reason code, collection completeness and outages, and the
  count of monotonicity repairs are all reported whatever they say.
- Every deviation from the plan is reported with its reason and its date.

### 3.10 Reproducibility and instrument validation

All code, configuration, the frozen plan, the sampling frame and the raw collected data are
in a public repository. Analysis parameters live in a single `settings.yaml`, so the plan can
reference exact values rather than describing them.

Two instrument checks are worth reporting in a methods paper, because both found real faults:

- **A leakage test with teeth.** An assertion that grouped cross-validation *passes* is not
  evidence of anything unless a leaky split *fails*. On a synthetic null cohort containing no
  signal, grouped CV scores 0.491 (chance) while a deliberately leaky split scores 0.804. The
  test is therefore capable of detecting the failure it exists to prevent.
- **Label unit tests.** The death-label function is tested against synthetic curves with
  analytically known death times. It is the one component where a silent bug invalidates
  every downstream number.

The suite contains 157 tests and runs in CI on every push.

---

## 4. Results

> **All numbers in this section are `[PENDING]` until Cohort A closes on 2026-09-16.**
> Appendix B contains a dress-rehearsal run, included as evidence that the pipeline executes
> end to end. It has 7 observed deaths and is not a result.

### 4.1 Sample and attrition

Generated by `python -m cdc.eval.report`.

| Stage | Posts |
|---|---|
| Observed on platform | `[PENDING]` |
| After `creator_daily_cap` | `[PENDING]` |
| After Cohort A cutoff | `[PENDING]` |
| − died before the landmark | `[PENDING]` |
| − not yet observed at the landmark | `[PENDING]` |
| − `insufficient_observations` | `[PENDING]` |
| − `no_positive_velocity` | `[PENDING]` |
| **Final analysis set** | `[PENDING]` |

Observed deaths `[PENDING]`; censored `[PENDING]` (`[PENDING]`%); creators `[PENDING]`;
monotonicity repairs `[PENDING]`.

Collection completeness over the window: `[PENDING]` — cycles executed against cycles
scheduled, with any outage listed by date and duration.

### 4.2 Descriptive survival

- **Figure 1.** Kaplan–Meier estimate of survival past the landmark, whole sample.
- **Figure 2.** Kaplan–Meier stratified by creator-size tier, with a log-rank test
  (exploratory).

Median time to attention death `[PENDING]`, with 95% CI.

### 4.3 H1 — model against naive baselines

| Model | C-index | 95% CI (creator bootstrap) | MAE log₁₀ |
|---|---|---|---|
| Weibull AFT | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| Random Survival Forest | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| Constant 48h | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| KM median | `[PENDING]` | `[PENDING]` | `[PENDING]` |

Paired Wilcoxon, model against best naive baseline: `[PENDING]`.

### 4.4 H2 — model against creator size alone

| Comparison | Δ C-index | 95% CI | Wilcoxon p |
|---|---|---|---|
| Weibull AFT − creator-size-only | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| RSF − creator-size-only | `[PENDING]` | `[PENDING]` | `[PENDING]` |

> **Note for the authors, to be removed before submission.** In the dress rehearsal the
> creator-size-only baseline *outscored* the full model (0.817 against 0.713) on seven
> deaths. At that sample size the intervals overlap almost entirely and this is noise. It is
> recorded here because H2 is the hypothesis that matters, and because noticing it now, in
> advance, is what stops it from becoming a result we quietly reframe later. If it holds at
> full sample, **that is the finding and it gets reported as the finding.**

### 4.5 H3 — exploratory stratification

Time-to-death by creator tier and by content category, with the caveat that the frame does
not cover the full range of either. `[PENDING]`

### 4.6 Robustness

#### 4.6.1 The second outcome definition

The plan commits to a second, mechanically different outcome: `t_saturation`, the time to
reach 90% of a fitted asymptote. The two labels fail differently on purpose — the velocity
label is non-parametric and local, sensitive to one noisy interval; the saturation label is
parametric and global, sensitive to the assumed functional form being wrong. Agreement
between them means a finding is not an artefact of either.

Reported for both labels: coverage of the curve fit, rank correlation between the two
durations, the typical ratio between them, and the whole evaluation re-run with saturation
as the outcome (`python -m cdc.eval.report --outcome saturation`).

| Quantity | Value |
|---|---|
| Saturation fit succeeded | `[PENDING]` |
| Fitted saturation beyond last observation (extrapolated) | `[PENDING]` |
| Saturated before the landmark | `[PENDING]` |
| Posts with both labels | `[PENDING]` |
| Spearman ρ between `t_death` and `t_saturation` | `[PENDING]` |
| Median `t_saturation` / `t_death` | `[PENDING]` |
| H1 verdict under the saturation outcome | `[PENDING]` |
| H2 verdict under the saturation outcome | `[PENDING]` |

Two caveats must accompany this table whatever it says, because both are structural rather
than sample-dependent:

- **Some fitted saturation times lie beyond the post's last actual observation.** Those are
  extrapolations of the fitted curve, not measurements. The count is reported so the reader
  can judge how much of the robustness check rests on them.
- **The saturation outcome has no censoring by construction** — every successful fit is an
  event. Censoring-aware models therefore cannot show their advantage under this label, so
  it is a weaker comparison than the primary one, not an equal second opinion.

#### 4.6.2 Other robustness checks

- Landmark sensitivity: results recomputed at landmarks of 3h and 12h. `[PENDING]`
- Threshold sensitivity: velocity fraction at 0.02, 0.05, 0.10. `[PENDING]`
- Feature coverage: which of the offered features actually survived the 50% coverage
  requirement. `[PENDING]`

### 4.7 Cohort B — temporal holdout

Evaluated once, after all model selection is complete. `[PENDING]`

The holdout ledger (`data/gold/holdout_evaluations.jsonl`) is reproduced here in full,
whatever it contains — number of evaluations, dates, and the code commit each ran against.
As of drafting it is **empty**: the freeze instant has not passed and Cohort B contains
0 posts. `[PENDING]`

A temporal generalisation gap, if one appears, is reported rather than explained away. Cohort B
posts are collected under the same instrument as Cohort A but at a later date, so a drop
between the two is evidence about stability over time — which is more informative than another
cross-validation fold.

### 4.8 Instagram — feasibility only

Instagram is a **cross-platform feasibility demonstration, not a second test set**, and no
hypothesis above is tested on it. Cohort A ran 16 rounds at 3-hour intervals across 5
accounts over 47.0 hours, inside a hard budget of 100 API credits per key. Yield:
`[PENDING]` labellable posts, `[PENDING]` observed deaths. Its purpose is to show that the
pipeline generalises across platforms and to supply one cross-platform figure.

---

## 5. Discussion

`[PENDING — write after §4]`

Points the discussion must cover regardless of which way the results fall:

- What a C-index of the observed magnitude means in practical terms for someone deciding
  when to stop promoting a post.
- Whether early dynamics added information over creator size, and what it means if they did
  not — a null on H2 would say decay timing is largely a property of the audience rather than
  the post, which is itself an interesting and reportable claim.
- Whether the two label definitions agreed, and what a disagreement would imply about the
  construct "attention death".

## 6. Limitations

Stated in advance in the frozen plan, not discovered afterwards.

1. **Observational design.** No causal claim. We predict decay; we do not explain it.
2. **Not a random sample.** Generalisation beyond the committed frame is not claimed. The
   frame over-represents frequent uploaders by construction, and neither the micro nor the
   mid tier spans its nominal range.
3. **Platform-reported metrics.** Views and likes are numbers the platform chooses to show
   us. We cannot audit them, and we observed retractions.
4. **Window truncation.** A 336-hour horizon truncates genuinely long-lived content. This is
   what censoring-aware modelling exists to handle, but it bounds what can be said about the
   tail.
5. **Unofficial third-party API for Instagram.** Scrape Creators is not an official endpoint.
   This is a stated limitation and one reason Instagram carries no hypothesis.
6. **Likes and views are not the same construct**, so the two platforms' death times are
   reported separately and never pooled.
7. **Statistical power.** `[PENDING — state the achieved number of deaths and what it does
   and does not support.]`

## 7. Ethics

Public content from public accounts only. No private data, no personal data beyond what a
logged-out visitor sees, and no attempt to identify individuals. Results are reported in
aggregate and creators are not named in the analysis. The YouTube Data API v3 is used within
its published quota; collection is rate-limited and metered by design. The use of an
unofficial third-party API for Instagram is disclosed plainly rather than buried. The study
is observational and involves no intervention on any user or creator.

## 8. Conclusion

`[PENDING]`

---

## References

*To be formatted in the department's required style. Entries marked ⚠ need bibliographic
verification against the source PDFs before submission.*

1. Szabo, G., & Huberman, B. A. (2010). Predicting the popularity of online content.
   *Communications of the ACM*, 53(8), 80–88.
2. Wu, F., & Huberman, B. A. (2007). Novelty and collective attention. *PNAS*, 104(45),
   17599–17601.
3. Pinto, H., Almeida, J. M., & Gonçalves, M. A. (2013). Using early view patterns to predict
   the popularity of YouTube videos. *WSDM 2013*.
4. Cheng, J., Adamic, L., Dow, P. A., Kleinberg, J., & Leskovec, J. (2014). Can cascades be
   predicted? *WWW 2014*.
5. Lee, J. G., Moon, S., & Salamatian, K. (2012). Modeling and predicting the popularity of
   online contents with Cox proportional hazard regression model. *Neurocomputing*, 76(1),
   134–145.
6. ⚠ Figueiredo, F., Almeida, J. M., Gonçalves, M. A., & Benevenuto, F. TrendLearner: Early
   prediction of popularity trends of user generated content. *Information Sciences*.
7. ⚠ HierCas — hierarchical temporal graph model for cascade popularity prediction. *Venue
   and year to be verified.*
8. ⚠ SMTPD — social media temporal popularity dataset. *Venue, year and sampling grid density
   to be verified.*
9. Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations.
   *JASA*, 53(282), 457–481.
10. Harrell, F. E., Califf, R. M., Pryor, D. B., Lee, K. L., & Rosati, R. A. (1982).
    Evaluating the yield of medical tests. *JAMA*, 247(18), 2543–2546.
11. Anderson, J. R., Cain, K. C., & Gelber, R. D. (1983). Analysis of survival by tumor
    response. *Journal of Clinical Oncology*, 1(11), 710–719. *(landmark analysis)*
12. van Houwelingen, H. C. (2007). Dynamic prediction by landmarking in event history
    analysis. *Scandinavian Journal of Statistics*, 34(1), 70–85.
13. Davidson-Pilon, C. (2019). lifelines: survival analysis in Python. *JOSS*, 4(40), 1317.

---

## Appendix A — Amendment log

Reproduced from `ANALYSIS_PLAN.md`. Both amendments were committed as new commits with stated
reasons; neither edits history.

### Amendment 1 — 2026-08-31: per-creator daily admission cap (YouTube only)

**Change.** At most 5 posts from one creator per UTC day enter the YouTube analysis sample.
The earliest published are kept; the remainder are excluded under `creator_daily_cap` and
counted in the attrition table. No cap is applied to Instagram.

**Reason.** One channel published 100 videos within 25 seconds and came to supply 59% of the
YouTube panel, carrying a median of 4 views against 1,152 for every other post. Those are not
100 independent observations. The contamination reached us through a measurement bug: our
upload-rate check paged only the 50 most recent videos, so any channel uploading faster than
about 7 per day reported exactly 11.67 per week regardless of its true rate, and the "at least
weekly" filter admitted firehoses alongside active creators. That bug is fixed separately; the
cap protects the design from the general case.

**Instagram exempt.** Its accounts were selected deliberately for frequent posting, no
hypothesis is tested on Instagram data, and its collection cost non-renewable credits. Capping
it would discard paid-for data to solve a problem it does not have.

**State of outcome data at amendment.** No death label computed, no model fitted.

### Amendment 2 — 2026-09-01: landmark design; outcome becomes remaining lifetime

**Change.** Predictions are made at a landmark of t = 7h. Only posts still alive and observed
at the landmark are eligible, and the outcome is time remaining from the landmark.

**Reason.** 53% of observed deaths occurred inside the 6-hour feature window, making
prediction circular. Ten posts were excluded on this ground; they were never genuinely
predictable.

**Why 7 and not 6.** The 30-minute scheduler places the nominal 6-hour measurement at a median
of 6.08h; a strict 6.00h cutoff discarded it for 72 of 86 posts. Usable feature columns went
from 4 to 8.

**Related defect fixed the same day.** `follower_count` was never recorded for YouTube —
`videos.list` does not return subscriber counts — so it was 0% populated and the
creator-size-only baseline, which H2 is defined against, had degenerated into a constant. It
is now joined from the frozen resolved frame.

**State of outcome data at amendment.** Seven observed deaths, no model selection performed,
no result reported or interpreted. The change was made because the previous design was
incoherent, not because a result was unwelcome.

---

## Appendix B — Dress-rehearsal output (NOT A RESULT)

Run 2026-09-01T19:40 UTC on partial data, reproduced to demonstrate that the analysis executes
end to end and that the power guard works. **7 observed deaths. Do not cite.**

```
  observed on this platform                215
  after creator_daily_cap                  116
  excluded: died before the landmark       -10
  excluded: insufficient_observations      -44
  excluded: no_positive_velocity            -2
  FINAL analysis set                        60

  posts 60 | observed deaths 7 | censored 53 (88%) | creators 26
  *** UNDERPOWERED - DRESS REHEARSAL, not a result ***

  features used (8): log_value_at_3h, log_value_at_6h, velocity_to_6h,
    log_growth_3_6h, log_follower_count, log_duration_sec,
    publish_hour_utc, title_len

  model                    C-index          95% CI     MAE log10
  weibull_aft                0.713  [0.445, 0.951]         1.683
  random_survival_forest     0.731  [0.514, 0.889]         1.005
  constant_48h               0.500  [0.500, 0.500]         1.203
  km_median                  0.561  [0.337, 0.781]         1.140
  subscriber_only            0.817  [0.620, 0.953]         2.308
  peak_velocity              0.604  [0.309, 0.872]         2.292

  Wilcoxon: n/a for all 8 comparisons (7 pairs, below the minimum of 10)
```

Two things not to read into the above, beyond the sample size:

- The forest's **MAE of 1.005 is the lowest in the table, and that is not evidence that it
  is the most accurate model.** RMST truncation biases its predictions downward (§3.7), and
  most durations in this rehearsal are short, so the bias happens to point the right way
  here. The C-index is the metric that is comparable across these rows.
- `subscriber_only` still leads on C-index. See the note in §4.4.
