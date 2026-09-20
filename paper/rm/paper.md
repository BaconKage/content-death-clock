# Content Death Clock: Predicting Time-to-Attention-Death for YouTube Videos from First-Session Engagement Signals

**Amogha V Prasad · S Anannya · Sanidhya Tiwari · Shubhang Srinivas Varda**
RV University, Bengaluru — 7th Semester, Research Methodology

> **Draft status.** Method (§3) is final and follows `ANALYSIS_PLAN.md`, frozen
> 2026-08-30, plus the dated amendments reproduced in Appendix A. Cohort A closed
> 2026-09-16T00:00:00Z and **§4 is now complete**, with two declared exceptions: the
> landmark and threshold sensitivity analyses (§4.6.2) are outstanding, and Cohort B
> (§4.7) is sealed and unevaluated by design. §5 Discussion and §8 Conclusion remain to be
> written against the results now in hand.

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

**Results.** Cohort A comprises 686 posts from 47 creators, 498 of them observed to reach
attention death (27.4% censored); median time to death from the landmark is 89.4h
(95% CI 89.2–113.0). Both primary models discriminate better than chance — Random Survival
Forest C-index 0.621 (95% CI 0.569–0.678), Weibull AFT 0.600 (0.540–0.657) — so H1 is
supported on discrimination. It is not supported on absolute error: neither model beats a
constant-lifetime baseline (p = 0.40, p = 0.55), and both are significantly *worse* than
the Kaplan–Meier median (p = 0.006, p = 0.0006), which attains the lowest error of any
method tested. Against a creator-size-only baseline (C-index 0.599) the evidence for H2 is
inconsistent: the forest's paired Wilcoxon is significant (p = 0.019) but its bootstrap
interval on the C-index difference includes zero (+0.022, 95% CI −0.015 to +0.065), and the
Weibull AFT is indistinguishable from the baseline on every measure (Δ = +0.0008,
p = 0.760). Pre-registered robustness analyses weaken this further: a mechanically
different outcome definition reproduces neither hypothesis, and H2's single passing test
survives neither moving the landmark to 3h or 12h nor loosening the death threshold to
0.10.

**Conclusion.** Time to attention death is predictable from first-session signals to a
small but measurable degree, and creator size accounts for most of that predictability.
The marginal contribution of early engagement *dynamics* is not robust to a change of
model, of test statistic, or of outcome definition, and absolute predictions are imprecise
by roughly a factor of two regardless of method. We report this as the result, as
pre-registered on 2026-08-30, rather than as a specification to be improved upon.

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

> Cohort A closed on 2026-09-16T00:00:00Z as pre-specified. All numbers below were
> generated on 2026-09-20 from `data/gold/evaluation_youtube.json` and are reproducible
> with `python -m cdc.eval.report --platform youtube`. Appendix B retains the earlier
> dress-rehearsal run (7 observed deaths) as evidence that the pipeline executed end to
> end before the freeze; it is not a result and is not cited as one.

### 4.1 Sample and attrition

Generated by `python -m cdc.eval.report`.

| Stage | Posts |
|---|---|
| Observed on platform | 1,086 |
| After `creator_daily_cap` | 987 |
| After Cohort A cutoff | 753 |
| − died before the landmark | −58 |
| − `insufficient_observations` | −7 |
| − `no_positive_velocity` | −2 |
| **Final analysis set** | **686** |

Observed deaths 498; censored 188 (27.4%); creators 47; monotonicity repairs 68.

The 234 posts removed at the Cohort A cutoff are not lost — they constitute Cohort B, the
sealed temporal holdout (§4.7).

Collection completeness over the window was **93.9%** of scheduled observations, 92.9%
of them inside the tolerance of the mark they satisfied, with mean lateness +0.15h. One
YouTube outage occurred: 7 consecutive hours from 2026-08-31T02. A separate 18-hour
Instagram interruption on 2026-09-04 is documented in the amendment log; it affects the
feasibility demonstration only (§4.8) and no hypothesis is tested on that data.

### 4.2 Descriptive survival

- **Figure 1.** Kaplan–Meier estimate of survival past the landmark, whole sample.
- **Figure 2.** Kaplan–Meier stratified by creator-size tier, with a log-rank test
  (exploratory).

Median time to attention death, measured from the 7-hour landmark, is **89.4 hours**
(95% CI 89.2–113.0). Posts therefore typically retain measurable attention for roughly
four days past the landmark, with a long right tail — 27.4% of the sample was still alive
when observation ended.

### 4.3 H1 — model against naive baselines

| Model | C-index | 95% CI (creator bootstrap) | MAE log₁₀ |
|---|---|---|---|
| Weibull AFT | 0.5995 | [0.5398, 0.6566] | 0.3491 |
| Random Survival Forest | **0.6207** | [0.5692, 0.6781] | 0.3288 |
| Constant 48h | 0.5000 | [0.5000, 0.5000] | 0.3368 |
| KM median | 0.4608 | [0.4181, 0.5077] | **0.3150** |

*(`peak_velocity`: C-index 0.5632 [0.5162, 0.6077], MAE 0.3377; `subscriber_only` is
reported in §4.4.)*

**H1 splits by metric, and the split is the finding.**

*On discrimination, H1 is supported.* Both primary models rank posts better than chance,
and both confidence intervals exclude 0.5: RSF 0.6207 [0.5692, 0.6781], Weibull AFT
0.5995 [0.5398, 0.6566]. A constant predictor cannot order posts at all, so it sits at
0.5000 by construction.

*On absolute error, H1 is not supported.* Paired Wilcoxon against `constant_48h` is
non-significant for both models (RSF median −0.0346, p = 0.399; Weibull median +0.0098,
p = 0.547). Worse, against `km_median` — the training-set median lifetime, the second
naive baseline named in the plan — **both models are significantly *worse***: Weibull
median +0.0269, p = 0.0006; RSF median +0.0155, p = 0.0062.

`km_median` has the lowest MAE of any method tested (0.3150). Every method, from the
random survival forest to a single constant, falls within a narrow band — back-transformed,
all of them are wrong by a factor of roughly 2.1 (km_median 2.07×, RSF 2.13×, constant_48h
2.17×, peak_velocity 2.18×, subscriber_only 2.21×, Weibull AFT 2.23×).

This is not a contradiction. A model that distributes its predictions across a range takes
on variance that a constant avoids, and when the recoverable signal is weak that variance
costs more than the signal gains. The models order posts better than a constant while
naming the time less accurately than one. We report both, because the plan specifies both
and because reporting only the favourable metric is the precise failure pre-registration
exists to prevent.

**Multiplicity.** H1 and H2 are the only confirmatory tests (§3.8). The remaining
comparisons in the tables are descriptive and are not used to support the primary claim.

### 4.4 H2 — model against creator size alone

The creator-size-only baseline scores C-index **0.5987** [0.5439, 0.6461], MAE 0.3436.

| Comparison | Δ C-index | 95% CI (paired creator bootstrap, 2,000) | Wilcoxon p |
|---|---|---|---|
| Weibull AFT − creator-size-only | +0.0008 | [−0.0394, +0.0473] | 0.760 |
| RSF − creator-size-only | +0.0220 | [−0.0146, +0.0654] | **0.019** |

**The two pre-specified tests disagree, and H2 is therefore equivocal at best.**

The Weibull AFT is indistinguishable from creator size on every measure: a C-index gap of
0.0008 — three ten-thousandths — with an interval straddling zero and a Wilcoxon p of
0.760. On this model, early engagement dynamics add nothing.

The Random Survival Forest is the only comparison that passes anything. Its paired
Wilcoxon on per-post absolute errors is significant (median −0.0101 favouring the model,
p = 0.019). But the plan also specifies bootstrap confidence intervals on metric
differences, resampled by creator, and that interval **includes zero**: Δ C-index +0.0220,
95% CI [−0.0146, +0.0654].

The two tests measure different things — Wilcoxon compares per-post absolute errors, the
bootstrap interval compares discrimination — so they are not required to agree, and here
they do not. One pre-specified test supports H2 for one of the two models; the other does
not support it for either. We record this as **weak and inconsistent evidence for H2, not
as support.**

The substantive reading is that creator size accounts for most of what is predictable
about time-to-attention-death, and that the marginal contribution of early engagement
dynamics is small enough to disappear under a change of model or of test.

> **Recorded in advance.** Before the freeze, the draft carried a note observing that in
> the 7-death dress rehearsal the creator-size baseline *outscored* the full model (0.817
> vs 0.713), and committing that if this held at full sample it would be reported as the
> finding. At full sample the gap closed rather than reversed — the baseline now sits
> level with the Weibull and marginally below the forest — but the direction of the
> conclusion is unchanged: creator size is not beaten by a margin we can defend. The note
> is retained rather than deleted because having written down the uncomfortable
> possibility beforehand is part of the evidence that it was not reframed afterwards.

### 4.5 H3 — exploratory stratification

Time-to-death by creator tier and by content category, with the caveat that the frame does
not cover the full range of either. **Exploratory: these comparisons are not used to
support H1 or H2, and no correction for multiplicity is applied because no confirmatory
claim rests on them.**

| Creator tier | n | Deaths | Median time to death (h) |
|---|---|---|---|
| micro (<10k) | 151 | 130 | **41.5** |
| mid (10k–500k) | 365 | 265 | 89.5 |
| large (>500k) | 170 | 103 | **113.0** |

The ordering is monotone in creator size and the spread is large: posts from micro
channels reach attention death in roughly a third of the time that posts from large
channels do. This is consistent with the H2 result — creator size carries most of the
recoverable signal — and it is the single clearest pattern in the data.

| Category | n | Deaths | Median time to death (h) |
|---|---|---|---|
| technology | 111 | 69 | 113.0 |
| education | 221 | 177 | 89.3 |
| news | 46 | 35 | 89.4 |
| science | 95 | 63 | 89.3 |
| gaming | 56 | 44 | 89.3 |
| talk | 26 | 19 | 77.0 |
| entertainment | 131 | 91 | 65.2 |

Category differences are weaker and confounded with tier, since the frame was not
stratified to be balanced across the two simultaneously. Several category medians coincide
at ~89h, which reflects the discreteness of the snapshot schedule rather than a genuine
clustering: with observations at 72h and 96h, an event located between them resolves to a
small number of possible values. We report the table and decline to interpret it further.

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
| Saturation fit succeeded | 686/686 (100%) |
| Fitted saturation beyond last observation (extrapolated) | 71 (10.3%) |
| Saturated before the landmark | 19 (2.8%) |
| Posts with both labels | 498 |
| Spearman ρ between `t_death` and `t_saturation` | **+0.718** (p ≈ 4×10⁻⁸⁰) |
| Median `t_saturation` / `t_death` | **0.633** [IQR 0.500, 0.857] |
| H1 verdict under the saturation outcome | **Not supported** |
| H2 verdict under the saturation outcome | **Not supported** |

**The two labels agree on ordering but not on timing.** Rank correlation is strong
(ρ = +0.718), so the labels broadly sort posts the same way. But the median ratio of 0.633
says saturation is reached at roughly two-thirds of the time at which velocity collapses,
consistently. That is a substantive observation rather than a defect: a post stops
*growing* appreciably well before it stops *receiving* attention. The two constructs are
related but not interchangeable, and a study reporting only one of them would be
overclaiming.

**The robustness check does not reproduce the primary finding, and this materially weakens
it.** Re-running the full evaluation with `t_saturation` as the outcome
(`python -m cdc.eval.report --outcome saturation`, n = 667):

| Model | C-index | 95% CI | MAE log₁₀ |
|---|---|---|---|
| Weibull AFT | 0.538 | [0.487, 0.580] | 0.400 |
| Random Survival Forest | 0.515 | [0.472, 0.558] | 0.997 |
| Constant 48h | 0.500 | [0.500, 0.500] | 0.391 |
| KM median | 0.453 | [0.404, 0.516] | 0.376 |
| Subscriber-only | 0.439 | [0.385, 0.503] | 0.379 |
| Peak velocity | 0.455 | [0.411, 0.517] | 0.378 |

Both primary models have confidence intervals that **include 0.5** under this outcome, so
H1 fails. No model significantly beats the subscriber-only baseline (Weibull p = 0.124),
so H2 fails. The forest's MAE of 0.997 — an error factor of roughly ten — indicates the
model degrades badly when every observation is an event; we report it and do not interpret
it as a finding about attention.

The caveats stated in advance apply and are the reason this check is weaker than the
primary analysis rather than an equal second opinion: the saturation outcome has **0%
censoring by construction**, so censoring-aware models cannot demonstrate their advantage,
and 71 of the fitted times are extrapolations beyond the post's last actual observation.
Nonetheless, a robustness label that was chosen in advance, and that fails to reproduce
the primary result, is evidence against the primary result. We weight it accordingly in
§5.

Two caveats must accompany this table whatever it says, because both are structural rather
than sample-dependent:

- **Some fitted saturation times lie beyond the post's last actual observation.** Those are
  extrapolations of the fitted curve, not measurements. The count is reported so the reader
  can judge how much of the robustness check rests on them.
- **The saturation outcome has no censoring by construction** — every successful fit is an
  event. Censoring-aware models therefore cannot show their advantage under this label, so
  it is a weaker comparison than the primary one, not an equal second opinion.

#### 4.6.2 Other robustness checks

- **Feature coverage.** Eleven features cleared the 50% coverage requirement and entered
  the models: `log_value_at_1h` (95.0% coverage), `log_value_at_3h` (97.5%),
  `log_value_at_6h` (95.9%), `velocity_to_6h` (95.9%), `log_growth_1_3h` (90.8%),
  `log_growth_3_6h` (93.0%), `decay_ratio_3_6_over_1_3` (91.1%), `log_follower_count`
  (100%), `log_duration_sec` (95.8%), `publish_hour_utc`, `title_len`. Coverage above 90%
  on every early-engagement feature confirms that the landmark amendment (Appendix A,
  Amendment 2) fixed the problem it was introduced for: under the previous 6.00h cutoff
  those columns were 0% available. `caption_len` and `hashtag_count` are Instagram-only
  and are 0% populated on YouTube, as expected.

**Landmark sensitivity.** Recomputed at landmarks of 3h and 12h
(`python -m cdc.eval.report --landmark 3.0`). Features are capped at the landmark, so an
earlier landmark cannot read a later observation; at 3h this leaves only 5 features with
adequate coverage, against 11 at 7h and 12h.

| Landmark | n | Deaths | RSF C-index | Weibull C-index | Creator-size C-index | RSF vs creator-size |
|---|---|---|---|---|---|---|
| 3h | 704 | 517 | 0.5932 [0.540, 0.652] | 0.6026 | 0.5972 | **+0.0204, p = 0.006 — favours the *baseline*** |
| **7h (registered)** | 686 | 498 | **0.6207** [0.569, 0.678] | 0.5995 | 0.5987 | −0.0101, p = 0.019 — favours the model |
| 12h | 685 | 497 | 0.6185 [0.563, 0.681] | 0.6004 | 0.6012 | −0.0116, p = 0.394 — no difference |

**Threshold sensitivity.** Recomputed at velocity fractions of 0.02 and 0.10.

| Threshold | n | Deaths | Censored | RSF C-index | Weibull C-index | Creator-size | RSF vs creator-size | Weibull vs creator-size |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 691 | 390 | 43.6% | 0.6632 [0.600, 0.727] | 0.6527 | 0.6363 | **p < 0.0001** | **p = 0.031** |
| **0.05 (registered)** | 686 | 498 | 27.4% | 0.6207 | 0.5995 | 0.5987 | p = 0.019 | p = 0.760 |
| 0.10 | 673 | 584 | 13.2% | 0.5954 [0.553, 0.645] | 0.5720 | 0.5677 | p = 0.369 | p = 0.828 |

**These analyses do not support the robustness of H2. They undermine it.**

The support for H2 appears only inside a narrow band of analyst choices, and both
directions of movement dissolve it:

- Move the landmark **earlier** and the forest becomes *significantly worse* than creator
  size (p = 0.006, sign reversed). Move it **later** and the difference vanishes
  (p = 0.394). The registered 7h landmark is the only one of the three at which H2's single
  passing test passes.
- Loosen the death threshold to 0.10 and neither model beats creator size. Tighten it to
  0.02 and both do, comfortably.

The threshold pattern is monotone rather than noisy, which makes it interpretable: a
stricter threshold pushes death later, raises censoring from 13% to 44%, and leaves a
harder discrimination problem on which the models have more room to help. That is a real
property of the estimand, not an artefact — but it means **the answer to H2 is a function
of where "dead" is defined**, and we pre-specified 0.05 without knowing that.

**On the appearance that 7h is the flattering choice.** The registered landmark is also
where the models look best on absolute error (MAE 0.329 at 7h against 0.447 at 3h and
0.443 at 12h), and a sceptical reader is right to ask whether it was selected for that.
It was not, and the record shows it was not: the move from 6h to 7h is Amendment 2, dated
2026-09-01, and its stated ground is a measurement one — the collector runs every 30
minutes, so the nominal 6-hour reading arrives at a median of 6.08h, and a strict 6.00h
cutoff discarded it for 72 of 86 posts, leaving every 6-hour feature 0% populated. That
amendment was committed fifteen days before Cohort A closed and before any death label
existed for the posts analysed here. The coincidence is real and we report it; the
chronology is what distinguishes it from tuning.

**Feature coverage.** Eleven features cleared the 50% coverage requirement at the
registered landmark and entered the models: `log_value_at_1h` (95.0% coverage),
`log_value_at_3h` (97.5%), `log_value_at_6h` (95.9%), `velocity_to_6h` (95.9%),
`log_growth_1_3h` (90.8%), `log_growth_3_6h` (93.0%), `decay_ratio_3_6_over_1_3` (91.1%),
`log_follower_count` (100%), `log_duration_sec` (95.8%), `publish_hour_utc`, `title_len`.
Coverage above 90% on every early-engagement feature confirms that Amendment 2 fixed the
problem it was introduced for. `caption_len` and `hashtag_count` are Instagram-only and
are 0% populated on YouTube, as expected.

### 4.7 Cohort B — temporal holdout

Evaluated once, after all model selection is complete. `[PENDING]` — **not yet evaluated,
and deliberately so.**

As of 2026-09-20, Cohort B contains **234 YouTube posts** published on or after the
2026-09-16 freeze instant, and collection continues. The holdout has not been opened. It
will be evaluated exactly once, after §4.6.2's outstanding sensitivity analyses are
complete and no further modelling decisions remain, using
`python -m cdc.eval.report --cohort B --unlock-holdout`.

The holdout ledger (`data/gold/holdout_evaluations.jsonl`) is reproduced here in full,
whatever it contains — number of evaluations, dates, and the code commit each ran against.
As of writing it is **empty**, which is the correct state and is itself the evidence that
no peeking has occurred.

A temporal generalisation gap, if one appears, is reported rather than explained away. Cohort B
posts are collected under the same instrument as Cohort A but at a later date, so a drop
between the two is evidence about stability over time — which is more informative than another
cross-validation fold.

### 4.8 Instagram — feasibility only

Instagram is a **cross-platform feasibility demonstration, not a second test set**, and no
hypothesis above is tested on it. Cohort A ran 16 rounds at 3-hour intervals across 5
accounts over 47.0 hours, inside a hard budget of 100 API credits per key. Cohort B then
ran 24 rounds at 3-hour intervals across 3 accounts over 72 hours (Appendix A, Amendment
3), interrupted for 18 hours by an API fault (Amendment 4).

Pooled yield across both cohorts: **396 posts collected, 77 labellable, 14 observed
deaths, 12 creators.** 276 posts were excluded for `insufficient_observations` — the
dominant failure mode, and the one Amendment 3 was written to address.

With 14 deaths against the 50 required for a reportable result, Instagram remains
**underpowered by design and by outcome**. Its purpose is to show that the pipeline
generalises across platforms and to supply one cross-platform figure; it is not evidence
for or against any hypothesis in this paper. Note also that Instagram's primary metric is
likes rather than views (§3.4), so its death times are not comparable with YouTube's and
the two are never pooled.

---

## 5. Discussion

### 5.1 What we set out to test, and what came back

We asked whether the time at which a post stops receiving meaningful attention can be
predicted from signals observable in its first session. The answer, across 686 posts and
498 observed deaths, is: **weakly, and almost entirely because of who posted it.**

Both primary models discriminate better than chance (RSF 0.621, Weibull AFT 0.600, both
intervals excluding 0.5), so H1 holds in the limited sense that attention death is not
unpredictable. But a creator-size-only baseline reaches 0.599 on its own. Everything our
feature set contributes beyond a single number describing the channel amounts to, at most,
two points of concordance — and that margin fails every robustness check we committed to
in advance.

### 5.2 A C-index near 0.6, in practical terms

The headline metric is easy to over-read, so we state it concretely. A concordance index
of 0.62 means that, shown two posts, the model identifies which will die first about 62
times in 100. Chance is 50. Knowing only the subscriber count gets 60.

For a creator deciding when to stop promoting a post, that is not a usable signal. The
absolute predictions are worse still: MAE of 0.329 on log₁₀ hours back-transforms to a
typical error of a factor of **2.1**, so a post that truly dies at 20 hours is routinely
predicted at 9 or at 43. Nothing in this study supports deploying such a prediction as a
countdown, and §4.3's most uncomfortable result — that a single constant, the Kaplan–Meier
median, achieves *lower* absolute error than either fitted model — should be read as the
honest ceiling on what this feature set can do.

The gap between the two metrics is itself informative. A model that spreads predictions
across a range buys ordering at the cost of variance, and when recoverable signal is thin
that trade is not worth making. Ranking improved; calibration did not.

### 5.3 The H2 null is the substantive finding

H2 was the hypothesis that mattered, and the one we pre-specified against a deliberately
hard baseline. It does not survive.

Of four pre-specified comparisons — two models × two tests — one passes. The forest's
paired Wilcoxon reaches p = 0.019; its bootstrap interval on the C-index difference
includes zero; the Weibull AFT is indistinguishable from creator size on both (Δ = +0.0008,
p = 0.760). Moving the landmark to 3h *reverses* the sign and makes the forest
significantly worse than the baseline (p = 0.006); moving it to 12h erases the difference
(p = 0.394). Loosening the death threshold to 0.10 removes it; tightening to 0.02 restores
it for both models. A mechanically different outcome definition reproduces neither
hypothesis.

A result that exists at one landmark out of three, at thresholds at or below the one we
happened to choose, on one of two models, under one of two tests, is not a finding. We
therefore report **H2 as not supported**, and treat the isolated significant test as what
it most likely is: the one cell of a small grid that fell below 0.05.

**What that null claims.** If early engagement dynamics add nothing beyond creator size,
then the timing of attention death is largely a property of the **audience** rather than
of the **post**. A channel's subscriber base appears to determine not only how much
attention a post receives but roughly how long it holds it, and the particular trajectory
of a post's first six hours tells us little more. §4.5 supports this directly and is the
clearest pattern in the data: median time to death rises monotonically from 41.5h for
micro channels to 89.5h for mid and 113.0h for large — a 2.7× spread across tiers,
dwarfing anything our dynamic features recover.

This is a more interesting claim than a working countdown would have been, and it runs
against the intuition that early velocity is diagnostic. It is also consistent with the
literature we build on: Szabó and Huberman's log-linear relationship works because early
magnitude predicts later magnitude, and Pinto et al.'s gains concentrate in trajectory
*shape* for predicting *volume*. Neither establishes that shape predicts *timing*, and our
result suggests it largely does not.

### 5.4 Two constructs, not one

The robustness label was chosen to fail differently from the primary one, and the
comparison earns its place. The two agree on ordering (Spearman ρ = +0.718) but disagree
systematically on timing: saturation arrives at a median of **0.633** of the time at which
velocity collapses.

That offset is not noise and not a bug. It says a post stops **growing** appreciably well
before it stops **receiving** attention — two distinct ideas that the phrase "attention
death" silently conflates. Any study reporting one of them alone, including a future
version of this one, is making a narrower claim than its language implies. We regard
pinning down which construct matters for a given application as a more valuable next step
than improving the C-index by a further two points.

The failure of the saturation outcome to reproduce H1 or H2 (§4.6.1) compounds this.
Under that label both models sit at chance. We note the structural reasons it is the weaker
test — zero censoring by construction, 71 extrapolated fits — without using them to dismiss
the result, because we selected the check in advance precisely so it could not be dismissed
afterwards.

### 5.5 What the method contributes independently of the result

Three elements of the design survive the null and would apply to any future attempt.

**Landmarking was not optional.** Before Amendment 2, 53% of observed deaths fell inside
the six-hour feature window, meaning the majority of "predictions" were determined by the
observations they were built from. Any study predicting content decay from early signals
faces this and, as far as we can tell from §2, none of the prior work we surveyed addresses
it. A reported accuracy that includes circular cases is not comparable to one that excludes
them.

**Censoring is not a technicality.** 27.4% of our analysis set was still alive when
observation ended, and these are systematically the longest-lived posts. Dropping them —
the natural move in a regression framing — would have biased every estimate toward
short-lived content. At the pre-registered threshold this is a quarter of the sample; at
0.02 it is 44%.

**Grouping by creator changes the answer.** Our 686 posts come from 47 creators. Random
k-fold would place the same channel in train and test and, given §4.5's tier effect, would
mostly measure the model's ability to recognise creators it had already seen.

### 5.6 Why the pre-registration did work here

This study produced a result its authors did not want, and the mechanism that made
reporting it straightforward was built before the result existed.

The analysis plan was frozen on 2026-08-30 with 179 posts collected, zero death labels
computed and zero models fitted. Every subsequent deviation is a dated commit with a stated
reason. The draft carried, in advance, a note recording that the creator-size baseline had
outscored the model in the dress rehearsal and committing to report that if it held. The
evaluation tooling refuses to print results below 50 observed deaths. Cohort B is sealed
behind a flag and a ledger.

None of that made the analysis correct. What it did was remove the decisions that would
otherwise have been made *after* seeing the numbers — which threshold, which landmark,
which of two tests to feature, whether to mention a robustness check that failed. §4.6.2
shows exactly how much latitude those choices carry: at 0.02 we could have reported both
models beating creator size at p < 0.0001. That run exists because we promised it, not
because we liked it.

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
7. **Statistical power.** The analysis achieved **498 observed deaths across 47 creators**,
   against a pre-set minimum of 50 deaths and 10 creators. The sample is therefore
   adequately powered for the primary comparisons, and the negative and equivocal findings
   in §4.3–§4.4 cannot be attributed to insufficient data. What the sample does *not*
   support is fine-grained subgroup inference: the exploratory tier and category
   breakdowns in §4.5 rest on as few as 19 deaths in one cell, and the effective unit of
   independence is the creator (47), not the post, which is why every confidence interval
   in this paper is bootstrapped by creator rather than by post. Power for detecting the
   *small* H2 effect specifically is the binding constraint — a Δ C-index of +0.022 with a
   half-width of roughly 0.04 means the study can rule out a large effect of early
   dynamics beyond creator size, but cannot resolve a genuinely small one.

## 7. Ethics

Public content from public accounts only. No private data, no personal data beyond what a
logged-out visitor sees, and no attempt to identify individuals. Results are reported in
aggregate and creators are not named in the analysis. The YouTube Data API v3 is used within
its published quota; collection is rate-limited and metered by design. The use of an
unofficial third-party API for Instagram is disclosed plainly rather than buried. The study
is observational and involves no intervention on any user or creator.

## 8. Conclusion

We asked when a social media post stops receiving meaningful attention, and whether that
moment can be predicted from the post's first session. Across a prospectively collected
panel of 686 YouTube posts from 47 channels, observed every 30 minutes for up to 14 days
and yielding 498 observed deaths, the answer is that it can be predicted — weakly — and
that creator size accounts for essentially all of the predictability.

Both censoring-aware models discriminate better than chance (Random Survival Forest
C-index 0.621, 95% CI 0.569–0.678; Weibull AFT 0.600, 0.540–0.657), so **H1 is supported
on discrimination and rejected on calibration**: neither model predicts the actual time
better than a constant, and both are significantly worse than the Kaplan–Meier median,
which attains the lowest absolute error of any method tested. Every approach we tried, from
a random survival forest to a single number, is wrong by a factor of roughly two.

**H2 is not supported.** A creator-size-only baseline reaches C-index 0.599 unaided. One of
four pre-specified comparisons favours a model, and it survives neither a change of
landmark, nor a loosening of the death threshold, nor a mechanically different outcome
definition. We report this as a null rather than as a marginal positive, because it is one.

The substantive claim we can make is therefore about audiences rather than posts: **how
long content holds attention appears to be determined largely by who published it, not by
how it began.** Median time to attention death rises from 41.5 hours for channels under
10,000 subscribers to 113.0 hours for those above 500,000 — a spread no feature derived
from a post's first six hours came close to matching.

Two secondary findings stand on their own. A post stops *growing* well before it stops
*receiving* attention: our two independently motivated labels agree on ordering
(ρ = +0.718) but the saturation label fires at a median of 63% of the velocity label's
time, so "attention death" names two distinguishable constructs rather than one. And
landmarking is not a refinement but a requirement: before we imposed it, 53% of observed
deaths fell inside the feature window, making the majority of predictions circular.

We make no causal claim. The design is observational, the sampling frame is a committed
query list rather than a random sample of YouTube, and generalisation beyond it is not
asserted. A temporal holdout of 234 posts remains sealed and will be evaluated exactly
once.

What we offer alongside the result is the apparatus that made reporting it
straightforward: a plan frozen in version control before any outcome existed, every
deviation logged as a dated amendment, tooling that refuses to present underpowered
numbers, and robustness analyses run because they were promised rather than because they
were welcome. The finding is modest and partly negative. The evidence that we did not go
looking for a better one is the part we would most want a reader to check.

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
