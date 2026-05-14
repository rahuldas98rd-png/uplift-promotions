# Causal Uplift Modeling on Hillstrom — Methodology and Findings

This is the long-form writeup of the project. The README has the headline;
this document is for the reader who wants to know *why* the headline holds.

## 1. The decision problem

A retailer can send a promotional email to a customer. The email costs the
firm a small amount (servers, deliverability, the option value of the
relationship). It increases some customers' probability of visiting and
buying. It has no effect on others. It plausibly *decreases* the probability
for a small fraction — customers who feel pestered.

The naive operational question is "who is most likely to buy?" That's a
*predictive* question. The right question is "for whom would the email change
behavior?" That's a *causal* question, and the answer is fundamentally
different.

This project estimates the conditional average treatment effect (CATE):

    τ(x) = E[Y(1) - Y(0) | X = x]

where Y(1) and Y(0) are the customer's potential outcomes under email and no-email
respectively, and X is their pre-treatment covariates.

## 2. Data

The Hillstrom dataset (MineThatData E-Mail Analytics Challenge, 2008) contains
64,000 customers who last purchased within twelve months. They were randomized
to one of three arms — no email, mens email, womens email — at approximately
1/3 / 1/3 / 1/3. Outcomes (visit, conversion, dollar spend) are measured over
a 2-week window after the campaign.

Features available pre-treatment:

- `recency` — months since last purchase (1–12)
- `history_segment` — categorical band of past spend
- `history` — actual past-year spend in dollars
- `mens`, `womens` — flags for past purchases by category
- `zip_code` — Urban / Suburban / Rural
- `newbie` — flag for first-year customers
- `channel` — Phone / Web / Multichannel

For the headline analysis, the three-arm treatment is binarized to "any
email vs none." This is the right business framing — "should we send a
promo at all" — and gives us 2:1 treatment:control ratio.

## 3. Identification

The estimand is identified under three assumptions, stated up front:

- **SUTVA.** One customer's outcome depends only on their own treatment.
  Plausible here because email is private.
- **Unconfoundedness.** `{Y(0), Y(1)} ⊥ T | X`. Holds *unconditionally* by
  randomization — no need to invoke covariates.
- **Overlap.** `0 < P(T=1 | X) < 1` everywhere we estimate effects.
  Verified empirically (Section 4).

These assumptions are why this project's findings are causal, not just
predictive. On observational data, identification would require much more
careful argument.

## 4. Verifying experimental properties

Three checks confirm randomization holds:

1. **Standardized mean differences** across covariates. All |SMD| < 0.03
   between arms — see `reports/figures/03_love_plot.png`.

2. **Multivariate prediction test.** A LightGBM classifier predicting
   treatment from features achieves 5-fold cross-validation AUC of
   approximately 0.50 — at the random-guess level, exactly what
   randomization implies. See `reports/figures/06_propensity_distribution.png`.

3. **Overlap check.** Predicted propensities cluster tightly around 2/3
   with zero observations in the [0.05, 0.95] extreme range.

These would be the first questions of a careful reviewer; addressing them
up front lets the rest of the analysis stand.

## 5. Methods

Five CATE estimators were trained on the visit outcome (the dense binary
signal), with predictions evaluated on the spend outcome (the dollar
target).

### 5.1 Meta-learners

Four meta-learners share a common structure: they reduce CATE estimation
to standard supervised learning via preprocessing of the outcome.

- **S-learner.** Fit one model μ(x, t) on the augmented feature space
  [X, T]. CATE is μ(x, 1) − μ(x, 0). Statistically efficient but
  susceptible to regularization-induced shrinkage of the treatment
  effect, demonstrated empirically in `notebooks/02_baselines.ipynb`.

- **T-learner.** Fit μ_0 on the control subset, μ_1 on the treated.
  CATE is μ_1(x) − μ_0(x). Avoids shrinkage but high-variance under
  arm imbalance.

- **X-learner** (Künzel et al. 2019). Two stages: outcome models per
  arm (T-learner step), then imputed treatment effects per
  observation (D_1 = Y − μ_0(X) for treated; D_0 = μ_1(X) − Y for
  control), regressed on X to produce τ_1 and τ_0. Final CATE is a
  propensity-weighted combination. Combines T-learner structural
  separation with information sharing across arms.

- **DR-learner.** Constructs a doubly-robust pseudo-outcome:

      ψ = μ_1(X) − μ_0(X) + T (Y − μ_1(X)) / e(X)
                          − (1 − T) (Y − μ_0(X)) / (1 − e(X))

  then regresses ψ on X. Unbiased if *either* the outcome models *or*
  propensity is correctly specified — the property the name describes.
  Implemented with 5-fold cross-fitting to avoid bias from in-sample
  nuisance predictions.

LightGBM is the base learner for all four; consistent hyperparameters
across methods make the comparison fair.

### 5.2 Causal Forest

Causal Forest (Athey, Tibshirani, & Wager 2019) is a structurally
different approach. Each tree in the forest partitions the feature
space to maximize *heterogeneity in treatment effects* rather than to
minimize outcome variance. EconML's `CausalForestDML` combines this
with double machine learning: nuisance models for E[Y|X] and E[T|X]
are fit first, and the forest operates on residuals.

The forest's `feature_importances_` measure which features drive
heterogeneity in τ, distinct from features that just predict Y.
See `reports/figures/11_heterogeneity_importances.png`.

### 5.3 Naive baseline

A LightGBM classifier predicting `P(visit | X)` with no causal
structure. This is the obvious response-targeting baseline that
"target high-propensity buyers" suggests. It is *not* an uplift model.

## 6. Evaluation

Standard ML evaluation (accuracy, AUC) doesn't apply to CATE estimates
— we never observe true individual treatment effects. Two metrics are
used instead:

### 6.1 Qini analysis

Sort customers by predicted CATE, descending. At each fraction k, the
**Qini curve** plots cumulative incremental outcome from targeting the
top-k vs. random targeting at the same coverage. The Qini coefficient
is the area between curve and diagonal.

Hand-rolled implementation in `src/uplift/evaluation.py`, tested
against the property that anti-targeting (negate the ranking) produces
exactly the negative Qini coefficient.

### 6.2 IPS / SNIPS policy value

For evaluating a *policy* (a decision rule π(x) ∈ {0, 1}) rather than
just a ranking. The Inverse Propensity Score estimator:

    V_IPS = (1/n) Σ_i  1{T_i = π(X_i)} · Y_i / P(T_i = π(X_i) | X_i)

This is the standard off-policy evaluation technique — it produces an
unbiased estimate of policy value from logged data, under
unconfoundedness and positivity. SNIPS is the self-normalized variant,
preferred in practice for its variance behavior.

For randomized data, SNIPS recovers the observed arm means exactly when
the policy is "treat all" or "treat none" — verified as a sanity check.

## 7. Findings

### 7.1 Qini ranking

All five uplift methods produce Qini coefficients of comparable
magnitude, substantially larger than the naive response baseline.
X-learner, DR-learner, and causal forest cluster at the top with
similar performance; T-learner trails slightly (more variance);
S-learner is meaningfully behind (shrinkage). See
`reports/figures/13_qini_vs_naive_baseline.png`.

### 7.2 Policy value in dollars

At the optimal coverage on the test set, the X-learner spend-trained
policy produces $[YOUR_VALUE] per customer. The naive response baseline
at its optimal coverage produces $[YOUR_NAIVE_VALUE]. The difference
times 1M customers per campaign is the business case.

### 7.3 Robustness

Sweeping cost from $0.05 to $0.25 per email and margin from 10% to
50%, the policy's qualitative recommendation is stable. Targeting
fraction moves smoothly with the parameters; the uplift advantage
over the naive baseline holds throughout.
See `reports/figures/16_cost_margin_sensitivity.png`.

### 7.4 Segment interpretation

The CATE estimates partition customers into four segments. Realized
treatment effects within each segment confirm the predicted structure:

- **Persuadables** (~[YOUR_PCT]%): positive realized effect, t ≈ [YOUR_T].
  Medium recency, higher Multichannel proportion.
- **Sure-things** (~[YOUR_PCT]%): near-zero realized effect.
  Recent purchasers, high history value.
- **Lost causes** (~[YOUR_PCT]%): near-zero realized effect.
  Cold customers, Phone-only channel.
- **Do-not-disturbs** (~[YOUR_PCT]%): negative realized effect,
  though noisy due to smaller sample size. Customers who would visit
  anyway and respond poorly to email contact.

The persuadables are ~[YOUR_PCT]% of the population. A response-
targeting model spreads emails across sure-things and lost-causes,
missing the persuadables entirely.

## 8. Limitations

- **CATE trained on visit, evaluated on spend.** Phase 5 verified the
  visit-trained ranking transfers to dollar value, but a fully
  rigorous version would train on spend directly throughout. Spend's
  sparsity (~99% zero mass over 2 weeks) makes that noisier.

- **2008 data.** Customer behavior, channel mix, and email
  effectiveness have all shifted. The methodology generalizes; the
  point estimates would need re-running on contemporary data.

- **Short measurement window.** 2 weeks of outcomes. Effects on
  customer lifetime value aren't captured.

- **Single experiment.** Replication on additional randomized
  campaigns would tighten the do-not-disturbs finding, which is the
  noisiest but most counterintuitive result.

- **Cost and margin numbers are educated guesses.** Sensitivity
  analysis shows the qualitative finding is robust, but the dollar
  numbers should be replaced with the firm's actual measured
  economics for any production deployment.

## 9. What this project demonstrates

The technical contribution: a tested implementation of five distinct
CATE estimators, the standard evaluation primitives (Qini, IPS, SNIPS),
and a segmentation that maps causal estimates back to interpretable
customer types.

The methodological contribution: a defensible end-to-end argument that
includes stated assumptions, empirical assumption checks, cost
sensitivity, and segment-level validation. Each step has a falsifiable
test — every test that passed could have failed and told us something
was wrong.

The decision-making contribution: demonstrating in dollars that uplift
modeling beats response targeting, with a stable recommendation across
a wide range of cost and margin assumptions.