# Causal Uplift Modeling for E-Commerce Promotions

**Decision problem:** which customers should receive a promotional email, given that
sending one is costly and some customers would have purchased anyway?

This project uses causal inference — not predictive ML — to identify the customers
whose behavior changes when targeted. Trained on the Hillstrom 64K-customer
randomized email experiment, evaluated with held-out test data and inverse
propensity score policy estimation.

![Headline chart — value vs. coverage](reports/figures/14_HEADLINE_value_vs_coverage.png)

## Key finding

Targeting customers by predicted *response* (the obvious baseline) captures only a
fraction of the value of targeting by predicted *treatment effect*. An uplift policy
applied to ~[YOUR_OPTIMAL_FRACTION]% of customers produces $[YOUR_UPLIFT_VALUE]
per customer in expected spend vs. $[YOUR_NAIVE_VALUE] for response-based targeting
at the same coverage — a [YOUR_PCT_IMPROVEMENT]% improvement.

Four customer segments emerge from the CATE estimates:

| Segment           | % of customers | Realized treatment effect on visit |
|-------------------|---------------:|-----------------------------------:|
| Persuadable       | [YOUR_PCT]%    | [YOUR_ATE] (t = [YOUR_T])          |
| Sure-thing        | [YOUR_PCT]%    | [YOUR_ATE] (t = [YOUR_T])          |
| Lost cause        | [YOUR_PCT]%    | [YOUR_ATE] (t = [YOUR_T])          |
| Do-not-disturb    | [YOUR_PCT]%    | [YOUR_ATE] (t = [YOUR_T])          |

The persuadables are the only group worth targeting. The do-not-disturbs respond
*negatively* to the email — an unintuitive but stable finding that argues for
restraint, not blanket targeting.

## Methods compared

Five distinct CATE estimators, each addressing a different statistical weakness:

- **S-learner** — single model on `[X, T]`. Susceptible to regularization shrinkage.
- **T-learner** — separate model per arm. High variance with imbalanced arms.
- **X-learner** (Künzel et al. 2019) — combines T-learner with cross-arm
  information sharing via imputed potential outcomes.
- **DR-learner** — doubly robust pseudo-outcome with 5-fold cross-fitting.
- **Causal Forest** (Athey et al. 2019) — tree-based heterogeneity discovery
  with double machine learning residualization. Via EconML.

Compared against the naive "target by predicted response" baseline (a LightGBM
classifier predicting visit probability with no causal structure).

## Evaluation

- **Qini curves and Qini coefficients** — ranking quality at each coverage level
- **IPS/SNIPS policy value estimators** — unbiased estimate of "value if deployed,"
  in dollars
- **Cost-aware policy** — sweep over (cost, margin) shows robustness to economic
  assumptions across a 5× range in both directions

## Quick start

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). On Windows
with PowerShell:

```powershell
git clone <repo-url>
cd uplift-promotions
uv sync

# Download the Hillstrom dataset (~3 MB)
$url = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
Invoke-WebRequest -Uri $url -OutFile data\raw\hillstrom.csv

# Verify everything works
uv run pytest tests/ -v
```

Then open the notebooks in order:

1. `notebooks/01_eda.ipynb` — causal EDA, randomization checks, heterogeneity preview
2. `notebooks/02_baselines.ipynb` — splits, propensity, S/T/X/DR-learners, causal forest
3. `notebooks/03_evaluation.ipynb` — Qini, IPS, headline chart, segment analysis

Each notebook runs end-to-end on a fresh kernel.

```bash
## Repository structure
uplift-promotions/
├── README.md
├── pyproject.toml             # uv project, pins econml + scikit-learn versions
├── src/uplift/                # importable Python package
│   ├── data.py                # hash-checked loader for Hillstrom CSV
│   ├── treatment.py           # estimand definition, feature encoding
│   ├── splits.py              # train/val/test stratified by treatment
│   ├── propensity.py          # propensity model + overlap diagnostics
│   ├── learners.py            # S, T, X, DR learners (hand-rolled)
│   ├── forest.py              # CausalForestDML wrapper
│   └── evaluation.py          # Qini, IPS/SNIPS, segment helpers
├── tests/                     # 49 passing tests including causal sanity checks
├── notebooks/                 # 3 analysis notebooks
├── reports/
│   ├── writeup.md             # methodological deep-dive
│   ├── segment_summary.txt    # auto-generated narrative summary
│   └── figures/               # 17 figures referenced by the writeup
├── configs/default.yaml       # cost, margin, seeds, splits
└── data/                      # raw (gitignored), processed (gitignored)
```
## Methodology highlights

- **Estimand stated explicitly** in `treatment.py` and the writeup. The
  CATE on visit probability, with the binarized "any email vs none"
  treatment, on all Hillstrom customers.
- **Identification assumptions** stated in `reports/writeup.md` and verified:
  randomization checked at the multivariate level (AUC ≈ 0.50 for
  predicting treatment from covariates), overlap verified
  (propensities cluster tightly at 2/3).
- **Cross-fitting** used for DR-learner pseudo-outcomes and causal forest
  nuisance models, ensuring out-of-fold predictions feed downstream
  estimation.
- **Policy value evaluated with SNIPS**, the self-normalized inverse
  propensity score estimator. The naive ATE-recovery sanity check
  passes (`ips_value(treat_all) ≈ E[Y | T=1]`).

## Limitations

This project demonstrates a methodology; production use would require:

- **Train CATE directly on dollar outcomes**, not visit probability as a proxy.
  The Phase 5 retraining on `spend` shows the proxy was reasonable but adds
  uncertainty around the dollar values.
- **SUTVA** is assumed (one customer's email doesn't affect another's outcome).
  Plausible for email; less so for social-network-mediated campaigns.
- **Stationary distributions**. CATE estimates from 2008 data won't transfer to
  2026 customers without retraining.
- **Single experiment**. The 14-day measurement window is short. Longer-term
  effects on customer lifetime value aren't captured.

The cost ($0.10/email) and margin (30%) are educated guesses, not measured
values — but the sensitivity analysis shows the qualitative conclusions hold
across a 5× range in either direction.

## References

- Athey, S., Tibshirani, J., & Wager, S. (2019). *Generalized Random Forests*. Annals of Statistics.
- Chernozhukov, V. et al. (2018). *Double/debiased machine learning for treatment and structural parameters*. The Econometrics Journal.
- Hillstrom, K. (2008). *MineThatData E-Mail Analytics and Data Mining Challenge*.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). *Metalearners for estimating heterogeneous treatment effects using machine learning*. PNAS.
- Radcliffe, N. (2007). *Using control groups to target on predicted lift*. Direct Marketing Analytics Journal.

