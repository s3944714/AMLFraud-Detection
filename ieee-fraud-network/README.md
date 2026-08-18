# AML Fraud Detection — IEEE-CIS Fraud Network

A fraud detection and fraud-ring identification pipeline built on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
Kaggle competition dataset (Vesta Corporation's real, anonymized e-commerce
transaction data — ~590K transactions, ~3.5% fraud rate).

This is the `ieee-fraud-network` half of the `AMLFraud-Detection` monorepo.
The other half, [`fraud-review-app`](../fraud-review-app), is a separate
FastAPI + vanilla JS case-review application that reads this pipeline's
exported output — it never touches the model, SHAP, or the entity graph
directly.

## Why this exists

A per-transaction fraud classifier can only ever say "this transaction looks
risky in isolation." Two things it can't do on its own: explain *why* to an
investigator in a way that's defensible rather than a black box, and notice
when a cluster of otherwise-unremarkable transactions are actually linked —
same card+address, same device — which is the actual signature of a
coordinated ring rather than independent bad actors. This pipeline addresses
both: SHAP-based reason codes for the first, an entity-link graph for the
second.

## Pipeline

```
data/loader.py          → time-based train/test split
models/baseline.py       → LightGBM classifier, PR-AUC + recall-at-budget
explainability/shap_report.py → per-transaction SHAP reason codes
network/entity_graph.py  → entity-link graph, candidate fraud rings
evals/backtest.py        → formal backtest report on the saved model
export/to_app.py         → assembles the 3-file contract for fraud-review-app
```

Run in that order:

```bash
conda activate AMLFraud-Detection
python -m data.loader
python -m models.baseline
python -m explainability.shap_report
python -m network.entity_graph
python -m evals.backtest
python -m export.to_app
```

## The time-based split (and why it's not random)

`TransactionDT` is a timedelta from an arbitrary reference point, not a real
calendar timestamp — but it's monotonically increasing with real transaction
order, so it's valid to sort and split on. The competition's own
documentation notes that train and test are separated by time, with a gap
between them. Transactions are not i.i.d. across that window: fraud
patterns, issuers, and even feature distributions (the `C1`-`C14` running
counts, for instance) drift over time. A random split would let rows that
are "in the future" relative to other training rows leak into training —
near-duplicate transactions on the same card landing on both sides of the
split — which inflates offline metrics in a way that won't survive a real,
forward-only deployment.

`data/loader.py` splits at the 80th percentile of `TransactionDT` and
hard-asserts no leakage across the boundary:

| | Rows | `TransactionDT` range |
|---|---|---|
| Train | 472,432 | 86,400 – 12,192,842 |
| Test | 118,108 | 12,192,900 – 15,811,131 |

Baseline fraud rate: **3.499%**. Identity-match rate: **23.84%** — most
transactions have no matching identity row, which is expected (identity
data is only captured for a subset of sessions), not a data quality issue.

## Model results

LightGBM, trained on raw features with no feature engineering beyond an
hour-of-day derivation — this is a deliberate baseline, not a tuned final
model. Evaluated with PR-AUC and recall-at-alert-budget rather than
accuracy: at a 3.5% fraud rate, a model that predicts "not fraud" for every
transaction scores 96.5% accuracy while catching zero fraud, so accuracy is
actively misleading here.

**PR-AUC: 0.5743** (best iteration: 986 of 1,000, `scale_pos_weight` softened
to the square root of the class ratio for training stability — see *Key
engineering decisions* below)

| Alert budget | Transactions reviewed | Recall (model) | Recall (random order) | Extra fraud caught | Lift |
|---|---|---|---|---|---|
| 1% | 1,182 | 26.1% | 1.2% | 1,014 | **22.6x** |
| 5% | 5,906 | 62.6% | 5.2% | 2,334 | **12.1x** |
| 10% | 11,811 | 75.2% | 9.5% | 2,668 | **7.9x** |

In plain terms: if a fraud investigation team can only manually review the
top 5% of transactions each day, reviewing them in model-risk order catches
62.6% of all fraud in that period — compared to 5.2% if reviewed in random
order. That's over 2,300 additional real fraud cases caught for the same
review effort.

## Explainability: SHAP reason codes

For the top 5% of test transactions by risk score,
`explainability/shap_report.py` computes SHAP values (`TreeExplainer`,
tree-path-dependent) and produces a plain-language justification — not just
a score, but a reason an investigator (or eventually a regulator) can
actually read.

**On the honesty of these explanations:** most of this dataset's columns
(`V1`-`V339`, `C1`-`C14`, `D1`-`D15`, most `id_` columns) are anonymized,
Vesta-engineered features — Kaggle does not disclose what they represent.
This pipeline does not invent plausible-sounding business meaning for them
(claiming e.g. "`V127` measures device fingerprint mismatches" would be a
fabrication). Named columns with real-world meaning (`TransactionAmt`,
`ProductCD`, card network/type, email domain, device type) get a proper
plain-language template; anonymized columns get an honest "anonymized
engineered feature, here's its value and direction" phrasing instead.

`V258` shows up as a dominant driver in a meaningful share of reason codes.
That's consistent with a discussion thread on the competition's 1st-place
solution write-up, where a competitor specifically raised `V258` as a
feature whose importance was invisible under LightGBM's default "split"
importance metric and only surfaced once they switched to "gain" — an
observation the 1st-place team engaged with directly. This is a real,
previously documented signal in this dataset, not an artifact of this
pipeline.

## Entity-link graph: fraud ring candidates

`network/entity_graph.py` links transactions sharing a `card1`+`addr1`
combination (a standard approximation for a unique client/card ID in this
dataset, since there's no direct customer identifier), a `DeviceInfo`
value, or a `P_emaildomain`, then runs connected components and flags
clusters with ≥3 transactions and a fraud rate ≥2x the dataset baseline as
candidate rings.

**1,254 candidate ring clusters found, covering 22,783 transactions, at an
average fraud rate of 36.3% — 10.4x the dataset baseline of 3.5%.**

This is unsupervised structure-finding, not a verified-fraud-ring detector.
A flagged cluster means "this group of transactions is unusually linked
*and* unusually fraud-heavy" — a lead for an investigator, not proof of
anything on its own. Shared attributes can also arise from mundane reasons
(a shared household, a shared office).

## Exported file contract (for `fraud-review-app`)

`export/to_app.py` assembles three files plus a metadata summary into
`export/output/`:

| File | Rows (real data) | Notes |
|---|---|---|
| `scored_transactions.csv` | 5,906 | Only transactions with a SHAP reason code; `cluster_id` is nullable, populated only for candidate-ring members |
| `shap_detail.csv` | 59,060 | Long format, top 10 SHAP features per transaction |
| `entity_clusters.csv` | 22,783 | Candidate-ring members only — non-ring clusters are intermediate data, not exported |
| `metadata.json` | — | `n_cases`, `n_cluster_members`, `n_candidate_rings`, `pr_auc` |

`isFraud` is deliberately excluded from the export, even though it exists
in this labeled historical dataset: a real production deployment wouldn't
have ground truth available at scoring time either, and the review app is
built to handle that gracefully.

## Testing

`pytest.ini` sets `pythonpath = .` so cross-package imports work regardless
of how pytest is invoked. Tests focus specifically on the data-join and
graph-construction logic — the two places a silent bug wouldn't show up in
any headline metric, it would just quietly produce wrong data that still
looks plausible:

```bash
pytest tests/ -v
```

- `tests/test_loader.py` (7 tests): left-join correctness (no dropped
  unmatched transactions, no row duplication), time-split ordering,
  behavior on shuffled/unsorted input, leakage-boundary handling
- `tests/test_entity_graph.py` (12 tests): edge formation per attribute
  type, null-value handling, fraud-rate computation, ring-flagging
  (size *and* rate thresholds independently), and the mega-component
  safeguard described below

## Key engineering decisions (and bugs caught along the way)

A few things surfaced only when validated against real data or realistic
scale — worth documenting since they're not obvious from the code alone:

**LightGBM's default metric silently overrides a custom one.** Early
training runs showed `best_iteration: 1` — boosting added zero value after
the first tree. The cause: LightGBM auto-tracks the objective's default
metric (`binary_logloss`) alongside a custom `eval_metric` callable unless
told not to, and early stopping's "best iteration" ended up governed by
`binary_logloss` instead of the PR-AUC metric that actually matters here.
Fixed with `metric="None"` in the model constructor plus
`first_metric_only=True`. PR-AUC went from 0.26 (wrong metric, effectively a
1-tree model) to 0.5743 (996 useful trees) after the fix.

**Entity-link mega-clusters.** A single common value (`DeviceInfo="Windows"`,
`P_emaildomain="gmail.com"`) can absorb a tight, genuinely fraud-heavy ring
into one enormous, mostly-unrelated component, diluting its fraud rate back
toward baseline and hiding it from ring detection entirely — confirmed
directly with an injected synthetic ring that was completely invisible
until fixed. `MAX_GROUP_SIZE=100` caps any single shared-value group from
forming edges past that size. A second, coarser `MAX_COMPONENT_SIZE=2000`
safeguard catches emergent mega-components formed by *chains* of
individually-small groups bridging across edge types (confirmed on the real
590K-row dataset: a 28,427-transaction component — 4.8% of the whole
dataset — formed this way despite every individual group staying under the
100 cap).

## Limitations and scope

**This is a portfolio/research exercise on public competition data, not a
production AML system.** A few things a real deployment would need that
this deliberately does not attempt:

- **Regulatory compliance:** a real Australian deployment would need
  AUSTRAC-compliant model documentation, ongoing model risk management, and
  formal validation — none of which this repo attempts to replicate.
- **Human review workflows:** this produces risk scores and reason codes;
  it doesn't implement case management, escalation paths, SAR/TTR filing
  workflows, or audit trails a real investigation team would need.
- **Feature engineering:** the baseline model deliberately uses raw
  features with minimal engineering. Known levers for improvement (visible
  in top public solutions for this exact competition) include treating
  `card1`/`addr1` as frequency-encoded categoricals, aggregation features,
  and UID-based client grouping.
- **Concept drift:** the model is trained and evaluated on one historical
  window. A production system would need ongoing monitoring for
  performance decay as fraud patterns evolve.
- **Candidate rings are leads, not verdicts:** cluster membership reflects
  unusual linkage and elevated fraud rate, not confirmed coordinated fraud.
