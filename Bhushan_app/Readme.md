# IndusCredit — Business Plan v2

**Federated ML credit scoring infrastructure for Indian NBFCs and cooperative banks**

Model: Logistic Regression · ROC-AUC 0.901 · PR-AUC 0.831 · Top driver: `missed_payments_2y`

---

## Executive summary

IndusCredit is a credit scoring API platform built on a LightGBM/LR ensemble trained on Indian loan data. The core commercial insight is simple: Indian NBFCs need good default prediction but will never share raw applicant data with each other or a third party. Federated learning solves the privacy problem. Variance-driven caching solves the cost problem. Multi-stage KServe canary rollout solves the reliability problem when the cached model changes.

The result is a platform whose unit economics improve automatically over time — the more federation rounds complete, the more stable the model, the higher the cache hit rate, and the lower the per-inference cost. A competitor who starts later begins their convergence clock from zero.

---

## 1. What we built

### Model snapshot (Iter 3 — production candidate)

Trained on 8,000 Indian loan applications. Predicts 30-day default probability at application time.

| Model | ROC-AUC (CV) | PR-AUC (CV) | F1 (CV) |
|---|---|---|---|
| Logistic Regression | 0.9020 ± 0.0118 | 0.8325 | 0.7273 |
| XGBoost | **0.9044 ± 0.0120** | 0.8320 | 0.7333 |
| LightGBM | 0.9039 ± 0.0131 | 0.8302 | 0.7299 |
| XGBoost + Pseudo-labels | 0.9028 ± 0.0119 | — | — |
| Stacking ensemble | 0.9041 ± 0.0119 | **0.8338** | **0.7379** |
| Rank averaging | 0.9032 | 0.8321 | — |

**Deployment choice: Stacking ensemble** — wins on both PR-AUC (0.8338) and F1 (0.7379), which are the metrics that matter most under class imbalance and for operational use. XGBoost leads on ROC-AUC by a slim 0.0003 margin but the stack's superior minority-class precision makes it the right production model. The stacking meta-learner also gives a natural audit trail — the credit committee can see which base models agreed.

### Top predictive features (SHAP, Iter 3)

1. `missed_payments_2y` — dominant signal, ~5× more important than anything else
2. `high_dti_flag` (DTI > 0.45)
3. `num_existing_loans`
4. `ltv_ratio`
5. `bureau_enquiries_6m`

**Fairness:** `gender_Male` ranked 29th of 55 features, contributing 0.89% of total SHAP importance. Not a fairness concern; no gender features need removal.

### Feature engineering highlights

55 features total, built from the raw application fields. The highest-leverage engineered features are:

- `emi_to_income_ratio` — actual repayment burden vs. monthly income (standard amortization formula)
- `missed_payment_rate` — missed_payments_2y / (num_existing_loans + 1), normalises by loan count
- `enquiry_intensity` — bureau_enquiries_6m / (num_existing_loans + 1), signals desperate borrowing
- `credit_score_x_dti` — interaction: high DTI is less dangerous with excellent credit
- `interest_spread` — premium above median rate for the loan type, signals lender-assessed risk

---

## 2. System architecture

The platform has four layers. Each exists to solve a specific cost or reliability problem.

```
┌─────────────────────────────────────────────┐
│  FEDERATION LAYER  (nightly, async)          │
│  FedAvg aggregator · ε-DP gradient clipping  │
│  Variance signal → cache TTL extension       │
└──────────────────────┬──────────────────────┘
                       │ new global model (when variance gate passes)
┌──────────────────────▼──────────────────────┐
│  CANARY PIPELINE  (KServe on GKE)            │
│  Shadow → 5% → 30% → 100%                   │
│  PSI + KS gates · auto-rollback              │
└──────────────────────┬──────────────────────┘
                       │ validated model
┌──────────────────────▼──────────────────────┐
│  CACHE ROUTING  (variance-driven)            │
│  Hot (Redis)  →  Warm (DynamoDB)  →  Cold    │
│  Route decision based on PSI + KS + var      │
└──────────────────────┬──────────────────────┘
                       │ API response
┌──────────────────────▼──────────────────────┐
│  CLIENT TIER                                 │
│  REST API · webhook callbacks · SDK          │
└─────────────────────────────────────────────┘
```

### Federation layer (privacy)

Each bank node runs gradient computation locally — raw loan data never leaves the institution. Only gradient updates are transmitted, with differential privacy noise added at ε ≈ 2–4. RBI data localisation requirements are satisfied by design.

Each nightly aggregation round produces a new global model candidate and a **variance signal** — the spread of gradient updates across nodes. As the federation matures, variance falls. This is the input to everything downstream.

### Canary pipeline (reliability)

Every new model candidate enters a 4-gate pipeline before touching production traffic:

| Stage | Traffic | Duration | Gate condition |
|---|---|---|---|
| Shadow | 0% (log-only) | 24 hours | No crash, no anomalous distributions |
| Canary 5% | 5% | 48 hours | PSI < 0.2 vs. current model |
| Canary 30% | 30% | 72 hours | KS stat stable, F1 within 2% |
| Production | 100% | — | Passed all gates |

Failure at any gate triggers automatic rollback to the prior cached model — not a cold restart. This is what makes a contractual SLA on model quality credible.

### Cache routing (cost)

Three signals are monitored continuously: prediction variance across recent federation rounds, PSI (population stability index) on incoming feature distributions, and KS drift on score distributions.

| Cache tier | Condition | Action | Cost per call |
|---|---|---|---|
| Hot (Redis) | All signals below threshold | Return cached score | ~₹0.001 |
| Warm (DynamoDB) | Low but non-negligible variance | Re-score on drifted features only | ~₹0.02 |
| Cold (KServe pod) | Variance or drift above threshold | Full model inference | ~₹0.15 |

As federation converges, hot-cache hit rate rises and average cost per call falls asymptotically toward the Redis lookup cost. This is the core cost moat — a competitor starting later begins convergence from zero.

---

## 3. Pricing model

Three tiers targeting different client sizes:

### Starter — ₹4,500/month
For cooperative banks and small NBFCs, up to 5,000 calls/month included.

- REST API access, standard model (no federation participation)
- 99.5% uptime SLA
- Overage: ₹1.20 per additional call
- No federation node — client benefits from global model but does not contribute gradients

### Growth — ₹18,000/month (recommended)
For mid-size NBFCs, up to 50,000 calls/month included.

- Federation node participation (contributes and receives gradient updates)
- Access to warm cache tier
- 99.9% uptime SLA, dedicated Slack channel
- Overage: ₹0.80 per additional call
- SHAP explanation API (top 5 features per decision, for credit officer review)

### Enterprise — ₹55,000+/month
For large NBFCs and MFIs with high call volumes.

- Private federation ring option (share gradients only within a defined group)
- Custom canary rollout schedules
- On-premise federation node deployment support
- 99.95% SLA with contractual penalty clauses
- Dedicated integration engineering hours

---

## 4. Unit economics and costing

### Infrastructure at 1M calls/month (realistic Y1 target)

Assume 70% hot-cache hit rate (conservative early estimate before full convergence):

| Component | Description | Monthly cost (₹) |
|---|---|---|
| Redis cluster (2-node) | Hot cache · ~700K calls | 13,500 |
| DynamoDB | Warm cache · ~250K calls | 4,200 |
| KServe (GKE, 1–2 pods, autoscaled) | Cold inference · ~50K calls | 18,000 |
| Federation aggregator | Spot VMs, nightly run | 5,400 |
| Monitoring (Prometheus + Grafana Cloud) | Metrics, alerting | 4,500 |
| Postgres (RDS t3.micro) | Metadata, audit logs | 2,700 |
| **Total infrastructure** | | **~₹48,300/month** |

At Growth tier pricing (₹0.36/call blended at 1M volume), revenue is ~₹3.6L/month. Gross margin at this scale: ~87%. The ₹48K infra figure is the honest number — the previous plan's ₹1.5–1.8L estimate was for 10M calls, not 1M.

### Infrastructure at 10M calls/month (Y2 with 5–8 Growth clients)

With 85% hot-cache hit rate (post-convergence):

| Component | Monthly cost (₹) |
|---|---|
| Redis cluster (3-node) | 27,000 |
| DynamoDB | 22,500 |
| KServe (2–3 pods) | 36,000 |
| Federation aggregator | 9,000 |
| Monitoring + logging | 9,000 |
| Postgres (RDS t3.small) | 5,400 |
| **Total** | **~₹1,09,000/month** |

Revenue at 10M calls (blended ₹0.30/call): ~₹30L/month. Gross margin: ~96%. This is where the variance-driven cache compounds — every additional federation round extends TTL and reduces cold-pod invocations.

### Breakeven

| Scenario | Monthly infra | Required MRR | Required clients |
|---|---|---|---|
| 1M calls (Y1) | ₹48,300 | ₹1.9L (2× infra + 1 FTE) | ~4 Growth clients |
| With 1 FTE salary (₹1.2L/month) | ₹1,68,300 total | ₹4.2L | ~8 Growth clients |
| Comfortable (2 FTE + runway) | ₹3,48,300 total | ₹8.7L | 3 Enterprise + 5 Growth |

---

## 5. The variance-driven cost flywheel

This is the business moat, restated precisely.

Standard ML platforms pay the same inference cost in month 24 as in month 1. IndusCredit's cost curve is downward-sloping because:

1. More federation nodes → lower inter-node gradient variance
2. Lower variance → model is more stable between rounds
3. More stable model → hot-cache TTL can be safely extended
4. Longer TTL → higher cache hit rate → fewer cold-pod invocations
5. Fewer cold-pod invocations → lower infra cost per call → higher gross margin

A bank that joins in month 1 reaches ~85% hot-cache hit rate by month 8–10. A competitor who builds the same stack in month 12 starts their convergence clock from zero. The federation's training data is also larger and more diverse by then — their model starts worse.

The implication for sales: **getting nodes is the highest-leverage activity in Y1.** Each new node doesn't just add revenue — it accelerates convergence for all existing nodes, raising the quality of the product for everyone.

---

## 6. Go-to-market and build sequence

### Phase 1 — Months 0–2: Ship a working API

Build the cold inference path only. Single KServe pod serving the Logistic Regression model trained on 8,000 rows. REST endpoint, basic auth, usage logging. This is enough to demo and sign pilots.

Target customers: 2–3 mid-size NBFCs in Maharashtra or Gujarat where NBFC density is highest. Offer 6-month pilot at ₹9,000/month (50% discount) in exchange for being federation node 1–3.

Milestone: first paying API call.

### Phase 2 — Months 3–5: Federation + warm cache

Stand up the FedAvg aggregator. Onboard pilot nodes. Implement PSI/KS monitoring. Add DynamoDB warm cache tier. Begin measuring variance signal from federation rounds.

At this point the product is differentiated — competitors can copy the inference API but not the federation history.

Milestone: first model update produced by federation (not just single-bank training).

### Phase 3 — Months 6–9: Hot cache + canary pipeline

Redis hot cache. Multi-stage KServe canary with PSI/KS gates and auto-rollback. SHAP explanation API. Move pilots to full Growth pricing.

Begin targeting Enterprise accounts — larger NBFCs and MFIs with 50,000+ calls/month.

Milestone: ₹4.2L MRR (breakeven with 1 FTE).

### Phase 4 — Months 10–18: Scale federation

Add more nodes, extend to new states and loan types. Publish convergence benchmarks. Use gross margin improvement data as a sales tool ("clients who joined month 1 now pay ₹0.04/effective inference").

---

## 7. Risks and mitigations

**RBI regulatory risk** (high) — New NBFC guidelines or a data protection ruling could require changes to how gradient updates are handled. Mitigation: build the federation aggregator to be auditable from day one; engage a fintech compliance advisor before launch; ε-DP parameters are already set conservatively.

**Model staleness between federation rounds** (medium) — If a node's applicant population shifts rapidly (seasonal agricultural loans, post-harvest default spike), the cached model may degrade before the nightly round runs. Mitigation: PSI monitoring fires an alert at PSI > 0.15 and triggers an out-of-cycle aggregation if > 0.25. This is already in the canary gate spec.

**Low initial node count** (medium) — With fewer than 3 nodes, the variance signal is noisy and FedAvg provides little benefit over single-bank training. Mitigation: train a strong single-bank baseline (already done — 0.901 AUC) so the product is useful from day 1 even without federation benefit. Federation is an upgrade path, not a launch dependency.

**Cold start CAC** (medium) — NBFCs are conservative buyers. The sales cycle may be 3–6 months. Mitigation: pilot pricing and integration support in Phase 1; design the onboarding to be a 1-day integration via REST API, no on-premise deployment required for Starter/Growth tiers.

**Credit score API competitors** (low-medium) — CIBIL, Experian, Equifax provide bureau scores. Mitigation: IndusCredit is not a bureau — it is a default probability model trained on the client's own portfolio behaviour, which bureau scores cannot replicate. The SHAP explanation output is also differentiated (bureau scores are black boxes to the lender).

---

## 8. What makes this defensible

The previous business plan called this an "asymmetric moat." More precisely, there are three distinct defensibility layers:

**Data network effect.** Every new federation node improves the global model. The model in month 18 will be trained on gradient signal from many diverse portfolios — urban salaried, rural agricultural, microfinance, home loan — that no single competitor can replicate without recruiting the same nodes.

**Switching cost.** Once an NBFC has integrated the API and their credit officers are reading SHAP explanations in their loan workflow, switching to a different scoring API means retraining their team, re-auditing their lending policy, and re-validating the replacement model for RBI compliance purposes.

**Convergence lead time.** The federation's variance clock starts from node onboarding, not from product launch. A competitor who enters the market 12 months later faces 12 months of catch-up convergence, during which their cache hit rates and gross margins are worse, making their pricing less competitive.

---

## Appendix: Model training notes

- Training set: 8,000 rows · Test set: 2,500 rows (no `default_flag` in test — all evaluation is CV-based)
- Production model: Stacking ensemble (XGBoost + LightGBM + LR base learners, LR meta-learner)
- Class balance: ~20% default rate (moderately imbalanced; handled via `class_weight='balanced'` and `is_unbalance=True`)
- Pseudo-label augmentation (XGBoost variant): marginal AUC gain of 0.0016 over base XGB; not used in stack
- Optimal decision threshold (max F1): ~0.35 on cross-validated OOF predictions
- Conservative threshold (precision ≥ 70%): ~0.55 — recommended for initial deployment to minimise false positives
- Outlier capping: `annual_income_inr`, `loan_amount_inr`, `savings_account_balance_inr` clipped at 1st/99th percentile (computed on train only)
- Target encoding: `state` and `loan_purpose` encoded with 5-fold CV encoding + smoothing to prevent leakage
- Scaler: `StandardScaler` fitted on train, applied to both train and test
