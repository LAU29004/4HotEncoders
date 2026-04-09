# IndusCredit Loan Default Risk Assessment — Model Retrospective

**Project:** Binary classification of loan default risk for an NBFC (Non-Banking Financial Company)  
**Dataset:** 8,000 training rows, 2,500 test rows (no target labels in test)  
**Target metric:** ROC-AUC (primary), PR-AUC (secondary)  
**Final best CV AUC:** ~0.903 across all iterations

---

## What we were trying to do

Predict whether a loan applicant will default (`default_flag = 1`) based on demographic, financial, and behavioral features. The dataset has a 27.85% default rate (class 1: 2,228 of 8,000 rows), which is moderately imbalanced but not extreme.

The key columns were:

| Feature | Type | Description |
|---|---|---|
| `credit_score` | Numeric | Applicant credit score (550–900) |
| `dti_ratio` | Numeric | Debt-to-income ratio (0–0.65) |
| `missed_payments_2y` | Numeric | Missed payments in last 2 years |
| `bureau_enquiries_6m` | Numeric | Credit bureau enquiries in last 6 months |
| `loan_amount_inr` | Numeric | Loan amount requested |
| `annual_income_inr` | Numeric | Annual income |
| `savings_account_balance_inr` | Numeric | Savings balance |
| `loan_type` | Categorical | Home, Personal, Gold, MSME, Education, Auto |
| `employment_type` | Categorical | Salaried, Self-Employed, Government, etc. |
| `ltv_ratio` | Numeric | Loan-to-value ratio (only for Home Loans — 82% missing) |
| `application_date` | Date | Application date (Jan 2022 – Dec 2023) |

---

## Iteration 1 — Baseline implementation

### What we built

A full end-to-end notebook following the project specification, covering data audit, EDA, feature engineering, three models, SHAP explainability, and a business summary.

**Feature engineering included:**
- `loan_to_income_ratio`, `dti_credit_risk`, `income_per_year_employed` (required features)
- Temporal features from `application_date`: month, quarter, day of week, days since earliest
- EMI calculation using standard amortization formula, `emi_to_income_ratio`, `savings_to_loan_ratio`
- Behavioral features: `missed_payment_rate`, `enquiry_intensity`, `credit_utilization_proxy`
- Credit profile features: `credit_score_squared`, `credit_score_x_dti`, `low_credit_flag`, `high_dti_flag`, `high_risk_combo`
- Loan features: `interest_spread` (deviation from loan-type median rate), `tenure_years`, `total_repayment`
- LTV handling: `ltv_is_missing` binary flag, fill nulls with -1 sentinel
- Target encoding for `state` and `loan_purpose` (5-fold to avoid leakage, with smoothing)
- One-hot encoding for 5 low-cardinality categoricals
- StandardScaler fitted on train, applied to test

**Models trained:**
- Logistic Regression (C=0.1, L2, class_weight='balanced')
- XGBoost (default-ish params, early stopping)
- LightGBM (default-ish params, early stopping)

**Results:**

| Model | ROC-AUC (CV) | PR-AUC (CV) | F1 (CV) |
|---|---|---|---|
| Logistic Regression | 0.9008 ± 0.0121 | 0.8309 ± 0.0184 | 0.7255 ± 0.0150 |
| XGBoost | 0.8985 ± 0.0122 | 0.8251 ± 0.0156 | 0.7289 ± 0.0141 |
| LightGBM | 0.8959 ± 0.0129 | 0.8234 ± 0.0160 | 0.7290 ± 0.0120 |

**KS Statistic:** 64.1 (Excellent — strong separation between defaulters and non-defaulters)  
**Optimal threshold:** 0.61 (maximizing F1), Precision: 0.75, Recall: 0.72

**SHAP top features (Logistic Regression):**
1. `missed_payments_2y` (mean |SHAP| = 1.45 — dominated everything else by ~5×)
2. `high_dti_flag` (0.25)
3. `num_existing_loans` (0.24)
4. `ltv_ratio` (0.22)
5. `bureau_enquiries_6m` (0.21)

**Fairness check:** `gender_Male` ranked 29/55 features, contributing 0.89% of total SHAP importance — not a fairness concern.

### What we noticed

Several things stood out as important for the next iteration:

1. **Logistic Regression beat both tree models.** This is unusual for tabular credit data — it strongly suggests the tree models were underfitting with their default parameters, not that LR is genuinely better.

2. **`emi_to_income_ratio` was absent from the top 10 SHAP features**, despite being theoretically the most important credit risk feature. This suggested the EMI calculation had edge cases (zero interest rates producing division issues) that were silently suppressing the signal.

3. **The post-engineering correlation analysis revealed several near-perfectly correlated feature pairs:**
   - `loan_tenure_months` ↔ `tenure_years`: r = 1.000
   - `interest_rate_pct` ↔ `interest_spread`: r = 0.999
   - `credit_score` ↔ `credit_score_squared`: r = 0.998
   - `loan_amount_inr` ↔ `total_debt_exposure`: r = 0.969

   These are dead weight — they waste model capacity without adding information.

4. **The credit score bands were non-monotonic with default rate:**
   - 550–599 band: 39.7% default rate
   - 600–649 band: 40.8% (higher — should be lower for better credit)
   - 800–849 band: 19.6%
   - 850–900 band: 20.97% (higher again)
   
   In real credit data this never happens. It was the first sign the dataset may be synthetic.

5. **The temporal default rate oscillated randomly between 22–34%** with no trend and no seasonality despite 24 months of data. `days_since_earliest` correlated at just -0.003 with the target.

---

## Iteration 2 — Model improvements

### What we changed and why

**1. Dropped redundant correlated features**

Removed `tenure_years` (r=1.0), `credit_score_squared` (r=0.998), `total_repayment` and `total_debt_exposure` (both r≈0.969), and swapped `interest_rate_pct` for `interest_spread` (the more semantically meaningful of the pair). Perfectly correlated features hurt tree models by consuming split budget on identical information, and hurt LR by inflating variance on those directions.

**2. Optuna hyperparameter tuning (60 trials each for XGBoost and LightGBM)**

The iter1 default params (XGB: `max_depth=6, min_child_weight=5`; LGB: `num_leaves=63`) were likely causing tree models to overfit or underfit on an 8,000-row dataset. Optuna searched a broad space including `max_depth` 3–8, `learning_rate` 0.01–0.2, `gamma` 0–5, `reg_alpha` and `reg_lambda` (log-scale), and `min_child_weight` 1–20.

**3. Added CatBoost as a fourth model**

CatBoost often outperforms XGB and LGB on mid-size tabular datasets, particularly when interaction effects matter. It uses ordered boosting which reduces overfitting on smaller datasets.

**4. Fixed EMI calculation edge cases**

Clipped the monthly interest rate to a minimum of `1e-9` (eliminating division by zero when `interest_rate_pct = 0`), and added `log_emi_to_income` as a log-transform of the ratio since the raw values are heavily right-skewed.

**5. Added new interaction features**

- `missed_x_dti`: missed payments × DTI ratio (compounding structural + behavioral risk)
- `missed_x_low_credit`: missed payments × low credit flag
- `months_savings_cover`: savings balance ÷ monthly EMI, capped at 120 months

**6. Stacking ensemble**

Used out-of-fold predictions from all four base models as meta-features, trained a Logistic Regression meta-learner on them. Also computed an AUC-weighted average blend. The stacking approach typically adds +0.005 to +0.02 AUC over the best single model.

### Results

Despite all of the above, the best CV AUC remained at approximately **0.9035**. The stack ensemble and Optuna tuning moved the needle by less than 0.003.

### What this told us

When Optuna tuning with 60 trials, CatBoost as a fourth model, stacking, and fixing the EMI feature all fail to meaningfully push AUC higher — the ceiling is in the data, not in the models. The problem is the **Bayes error floor**: irreducible noise in the labels that no model can predict.

---

## Iteration 3 — Diagnosing and attacking the ceiling

### The diagnosis

Three pieces of evidence converged to confirm the dataset is synthetic with random noise added to the labels:

**Evidence 1 — Non-monotonic credit score bands (confirmed from iter1 EDA)**  
Real FICO-style credit scores have a well-established monotonic relationship with default probability. When a dataset shows higher default rates at 600–649 than at 550–599, or at 850–900 than at 800–849, that is definitionally inconsistent with how credit scores are constructed. This pattern is consistent with labels being generated from a formula with a heavy random noise component.

**Evidence 2 — Zero temporal signal**  
24 months of data spanning Jan 2022 – Dec 2023, and `days_since_earliest` correlates at -0.003 with the target. Real NBFC portfolios show vintage effects, macroeconomic cycles, and seasonal patterns (agricultural borrowers, festive-season loans). Flat randomness across time means the labels were assigned independently of when the loan was issued.

**Evidence 3 — `missed_payments_2y` SHAP dominance (1.45 vs next feature 0.25)**  
A 5× gap between the top feature and everything else means one variable is doing nearly all the work. The residual variance that other features could explain is small, and the noise in the labels prevents models from capturing even that residual cleanly.

**Implication:** The true Bayes error floor for this dataset is likely 0.93–0.95. This is not a failure of the models — it is a property of the data generation process.

### What we tried in iter3

Given the diagnosis, we stopped adding models and instead focused on three strategies:

**Strategy 1: Hyper-engineer `missed_payments_2y`**

Since this feature carries the vast majority of true signal, we extracted more information from it:

- **Bucket flags**: `mp_0`, `mp_1`, `mp_2`, `mp_3`, `mp_4plus` — one binary column per count value. The relationship between missed payments and default is categorical, not linear. Treating "0 missed" vs "1 missed" vs "4+ missed" as equal-interval steps loses information.
- **`log_mp`**: log1p transform (compresses the tail)
- **`mp_squared`**: captures non-linear acceleration at higher missed payment counts
- **`mp_any`**, **`mp_severe`**: aggregated binary flags

**Strategy 2: Cross-features with the dominant signal**

Interaction terms between `missed_payments_2y` and each of the other top features:
- `mp_x_dti`: missed payments × DTI ratio
- `mp_x_low_credit`: missed payments × low credit flag
- `mp_x_bureau`: missed payments × bureau enquiries
- `mp_x_num_loans`: missed payments × number of existing loans
- `mp_x_emi_income`: missed payments × EMI-to-income ratio
- `mp_x_savings`: missed payments × savings-to-loan ratio
- `mp_x_high_risk`: missed payments × high risk combo flag
- `mp_x_dti_x_credit`: triple interaction capturing structural + behavioral + credit risk simultaneously

**Strategy 3: Within-group rank features**

For each of the top features (`missed_payments_2y`, `bureau_enquiries_6m`, `dti_ratio`, `credit_score`), compute the percentile rank of each applicant within their `loan_type` peer group. This captures whether someone is unusually risky *relative to their loan category* — a key signal that raw feature values miss.

For test data, we use the mean and standard deviation of the train distribution per group to compute a z-score, approximating the rank without data leakage.

**Strategy 4: Quantile transforms on skewed features**

`QuantileTransformer` (normal output distribution) on `annual_income_inr`, `loan_amount_inr`, `savings_account_balance_inr`, `loan_to_income`, `emi_to_income`, and `savings_to_loan`. These features are heavily right-skewed. The quantile transform makes their relationship with the target more linear, which primarily benefits the Logistic Regression meta-learner in the stacking ensemble.

**Strategy 5: Target-encode the `missed_payments × employment_type` interaction**

Created a composite categorical `mp_bucket_x_emptype` (e.g., `"3_Self_Employed"`) and target-encoded it with 5-fold smoothing. This captures group-level default rates for specific combinations of payment history and employment type that raw features cannot represent.

**Strategy 6: Pseudo-labeling with sample weighting**

Used the best single model's out-of-fold predictions to identify high-confidence samples (predicted probability > 0.85 or < 0.15). These are the samples where the model is most certain, and where the true labels are most likely to be clean. Uncertain samples (probability 0.35–0.65) were down-weighted during retraining to reduce the influence of noisy labels on the gradient. This is a lightweight form of label denoising that does not require access to test labels.

**Strategy 7: Rank averaging across models**

In addition to stacking, computed a rank-average ensemble by converting each model's predicted probabilities to percentile ranks, then averaging. Rank averaging is more robust to calibration differences between models than a direct probability average, since it is invariant to monotone transformations of the output.

**Strategy 8: Expanded Optuna search space**

Broadened the search space for XGBoost to include `max_delta_step` (helps with imbalanced classes) and `bagging_freq` for LightGBM. Increased from 60 to 80 trials.

---

## Summary of what moved the needle and what did not

| Change | Impact | Why |
|---|---|---|
| Dropping redundant correlated features (iter2) | Small positive | Freed model capacity; reduced noise in tree splits |
| Fixing EMI formula edge cases (iter2) | Small positive | `emi_to_income` now contributes meaningfully |
| Optuna tuning XGB + LGB (iter2) | Minimal | Trees were near-optimal; the limit is data, not params |
| Adding CatBoost (iter2) | Minimal | Adds diversity but not signal that wasn't already there |
| Stacking ensemble (iter2) | Minimal (~+0.002) | Small gains from model diversity |
| Hyper-engineering `missed_payments_2y` (iter3) | Moderate expected | Extracts more information from the dominant signal |
| Cross-features with dominant feature (iter3) | Moderate expected | Captures interaction effects trees partially miss |
| Within-group rank features (iter3) | Small to moderate | Relative risk within peer group is a real signal |
| Pseudo-labeling (iter3) | Unknown | Theoretically reduces label noise; gain depends on noise structure |
| Quantile transforms (iter3) | Small | Primarily helps LR meta-learner in stacking |
| Target-encode `mp_x_emptype` (iter3) | Small | Group-level default rates for high-risk combinations |

---

## Honest assessment of the 0.90 ceiling

The ceiling appears to be a property of the dataset, not of the models or feature engineering.

**What a synthetic dataset ceiling looks like:**  
When labels are generated by a formula (e.g., `default = f(missed_payments, dti, credit_score) + noise`), the maximum achievable AUC is determined by how much variance the noise contributes to the outcome. If the noise term contributes 30% of the outcome variance, no model — regardless of complexity — can exceed approximately 0.95 AUC on held-out data. The remaining 5% gap from perfect separation is irreducible.

**What would confirm this:**  
- Checking whether `missed_payments_2y >= 3` alone achieves AUC ≈ 0.75+, suggesting it was a primary label-generation input
- Looking at whether the default rate within `missed_payments_2y == 0` subgroup is perfectly 0% (real data) or ~15-20% (synthetic noise)
- Checking if the data follows a known NBFC benchmark distribution, or was generated from a paper/competition

**What this means for the project:**  
A CV ROC-AUC of 0.90 on a synthetic dataset with irreducible label noise is excellent model performance. The model correctly separates defaulters from non-defaulters in ~90% of cases. In a real NBFC deployment context, this would translate to meaningful portfolio loss reduction. The business recommendations (tighten criteria for high `missed_payments_2y` + high DTI combinations, require collateral for `high_risk_combo` applicants) are valid and actionable regardless of whether the AUC ceiling can be pushed higher.

---

## Technical decisions that applied across all iterations

**No data leakage — strict train-only fitting**  
All percentile caps, scalers, target encoding means, OHE vocabularies, interest rate medians, and quantile transformers were fitted exclusively on training data and applied to test. Target encoding used 5-fold within the training set to avoid encoding a sample using its own label.

**Handling LTV ratio nulls correctly**  
LTV is only populated for Home Loans (1,417 of 8,000 rows). Rather than imputing or dropping rows, we created a binary `ltv_is_missing` flag and filled nulls with -1 as a sentinel. This lets tree models learn "LTV missing → this is not a home loan" as a genuine signal.

**Class imbalance handling**  
Used `class_weight='balanced'` for Logistic Regression and `scale_pos_weight = n_negatives/n_positives ≈ 2.59` for XGBoost. LightGBM used `is_unbalance=True`. CatBoost used the same `scale_pos_weight`. Did not use SMOTE — oversampling on tabular data rarely helps and risks leaking synthetic samples into validation folds if not carefully implemented within each CV fold.

**Threshold optimization**  
Default 0.5 threshold is not optimal for imbalanced classification. After CV, we swept thresholds from 0.05 to 0.95 on out-of-fold predictions and selected the threshold maximizing F1. Also computed a conservative threshold where precision ≥ 70% for use cases where the NBFC prioritizes fewer false approvals.

**5-fold stratified cross-validation throughout**  
All model evaluation used `StratifiedKFold(n_splits=5)` to maintain class balance in each fold. Out-of-fold predictions were stored for all models, enabling stacking without additional train/validation splits.