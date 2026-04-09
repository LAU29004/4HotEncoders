
| Metric | Value |
|--------|-------|
| **Final Model** | Stacking Classifier (Random Forest + XGBoost) |
| **Performance** | ~0.9067 AUC-ROC |
| **Optimization** | Bayesian Optimization via Optuna (15+ trials) |
| **Explainability** | SHAP (Waterfall & Summary plots) |
| **Target NPA Reduction** | 4.8% → 3.5% |


## 🧠 Feature Engineering (Task 2.1 Compliant)

The pipeline implements specific financial risk metrics to enhance predictive power:

### Key Financial Risk Metrics

| Feature | Formula | Purpose |
|---------|---------|---------|
| **Loan-to-Income Ratio** | `loan_amount_inr / annual_income_inr` | Measures total debt burden relative to earning capacity |
| **DTI Credit Risk** | `dti_ratio / (credit_score / 700)` | Non-linear feature that penalizes high debt more severely for lower credit tiers |
| **Income Per Year Employed** | `annual_income_inr / (employment_years + 1)` | Normalized stability metric accounting for employment history |
| **LTV Sentinel Logic** | Binary flag + sentinel value (-1) | Handles missing Loan-to-Value ratios for non-mortgage assets |

### Feature Engineering Highlights

- **Loan-to-Income Ratio:** Captures debt sustainability relative to annual income
- **DTI Credit Risk:** Non-linear penalty function that amplifies risk for applicants with poor credit scores and high DTI
- **Income Stability:** Normalizes income by employment duration to identify consistent earners
- **LTV Handling:** Implements binary classification (`is_home_loan`) with sentinel value (-1) for missing values

---

## 🏗️ Model Architecture

Our **Stacking Ensemble** approach captures both linear and non-linear patterns in credit risk data:

```
┌─────────────────────────────────────┐
│      Training Data                  │
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────────┐  ┌──────────┐
│ Random     │  │ XGBoost  │
│ Forest     │  │ (Tuned)  │
└────────┬───┘  └────┬─────┘
         │           │
    ┌────┴───────────┴────┐
    │ Meta Features       │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Logistic Regression │
    │ (Meta-Learner)      │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ Default Probability │
    │ (0-1)               │
    └─────────────────────┘
```

### Component Details

1. **Base Model 1: RandomForestClassifier**
   - Handles high-dimensional feature interactions
   - Robust to outliers and non-linear relationships
   - Provides feature importance insights

2. **Base Model 2: XGBoost**
   - Optimized via Gradient Boosting for skewed default distributions
   - Tuned hyperparameters: learning_rate, max_depth, n_estimators
   - Handles class imbalance effectively

3. **Meta-Model: Logistic Regression**
   - Aggregates base model predictions
   - Outputs final default probability
   - Interpretable decision boundary

---

## 🔍 Explainability & Insights

Using **SHAP (SHapley Additive exPlanations)**, we identified the key drivers of loan default:

### Key Findings

- **Top Predictors:** 
  - `dti_credit_risk` - Consistently shows highest impact on model output
  - `missed_payments_2y` - Recent payment history is a strong default indicator
  - `credit_score` - Non-linear relationship with default probability

- **Local Interpretability:** 
  - Waterfall plots generated for individual applicants
  - Explains **why** the model classified a specific applicant as "High Risk"
  - Example insights: "Recent bureau enquiries increased risk despite good credit score"

- **Global Interpretability:** 
  - Summary plots show aggregate feature influence across entire dataset
  - Identify consistent risk patterns across borrower segments

---

## 🚀 How to Run

### 1. Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv credit_risk_env
source credit_risk_env/bin/activate  # On Windows: credit_risk_env\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost optuna shap matplotlib seaborn
```

### 2. Training & Hyperparameter Optimization

```bash
# Run the main training pipeline
python final_stacking_code.ipynb
```

The notebook will:
- Load `loan_train.csv`
- Execute Optuna optimization (15+ trials)
- Train base models (Random Forest + XGBoost)
- Train meta-learner
- Generate SHAP explainability plots
- Save the optimized model

### 3. Inference on Test Set

```bash
# Generate predictions on test data
# The script will:
# - Load test_predictions.csv
# - Apply optimized threshold
# - Generate test_predictions_report.csv with risk scores and classifications
```

### 4. Expected Output Files

- `model_weights.pkl` - Trained stacking ensemble
- `optuna_study.pkl` - Hyperparameter optimization history
- `test_predictions_report.csv` - Predictions with confidence scores
- `shap_waterfall_plots/` - Individual-level explainability
- `shap_summary_plot.png` - Global feature importance

---

## 📈 Business Impact

The model enables the NBFC to implement a strategic roadmap for NPA (Non-Performing Assets) reduction:

### Key Metrics

| Metric | Target | Strategy |
|--------|--------|----------|
| **NPA Reduction** | 4.8% → 3.5% | Risk-based lending decisions |
| **Portfolio Quality** | +15% better credit mix | DTI-capped, risk-segmented origination |
| **Interest Premium** | Dynamic by risk tier | "Medium Risk" cluster charging 2-3% higher rates |

### Implementation Strategy

1. **Hard DTI Cap:** Set maximum DTI ratio at 40% to filter high-risk applicants
2. **Risk-Based Pricing:** 
   - Green (Low Risk): Standard rate
   - Amber (Medium Risk): +2-3% premium
   - Red (High Risk): Declined or +5%+ premium
3. **Portfolio Monitoring:** Quarterly performance tracking using SHAP insights

### Expected Outcomes

- Reduce portfolio default rate from 4.8% to 3.5% within 12 months
- Increase portfolio profitability by 8-12% through risk-adjusted pricing
- Improve customer satisfaction by transparent, explainable credit decisions

---


## 🔧 Configuration & Hyperparameters

### Random Forest Base Model
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
```

### XGBoost Base Model
```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,  # Handle class imbalance
    random_state=42
)
```

### Optuna Optimization
- **Objective:** Maximize AUC-ROC on validation fold
- **Sampler:** TPESampler (Tree-structured Parzen Estimator)
- **Trials:** 15+ iterations with cross-validation (5-fold)
- **Trial Timeout:** 10 minutes per trial



### Validation Strategy
- Stratified 5-fold cross-validation
- Class weights applied to handle imbalanced target
- Threshold optimization via ROC curve analysis

