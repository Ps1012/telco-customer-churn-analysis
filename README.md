# Customer Churn Prediction & Risk Analysis — Telco

> Predicting which telecom customers are likely to churn using machine learning,
> and identifying the key business drivers behind customer attrition.

---

## Overview

Customer churn is one of the most expensive problems in the telecom industry.
Acquiring a new customer costs 5–7x more than retaining an existing one.
This project builds an end-to-end pipeline to identify high-risk customers
before they leave — giving the business time to intervene with targeted
retention strategies.

**Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 features  
**Tools:** Python · Pandas · Scikit-learn · XGBoost · SMOTE · Plotly  
**Best Model:** XGBoost · ROC-AUC: `[fill after running]` · Recall: `[fill after running]`

---

## Key Findings

- 📌 **Month-to-month contracts** churn at **42.7%** vs just **2.8%** on two-year plans — the single strongest predictor
- 📌 **Fiber optic customers** churn at **41.9%** — nearly double the DSL rate of 19%
- 📌 **Electronic check payment** users churn at **45.3%** — the highest of any payment method
- 📌 The **first 12 months** of tenure is the highest-risk window — churned customers average 18 months vs 37 for retained customers
- 📌 Customers **without Tech Support or Online Security** churn significantly more than those who have these services bundled

---

## Project Structure

```
telco-customer-churn-analysis/
│
├── customer_churn.ipynb              ← Main analysis notebook
├── churn_predictions.csv             ← Model predictions export (for Power BI)
├── requirements.txt                  ← Python dependencies
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── images/
│   ├── churn_distribution.png
│   ├── feature_importance.png
│   ├── roc_curve.png
│   └── risk_heatmap.png
│
└── README.md
```

---

## Notebook Walkthrough

The notebook is divided into 10 clean sections, each with markdown explanations:

| Section | What it covers |
|---|---|
| 1. Setup | Libraries and shared chart configuration |
| 2. Load Data | Read CSV, inspect shape, column types, first rows |
| 3. Data Cleaning | Fix TotalCharges dtype, uncover 11 hidden nulls, check duplicates |
| 4. EDA | 7 interactive Plotly charts — distributions, category churn rates, correlation heatmap, tenure cohorts, scatter analysis |
| 5. Feature Engineering | 3 new features created, binary + one-hot encoding, StandardScaler, SMOTE |
| 6. Modelling | Train and cross-validate Logistic Regression, Random Forest, XGBoost |
| 7. Evaluation | Confusion matrix, ROC curves, Precision-Recall curve, CV scores, feature importance |
| 8. Business Insights | Risk segment heatmap, payment method analysis, probability distribution |
| 9. Export | Predictions CSV with Risk_Level column ready for Power BI |
| 10. Summary | Full results recap and business recommendations |

---

## Data Quality Issues Found & Fixed

| Issue | What was wrong | How it was fixed |
|---|---|---|
| `TotalCharges` dtype | Stored as string instead of float | Converted using `pd.to_numeric(errors='coerce')` |
| Hidden nulls | 11 nulls appeared only after dtype conversion | Investigated — all had `tenure = 0` (new customers never billed). Filled with `0`. |
| `customerID` | Random identifier with no predictive value | Dropped before modelling |

> Detecting a hidden null that only surfaces after a dtype conversion — and explaining the business reason behind it — is the kind of analytical thinking that distinguishes a good data analyst from someone who just runs code.

---

## Feature Engineering

Three new features were created from existing columns:

| Feature | Formula | Why it helps |
|---|---|---|
| `AvgMonthlySpend` | `TotalCharges / (tenure + 1)` | Normalises spend by customer age — a $500 spend in 5 months vs 50 months signals very different risk |
| `IsNewCustomer` | `1 if tenure <= 12 else 0` | Directly flags the highest-risk window identified in EDA |
| `ChargesPerService` | `MonthlyCharges / active services` | Measures value for money — high charge per service = likely feeling overcharged |

---

## Model Results

Three models were trained and compared using 5-fold cross-validation:

| Model | Accuracy | F1-Score | ROC-AUC | CV F1 (mean) |
|---|---|---|---|---|
| Logistic Regression | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| Random Forest | `[fill]` | `[fill]` | `[fill]` | `[fill]` |
| XGBoost | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

> **Why not just use accuracy?**  
> The dataset has a 73.5% / 26.5% class split. A model that always predicts "No Churn"
> would score 73.5% accuracy while catching zero churners — completely useless.
> F1-Score and ROC-AUC are the meaningful metrics here.

**Class imbalance** was handled using SMOTE — applied on the training set only.
Applying SMOTE before splitting would contaminate the test set and produce
misleadingly inflated scores.

---

## Charts Included

All charts are built with **Plotly** — fully interactive (hover, zoom, filter).

| Chart | What it shows |
|---|---|
| Churn Distribution | Bar + donut chart of churn vs retained customers |
| Numeric Features vs Churn | Overlapping histograms and box plots for tenure, monthly charges, total charges |
| Categorical Churn Rates | 6-panel bar chart — contract, internet service, payment method, tech support, security, senior citizen |
| Correlation Heatmap | Numeric feature relationships and multicollinearity check |
| Tenure Cohort Analysis | Dual-axis chart — customer count and churn rate by 12-month tenure buckets |
| Scatter Plot | Individual customers plotted by tenure vs monthly charges, coloured by churn |
| Model Comparison | Side-by-side accuracy, F1, and AUC for all three models |
| Confusion Matrix | Annotated heatmap for the best model |
| ROC Curves | All 3 models overlaid on one chart |
| Precision-Recall Curve | Trade-off analysis across models |
| Cross-Validation Scores | 5-fold F1 scores per model with mean lines |
| Feature Importance | Top 20 features ranked by importance |
| Risk Segment Heatmap | Churn rate by contract type × internet service combination |
| Payment & Contract Bars | Churn rate by payment method and contract type |
| Probability Distribution | Model confidence score distribution for churned vs retained |

---

## Business Recommendations

Based on the analysis, five actions are recommended for the retention team:

1. **Target new customers (0–12 months)** — run proactive check-in calls at months 3 and 6, offer loyalty rewards early in the relationship
2. **Incentivise contract upgrades** — month-to-month customers churn at 42.7%; even a small discount for switching to an annual plan would significantly reduce attrition
3. **Investigate fiber optic service quality** — a 41.9% churn rate suggests pricing or reliability issues that need a dedicated review
4. **Promote auto-payment methods** — electronic check users churn at 45.3%; offer a small monthly discount for switching to credit card or bank transfer
5. **Bundle support services** — customers without Tech Support and Online Security churn far more; include these in base plans or offer a free trial period

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Ps1012/telco-customer-churn-analysis
cd telco-customer-churn-analysis

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook customer_churn.ipynb
# or open directly in Google Colab
```

To run in **Google Colab**:
1. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` using the Colab file panel
2. Run all cells in order from top to bottom
3. `churn_predictions.csv` will be saved and ready to download

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
plotly
```

---

## About

**Prabhjot Singh**  
Aspiring Data Analyst | Python · SQL · Power BI  
📧 ps844601@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/prabhjot-singh-35b17732a)  
🐙 [GitHub](https://github.com/Ps1012)

---

*IBM Telco Customer Churn dataset sourced from Kaggle. Used for portfolio and learning purposes.*
