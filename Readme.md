# 📉 Customer Churn Prediction — IBM Telco Dataset

> Predicting which telecom customers are likely to churn using supervised machine learning, with a focus on maximizing recall on the churn class for real business value.

<!--[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)](https://lightgbm.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deploy-FF4B4B?logo=streamlit)](https://streamlit.io)

---

## 🔗 Links

| Resource | Link |
|---|---|
| 📓 Notebook | [customer-churn-telecom.ipynb](./customer-churn-telecom.ipynb) |
| 📊 Dataset | [IBM Telco Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| 🚀 Live Demo | *Coming soon — Streamlit deployment* |

---
-->
## 🧩 Problem Statement

Customer churn — when a customer stops using a service — is one of the most expensive problems in the telecom industry. Acquiring a new customer costs 5–10x more than retaining an existing one.

**Goal:** Build a model that identifies customers at high risk of churning so the business can intervene with targeted retention offers *before* they leave.

**Why F1 and recall matter more than accuracy here:**
A model that predicts "No Churn" for everyone would achieve 73.5% accuracy on this dataset — but catch zero churners. We optimize for recall on the churn class: catching real churners matters more than avoiding false alarms.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | IBM Telco Customer Churn (Kaggle) |
| Rows | 7,043 customers |
| Features | 21 (demographics, services, account info) |
| Target | `Churn` — Yes (1) / No (0) |
| Class balance | 73.5% No Churn / 26.5% Churn |

---

## 🔍 Key EDA Findings

| Finding | Insight |
|---|---|
| **Contract type** | Month-to-month customers churn at 43% vs 3% for 2-year contracts — strongest single predictor |
| **Tenure** | 47.7% of customers in their first year churn — early months are highest risk |
| **Fiber optic** | Churn rate is higher than DSL despite customers paying more — signals service quality issues |
| **Monthly charges** | Churners pay higher monthly charges on average |
| **Add-on services** | Customers without Online Security or Tech Support churn significantly more |

---

## ⚙️ ML Pipeline

```
Raw Data
   │
   ├── Fix TotalCharges (object → float, fill 11 NaNs with median)
   ├── Drop customerID (no predictive value)
   │
   ├── EDA (on raw data, before any encoding or splitting)
   │
   ├── Feature Engineering
   │     └── tenure_group: bin tenure into 5 groups (0-1yr, 1-2yr, 2-4yr, 4-5yr, 5-6yr)
   │
   ├── Encoding
   │     ├── Binary cols (Yes/No, Male/Female): map() → 0/1
   │     └── Multi-class cols (Contract, InternetService, etc.): get_dummies()
   │
   ├── Train-Test Split (80/20, stratified)
   │
   ├── Class Imbalance
   │     └── class_weight='balanced' in models (not SMOTE — avoids fake one-hot samples)
   │     └── scale_pos_weight=2.77 for XGBoost
   │
   ├── StandardScaler (fit on train only, applied to test — no leakage)
   │
   └── Model Training → Evaluation → Hyperparameter Tuning → Save Best Model
```

---

## 🤖 Model Results

| Model | CV F1 ± Std | Test F1 | Test Accuracy | Churn Recall |
|---|---|---|---|---|
| **XGBoost (tuned)** | 0.639 ± — | **0.622** | 74.0% | **0.80** |
| LightGBM | 0.612 ± 0.019 | 0.620 | 75.9% | 0.74 |
| Logistic Regression | 0.627 ± 0.025 | 0.613 | 73.5% | 0.79 |
| XGBoost (default) | 0.589 ± 0.022 | 0.599 | 75.7% | 0.68 |
| Random Forest | 0.534 ± 0.031 | 0.545 | **79.0%** | 0.48 |
| Decision Tree | 0.495 ± 0.023 | 0.490 | 73.4% | 0.48 |

### Why XGBoost is saved over Random Forest

Random Forest achieves the highest accuracy (79%) but only catches **48% of actual churners**. XGBoost Tuned catches **80% of churners** with a higher F1 (0.622 vs 0.545).

In churn prediction, missing a churner (false negative) is more costly than a false alarm (false positive) — a missed churner means permanent revenue loss. XGBoost is the correct model to deploy.

---

## 🏆 Best Model — XGBoost (Tuned)

**Best hyperparameters found via GridSearchCV (144 combinations, 3-fold CV):**

```python
{
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_child_weight': 3,
    'n_estimators': 100,
    'subsample': 0.8,
    'scale_pos_weight': 2.77
}
```

**Test set performance:**

```
              precision    recall  f1-score
No Churn          0.91      0.72      0.80
Churn             0.51      0.80      0.62
Accuracy                              0.74
ROC-AUC                               0.843
```

---

## 📈 Top Predictive Features (XGBoost)

1. `tenure` — longer-tenured customers rarely churn
2. `Contract_Two year` — 2-year contract = lowest churn risk
3. `MonthlyCharges` — higher charges increase churn probability
4. `InternetService_Fiber optic` — fiber customers churn more
5. `tenure_group_0-1yr` — newest customers are highest risk
6. `TotalCharges` — proxy for customer lifetime value
7. `Contract_One year` — still protective vs month-to-month
8. `OnlineSecurity_Yes` — add-on services reduce churn

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── customer-churn-telecom.ipynb   # Full ML pipeline notebook
├── best_churn_model.pkl           # Saved XGBoost model (joblib)
├── scaler.pkl                     # Saved StandardScaler
├── app.py                         # Streamlit web app
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook customer-churn-telecom.ipynb

# 4. Run the Streamlit app
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Pandas & NumPy | Data manipulation |
| Seaborn & Matplotlib | EDA visualizations |
| Scikit-learn | Preprocessing, models, evaluation |
| XGBoost | Best performing model |
| LightGBM | Gradient boosting comparison |
| Joblib | Model serialization |
| Streamlit | Web app deployment |

---

## 💡 Business Recommendations

Based on model findings, the highest-impact retention actions are:

1. **Offer long-term contract incentives** to month-to-month customers — the single biggest lever
2. **Proactive outreach in months 1–12** — almost half of first-year customers churn
3. **Investigate fiber optic service quality** — customers paying more are churning more
4. **Bundle add-on services** (Online Security, Tech Support) during onboarding — they reduce churn significantly
5. **Flag high monthly charge + month-to-month** customers for immediate retention campaigns

---

## 👤 Author

**Akula Durga Sri Praveen Kumar**
[GitHub](https://github.com/praveenakula9) 

---

*Dataset source: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) via Kaggle*