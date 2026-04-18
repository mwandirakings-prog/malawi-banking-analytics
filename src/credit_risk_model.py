# ============================================================
# Malawi Banking Analytics Project
# File: src/credit_risk_model.py
# Author: Kings
# Description: Improved XGBoost Credit Risk Model
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import shap

print("=" * 60)
print("  Malawi Banking Credit Risk Model")
print("  XGBoost + SHAP Explainability")
print("=" * 60)

# ── 1. Generate Better Training Data Directly ─────────────────
print("\n[1/7] Generating high-quality training data...")

np.random.seed(42)
N = 100_000

# Generate features with strong signal
repayment_score     = np.random.uniform(1, 100, N)
loan_to_value       = np.random.uniform(0.3, 0.95, N)
debt_to_income      = np.random.uniform(0.1, 0.9, N)
interest_rate       = np.random.uniform(18, 45, N)
credit_utilization  = np.random.uniform(10, 95, N)
months_last_default = np.random.choice(
    [0,6,12,24,36,48,60,999], N,
    p=[0.10,0.05,0.08,0.10,0.10,0.10,0.12,0.35])
loan_term           = np.random.choice([6,12,18,24,36,48,60], N)
prev_loans          = np.random.choice(range(0, 8), N)
loan_amount         = np.random.lognormal(13.5, 1.2, N)
collateral          = np.random.lognormal(13.8, 1.3, N)

banks = ["National Bank of Malawi","Standard Bank Malawi",
         "First Capital Bank","NBS Bank","FDH Bank","Ecobank Malawi"]
sectors = ["Agriculture","Retail","SME","Corporate",
           "Personal","Government","Real Estate"]
purposes = ["Working Capital","Asset Finance","Mortgage",
            "Personal Loan","Trade Finance",
            "Agricultural Input","Business Expansion"]

bank_arr    = np.random.choice(banks, N)
sector_arr  = np.random.choice(sectors, N,
              p=[0.25,0.20,0.20,0.10,0.15,0.05,0.05])
purpose_arr = np.random.choice(purposes, N)

# ── Very strong default signal ────────────────────────────────
# Normalize repayment score to 0-1 (lower = more risky)
repayment_risk = (100 - repayment_score) / 100

# Build strong linear combination
log_odds = (
    -3.0 +                              # intercept
    3.5  * repayment_risk +             # strongest predictor
    2.0  * loan_to_value +              # LTV risk
    2.0  * debt_to_income +             # debt burden
    1.5  * (interest_rate / 45) +       # rate risk
    1.2  * (credit_utilization / 100) + # utilization
    1.8  * (months_last_default < 12).astype(float) +
    1.0  * (months_last_default < 24).astype(float) +
    0.8  * (sector_arr == "Agriculture").astype(float) +
    0.5  * (loan_term > 48).astype(float) +
    -0.5 * (prev_loans > 3).astype(float)  # experience reduces risk
)

# Convert log odds to probability
default_prob = 1 / (1 + np.exp(-log_odds))
default_prob = np.clip(default_prob, 0, 1)
default_status = np.random.binomial(1, default_prob, N)

# Build DataFrame
df = pd.DataFrame({
    "loan_amount_mwk":            np.round(loan_amount, 2),
    "loan_term_months":           loan_term,
    "interest_rate_pct":          np.round(interest_rate, 2),
    "loan_to_value_ratio":        np.round(loan_to_value, 2),
    "repayment_score":            np.round(repayment_score, 1),
    "months_since_last_default":  months_last_default,
    "number_of_previous_loans":   prev_loans,
    "collateral_value_mwk":       np.round(collateral, 2),
    "debt_to_income_ratio":       np.round(debt_to_income, 2),
    "credit_utilization_pct":     np.round(credit_utilization, 1),
    "bank_name":                  bank_arr,
    "sector":                     sector_arr,
    "loan_purpose":               purpose_arr,
    "default_status":             default_status,
})

print(f"      Records created : {N:,}")
print(f"      Default rate    : {default_status.mean()*100:.1f}%")

# ── 2. Encode Categoricals ────────────────────────────────────
print("[2/7] Encoding features...")
le = LabelEncoder()
for col in ["bank_name", "sector", "loan_purpose"]:
    df[col] = le.fit_transform(df[col])

features = [c for c in df.columns if c != "default_status"]
X = df[features]
y = df["default_status"]

# ── 3. Train Test Split ───────────────────────────────────────
print("[3/7] Splitting data 80/20...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 4. Train XGBoost ──────────────────────────────────────────
print("[4/7] Training XGBoost model...")

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="auc",
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)
print("      Training complete!")

# ── 5. Evaluate ───────────────────────────────────────────────
print("[5/7] Evaluating performance...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc   = roc_auc_score(y_test, y_prob)
gini  = 2 * auc - 1

print("\n" + "=" * 60)
print("  MODEL PERFORMANCE RESULTS")
print("=" * 60)
print(f"  AUC-ROC Score : {auc:.4f}")
print(f"  Gini Score    : {gini:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Performing','Default'])}")

# ── 6. Cross Validation ───────────────────────────────────────
print("[6/7] Cross validation (5 folds)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"      CV Scores : {cv_scores.round(4)}")
print(f"      Mean AUC  : {cv_scores.mean():.4f}")
print(f"      Std AUC   : {cv_scores.std():.4f}")

# ── 7. SHAP ───────────────────────────────────────────────────
print("[7/7] Calculating SHAP values...")
explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_test[:2000])

os.makedirs("data/processed/plots", exist_ok=True)

plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_test[:2000],
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance — Malawi Credit Risk",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("data/processed/plots/shap_importance.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("      SHAP chart saved!")

# ── Save Predictions ──────────────────────────────────────────
out = X_test.copy()
out["actual_default"]      = y_test.values
out["predicted_default"]   = y_pred
out["default_probability"] = y_prob.round(4)
out["risk_tier"] = pd.cut(
    out["default_probability"],
    bins=[0, 0.1, 0.3, 0.5, 1.0],
    labels=["Low Risk","Medium Risk","High Risk","Very High Risk"]
)
out.to_csv("data/processed/model_predictions.csv", index=False)

print("\n" + "=" * 60)
print("  CREDIT RISK MODEL COMPLETE!")
print("=" * 60)
print(f"  AUC Score   : {auc:.4f}")
print(f"  Gini Score  : {gini:.4f}")
print(f"  CV Mean AUC : {cv_scores.mean():.4f}")
print("  Predictions : data/processed/model_predictions.csv")
print("  SHAP Plot   : data/processed/plots/shap_importance.png")
print("=" * 60)