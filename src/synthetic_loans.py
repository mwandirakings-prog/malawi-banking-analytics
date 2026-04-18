# ============================================================
# Malawi Banking Analytics Project
# File: src/synthetic_loans.py
# Author: Kings
# Description: Generates 100,000 synthetic loan records
#              with stronger default patterns for ML modeling
# ============================================================

import pandas as pd
import numpy as np
import os

print("Generating 100,000 synthetic loan records...")

np.random.seed(42)
NUM_RECORDS = 100_000

banks = [
    "National Bank of Malawi",
    "Standard Bank Malawi",
    "First Capital Bank",
    "NBS Bank",
    "FDH Bank",
    "Ecobank Malawi"
]

sectors = [
    "Agriculture","Retail","SME",
    "Corporate","Personal","Government","Real Estate"
]

loan_purposes = [
    "Working Capital","Asset Finance","Mortgage",
    "Personal Loan","Trade Finance",
    "Agricultural Input","Business Expansion"
]

# ── Generate Base Features ────────────────────────────────────
repayment_score      = np.round(np.random.uniform(1, 100, NUM_RECORDS), 1)
loan_to_value        = np.round(np.random.uniform(0.3, 0.95, NUM_RECORDS), 2)
interest_rate        = np.round(np.random.uniform(18, 45, NUM_RECORDS), 2)
debt_to_income       = np.round(np.random.uniform(0.1, 0.9, NUM_RECORDS), 2)
credit_utilization   = np.round(np.random.uniform(10, 95, NUM_RECORDS), 1)
months_last_default  = np.random.choice(
    [0,6,12,24,36,48,60,999], NUM_RECORDS,
    p=[0.15,0.05,0.08,0.10,0.10,0.10,0.12,0.30])
loan_term            = np.random.choice([6,12,18,24,36,48,60], NUM_RECORDS)
prev_loans           = np.random.choice(range(0, 8), NUM_RECORDS)
sector_arr           = np.random.choice(sectors, NUM_RECORDS,
                        p=[0.25,0.20,0.20,0.10,0.15,0.05,0.05])

# ── Strong Default Probability ────────────────────────────────
default_probability = (
    0.03 +
    (loan_to_value > 0.8)          * 0.15 +
    (loan_to_value > 0.9)          * 0.10 +
    (repayment_score < 30)         * 0.25 +
    (repayment_score < 50)         * 0.10 +
    (interest_rate > 35)           * 0.10 +
    (debt_to_income > 0.7)         * 0.12 +
    (debt_to_income > 0.5)         * 0.06 +
    (months_last_default < 12)     * 0.20 +
    (months_last_default < 24)     * 0.08 +
    (sector_arr == "Agriculture")  * 0.06 +
    (loan_term > 48)               * 0.05 +
    (credit_utilization > 80)      * 0.08 +
    (prev_loans == 0)              * 0.05
)

default_probability = np.clip(default_probability, 0, 1)
default_status = np.random.binomial(n=1, p=default_probability,
                                    size=NUM_RECORDS)

# ── Build DataFrame ───────────────────────────────────────────
df = pd.DataFrame({
    "loan_id": [f"MLW-{str(i).zfill(6)}"
                for i in range(1, NUM_RECORDS + 1)],
    "bank_name": np.random.choice(banks, NUM_RECORDS),
    "sector": sector_arr,
    "loan_purpose": np.random.choice(loan_purposes, NUM_RECORDS),
    "year_originated": np.random.choice(
                [2018,2019,2020,2021,2022,2023], NUM_RECORDS),
    "loan_amount_mwk": np.round(
                np.random.lognormal(13.5, 1.2, NUM_RECORDS), 2),
    "loan_term_months": loan_term,
    "interest_rate_pct": interest_rate,
    "loan_to_value_ratio": loan_to_value,
    "repayment_score": repayment_score,
    "months_since_last_default": months_last_default,
    "number_of_previous_loans": prev_loans,
    "collateral_value_mwk": np.round(
                np.random.lognormal(13.8, 1.3, NUM_RECORDS), 2),
    "debt_to_income_ratio": debt_to_income,
    "credit_utilization_pct": credit_utilization,
    "default_status": default_status,
})

# ── IFRS 9 Staging ────────────────────────────────────────────
def assign_stage(row):
    if row["default_status"] == 1:
        return "Stage 3 - Credit Impaired"
    elif (row["repayment_score"] < 40 or
          row["months_since_last_default"] < 24):
        return "Stage 2 - Significant Risk Increase"
    else:
        return "Stage 1 - Performing"

df["ifrs9_stage"] = df.apply(assign_stage, axis=1)

df["risk_category"] = pd.cut(
    df["repayment_score"],
    bins=[0, 25, 50, 75, 100],
    labels=["Very High Risk","High Risk","Medium Risk","Low Risk"]
)

# ── Save ──────────────────────────────────────────────────────
output_path = os.path.join("data","synthetic","loan_records.csv")
df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("  Synthetic Loan Data Generated Successfully!")
print("=" * 60)
print(f"  Total Records : {len(df):,}")
print(f"  Default Rate  : {df['default_status'].mean()*100:.1f}%")
print(f"  Saved to      : {output_path}")
print("=" * 60)