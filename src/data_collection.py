# ============================================================
# Malawi Banking Analytics Project
# File: src/data_collection.py
# Author: Kings
# Description: Creates and cleans Malawian banking data
# ============================================================

import pandas as pd
import numpy as np
import os

# ── 1. Define the 6 Malawian Banks ──────────────────────────
banks = [
    "National Bank of Malawi",
    "Standard Bank Malawi",
    "First Capital Bank",
    "NBS Bank",
    "FDH Bank",
    "Economic Bank"
]

years = [2018, 2019, 2020, 2021, 2022, 2023]

# ── 2. Create Realistic Banking Data ────────────────────────
np.random.seed(42)

records = []

for bank in banks:
    for year in years:
        record = {
            "bank_name": bank,
            "year": year,
            "total_assets_mwk_bn": round(np.random.uniform(50, 500), 2),
            "total_loans_mwk_bn": round(np.random.uniform(20, 200), 2),
            "total_deposits_mwk_bn": round(np.random.uniform(30, 300), 2),
            "total_equity_mwk_bn": round(np.random.uniform(5, 50), 2),
            "net_interest_income_mwk_bn": round(np.random.uniform(5, 40), 2),
            "operating_expenses_mwk_bn": round(np.random.uniform(3, 30), 2),
            "profit_after_tax_mwk_bn": round(np.random.uniform(1, 20), 2),
            "npl_amount_mwk_bn": round(np.random.uniform(1, 25), 2),
            "capital_adequacy_ratio_pct": round(np.random.uniform(10, 25), 2),
            "liquidity_ratio_pct": round(np.random.uniform(30, 70), 2),
        }
        records.append(record)

# ── 3. Create DataFrame ──────────────────────────────────────
df = pd.DataFrame(records)

# ── 4. Calculate CAMELS Ratios ───────────────────────────────
df["return_on_assets_pct"] = round(
    (df["profit_after_tax_mwk_bn"] / df["total_assets_mwk_bn"]) * 100, 2)

df["return_on_equity_pct"] = round(
    (df["profit_after_tax_mwk_bn"] / df["total_equity_mwk_bn"]) * 100, 2)

df["npl_ratio_pct"] = round(
    (df["npl_amount_mwk_bn"] / df["total_loans_mwk_bn"]) * 100, 2)

df["cost_to_income_ratio_pct"] = round(
    (df["operating_expenses_mwk_bn"] / df["net_interest_income_mwk_bn"]) * 100, 2)

df["loan_to_deposit_ratio_pct"] = round(
    (df["total_loans_mwk_bn"] / df["total_deposits_mwk_bn"]) * 100, 2)

# ── 5. Add Risk Flag ─────────────────────────────────────────
df["risk_flag"] = df["npl_ratio_pct"].apply(
    lambda x: "HIGH RISK" if x > 10 else ("WATCH" if x > 5 else "HEALTHY"))

# ── 6. Save to processed folder ─────────────────────────────
output_path = os.path.join("data", "processed", "malawi_banking_data.csv")
df.to_csv(output_path, index=False)

print("=" * 55)
print(" Malawi Banking Data Created Successfully!")
print("=" * 55)
print(f" Banks: {len(banks)}")
print(f" Years: {years[0]} to {years[-1]}")
print(f" Total Records: {len(df)}")
print(f" Columns: {len(df.columns)}")
print(f" Saved to: {output_path}")
print("=" * 55)
print("\n CAMELS Summary:")
print(df[["bank_name", "year", "return_on_assets_pct",
          "npl_ratio_pct", "capital_adequacy_ratio_pct",
          "risk_flag"]].to_string(index=False))