# ============================================================
# Malawi Banking Analytics Project
# File: tests/test_data_cleaning.py
# Author: Kings Mwandira
# Description: Unit tests for data cleaning and
#              financial ratio calculations
# ============================================================

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path so we can import functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Test Data Setup ───────────────────────────────────────────
@pytest.fixture
def sample_banking_data():
    """Create a small sample banking dataset for testing"""
    data = {
        'bank_name': ['National Bank of Malawi', 'FDH Bank',
                      'NBS Bank', 'Ecobank Malawi'],
        'year': [2022, 2022, 2022, 2022],
        'total_assets_mwk_bn': [500.0, 200.0, 300.0, 150.0],
        'total_loans_mwk_bn': [200.0, 80.0, 120.0, 60.0],
        'total_deposits_mwk_bn': [350.0, 140.0, 210.0, 105.0],
        'total_equity_mwk_bn': [50.0, 20.0, 30.0, 15.0],
        'net_interest_income_mwk_bn': [40.0, 16.0, 24.0, 12.0],
        'operating_expenses_mwk_bn': [25.0, 10.0, 15.0, 7.5],
        'profit_after_tax_mwk_bn': [15.0, 6.0, 9.0, 4.5],
        'npl_amount_mwk_bn': [20.0, 8.0, 12.0, 6.0],
        'capital_adequacy_ratio_pct': [15.0, 12.0, 14.0, 13.0],
        'liquidity_ratio_pct': [45.0, 42.0, 48.0, 44.0],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_loan_data():
    """Create a small sample loan dataset for testing"""
    data = {
        'loan_id': ['MLW-000001', 'MLW-000002',
                    'MLW-000003', 'MLW-000004', 'MLW-000005'],
        'bank_name': ['NBS Bank', 'FDH Bank', 'NBS Bank',
                      'Ecobank Malawi', 'FDH Bank'],
        'sector': ['Agriculture', 'Retail', 'SME',
                   'Corporate', 'Personal'],
        'loan_amount_mwk': [500000, 250000, 1000000,
                            2000000, 150000],
        'default_status': [1, 0, 0, 1, 0],
        'repayment_score': [25.0, 75.0, 60.0, 20.0, 80.0],
        'loan_to_value_ratio': [0.85, 0.50, 0.65, 0.90, 0.45],
        'interest_rate_pct': [35.0, 22.0, 28.0, 40.0, 20.0],
        'debt_to_income_ratio': [0.80, 0.35, 0.50, 0.85, 0.30],
    }
    return pd.DataFrame(data)

# ── Test 1: Data Loading ──────────────────────────────────────
class TestDataLoading:

    def test_banking_data_has_correct_columns(self,
                                    sample_banking_data):
        """Test that banking data has all required columns"""
        required_cols = [
            'bank_name', 'year', 'total_assets_mwk_bn',
            'total_loans_mwk_bn', 'total_deposits_mwk_bn',
            'total_equity_mwk_bn', 'capital_adequacy_ratio_pct'
        ]
        for col in required_cols:
            assert col in sample_banking_data.columns, \
                f"Missing column: {col}"

    def test_banking_data_has_correct_shape(self,
                                    sample_banking_data):
        """Test that data has expected number of rows"""
        assert len(sample_banking_data) == 4
        assert len(sample_banking_data.columns) == 12

    def test_loan_data_has_correct_columns(self,
                                    sample_loan_data):
        """Test that loan data has all required columns"""
        required_cols = [
            'loan_id', 'bank_name', 'sector',
            'loan_amount_mwk', 'default_status'
        ]
        for col in required_cols:
            assert col in sample_loan_data.columns, \
                f"Missing column: {col}"

# ── Test 2: Data Quality ──────────────────────────────────────
class TestDataQuality:

    def test_no_missing_values_in_key_columns(self,
                                    sample_banking_data):
        """Test that key columns have no missing values"""
        key_cols = ['bank_name', 'year',
                    'total_assets_mwk_bn',
                    'capital_adequacy_ratio_pct']
        for col in key_cols:
            assert sample_banking_data[col].isnull().sum() == 0, \
                f"Missing values found in {col}"

    def test_all_financial_values_positive(self,
                                    sample_banking_data):
        """Test that all financial values are positive"""
        financial_cols = [
            'total_assets_mwk_bn',
            'total_loans_mwk_bn',
            'total_deposits_mwk_bn'
        ]
        for col in financial_cols:
            assert (sample_banking_data[col] > 0).all(), \
                f"Negative values found in {col}"

    def test_capital_adequacy_within_range(self,
                                    sample_banking_data):
        """Test that CAR is within realistic range 0-100"""
        assert (sample_banking_data[
            'capital_adequacy_ratio_pct'] >= 0).all()
        assert (sample_banking_data[
            'capital_adequacy_ratio_pct'] <= 100).all()

    def test_default_status_is_binary(self, sample_loan_data):
        """Test that default status is only 0 or 1"""
        unique_values = set(
            sample_loan_data['default_status'].unique())
        assert unique_values.issubset({0, 1}), \
            "Default status must be 0 or 1 only"

    def test_loan_ids_are_unique(self, sample_loan_data):
        """Test that all loan IDs are unique"""
        assert sample_loan_data['loan_id'].nunique() == \
               len(sample_loan_data), \
               "Duplicate loan IDs found"

    def test_repayment_score_range(self, sample_loan_data):
        """Test repayment score is between 1 and 100"""
        assert (sample_loan_data['repayment_score'] >= 1).all()
        assert (sample_loan_data[
            'repayment_score'] <= 100).all()

# ── Test 3: Financial Ratio Calculations ─────────────────────
class TestFinancialRatios:

    def test_roa_calculation(self, sample_banking_data):
        """Test Return on Assets calculation is correct"""
        df = sample_banking_data.copy()
        df['roa'] = (df['profit_after_tax_mwk_bn'] /
                     df['total_assets_mwk_bn']) * 100
        expected_roa = (15.0 / 500.0) * 100  # NBM
        assert abs(df['roa'].iloc[0] - expected_roa) < 0.01, \
            "ROA calculation is incorrect"

    def test_roe_calculation(self, sample_banking_data):
        """Test Return on Equity calculation is correct"""
        df = sample_banking_data.copy()
        df['roe'] = (df['profit_after_tax_mwk_bn'] /
                     df['total_equity_mwk_bn']) * 100
        expected_roe = (15.0 / 50.0) * 100  # NBM = 30%
        assert abs(df['roe'].iloc[0] - expected_roe) < 0.01, \
            "ROE calculation is incorrect"

    def test_npl_ratio_calculation(self, sample_banking_data):
        """Test NPL ratio calculation is correct"""
        df = sample_banking_data.copy()
        df['npl_ratio'] = (df['npl_amount_mwk_bn'] /
                           df['total_loans_mwk_bn']) * 100
        expected_npl = (20.0 / 200.0) * 100  # NBM = 10%
        assert abs(df['npl_ratio'].iloc[0] - expected_npl) < 0.01, \
            "NPL ratio calculation is incorrect"

    def test_cost_to_income_calculation(self,
                                    sample_banking_data):
        """Test Cost to Income ratio calculation"""
        df = sample_banking_data.copy()
        df['cir'] = (df['operating_expenses_mwk_bn'] /
                     df['net_interest_income_mwk_bn']) * 100
        expected_cir = (25.0 / 40.0) * 100  # NBM = 62.5%
        assert abs(df['cir'].iloc[0] - expected_cir) < 0.01, \
            "Cost to Income calculation is incorrect"

    def test_loan_to_deposit_calculation(self,
                                    sample_banking_data):
        """Test Loan to Deposit ratio calculation"""
        df = sample_banking_data.copy()
        df['ldr'] = (df['total_loans_mwk_bn'] /
                     df['total_deposits_mwk_bn']) * 100
        expected_ldr = (200.0 / 350.0) * 100  # NBM
        assert abs(df['ldr'].iloc[0] - expected_ldr) < 0.01, \
            "Loan to Deposit calculation is incorrect"

# ── Test 4: Risk Classification ───────────────────────────────
class TestRiskClassification:

    def test_risk_flag_high_risk(self):
        """Test HIGH RISK flag when NPL > 10%"""
        npl_ratio = 15.0
        risk_flag = ("HIGH RISK" if npl_ratio > 10
                     else ("WATCH" if npl_ratio > 5
                           else "HEALTHY"))
        assert risk_flag == "HIGH RISK"

    def test_risk_flag_watch(self):
        """Test WATCH flag when NPL between 5% and 10%"""
        npl_ratio = 7.0
        risk_flag = ("HIGH RISK" if npl_ratio > 10
                     else ("WATCH" if npl_ratio > 5
                           else "HEALTHY"))
        assert risk_flag == "WATCH"

    def test_risk_flag_healthy(self):
        """Test HEALTHY flag when NPL below 5%"""
        npl_ratio = 3.0
        risk_flag = ("HIGH RISK" if npl_ratio > 10
                     else ("WATCH" if npl_ratio > 5
                           else "HEALTHY"))
        assert risk_flag == "HEALTHY"

    def test_rbm_minimum_car_threshold(self):
        """Test RBM minimum CAR requirement of 10%"""
        rbm_minimum = 10.0
        bank_car = 8.5
        assert bank_car < rbm_minimum, \
            "Bank below RBM minimum CAR"

    def test_rbm_npl_threshold(self):
        """Test RBM NPL threshold of 5%"""
        rbm_npl_threshold = 5.0
        bank_npl = 12.0
        assert bank_npl > rbm_npl_threshold, \
            "Bank exceeds RBM NPL threshold"

# ── Test 5: Credit Risk Model Inputs ─────────────────────────
class TestCreditRiskInputs:

    def test_high_risk_borrower_detection(self,
                                    sample_loan_data):
        """Test detection of high risk borrowers"""
        high_risk = sample_loan_data[
            (sample_loan_data['repayment_score'] < 30) |
            (sample_loan_data['loan_to_value_ratio'] > 0.85)
        ]
        assert len(high_risk) >= 1, \
            "Should detect at least one high risk borrower"

    def test_default_rate_calculation(self,
                                    sample_loan_data):
        """Test overall default rate calculation"""
        default_rate = (
            sample_loan_data['default_status'].sum() /
            len(sample_loan_data)) * 100
        assert 0 <= default_rate <= 100, \
            "Default rate must be between 0 and 100"

    def test_agriculture_higher_default(self,
                                    sample_loan_data):
        """Test agriculture sector default detection"""
        agri_loans = sample_loan_data[
            sample_loan_data['sector'] == 'Agriculture']
        if len(agri_loans) > 0:
            agri_default = agri_loans['default_status'].mean()
            assert agri_default >= 0, \
                "Agriculture default rate must be non-negative"

    def test_loan_amount_positive(self, sample_loan_data):
        """Test all loan amounts are positive"""
        assert (sample_loan_data['loan_amount_mwk'] > 0).all(), \
            "All loan amounts must be positive"

# ── Test 6: IFRS 9 Staging ────────────────────────────────────
class TestIFRS9Staging:

    def test_stage3_for_defaulted_loans(self,
                                    sample_loan_data):
        """Test Stage 3 assigned to defaulted loans"""
        def assign_stage(row):
            if row['default_status'] == 1:
                return 'Stage 3 - Credit Impaired'
            elif (row['repayment_score'] < 40 or
                  row['loan_to_value_ratio'] > 0.8):
                return 'Stage 2 - Significant Risk Increase'
            else:
                return 'Stage 1 - Performing'

        df = sample_loan_data.copy()
        df['stage'] = df.apply(assign_stage, axis=1)
        stage3 = df[df['default_status'] == 1]['stage']
        assert (stage3 == 'Stage 3 - Credit Impaired').all(), \
            "All defaulted loans must be Stage 3"

    def test_stage_values_valid(self, sample_loan_data):
        """Test all IFRS 9 stages are valid values"""
        valid_stages = {
            'Stage 1 - Performing',
            'Stage 2 - Significant Risk Increase',
            'Stage 3 - Credit Impaired'
        }
        def assign_stage(row):
            if row['default_status'] == 1:
                return 'Stage 3 - Credit Impaired'
            elif row['repayment_score'] < 40:
                return 'Stage 2 - Significant Risk Increase'
            else:
                return 'Stage 1 - Performing'

        df = sample_loan_data.copy()
        df['stage'] = df.apply(assign_stage, axis=1)
        assert set(df['stage'].unique()).issubset(valid_stages), \
            "Invalid IFRS 9 stage values found"