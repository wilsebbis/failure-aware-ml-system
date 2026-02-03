"""
Lending Club Loan Adapter.

Demonstrates business value mastery through:
- Profit-focused optimization (IRR over binary default)
- Portfolio-level thinking (maximize risk-adjusted returns)
- Moving beyond accuracy to actual business metrics

Dataset: Lending Club Loan Data
Source: https://www.kaggle.com/wordsforthewise/lending-club
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .base import RiskDataAdapter, AdapterMetadata


class LendingClubAdapter(RiskDataAdapter):
    """
    Adapter for Lending Club dataset with profit-focused optimization.
    
    This adapter demonstrates business value skills by:
    1. Changing target from binary (default/not) to continuous (IRR)
    2. Building a profit-maximizing model, not just accuracy-maximizing
    3. Accounting for the fact that risky loans can be profitable
    
    Key insight: A loan that defaults after 3 years at 20% APR may be
    more profitable than a safe loan at 3% APR.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = Path(config.get("path", "data/raw/lending_club"))
        
        # Target mode: "binary" (default prediction) or "irr" (profitability)
        self.target_mode = config.get("target_mode", "irr")
        self.primary_key = "id"
        
        # Sample size to avoid OOM on 1.6GB dataset (None = load all)
        self.sample_size = config.get("sample_size", 100000)
    
    def load_raw(self) -> None:
        """Load Lending Club loan data."""
        # Try multiple possible file names and locations
        # Kaggle sometimes creates nested directory structures
        possible_paths = [
            self.data_path / "accepted_2007_to_2018Q4.csv",
            self.data_path / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv",  # Nested
            self.data_path / "accepted_2007_to_2018Q4.csv.gz",  # Compressed
            self.data_path / "lending_club_loan_full.csv",
            self.data_path / "loan.csv",
        ]
        
        data_path = None
        for path in possible_paths:
            if path.exists() and path.is_file():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                f"Lending Club data not found in {self.data_path}. "
                f"Download from Kaggle: kaggle datasets download -d wordsforthewise/lending-club"
            )
        
        # Load with low_memory=False due to mixed types
        # Use nrows and usecols to limit memory usage on large datasets
        # Essential columns for credit risk modeling
        essential_cols = [
            # Target columns
            "loan_status", "total_pymnt", "total_rec_prncp",
            # Loan characteristics
            "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
            # Borrower info
            "emp_title", "emp_length", "home_ownership", "annual_inc", 
            "verification_status", "issue_d", "purpose", "title",
            # Credit history
            "dti", "delinq_2yrs", "earliest_cr_line", "inq_last_6mths",
            "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
            # Address
            "addr_state", "zip_code",
            # Other
            "initial_list_status", "application_type", "id",
        ]
        
        self.df = pd.read_csv(
            data_path, 
            low_memory=False,
            nrows=self.sample_size,
            usecols=lambda c: c in essential_cols
        )
        
        # Filter to completed loans only (we need outcomes)
        completed_statuses = [
            "Fully Paid", "Charged Off", "Default",
            "Late (31-120 days)", "Late (16-30 days)"
        ]
        
        if "loan_status" in self.df.columns:
            self.df = self.df[self.df["loan_status"].isin(completed_statuses)].copy()
        
        self._metadata = AdapterMetadata(
            name="lending_club",
            num_samples=len(self.df),
            num_features=len(self.df.columns) - 1,
            target_rate=None,  # Will be set after target calculation
            temporal_column="issue_d",
            primary_key=self.primary_key
        )
    
    def feature_engineer(self) -> None:
        """
        Apply Lending Club specific feature engineering.
        
        Key transformations:
        - Calculate IRR (Internal Rate of Return) as target
        - Create credit risk features
        - Handle percentage strings and dates
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        
        # --- Clean Percentage Columns ---
        pct_cols = ["int_rate", "revol_util"]
        for col in pct_cols:
            if col in self.df.columns and self.df[col].dtype == "object":
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.replace("%", "")
                    .astype(float)
                )
        
        # --- Calculate IRR (the "business value" target) ---
        self._calculate_irr()
        
        # --- Create Binary Default Target (for comparison) ---
        if "loan_status" in self.df.columns:
            self.df["is_default"] = self.df["loan_status"].isin([
                "Charged Off", "Default", 
                "Late (31-120 days)", "Late (16-30 days)"
            ]).astype(int)
        
        # --- Credit Risk Features ---
        # Debt-to-income is key risk indicator
        if "dti" in self.df.columns:
            self.df["dti_binned"] = pd.cut(
                self.df["dti"], 
                bins=[0, 10, 20, 30, 40, 100],
                labels=["very_low", "low", "medium", "high", "very_high"]
            )
        
        # Income to loan ratio
        if "annual_inc" in self.df.columns and "loan_amnt" in self.df.columns:
            self.df["income_to_loan"] = (
                self.df["annual_inc"] / self.df["loan_amnt"].replace(0, np.nan)
            )
        
        # Credit utilization
        if "revol_util" in self.df.columns:
            self.df["high_utilization"] = (self.df["revol_util"] > 80).astype(int)
        
        # Employment length (numeric)
        if "emp_length" in self.df.columns:
            emp_map = {
                "< 1 year": 0.5, "1 year": 1, "2 years": 2, "3 years": 3,
                "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
                "8 years": 8, "9 years": 9, "10+ years": 10, "n/a": np.nan
            }
            self.df["emp_length_years"] = self.df["emp_length"].map(emp_map)
        
        # Grade ordinal encoding
        if "grade" in self.df.columns:
            grade_map = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
            self.df["grade_numeric"] = self.df["grade"].map(grade_map)
        
        # --- Date Features ---
        if "issue_d" in self.df.columns:
            self.df["issue_d"] = pd.to_datetime(
                self.df["issue_d"], format="%b-%Y", errors="coerce"
            )
            self.df["issue_year"] = self.df["issue_d"].dt.year
            self.df["issue_month"] = self.df["issue_d"].dt.month
        
        # Update metadata with target rate
        if self.target_mode == "binary" and "is_default" in self.df.columns:
            self._metadata = AdapterMetadata(
                name="lending_club",
                num_samples=len(self.df),
                num_features=len(self.df.columns) - 1,
                target_rate=self.df["is_default"].mean(),
                temporal_column="issue_d",
                primary_key=self.primary_key
            )
    
    def _calculate_irr(self) -> None:
        """
        Calculate Internal Rate of Return for each loan.
        
        IRR captures the true profitability of a loan, accounting for:
        - Interest rate
        - Payments received before default
        - Principal lost on default
        
        This is the "business value flex" - moving beyond binary classification
        to actual profit optimization.
        """
        if "loan_amnt" not in self.df.columns:
            self.df["irr"] = np.nan
            return
        
        # Simplified IRR calculation
        # Real IRR would require monthly payment schedule reconstruction
        
        # Get total payments received
        total_pymnt = self.df.get("total_pymnt", self.df.get("total_rec_prncp", 0))
        if isinstance(total_pymnt, int):
            total_pymnt = pd.Series([total_pymnt] * len(self.df))
        
        loan_amnt = self.df["loan_amnt"]
        
        # Calculate return
        self.df["total_return"] = (total_pymnt - loan_amnt) / loan_amnt.replace(0, np.nan)
        
        # Annualize based on loan term (assume 36 months if not specified)
        term_months = 36
        if "term" in self.df.columns:
            term_months = (
                self.df["term"]
                .astype(str)
                .str.extract(r"(\d+)")
                .astype(float)
                .fillna(36)
            )
        
        # Approximate IRR (annualized return)
        self.df["irr"] = (
            (1 + self.df["total_return"]) ** (12 / term_months) - 1
        )
        
        # Cap extreme values
        self.df["irr"] = self.df["irr"].clip(-1, 1)
        
        # Binary profitability flag
        self.df["is_profitable"] = (self.df["irr"] > 0).astype(int)
    
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return X and y for Lending Club dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        
        # Define target based on mode
        if self.target_mode == "irr":
            target_col = "irr"
        elif self.target_mode == "profitable":
            target_col = "is_profitable"
        else:
            target_col = "is_default"
        
        # Drop targets, IDs, and non-predictive columns
        drop_cols = [
            self.primary_key, "irr", "is_default", "is_profitable",
            "total_return", "loan_status", "issue_d",
            "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
            "total_rec_int", "total_rec_late_fee", "recoveries",
            "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
            "next_pymnt_d", "last_credit_pull_d", "out_prncp", "out_prncp_inv",
        ]
        
        X = self.df.drop(
            columns=[c for c in drop_cols if c in self.df.columns],
            errors="ignore"
        )
        y = self.df[target_col]
        
        # Encode categorical columns (object dtype)
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            X[col] = X[col].astype("category").cat.codes
        
        # Drop datetime columns (not supported by sklearn)
        datetime_cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
        X = X.drop(columns=datetime_cols, errors="ignore")
        
        # Fill remaining NaN with 0
        X = X.fillna(0)
        
        return X, y
    
    def get_categorical_features(self) -> List[str]:
        """Return categorical features for Lending Club dataset."""
        cat_cols = [
            "term", "grade", "sub_grade", "emp_title", "emp_length",
            "home_ownership", "verification_status", "purpose", "title",
            "addr_state", "initial_list_status", "application_type",
            "dti_binned",
        ]
        
        if self.df is None:
            return cat_cols
        
        return [c for c in cat_cols if c in self.df.columns]
    
    def get_split_strategy(self) -> str:
        """Lending Club has temporal component - use temporal split."""
        return "temporal"
    
    def get_temporal_column(self) -> Optional[str]:
        """Return issue_d for time-based splitting."""
        return "issue_d"
