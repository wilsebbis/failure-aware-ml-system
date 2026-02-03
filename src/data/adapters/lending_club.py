"""
Lending Club Loan Adapter.
Dataset: Lending Club Loan Data
Source: https://www.kaggle.com/wordsforthewise/lending-club
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .base import RiskDataAdapter, AdapterMetadata


class LendingClubAdapter(RiskDataAdapter):
    """Adapter for Lending Club dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = Path(config.get("path", "data/raw/lending_club"))
        self.target_mode = config.get("target_mode", "binary")
        self.primary_key = "id"
    
    def load_raw(self) -> None:
        """Load Lending Club loan data."""
        # Kaggle nested structure
        csv_path = self.data_path / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Lending Club data not found at {csv_path}. "
                f"Download from Kaggle: kaggle datasets download -d wordsforthewise/lending-club"
            )
        
        self.df = pd.read_csv(csv_path, low_memory=False)
        
        # Filter to completed loans
        completed = ["Fully Paid", "Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"]
        if "loan_status" in self.df.columns:
            self.df = self.df[self.df["loan_status"].isin(completed)].copy()
        
        self._metadata = AdapterMetadata(
            name="lending_club",
            num_samples=len(self.df),
            num_features=len(self.df.columns) - 1,
            target_rate=None,
            temporal_column="issue_d",
            primary_key=self.primary_key
        )
    
    def feature_engineer(self) -> None:
        """Apply feature engineering."""
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        # Clean percentage columns
        for col in ["int_rate", "revol_util"]:
            if col in self.df.columns and self.df[col].dtype == "object":
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str).str.replace("%", "", regex=False),
                    errors="coerce"
                )
        
        # Binary default target
        if "loan_status" in self.df.columns:
            self.df["is_default"] = self.df["loan_status"].isin([
                "Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"
            ]).astype(int)
        
        # Simple features
        if "dti" in self.df.columns:
            self.df["dti_binned"] = pd.cut(
                self.df["dti"], bins=[0, 10, 20, 30, 40, 100],
                labels=["very_low", "low", "medium", "high", "very_high"]
            )
        
        if "annual_inc" in self.df.columns and "loan_amnt" in self.df.columns:
            self.df["income_to_loan"] = self.df["annual_inc"] / self.df["loan_amnt"].replace(0, np.nan)
        
        if "grade" in self.df.columns:
            self.df["grade_numeric"] = self.df["grade"].map({"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1})
        
        if "issue_d" in self.df.columns:
            self.df["issue_d"] = pd.to_datetime(self.df["issue_d"], format="%b-%Y", errors="coerce")
            self.df["issue_year"] = self.df["issue_d"].dt.year
        
        # Update metadata
        if "is_default" in self.df.columns:
            self._metadata = AdapterMetadata(
                name="lending_club",
                num_samples=len(self.df),
                num_features=len(self.df.columns) - 1,
                target_rate=self.df["is_default"].mean(),
                temporal_column="issue_d",
                primary_key=self.primary_key
            )
    
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return X and y."""
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        target_col = "is_default"
        
        drop_cols = [
            self.primary_key, "is_default", "loan_status", "issue_d",
            "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
            "total_rec_int", "total_rec_late_fee", "recoveries",
            "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
            "next_pymnt_d", "last_credit_pull_d", "out_prncp", "out_prncp_inv",
        ]
        
        X = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], errors="ignore")
        y = self.df[target_col]
        
        # Encode categoricals
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes
        
        # Drop datetime
        X = X.drop(columns=X.select_dtypes(include=["datetime64"]).columns, errors="ignore")
        X = X.fillna(0)
        
        return X, y
    
    def get_categorical_features(self) -> List[str]:
        """Return categorical features."""
        cats = ["term", "grade", "sub_grade", "emp_length", "home_ownership", 
                "verification_status", "purpose", "addr_state", "application_type"]
        return [c for c in cats if self.df is not None and c in self.df.columns]
    
    def get_split_strategy(self) -> str:
        return "temporal"
    
    def get_temporal_column(self) -> Optional[str]:
        return "issue_d"
