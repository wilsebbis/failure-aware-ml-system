"""
Home Credit Default Risk Adapter.

Demonstrates data engineering mastery through:
- Multi-table SQL-style joins (7 tables → 1 feature vector)
- Complex aggregations (avg days past due, loan counts, etc.)
- Handling of domain-specific missing value encodings

Dataset: Home Credit Default Risk
Source: https://www.kaggle.com/c/home-credit-default-risk
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .base import RiskDataAdapter, AdapterMetadata


class HomeCreditAdapter(RiskDataAdapter):
    """
    Adapter for Home Credit Default Risk dataset.
    
    This adapter demonstrates ETL pipeline skills by:
    1. Loading 7 relational tables
    2. Performing aggregations on historical records
    3. Joining into a single feature vector per applicant
    
    Tables used:
    - application_train.csv (main table)
    - bureau.csv (credit bureau records)
    - bureau_balance.csv (monthly bureau balances)
    - previous_application.csv (past loan applications)
    - POS_CASH_balance.csv (monthly POS/cash loan balances)
    - credit_card_balance.csv (monthly credit card balances)
    - installments_payments.csv (payment history)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = Path(config.get("path", "data/raw/home_credit"))
        self.target_col = "TARGET"
        self.primary_key = "SK_ID_CURR"
        
        # Store intermediate tables
        self._bureau_agg: Optional[pd.DataFrame] = None
        self._prev_app_agg: Optional[pd.DataFrame] = None
        self._pos_agg: Optional[pd.DataFrame] = None
        self._cc_agg: Optional[pd.DataFrame] = None
        self._install_agg: Optional[pd.DataFrame] = None
    
    def load_raw(self) -> None:
        """
        Load and join Home Credit tables.
        
        This is the "Data Engineering Flex" - demonstrating ability to
        consolidate disparate credit bureau sources into a unified risk view.
        """
        # Main application table
        app_path = self.data_path / "application_train.csv"
        if not app_path.exists():
            raise FileNotFoundError(
                f"Home Credit data not found at {app_path}. "
                f"Download from Kaggle: kaggle competitions download -c home-credit-default-risk"
            )
        
        self.df = pd.read_csv(app_path)
        
        # Load and aggregate bureau records
        self._load_bureau()
        
        # Load and aggregate previous applications
        self._load_previous_applications()
        
        # Load and aggregate POS/cash balance
        self._load_pos_cash()
        
        # Load and aggregate credit card balance
        self._load_credit_card()
        
        # Load and aggregate installment payments
        self._load_installments()
        
        # Join all aggregations to main table
        self._join_all_tables()
        
        self._metadata = AdapterMetadata(
            name="home_credit",
            num_samples=len(self.df),
            num_features=len(self.df.columns) - 1,
            target_rate=self.df[self.target_col].mean(),
            temporal_column=None,
            primary_key=self.primary_key
        )
    
    def _load_bureau(self) -> None:
        """Aggregate credit bureau records per applicant."""
        bureau_path = self.data_path / "bureau.csv"
        if not bureau_path.exists():
            return
        
        bureau = pd.read_csv(bureau_path)
        
        # Aggregate bureau records
        self._bureau_agg = bureau.groupby(self.primary_key).agg({
            "DAYS_CREDIT": ["count", "mean", "min", "max"],
            "DAYS_CREDIT_ENDDATE": ["mean", "min"],
            "DAYS_CREDIT_UPDATE": ["mean"],
            "AMT_CREDIT_SUM": ["sum", "mean"],
            "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
            "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
            "CREDIT_DAY_OVERDUE": ["mean", "max"],
        })
        
        # Flatten column names
        self._bureau_agg.columns = [
            f"BUREAU_{col[0]}_{col[1].upper()}" 
            for col in self._bureau_agg.columns
        ]
        self._bureau_agg = self._bureau_agg.reset_index()
    
    def _load_previous_applications(self) -> None:
        """Aggregate previous loan applications per applicant."""
        prev_path = self.data_path / "previous_application.csv"
        if not prev_path.exists():
            return
        
        prev = pd.read_csv(prev_path)
        
        self._prev_app_agg = prev.groupby(self.primary_key).agg({
            "SK_ID_PREV": "count",  # Number of previous applications
            "AMT_APPLICATION": ["mean", "sum"],
            "AMT_CREDIT": ["mean", "sum"],
            "DAYS_DECISION": ["mean", "min"],
            "CNT_PAYMENT": ["mean", "sum"],
        })
        
        self._prev_app_agg.columns = [
            f"PREV_{col[0]}_{col[1].upper()}" 
            for col in self._prev_app_agg.columns
        ]
        self._prev_app_agg = self._prev_app_agg.reset_index()
        
        # Approval rate (count Approved / total)
        approval = prev.groupby(self.primary_key)["NAME_CONTRACT_STATUS"].apply(
            lambda x: (x == "Approved").mean()
        ).reset_index()
        approval.columns = [self.primary_key, "PREV_APPROVAL_RATE"]
        self._prev_app_agg = self._prev_app_agg.merge(approval, on=self.primary_key, how="left")
    
    def _load_pos_cash(self) -> None:
        """Aggregate POS/cash loan balance history."""
        pos_path = self.data_path / "POS_CASH_balance.csv"
        if not pos_path.exists():
            return
        
        pos = pd.read_csv(pos_path)
        
        self._pos_agg = pos.groupby(self.primary_key).agg({
            "MONTHS_BALANCE": ["count", "min"],
            "SK_DPD": ["mean", "max"],  # Days past due
            "SK_DPD_DEF": ["mean", "max"],  # Days past due (with tolerance)
        })
        
        self._pos_agg.columns = [
            f"POS_{col[0]}_{col[1].upper()}" 
            for col in self._pos_agg.columns
        ]
        self._pos_agg = self._pos_agg.reset_index()
    
    def _load_credit_card(self) -> None:
        """Aggregate credit card balance history."""
        cc_path = self.data_path / "credit_card_balance.csv"
        if not cc_path.exists():
            return
        
        cc = pd.read_csv(cc_path)
        
        self._cc_agg = cc.groupby(self.primary_key).agg({
            "MONTHS_BALANCE": ["count", "min"],
            "AMT_BALANCE": ["mean", "max"],
            "AMT_CREDIT_LIMIT_ACTUAL": ["mean"],
            "AMT_DRAWINGS_CURRENT": ["mean", "sum"],
            "AMT_PAYMENT_CURRENT": ["mean", "sum"],
            "SK_DPD": ["mean", "max"],
        })
        
        self._cc_agg.columns = [
            f"CC_{col[0]}_{col[1].upper()}" 
            for col in self._cc_agg.columns
        ]
        self._cc_agg = self._cc_agg.reset_index()
    
    def _load_installments(self) -> None:
        """Aggregate installment payment history."""
        install_path = self.data_path / "installments_payments.csv"
        if not install_path.exists():
            return
        
        install = pd.read_csv(install_path)
        
        # Calculate payment delay (negative = early, positive = late)
        install["PAYMENT_DELAY"] = install["DAYS_ENTRY_PAYMENT"] - install["DAYS_INSTALMENT"]
        
        # Calculate payment shortage
        install["PAYMENT_SHORTAGE"] = install["AMT_INSTALMENT"] - install["AMT_PAYMENT"]
        
        self._install_agg = install.groupby(self.primary_key).agg({
            "NUM_INSTALMENT_NUMBER": ["count", "max"],
            "PAYMENT_DELAY": ["mean", "max", "sum"],
            "PAYMENT_SHORTAGE": ["mean", "max", "sum"],
            "AMT_PAYMENT": ["sum", "mean"],
        })
        
        self._install_agg.columns = [
            f"INSTALL_{col[0]}_{col[1].upper()}" 
            for col in self._install_agg.columns
        ]
        self._install_agg = self._install_agg.reset_index()
    
    def _join_all_tables(self) -> None:
        """Join all aggregated tables to main application table."""
        for agg_df in [
            self._bureau_agg, 
            self._prev_app_agg, 
            self._pos_agg, 
            self._cc_agg, 
            self._install_agg
        ]:
            if agg_df is not None:
                self.df = self.df.merge(agg_df, on=self.primary_key, how="left")
        
        # Defragment DataFrame after all joins to improve performance
        self.df = self.df.copy()
    
    def feature_engineer(self) -> None:
        """Apply Home Credit-specific feature engineering."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        
        # Handle domain-specific missing value encoding
        # In Home Credit, 365243 in DAYS_EMPLOYED means unemployed/retired
        if "DAYS_EMPLOYED" in self.df.columns:
            self.df["DAYS_EMPLOYED_ANOMALY"] = (
                self.df["DAYS_EMPLOYED"] == 365243
            ).astype(int)
            self.df["DAYS_EMPLOYED"] = self.df["DAYS_EMPLOYED"].replace(
                365243, np.nan
            )
        
        # Income ratios
        if "AMT_INCOME_TOTAL" in self.df.columns:
            if "AMT_CREDIT" in self.df.columns:
                self.df["INCOME_CREDIT_RATIO"] = (
                    self.df["AMT_INCOME_TOTAL"] / 
                    self.df["AMT_CREDIT"].replace(0, np.nan)
                )
            
            if "AMT_ANNUITY" in self.df.columns:
                self.df["INCOME_ANNUITY_RATIO"] = (
                    self.df["AMT_INCOME_TOTAL"] / 
                    self.df["AMT_ANNUITY"].replace(0, np.nan)
                )
        
        # Credit-to-goods ratio
        if "AMT_CREDIT" in self.df.columns and "AMT_GOODS_PRICE" in self.df.columns:
            self.df["CREDIT_GOODS_RATIO"] = (
                self.df["AMT_CREDIT"] / 
                self.df["AMT_GOODS_PRICE"].replace(0, np.nan)
            )
        
        # Age in years
        if "DAYS_BIRTH" in self.df.columns:
            self.df["AGE_YEARS"] = -self.df["DAYS_BIRTH"] / 365
        
        # Employment length in years
        if "DAYS_EMPLOYED" in self.df.columns:
            self.df["EMPLOYED_YEARS"] = -self.df["DAYS_EMPLOYED"] / 365
        
        # Fill NaN in aggregated columns with 0 (no history = 0 count/sum)
        agg_cols = [c for c in self.df.columns if any(
            prefix in c for prefix in ["BUREAU_", "PREV_", "POS_", "CC_", "INSTALL_"]
        )]
        self.df[agg_cols] = self.df[agg_cols].fillna(0)
    
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return X and y for Home Credit dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        
        # Drop ID and target
        drop_cols = [self.primary_key, self.target_col]
        X = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns])
        y = self.df[self.target_col]
        
        # Encode categorical columns (object dtype)
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            X[col] = X[col].astype("category").cat.codes
        
        # Fill remaining NaN with 0
        X = X.fillna(0)
        
        return X, y
    
    def get_categorical_features(self) -> List[str]:
        """Return categorical features for Home Credit dataset."""
        if self.df is None:
            return []
        
        # Object columns are categorical
        cat_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        
        # Remove ID-like columns
        cat_cols = [c for c in cat_cols if self.primary_key not in c]
        
        return cat_cols
    
    def get_split_strategy(self) -> str:
        """Home Credit has no temporal component - use stratified random split."""
        return "stratified"
