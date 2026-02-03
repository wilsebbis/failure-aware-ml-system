"""
UCI Credit Card Default Adapter.

Migrates existing UCI dataset logic to the adapter pattern for backward compatibility.
This is the original "bootcamp" dataset - kept for testing and demonstration purposes.

Dataset: UCI Default of Credit Card Clients
Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

from .base import RiskDataAdapter, AdapterMetadata


class UCICreditAdapter(RiskDataAdapter):
    """
    Adapter for UCI Credit Card Default dataset.
    
    This is a single-table dataset (simple) kept for:
    - Backward compatibility with existing pipeline
    - Quick testing and demonstration
    - Baseline comparison against more complex datasets
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_col = config.get("target_col", "default.payment" )
        self.data_path = config.get("path", "data/raw/UCI_Credit_Card.csv")
    
    def load_raw(self) -> None:
        """Load the UCI Credit Card CSV."""
        self.df = pd.read_csv(self.data_path)
        
        # Rename target column if needed (UCI has inconsistent naming)
        if "default.payment.next.month" in self.df.columns:
            self.df = self.df.rename(columns={
                "default.payment.next.month": "default_payment"
            })
        elif "default payment next month" in self.df.columns:
            self.df = self.df.rename(columns={
                "default payment next month": "default_payment"
            })
        else:
            # Assume last column is target
            self.df = self.df.rename(columns={self.df.columns[-1]: "default_payment"})
        
        # Drop ID column if present
        if "ID" in self.df.columns:
            self.df = self.df.drop(columns=["ID"])
        
        self._metadata = AdapterMetadata(
            name="uci_credit",
            num_samples=len(self.df),
            num_features=len(self.df.columns) - 1,
            target_rate=self.df["default_payment"].mean(),
            temporal_column=None,
            primary_key=None
        )
    
    def feature_engineer(self) -> None:
        """Apply minimal feature engineering for UCI dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        
        # Create utilization ratio
        if "LIMIT_BAL" in self.df.columns and "BILL_AMT1" in self.df.columns:
            self.df["utilization_ratio"] = (
                self.df["BILL_AMT1"] / self.df["LIMIT_BAL"].replace(0, np.nan)
            ).fillna(0).clip(0, 5)
        
        # Create max payment delay
        pay_cols = [c for c in self.df.columns if c.startswith("PAY_") and c != "PAY_AMT1"]
        if pay_cols:
            self.df["max_delay"] = self.df[pay_cols].max(axis=1)
        
        # Create average bill amount
        bill_cols = [c for c in self.df.columns if c.startswith("BILL_AMT")]
        if bill_cols:
            self.df["avg_bill"] = self.df[bill_cols].mean(axis=1)
        
        # Create payment-to-bill ratio
        pay_amt_cols = [c for c in self.df.columns if c.startswith("PAY_AMT")]
        if pay_amt_cols and bill_cols:
            self.df["payment_ratio"] = (
                self.df[pay_amt_cols].sum(axis=1) / 
                self.df[bill_cols].sum(axis=1).replace(0, np.nan)
            ).fillna(0).clip(0, 10)
    
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return X and y for UCI dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        
        X = self.df.drop(columns=["default_payment"])
        y = self.df["default_payment"]
        return X, y
    
    def get_categorical_features(self) -> List[str]:
        """Return categorical features for UCI dataset."""
        return ["SEX", "EDUCATION", "MARRIAGE"]
    
    def get_split_strategy(self) -> str:
        """UCI has no temporal component - use stratified random split."""
        return "stratified"
    
    def get_temporal_column(self) -> Optional[str]:
        """UCI has no temporal column."""
        return None
