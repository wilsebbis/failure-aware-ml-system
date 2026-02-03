"""
IEEE-CIS Fraud Detection Adapter.

Demonstrates ML Ops mastery through:
- Time-based train/test splits (prevents future leakage)
- Handling of 339 anonymized "V" features
- High-velocity transaction stream processing

Dataset: IEEE-CIS Fraud Detection
Source: https://www.kaggle.com/c/ieee-fraud-detection
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .base import RiskDataAdapter, AdapterMetadata


class IEEECISAdapter(RiskDataAdapter):
    """
    Adapter for IEEE-CIS Fraud Detection dataset.
    
    This adapter demonstrates ML Ops skills by:
    1. Using temporal splits (TransactionDT) to prevent data leakage
    2. Handling 339 anonymized "V" features that mimic real bank data
    3. Processing high-velocity transaction streams (500k+ rows)
    
    Key insight: Random K-Fold on time-series data causes future leakage.
    We explicitly split by time to mirror production deployment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_path = Path(config.get("path", "data/raw/ieee_cis"))
        self.target_col = "isFraud"
        self.temporal_col = "TransactionDT"
        self.primary_key = "TransactionID"
        
        # Time split ratio (e.g., first 80% for train, last 20% for test)
        self.train_time_ratio = config.get("train_time_ratio", 0.8)
    
    def load_raw(self) -> None:
        """
        Load IEEE-CIS transaction and identity tables.
        
        The dataset consists of:
        - train_transaction.csv: Transaction details + target
        - train_identity.csv: Device/browser info (optional join)
        """
        # Load transaction data
        trans_path = self.data_path / "train_transaction.csv"
        if not trans_path.exists():
            raise FileNotFoundError(
                f"IEEE-CIS data not found at {trans_path}. "
                f"Download from Kaggle: kaggle competitions download -c ieee-fraud-detection"
            )
        
        self.df = pd.read_csv(trans_path)
        
        # Load and join identity data (optional)
        identity_path = self.data_path / "train_identity.csv"
        if identity_path.exists():
            identity = pd.read_csv(identity_path)
            self.df = self.df.merge(identity, on=self.primary_key, how="left")
        
        # Defragment DataFrame after joins to improve performance
        self.df = self.df.copy()
        
        self._metadata = AdapterMetadata(
            name="ieee_cis",
            num_samples=len(self.df),
            num_features=len(self.df.columns) - 1,
            target_rate=self.df[self.target_col].mean(),
            temporal_column=self.temporal_col,
            primary_key=self.primary_key
        )
    
    def feature_engineer(self) -> None:
        """
        Apply IEEE-CIS specific feature engineering.
        
        Key transformations:
        - Time-based features from TransactionDT
        - Aggregations over V-features
        - Card/device fingerprinting
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        
        # --- Time Features ---
        # TransactionDT is seconds from reference point
        if self.temporal_col in self.df.columns:
            # Convert to interpretable time (hour of day, day of week)
            # Reference: TransactionDT starts from some epoch
            self.df["TransactionHour"] = (
                self.df[self.temporal_col] // 3600
            ) % 24
            
            self.df["TransactionDay"] = (
                self.df[self.temporal_col] // (3600 * 24)
            ) % 7
        
        # --- V-Feature Aggregations ---
        # The 339 V-features are anonymized, but we can create aggregations
        v_cols = [c for c in self.df.columns if c.startswith("V")]
        
        if v_cols:
            # Summary statistics across V-features
            self.df["V_mean"] = self.df[v_cols].mean(axis=1)
            self.df["V_std"] = self.df[v_cols].std(axis=1)
            self.df["V_nan_count"] = self.df[v_cols].isna().sum(axis=1)
            
            # Reduce V-features dimensionality by grouping
            # (real deployment might use PCA, but this keeps it interpretable)
            v_groups = {
                "V_1_11": [f"V{i}" for i in range(1, 12) if f"V{i}" in v_cols],
                "V_12_34": [f"V{i}" for i in range(12, 35) if f"V{i}" in v_cols],
                "V_35_52": [f"V{i}" for i in range(35, 53) if f"V{i}" in v_cols],
                "V_53_74": [f"V{i}" for i in range(53, 75) if f"V{i}" in v_cols],
                "V_75_94": [f"V{i}" for i in range(75, 95) if f"V{i}" in v_cols],
            }
            
            for group_name, group_cols in v_groups.items():
                valid_cols = [c for c in group_cols if c in self.df.columns]
                if valid_cols:
                    self.df[f"{group_name}_mean"] = self.df[valid_cols].mean(axis=1)
        
        # --- Amount Features ---
        if "TransactionAmt" in self.df.columns:
            self.df["TransactionAmt_log"] = np.log1p(self.df["TransactionAmt"])
            
            # Amount deviation from card average (if card info available)
            if "card1" in self.df.columns:
                card_means = self.df.groupby("card1")["TransactionAmt"].transform("mean")
                self.df["Amt_deviation"] = self.df["TransactionAmt"] - card_means
        
        # --- Device/Browser Fingerprint ---
        # Create composite keys for device fingerprinting
        device_cols = ["DeviceType", "DeviceInfo"]
        for col in device_cols:
            if col in self.df.columns:
                # Fill NaN with "unknown"
                self.df[col] = self.df[col].fillna("unknown")
        
        # --- Email Domain Features ---
        email_cols = ["P_emaildomain", "R_emaildomain"]
        for col in email_cols:
            if col in self.df.columns:
                # Extract top-level domain
                self.df[f"{col}_suffix"] = (
                    self.df[col]
                    .fillna("unknown")
                    .apply(lambda x: x.split(".")[-1] if pd.notna(x) else "unknown")
                )
    
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Return X and y for IEEE-CIS dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        
        # Drop ID, target, and temporal column (used for splitting, not features)
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
        """Return categorical features for IEEE-CIS dataset."""
        if self.df is None:
            return []
        
        # Explicitly defined categorical columns
        cat_cols = [
            "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "P_emaildomain", "R_emaildomain",
            "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
            "DeviceType", "DeviceInfo",
        ]
        
        # Filter to columns that actually exist
        return [c for c in cat_cols if c in self.df.columns]
    
    def get_split_strategy(self) -> str:
        """IEEE-CIS MUST use temporal split to prevent leakage."""
        return "temporal"
    
    def get_temporal_column(self) -> Optional[str]:
        """Return TransactionDT for time-based splitting."""
        return self.temporal_col
    
    def get_temporal_split_indices(self) -> Tuple[pd.Index, pd.Index]:
        """
        Return train/test indices based on time.
        
        This is the key MLOps insight: we split by time, not randomly,
        to prevent future data leakage.
        
        Returns:
            train_idx: Indices for training set (earlier transactions)
            test_idx: Indices for test set (later transactions)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")
        
        if self.temporal_col not in self.df.columns:
            raise ValueError(f"Temporal column {self.temporal_col} not found")
        
        # Sort by time and split
        sorted_df = self.df.sort_values(self.temporal_col)
        split_point = int(len(sorted_df) * self.train_time_ratio)
        
        train_idx = sorted_df.index[:split_point]
        test_idx = sorted_df.index[split_point:]
        
        return train_idx, test_idx
