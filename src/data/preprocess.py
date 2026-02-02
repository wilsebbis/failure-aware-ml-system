"""
Data Preprocessing Module

Handles encoding, scaling, and transformation of raw data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional
import logging
import pickle
from pathlib import Path

from src.config import PROCESSED_DATA_DIR, data_config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessing pipeline for credit default data.
    
    Handles:
    - Categorical encoding (one-hot for EDUCATION, MARRIAGE)
    - Continuous feature scaling (StandardScaler)
    - Preserves fit state for inference consistency
    """
    
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
        
        # Columns to scale
        self.continuous_cols = [
            "LIMIT_BAL", "AGE",
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        ]
        
        # Columns to one-hot encode
        self.categorical_cols = ["EDUCATION", "MARRIAGE"]
        
        # Payment history columns (ordinal, keep as-is)
        self.ordinal_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        
        # Binary column (already numeric)
        self.binary_cols = ["SEX"]
    
    def _clean_education(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean EDUCATION column.
        
        Original values: 1=grad school, 2=university, 3=high school, 4+=others/unknown
        Group 4, 5, 6, 0 → 4 (Other)
        """
        df = df.copy()
        df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
        return df
    
    def _clean_marriage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean MARRIAGE column.
        
        Original values: 1=married, 2=single, 3=others, 0=unknown
        Group 0, 3 → 3 (Other)
        """
        df = df.copy()
        df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
        return df
    
    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit preprocessing transformations.
        
        Args:
            df: Training data (features only, no target)
            
        Returns:
            Self for chaining
        """
        logger.info("Fitting preprocessor...")
        
        # Fit scaler on continuous columns
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.continuous_cols])
        
        self.is_fitted = True
        logger.info("Preprocessor fitted")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming")
        
        df = df.copy()
        
        # Clean categorical columns
        df = self._clean_education(df)
        df = self._clean_marriage(df)
        
        # Scale continuous columns
        df[self.continuous_cols] = self.scaler.transform(df[self.continuous_cols])
        
        # One-hot encode categorical columns
        df = pd.get_dummies(
            df, 
            columns=self.categorical_cols, 
            prefix=self.categorical_cols,
            drop_first=True  # Avoid multicollinearity
        )
        
        # Convert boolean columns to int
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
        
        logger.info(f"Transformed data: {df.shape[1]} features")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save fitted preprocessor to disk."""
        path = path or PROCESSED_DATA_DIR / "preprocessor.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DataPreprocessor":
        """Load fitted preprocessor from disk."""
        path = path or PROCESSED_DATA_DIR / "preprocessor.pkl"
        with open(path, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
    
    def get_feature_names(self, df: pd.DataFrame) -> list[str]:
        """
        Get feature names after transformation.
        
        Args:
            df: Sample DataFrame to transform
            
        Returns:
            List of feature column names
        """
        transformed = self.transform(df.head(1))
        return list(transformed.columns)


def preprocess_data(
    df: pd.DataFrame,
    preprocessor: Optional[DataPreprocessor] = None,
    fit: bool = True
) -> tuple[pd.DataFrame, pd.Series, DataPreprocessor]:
    """
    Main preprocessing entry point.
    
    Args:
        df: Raw DataFrame with features and target
        preprocessor: Existing preprocessor (for inference)
        fit: If True, fit a new preprocessor
        
    Returns:
        Tuple of (features DataFrame, target Series, preprocessor)
    """
    # Separate features and target
    target_col = data_config.target_column
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Create or use preprocessor
    if fit:
        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X)
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor required when fit=False")
        X_processed = preprocessor.transform(X)
    
    return X_processed, y, preprocessor
