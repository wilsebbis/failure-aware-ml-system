"""
Data Loading Module

Handles loading the UCI Credit Card Default dataset with schema validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

from ucimlrepo import fetch_ucirepo

from src.config import data_config, RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def fetch_credit_default_data(
    use_cache: bool = True,
    cache_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Fetch the UCI Credit Card Default dataset.
    
    Args:
        use_cache: If True, load from cache if available
        cache_path: Custom cache path (defaults to RAW_DATA_DIR)
        
    Returns:
        DataFrame with features and target combined
    """
    cache_path = cache_path or RAW_DATA_DIR / "credit_default_raw.csv"
    
    if use_cache and cache_path.exists():
        logger.info(f"Loading cached data from {cache_path}")
        return pd.read_csv(cache_path)
    
    logger.info("Fetching UCI Credit Card Default dataset...")
    
    # Fetch from UCI repository
    dataset = fetch_ucirepo(id=data_config.uci_dataset_id)
    
    # Combine features and target
    X = dataset.data.features
    y = dataset.data.targets
    
    # Rename target column if needed
    if "default.payment.next.month" in y.columns:
        y = y.rename(columns={"default.payment.next.month": data_config.target_column})
    elif "Y" in y.columns:
        y = y.rename(columns={"Y": data_config.target_column})
    
    df = pd.concat([X, y], axis=1)
    
    # The UCI dataset returns X1, X2, ... format - rename to standard names
    column_mapping = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
    }
    
    # Only rename columns that exist
    existing_renames = {k: v for k, v in column_mapping.items() if k in df.columns}
    if existing_renames:
        df = df.rename(columns=existing_renames)
        logger.info(f"Renamed {len(existing_renames)} columns to standard format")
    
    # Cache for future use
    df.to_csv(cache_path, index=False)
    logger.info(f"Data cached to {cache_path}")
    
    return df


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate data against expected schema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors = []
    
    # Check target column exists
    if data_config.target_column not in df.columns:
        errors.append(f"Target column '{data_config.target_column}' not found")
    
    # Expected feature columns
    expected_cols = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]
    
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")
    
    # Validate ranges
    if "LIMIT_BAL" in df.columns:
        if (df["LIMIT_BAL"] <= 0).any():
            n_invalid = (df["LIMIT_BAL"] <= 0).sum()
            errors.append(f"LIMIT_BAL has {n_invalid} non-positive values")
    
    if "AGE" in df.columns:
        invalid_age = (df["AGE"] < 18) | (df["AGE"] > 100)
        if invalid_age.any():
            n_invalid = invalid_age.sum()
            errors.append(f"AGE has {n_invalid} values outside [18, 100]")
    
    if "SEX" in df.columns:
        invalid_sex = ~df["SEX"].isin([1, 2])
        if invalid_sex.any():
            n_invalid = invalid_sex.sum()
            errors.append(f"SEX has {n_invalid} values not in [1, 2]")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        cols_with_missing = missing_counts[missing_counts > 0].to_dict()
        errors.append(f"Missing values found: {cols_with_missing}")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info("Schema validation passed")
    else:
        logger.warning(f"Schema validation failed with {len(errors)} errors")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return is_valid, errors


def load_data(validate: bool = True) -> pd.DataFrame:
    """
    Main entry point for loading data.
    
    Args:
        validate: If True, validate schema after loading
        
    Returns:
        DataFrame with validated data
        
    Raises:
        ValueError: If validation fails
    """
    df = fetch_credit_default_data()
    
    # Drop ID column if present
    for col in data_config.drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"Dropped column: {col}")
    
    if validate:
        is_valid, errors = validate_schema(df)
        if not is_valid:
            raise ValueError(f"Schema validation failed: {errors}")
    
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Target distribution: {df[data_config.target_column].value_counts(normalize=True).to_dict()}")
    
    return df


def get_class_weight(y: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost based on class imbalance.
    
    Args:
        y: Target series
        
    Returns:
        Ratio of negative to positive samples
    """
    n_positive = (y == 1).sum()
    n_negative = (y == 0).sum()
    
    weight = n_negative / n_positive
    logger.info(f"Computed scale_pos_weight: {weight:.2f} (neg/pos ratio)")
    
    return weight
