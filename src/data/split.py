"""
Data Splitting Module

Handles stratified train/val/test splits with optional synthetic distribution shift.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional
import logging

from src.config import data_config, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    random_state: int = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Perform stratified train/validation/test split.
    
    Args:
        X: Features DataFrame
        y: Target Series
        train_ratio: Training set ratio (default from config)
        val_ratio: Validation set ratio (default from config)
        test_ratio: Test set ratio (default from config)
        random_state: Random seed (default from config)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Use config defaults
    train_ratio = train_ratio or data_config.train_ratio
    val_ratio = val_ratio or data_config.val_ratio
    test_ratio = test_ratio or data_config.test_ratio
    random_state = random_state or data_config.random_state
    
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    logger.info(f"Splitting data: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=val_test_ratio,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: val vs test
    val_of_temp = val_ratio / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_of_temp),
        stratify=y_temp,
        random_state=random_state
    )
    
    # Log split statistics
    logger.info(f"Train: {len(X_train)} samples, {y_train.mean():.2%} positive")
    logger.info(f"Val: {len(X_val)} samples, {y_val.mean():.2%} positive")
    logger.info(f"Test: {len(X_test)} samples, {y_test.mean():.2%} positive")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_shifted_test_set(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    shift_feature: str = "LIMIT_BAL",
    shift_percentile: float = 0.75,
    random_state: int = None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create a synthetically shifted test set for failure mode analysis.
    
    Simulates distribution shift by selecting samples where a feature
    is above a threshold, representing a demographic shift scenario.
    
    Args:
        X_test: Original test features
        y_test: Original test target
        shift_feature: Feature to shift on
        shift_percentile: Keep samples above this percentile
        random_state: Random seed
        
    Returns:
        Tuple of (shifted X, shifted y)
    """
    random_state = random_state or data_config.random_state
    
    # Find threshold
    threshold = X_test[shift_feature].quantile(shift_percentile)
    
    # Select shifted samples
    mask = X_test[shift_feature] >= threshold
    X_shifted = X_test[mask].copy()
    y_shifted = y_test[mask].copy()
    
    logger.info(f"Created shifted test set: {len(X_shifted)} samples")
    logger.info(f"Shift condition: {shift_feature} >= {threshold:.2f} (p{shift_percentile*100:.0f})")
    logger.info(f"Original positive rate: {y_test.mean():.2%}")
    logger.info(f"Shifted positive rate: {y_shifted.mean():.2%}")
    
    return X_shifted, y_shifted


def save_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> None:
    """Save split data to CSV files."""
    
    # Combine X and y for each split
    train_df = X_train.copy()
    train_df[data_config.target_column] = y_train
    
    val_df = X_val.copy()
    val_df[data_config.target_column] = y_val
    
    test_df = X_test.copy()
    test_df[data_config.target_column] = y_test
    
    # Save to processed directory
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    
    logger.info(f"Splits saved to {PROCESSED_DATA_DIR}")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load previously saved splits."""
    
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    target_col = data_config.target_column
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    logger.info("Splits loaded from disk")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
