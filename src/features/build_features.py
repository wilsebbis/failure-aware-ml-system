"""
Feature Engineering Module

Builds derived features from raw data.
Includes ratio-first feature engineering for credit risk.
"""

import pandas as pd
import numpy as np
import logging

from .interaction_features import add_ratio_features

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame, dataset: str = "auto") -> pd.DataFrame:
    """
    Build derived features from preprocessed data.
    
    Args:
        df: Preprocessed DataFrame with original features
        dataset: Dataset name for ratio feature selection
        
    Returns:
        DataFrame with added derived features
    """
    df = df.copy()
    
    # Add ratio-first features (most important for credit risk)
    df = add_ratio_features(df, dataset=dataset)
    
    # Payment history columns
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    
    # Check which columns exist (some may be renamed after preprocessing)
    existing_pay_cols = [c for c in pay_cols if c in df.columns]
    existing_bill_cols = [c for c in bill_cols if c in df.columns]
    existing_pay_amt_cols = [c for c in pay_amt_cols if c in df.columns]
    
    # Utilization ratio
    if "BILL_AMT1" in df.columns and "LIMIT_BAL" in df.columns:
        # Handle scaled values - use relative ratio
        df["utilization_ratio"] = df["BILL_AMT1"] / (df["LIMIT_BAL"].abs() + 1e-6)
        logger.debug("Created: utilization_ratio")
    
    # Average payment delay
    if existing_pay_cols:
        df["avg_payment_delay"] = df[existing_pay_cols].mean(axis=1)
        logger.debug("Created: avg_payment_delay")
    
    # Maximum payment delay
    if existing_pay_cols:
        df["max_delay"] = df[existing_pay_cols].max(axis=1)
        logger.debug("Created: max_delay")
    
    # Severe delay flag (3+ months)
    if "max_delay" in df.columns:
        # Note: In scaled data, we need to use relative thresholds
        # Assuming PAY columns are NOT scaled (they're ordinal)
        df["has_severe_delay"] = (df["max_delay"] >= 3).astype(int)
        logger.debug("Created: has_severe_delay")
    
    # Payment ratio (recent payment / recent bill)
    if "PAY_AMT1" in df.columns and "BILL_AMT1" in df.columns:
        df["payment_ratio"] = df["PAY_AMT1"] / (df["BILL_AMT1"].abs() + 1e-6)
        # Clip extreme values
        df["payment_ratio"] = df["payment_ratio"].clip(-10, 10)
        logger.debug("Created: payment_ratio")
    
    # Total bill over 6 months
    if existing_bill_cols:
        df["total_bill"] = df[existing_bill_cols].sum(axis=1)
        logger.debug("Created: total_bill")
    
    # Total payment over 6 months
    if existing_pay_amt_cols:
        df["total_payment"] = df[existing_pay_amt_cols].sum(axis=1)
        logger.debug("Created: total_payment")
    
    # Payment consistency (std of monthly payments)
    if existing_pay_amt_cols:
        df["payment_consistency"] = df[existing_pay_amt_cols].std(axis=1)
        logger.debug("Created: payment_consistency")
    
    # Count of delayed payments
    if existing_pay_cols:
        df["n_delayed_payments"] = (df[existing_pay_cols] > 0).sum(axis=1)
        logger.debug("Created: n_delayed_payments")
    
    # Recent vs old payment trend
    if "PAY_AMT1" in df.columns and "PAY_AMT6" in df.columns:
        df["payment_trend"] = df["PAY_AMT1"] - df["PAY_AMT6"]
        logger.debug("Created: payment_trend")
    
    # Bill growth trend
    if "BILL_AMT1" in df.columns and "BILL_AMT6" in df.columns:
        df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]
        logger.debug("Created: bill_trend")
    
    n_derived = len(df.columns) - len(existing_pay_cols) - len(existing_bill_cols) - len(existing_pay_amt_cols)
    logger.info(f"Built {n_derived} derived features, total: {len(df.columns)} features")
    
    return df


def get_feature_importances_template() -> dict[str, str]:
    """
    Get expected feature importance interpretations for audit.
    
    Returns:
        Dict mapping feature names to expected behavior descriptions
    """
    return {
        "max_delay": "Higher values (more delay) should increase default probability",
        "has_severe_delay": "Presence of severe delay should strongly increase default probability",
        "avg_payment_delay": "Higher average delay should increase default probability",
        "utilization_ratio": "Higher utilization should increase default probability",
        "payment_ratio": "Lower payment ratio should increase default probability",
        "n_delayed_payments": "More delayed payments should increase default probability",
        "LIMIT_BAL": "Lower credit limit may indicate higher risk (bank assessment)",
        "AGE": "Relationship may be non-linear; very young may be higher risk",
        "payment_consistency": "Higher variance may indicate financial instability",
    }


def validate_feature_behavior(
    feature_importances: dict[str, float],
    shap_values: np.ndarray = None
) -> dict[str, bool]:
    """
    Validate that feature importances align with domain expectations.
    
    This is an audit check to ensure the model learned sensible patterns.
    
    Args:
        feature_importances: Dict of feature name to importance score
        shap_values: Optional SHAP values for direction validation
        
    Returns:
        Dict of feature name to validation result (True = expected behavior)
    """
    expected = get_feature_importances_template()
    results = {}
    
    # Check that high-risk features have notable importance
    risk_features = ["max_delay", "has_severe_delay", "avg_payment_delay", "PAY_0"]
    
    for feature in risk_features:
        if feature in feature_importances:
            # Should be in top 50% of features by importance
            importance = feature_importances[feature]
            median_importance = np.median(list(feature_importances.values()))
            results[feature] = importance >= median_importance
            
            if not results[feature]:
                logger.warning(
                    f"Feature '{feature}' has lower than expected importance: "
                    f"{importance:.4f} < median {median_importance:.4f}"
                )
    
    return results
