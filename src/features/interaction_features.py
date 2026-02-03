"""
Ratio-First Feature Engineering.

In credit risk, raw numbers (e.g., Income: $50k) are useless.
The signal is in the RATIOS (e.g., Debt-to-Income).

This module provides explicit ratio/interaction features that give
interpretable models the non-linear insight they need.

Key Features:
- Payment Stress: How much of income goes to debt service
- Utilization Velocity: How fast is debt growing
- Employment Stability: Job tenure relative to age
- Credit Burden: Total credit relative to capacity

Example:
    from src.features.interaction_features import add_ratio_features
    
    X_enhanced = add_ratio_features(X, dataset="home_credit")
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# Dataset-specific ratio features
HOME_CREDIT_RATIOS = [
    # Payment Stress (most important for default)
    {
        "name": "PAYMENT_STRESS",
        "numerator": ["AMT_CREDIT", "AMT_ANNUITY"],
        "denominator": "AMT_INCOME_TOTAL",
        "operation": "sum_divide",
        "description": "(Credit + Annuity) / Income - Total debt burden"
    },
    {
        "name": "CREDIT_INCOME_RATIO",
        "numerator": "AMT_CREDIT",
        "denominator": "AMT_INCOME_TOTAL",
        "operation": "divide",
        "description": "Credit / Income - Leverage ratio"
    },
    {
        "name": "ANNUITY_INCOME_RATIO",
        "numerator": "AMT_ANNUITY",
        "denominator": "AMT_INCOME_TOTAL",
        "operation": "divide",
        "description": "Annuity / Income - Monthly payment burden"
    },
    # Credit Utilization
    {
        "name": "CREDIT_GOODS_RATIO",
        "numerator": "AMT_CREDIT",
        "denominator": "AMT_GOODS_PRICE",
        "operation": "divide",
        "description": "Credit / Goods Price - Financing ratio"
    },
    # Employment Stability
    {
        "name": "EMPLOYMENT_RATIO",
        "numerator": "DAYS_EMPLOYED",
        "denominator": "DAYS_BIRTH",
        "operation": "divide",
        "description": "Employment tenure / Age - Job stability"
    },
    # External Credit Requests
    {
        "name": "CREDIT_REQUESTS_INTENSITY",
        "numerator": "AMT_REQ_CREDIT_BUREAU_YEAR",
        "denominator": "AMT_INCOME_TOTAL",
        "operation": "divide",
        "description": "Credit requests / Income - Credit seeking intensity"
    },
]

UCI_CREDIT_RATIOS = [
    # Utilization Velocity (is debt growing fast?)
    {
        "name": "UTILIZATION_VELOCITY",
        "numerator": ["BILL_AMT3", "-BILL_AMT1"],
        "denominator": "LIMIT_BAL",
        "operation": "diff_divide",
        "description": "(Bill3 - Bill1) / Limit - Debt growth rate"
    },
    # Payment Behavior
    {
        "name": "PAYMENT_RATIO_1",
        "numerator": "PAY_AMT1",
        "denominator": "BILL_AMT1",
        "operation": "divide",
        "description": "Payment / Bill - Payment discipline"
    },
    {
        "name": "PAYMENT_RATIO_2",
        "numerator": "PAY_AMT2",
        "denominator": "BILL_AMT2",
        "operation": "divide",
        "description": "Payment / Bill - Payment discipline"
    },
    # Utilization
    {
        "name": "UTILIZATION_RATE",
        "numerator": "BILL_AMT1",
        "denominator": "LIMIT_BAL",
        "operation": "divide",
        "description": "Bill / Limit - Credit utilization"
    },
    # Age-Income Interaction
    {
        "name": "INCOME_AGE_RATIO",
        "numerator": "LIMIT_BAL",
        "denominator": "AGE",
        "operation": "divide",
        "description": "Limit / Age - Credit capacity relative to life stage"
    },
]

IEEE_CIS_RATIOS = [
    # Transaction Patterns
    {
        "name": "AMT_DEVIATION_RATIO",
        "numerator": "TransactionAmt",
        "denominator": "card1_TransactionAmt_mean",
        "operation": "divide",
        "description": "Transaction / Card Mean - Unusual amount detection"
    },
]

LENDING_CLUB_RATIOS = [
    # Debt Stress
    {
        "name": "DTI_UTILIZATION",
        "numerator": "dti",
        "denominator": "revol_util",
        "operation": "multiply",
        "description": "DTI * Utilization - Combined debt stress"
    },
    {
        "name": "LOAN_INCOME_RATIO",
        "numerator": "loan_amnt",
        "denominator": "annual_inc",
        "operation": "divide",
        "description": "Loan / Income - Loan burden"
    },
    {
        "name": "INSTALLMENT_INCOME_RATIO",
        "numerator": "installment",
        "denominator": "annual_inc",
        "operation": "divide_monthly",  # Divide income by 12 first
        "description": "Installment / Monthly Income - Payment burden"
    },
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe division avoiding divide by zero."""
    return numerator / denominator.replace(0, np.nan)


def _compute_ratio(df: pd.DataFrame, spec: dict) -> Optional[pd.Series]:
    """Compute a single ratio feature from specification."""
    name = spec["name"]
    operation = spec["operation"]
    
    try:
        if operation == "divide":
            num_col = spec["numerator"]
            den_col = spec["denominator"]
            if num_col not in df.columns or den_col not in df.columns:
                return None
            return _safe_divide(df[num_col], df[den_col])
            
        elif operation == "sum_divide":
            num_cols = spec["numerator"]
            den_col = spec["denominator"]
            if den_col not in df.columns:
                return None
            # Sum available numerator columns
            available = [c for c in num_cols if c in df.columns]
            if not available:
                return None
            numerator = df[available].sum(axis=1)
            return _safe_divide(numerator, df[den_col])
            
        elif operation == "diff_divide":
            # For velocity: (col2 - col1) / denominator
            num_cols = spec["numerator"]
            den_col = spec["denominator"]
            if den_col not in df.columns:
                return None
            # Parse column names (handle "-" prefix for subtraction)
            col1 = num_cols[1].replace("-", "") if num_cols[1].startswith("-") else num_cols[0]
            col2 = num_cols[0] if num_cols[1].startswith("-") else num_cols[1]
            if col1 not in df.columns or col2 not in df.columns:
                return None
            diff = df[col2] - df[col1]
            return _safe_divide(diff, df[den_col])
            
        elif operation == "multiply":
            num_col = spec["numerator"]
            den_col = spec["denominator"]
            if num_col not in df.columns or den_col not in df.columns:
                return None
            return df[num_col] * df[den_col]
            
        elif operation == "divide_monthly":
            num_col = spec["numerator"]
            den_col = spec["denominator"]
            if num_col not in df.columns or den_col not in df.columns:
                return None
            return _safe_divide(df[num_col], df[den_col] / 12)
            
    except Exception as e:
        logger.warning(f"Failed to compute ratio {name}: {e}")
        return None
    
    return None


def add_ratio_features(
    df: pd.DataFrame,
    dataset: str = "auto",
    custom_ratios: Optional[List[dict]] = None,
) -> pd.DataFrame:
    """
    Add ratio/interaction features to DataFrame.
    
    These are the "non-linear insights" that make interpretable models
    (like Logistic Regression) competitive with black-box models.
    
    Args:
        df: Input DataFrame
        dataset: Dataset name ("home_credit", "uci_credit", "ieee_cis", "lending_club", "auto")
        custom_ratios: Optional custom ratio specifications
        
    Returns:
        DataFrame with ratio features added
    """
    # Select ratio specs based on dataset
    if dataset == "home_credit":
        ratio_specs = HOME_CREDIT_RATIOS
    elif dataset == "uci_credit":
        ratio_specs = UCI_CREDIT_RATIOS
    elif dataset == "ieee_cis":
        ratio_specs = IEEE_CIS_RATIOS
    elif dataset == "lending_club":
        ratio_specs = LENDING_CLUB_RATIOS
    elif dataset == "auto":
        # Try all - use whichever columns are available
        ratio_specs = HOME_CREDIT_RATIOS + UCI_CREDIT_RATIOS + IEEE_CIS_RATIOS + LENDING_CLUB_RATIOS
    else:
        ratio_specs = []
    
    # Add custom ratios
    if custom_ratios:
        ratio_specs = ratio_specs + custom_ratios
    
    # Compute each ratio
    added_features = []
    for spec in ratio_specs:
        result = _compute_ratio(df, spec)
        if result is not None:
            feature_name = f"RATIO_{spec['name']}"
            df[feature_name] = result
            added_features.append(feature_name)
    
    # Fill NaN with 0 (safe default for ratios)
    for col in added_features:
        df[col] = df[col].fillna(0)
        # Clip extreme values (> 100x is likely data error)
        df[col] = df[col].clip(-100, 100)
    
    if added_features:
        logger.info(f"Added {len(added_features)} ratio features: {added_features}")
    
    return df


def get_top_interactions(
    df: pd.DataFrame,
    target: pd.Series,
    n_features: int = 20,
) -> List[str]:
    """
    Automatically discover top interaction features using correlation.
    
    This is a fallback when domain-specific ratios aren't available.
    
    Args:
        df: Feature DataFrame
        target: Target series
        n_features: Number of top features to return
        
    Returns:
        List of feature names with highest target correlation
    """
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return []
    
    # Compute correlations with target
    correlations = {}
    for col in numeric_cols:
        try:
            corr = df[col].corr(target)
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        except:
            pass
    
    # Sort by absolute correlation
    sorted_cols = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    return [col for col, _ in sorted_cols[:n_features]]
