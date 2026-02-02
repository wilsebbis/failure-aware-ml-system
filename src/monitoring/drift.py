"""
Distribution Drift Detection

Monitors feature and prediction drift using PSI and statistical tests.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

from src.config import monitoring_config

logger = logging.getLogger(__name__)


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI).
    
    PSI measures shift between two distributions:
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.25: Moderate shift - investigate
    - PSI >= 0.25: Significant shift - action required
    """
    eps = 1e-10
    
    # Create bins from expected distribution
    _, bin_edges = np.histogram(expected, bins=n_bins)
    
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi)


def compute_feature_drift(
    X_reference: pd.DataFrame,
    X_current: pd.DataFrame,
    features: list[str] = None
) -> pd.DataFrame:
    """Compute PSI for each feature."""
    features = features or list(X_reference.columns)
    
    results = []
    for feature in features:
        if feature in X_reference.columns and feature in X_current.columns:
            psi = compute_psi(X_reference[feature].values, X_current[feature].values)
            status = "ok" if psi < monitoring_config.psi_warning else "warning" if psi < monitoring_config.psi_critical else "critical"
            results.append({"feature": feature, "psi": psi, "status": status})
    
    return pd.DataFrame(results).sort_values("psi", ascending=False)


def detect_drift(X_reference: pd.DataFrame, X_current: pd.DataFrame) -> dict:
    """Full drift detection report."""
    drift_df = compute_feature_drift(X_reference, X_current)
    
    n_warning = (drift_df["status"] == "warning").sum()
    n_critical = (drift_df["status"] == "critical").sum()
    
    return {
        "overall_status": "critical" if n_critical > 0 else "warning" if n_warning > 0 else "ok",
        "features_warning": n_warning,
        "features_critical": n_critical,
        "top_drifted": drift_df.head(5).to_dict("records"),
        "max_psi": drift_df["psi"].max() if len(drift_df) > 0 else 0
    }
