"""
Evaluation Metrics Module

Recall-focused metrics for asymmetric error costs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Optional
import logging

from src.config import eval_config

logger = logging.getLogger(__name__)


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall (sensitivity) for positive class."""
    return recall_score(y_true, y_pred, pos_label=1)


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute precision for positive class."""
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def compute_fn_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute false negative rate.
    
    FN Rate = FN / (FN + TP) = 1 - Recall
    
    This is the primary error metric for asymmetric cost scenarios.
    """
    return 1.0 - compute_recall(y_true, y_pred)


def compute_fp_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute false positive rate.
    
    FP Rate = FP / (FP + TN)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = None
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match actual outcomes.
    Lower ECE = better calibrated model.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        n_bins: Number of bins for calibration curve
        
    Returns:
        ECE value (0 = perfect calibration)
    """
    n_bins = n_bins or eval_config.ece_n_bins
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = y_proba[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    return ece


def compute_cost_weighted_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = None,
    cost_fp: float = None
) -> float:
    """
    Compute total cost-weighted error.
    
    Total Cost = (FN * cost_fn) + (FP * cost_fp)
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        cost_fn: Cost of false negative (default from config)
        cost_fp: Cost of false positive (default from config)
        
    Returns:
        Total weighted cost
    """
    cost_fn = cost_fn or eval_config.cost_fn
    cost_fp = cost_fp or eval_config.cost_fp
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    
    return total_cost


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> dict:
    """
    Compute comprehensive metrics suite.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities (optional, for calibration metrics)
        
    Returns:
        Dictionary with all metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        # Primary metrics (asymmetric focus)
        "recall": compute_recall(y_true, y_pred),
        "fn_rate": compute_fn_rate(y_true, y_pred),
        "precision": compute_precision(y_true, y_pred),
        "fp_rate": compute_fp_rate(y_true, y_pred),
        
        # Secondary metrics
        "f1": f1_score(y_true, y_pred, pos_label=1),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        
        # Confusion matrix
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        
        # Cost
        "weighted_cost": compute_cost_weighted_error(y_true, y_pred),
    }
    
    # Probability-based metrics
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["ece"] = compute_expected_calibration_error(y_true, y_proba)
    
    return metrics


def compute_coverage_vs_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None
) -> pd.DataFrame:
    """
    Compute coverage vs accuracy curve for abstention analysis.
    
    Coverage = proportion of samples where model makes a decision
    Accuracy = accuracy on covered samples
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        thresholds: Confidence thresholds for abstention
        
    Returns:
        DataFrame with threshold, coverage, accuracy columns
    """
    if thresholds is None:
        thresholds = np.linspace(0.5, 1.0, 11)
    
    results = []
    
    for conf_threshold in thresholds:
        # Confidence = distance from 0.5
        confidence = np.abs(y_proba - 0.5) * 2
        
        # Samples above confidence threshold
        covered = confidence >= (conf_threshold - 0.5) * 2
        coverage = covered.mean()
        
        if coverage > 0:
            # Accuracy on covered samples
            y_pred_covered = (y_proba[covered] >= 0.5).astype(int)
            accuracy = (y_pred_covered == y_true[covered]).mean()
        else:
            accuracy = np.nan
        
        results.append({
            "threshold": conf_threshold,
            "coverage": coverage,
            "accuracy": accuracy
        })
    
    return pd.DataFrame(results)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Classification Report"
) -> str:
    """Print formatted classification report."""
    report = classification_report(
        y_true, y_pred,
        target_names=["No Default", "Default"],
        digits=4
    )
    
    logger.info(f"\n{title}\n{report}")
    return report
