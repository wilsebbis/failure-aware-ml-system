"""
Threshold Optimization Module

Finds optimal decision thresholds for asymmetric error costs.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from typing import Optional
import logging

from src.config import eval_config, decision_config
from src.evaluation.metrics import compute_recall, compute_fn_rate, compute_cost_weighted_error

logger = logging.getLogger(__name__)


def find_optimal_threshold_for_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = None
) -> tuple[float, dict]:
    """
    Find threshold that achieves target recall.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        target_recall: Minimum recall to achieve
        
    Returns:
        Tuple of (optimal threshold, metrics at threshold)
    """
    target_recall = target_recall or decision_config.min_recall
    
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find threshold that achieves target recall
    valid_idx = np.where(recall >= target_recall)[0]
    
    if len(valid_idx) == 0:
        logger.warning(f"Cannot achieve target recall {target_recall:.2%}")
        # Return lowest threshold
        optimal_threshold = float(thresholds[0])
    else:
        # Get highest threshold (lowest FP) that still achieves recall
        best_idx = valid_idx[0]  # Highest recall above target
        for idx in valid_idx:
            if idx < len(thresholds) and precision[idx] > precision[best_idx]:
                best_idx = idx
        
        optimal_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    
    # Compute metrics at this threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    metrics = {
        "threshold": optimal_threshold,
        "recall": compute_recall(y_true, y_pred),
        "precision": float(precision[min(best_idx, len(precision) - 1)]),
        "fn_rate": compute_fn_rate(y_true, y_pred),
    }
    
    logger.info(
        f"Optimal threshold for {target_recall:.0%} recall: "
        f"{optimal_threshold:.4f} (actual recall: {metrics['recall']:.2%})"
    )
    
    return optimal_threshold, metrics


def find_cost_minimizing_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = None,
    cost_fp: float = None,
    threshold_range: tuple[float, float] = (0.1, 0.9),
    n_thresholds: int = 100
) -> tuple[float, dict]:
    """
    Find threshold that minimizes weighted error cost.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        cost_fn: Cost of false negative
        cost_fp: Cost of false positive
        threshold_range: Range of thresholds to search
        n_thresholds: Number of thresholds to evaluate
        
    Returns:
        Tuple of (optimal threshold, metrics at threshold)
    """
    cost_fn = cost_fn or eval_config.cost_fn
    cost_fp = cost_fp or eval_config.cost_fp
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    
    best_threshold = 0.5
    best_cost = float("inf")
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cost = compute_cost_weighted_error(y_true, y_pred, cost_fn, cost_fp)
        
        results.append({
            "threshold": threshold,
            "cost": cost,
            "recall": compute_recall(y_true, y_pred),
            "fn_rate": compute_fn_rate(y_true, y_pred)
        })
        
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    
    # Get metrics at best threshold
    y_pred = (y_proba >= best_threshold).astype(int)
    
    metrics = {
        "threshold": best_threshold,
        "cost": best_cost,
        "recall": compute_recall(y_true, y_pred),
        "fn_rate": compute_fn_rate(y_true, y_pred)
    }
    
    logger.info(
        f"Cost-minimizing threshold: {best_threshold:.4f} "
        f"(cost: {best_cost:.2f}, recall: {metrics['recall']:.2%})"
    )
    
    return best_threshold, metrics


def compute_threshold_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = None,
    cost_fp: float = None,
    n_thresholds: int = 50
) -> pd.DataFrame:
    """
    Compute metrics across threshold range.
    
    Useful for visualizing threshold trade-offs.
    
    Returns:
        DataFrame with threshold, recall, precision, cost, fn_rate
    """
    cost_fn = cost_fn or eval_config.cost_fn
    cost_fp = cost_fp or eval_config.cost_fp
    
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        results.append({
            "threshold": threshold,
            "recall": recall,
            "precision": precision,
            "fn_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "fp_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "cost": fn * cost_fn + fp * cost_fp,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        })
    
    return pd.DataFrame(results)


def find_abstention_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    max_review_rate: float = None,
    min_recall: float = None
) -> tuple[float, float, dict]:
    """
    Find thresholds for three-way decision (pass/flag/review).
    
    Optimizes for:
    1. Minimum recall on flagged positives
    2. Maximum review rate constraint
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        max_review_rate: Maximum proportion for human review
        min_recall: Minimum recall to achieve on auto-decisions
        
    Returns:
        Tuple of (threshold_negative, threshold_positive, metrics)
    """
    max_review_rate = max_review_rate or decision_config.max_review_rate
    min_recall = min_recall or decision_config.min_recall
    
    # Grid search over threshold pairs
    low_thresholds = np.linspace(0.05, 0.40, 20)
    high_thresholds = np.linspace(0.50, 0.95, 20)
    
    best_config = None
    best_score = -float("inf")
    
    for t_low in low_thresholds:
        for t_high in high_thresholds:
            if t_low >= t_high:
                continue
            
            # Classify samples
            auto_pass = y_proba < t_low
            auto_flag = y_proba >= t_high
            review = (~auto_pass) & (~auto_flag)
            
            review_rate = review.mean()
            
            # Skip if review rate too high
            if review_rate > max_review_rate:
                continue
            
            # Calculate recall on auto-flagged samples
            auto_flagged_positives = (y_true == 1) & auto_flag
            all_positives_in_auto = (y_true == 1) & (auto_pass | auto_flag)
            
            if all_positives_in_auto.sum() == 0:
                continue
            
            # Recall among auto-processed samples
            auto_recall = auto_flagged_positives.sum() / (y_true == 1).sum() if (y_true == 1).sum() > 0 else 0
            
            # Score: maximize recall, minimize review rate
            score = auto_recall - 0.5 * review_rate
            
            if score > best_score and auto_recall >= min_recall * 0.5:  # Relaxed for human review
                best_score = score
                best_config = {
                    "threshold_negative": t_low,
                    "threshold_positive": t_high,
                    "review_rate": review_rate,
                    "auto_recall": auto_recall,
                    "auto_pass_rate": auto_pass.mean(),
                    "auto_flag_rate": auto_flag.mean()
                }
    
    if best_config is None:
        logger.warning("Could not find valid threshold configuration")
        return 0.15, 0.60, {"error": "No valid configuration found"}
    
    logger.info(
        f"Abstention thresholds: [{best_config['threshold_negative']:.2f}, "
        f"{best_config['threshold_positive']:.2f}] "
        f"(review: {best_config['review_rate']:.1%})"
    )
    
    return best_config["threshold_negative"], best_config["threshold_positive"], best_config
