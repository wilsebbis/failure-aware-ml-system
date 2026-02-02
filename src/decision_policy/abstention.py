"""
Abstention Logic

Determines when model should defer to human review.
"""

import numpy as np
from typing import Optional
import logging

from src.config import decision_config

logger = logging.getLogger(__name__)


def compute_abstention_rate(
    probabilities: np.ndarray,
    threshold_negative: float = None,
    threshold_positive: float = None
) -> float:
    """Compute proportion of samples that would be abstained."""
    t_neg = threshold_negative or decision_config.threshold_negative
    t_pos = threshold_positive or decision_config.threshold_positive
    
    abstain = (probabilities >= t_neg) & (probabilities < t_pos)
    return abstain.mean()


def get_abstention_mask(
    probabilities: np.ndarray,
    threshold_negative: float = None,
    threshold_positive: float = None
) -> np.ndarray:
    """Get boolean mask of samples to abstain on."""
    t_neg = threshold_negative or decision_config.threshold_negative
    t_pos = threshold_positive or decision_config.threshold_positive
    return (probabilities >= t_neg) & (probabilities < t_pos)


def analyze_abstention_quality(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold_negative: float = None,
    threshold_positive: float = None
) -> dict:
    """Analyze the quality of abstention decisions."""
    abstain_mask = get_abstention_mask(probabilities, threshold_negative, threshold_positive)
    
    n_abstain = abstain_mask.sum()
    n_total = len(probabilities)
    
    if n_abstain == 0:
        return {"abstention_rate": 0.0, "message": "No abstentions"}
    
    # Of abstained samples, how many were actually positive?
    positive_rate_in_abstain = y_true[abstain_mask].mean()
    
    return {
        "abstention_rate": n_abstain / n_total,
        "n_abstained": int(n_abstain),
        "positive_rate_in_abstained": float(positive_rate_in_abstain),
        "mean_prob_in_abstained": float(probabilities[abstain_mask].mean())
    }
