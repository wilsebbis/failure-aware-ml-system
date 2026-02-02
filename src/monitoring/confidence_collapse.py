"""
Confidence Collapse Detection

Detects when model confidence degrades under distribution shift.
"""

import numpy as np
from typing import Optional
import logging

from src.config import monitoring_config
from src.evaluation.metrics import compute_expected_calibration_error

logger = logging.getLogger(__name__)


def detect_confidence_collapse(
    proba_reference: np.ndarray,
    proba_current: np.ndarray,
    y_reference: np.ndarray = None,
    y_current: np.ndarray = None
) -> dict:
    """
    Detect confidence collapse between reference and current predictions.
    
    Indicators:
    - Drop in mean confidence (distance from 0.5)
    - Increase in calibration error
    - Shift in prediction distribution
    """
    # Confidence = distance from 0.5
    conf_ref = np.abs(proba_reference - 0.5) * 2
    conf_cur = np.abs(proba_current - 0.5) * 2
    
    mean_conf_ref = conf_ref.mean()
    mean_conf_cur = conf_cur.mean()
    conf_drop = mean_conf_ref - mean_conf_cur
    
    result = {
        "mean_confidence_reference": float(mean_conf_ref),
        "mean_confidence_current": float(mean_conf_cur),
        "confidence_drop": float(conf_drop),
        "collapse_detected": conf_drop > monitoring_config.confidence_drop_threshold
    }
    
    # ECE comparison if labels available
    if y_reference is not None and y_current is not None:
        ece_ref = compute_expected_calibration_error(y_reference, proba_reference)
        ece_cur = compute_expected_calibration_error(y_current, proba_current)
        result["ece_reference"] = float(ece_ref)
        result["ece_current"] = float(ece_cur)
        result["ece_spike"] = ece_cur - ece_ref > monitoring_config.ece_spike_threshold
    
    if result["collapse_detected"]:
        logger.warning(f"Confidence collapse detected! Drop: {conf_drop:.2%}")
    
    return result


def analyze_prediction_shift(proba_reference: np.ndarray, proba_current: np.ndarray) -> dict:
    """Analyze how prediction distribution has shifted."""
    return {
        "mean_proba_reference": float(proba_reference.mean()),
        "mean_proba_current": float(proba_current.mean()),
        "std_proba_reference": float(proba_reference.std()),
        "std_proba_current": float(proba_current.std()),
        "median_shift": float(np.median(proba_current) - np.median(proba_reference))
    }
