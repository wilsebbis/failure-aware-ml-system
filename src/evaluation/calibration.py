"""
Probability Calibration Module

Ensures predicted probabilities match actual outcome frequencies.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Literal
import logging

from src.config import eval_config, FIGURES_DIR

logger = logging.getLogger(__name__)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = None,
    strategy: Literal["uniform", "quantile"] = "uniform"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy
        
    Returns:
        Tuple of (mean predicted probability, fraction of positives)
    """
    n_bins = n_bins or eval_config.n_calibration_bins
    
    fraction_positives, mean_predicted = calibration_curve(
        y_true, y_proba,
        n_bins=n_bins,
        strategy=strategy
    )
    
    return mean_predicted, fraction_positives


class ProbabilityCalibrator:
    """
    Standalone probability calibrator for post-hoc calibration.
    
    Useful when:
    - Calibrating an existing model without retraining
    - Comparing different calibration methods
    - Recalibrating after distribution shift
    """
    
    def __init__(self, method: Literal["isotonic", "sigmoid"] = None):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method
                - "isotonic": Non-parametric, more flexible
                - "sigmoid": Platt scaling, parametric
        """
        self.method = method or eval_config.calibration_method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_proba: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fit calibrator on validation data.
        
        Args:
            y_true: True binary labels
            y_proba: Uncalibrated probabilities
            
        Returns:
            Self for chaining
        """
        logger.info(f"Fitting {self.method} calibrator")
        
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        else:
            # Platt scaling via logistic regression
            self.calibrator = LogisticRegression(C=1.0)
        
        if self.method == "isotonic":
            self.calibrator.fit(y_proba, y_true)
        else:
            # Reshape for logistic regression
            self.calibrator.fit(y_proba.reshape(-1, 1), y_true)
        
        self.is_fitted = True
        logger.info("Calibrator fitted")
        
        return self
    
    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.
        
        Args:
            y_proba: Uncalibrated probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before transforming")
        
        if self.method == "isotonic":
            return self.calibrator.predict(y_proba)
        else:
            return self.calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
    
    def fit_transform(self, y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_true, y_proba)
        return self.transform(y_proba)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba_dict: dict[str, np.ndarray],
    title: str = "Calibration Curve",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot calibration curves for multiple models.
    
    Args:
        y_true: True binary labels
        y_proba_dict: Dict mapping model name to probabilities
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(y_proba_dict)))
    
    for (name, y_proba), color in zip(y_proba_dict.items(), colors):
        mean_pred, frac_pos = compute_calibration_curve(y_true, y_proba)
        ax.plot(mean_pred, frac_pos, "o-", color=color, label=name)
    
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Calibration plot saved to {save_path}")
    
    return fig


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = None,
    title: str = "Reliability Diagram",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot reliability diagram with histogram of predictions.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_bins = n_bins or eval_config.n_calibration_bins
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1])
    
    # Calibration curve
    mean_pred, frac_pos = compute_calibration_curve(y_true, y_proba, n_bins)
    
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax1.plot(mean_pred, frac_pos, "o-", color="steelblue", label="Model")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of predictions
    ax2.hist(y_proba, bins=n_bins, range=(0, 1), color="steelblue", alpha=0.7)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Reliability diagram saved to {save_path}")
    
    return fig
