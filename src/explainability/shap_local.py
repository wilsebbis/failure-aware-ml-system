"""
SHAP Local Explainability

Instance-level explanations for audit and review.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Any
import logging

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def explain_instance(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    instance_idx: int,
    expected_value: float = None
) -> dict:
    """
    Get detailed explanation for a single instance.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        instance_idx: Index of instance to explain
        expected_value: Base prediction value
        
    Returns:
        Dict with feature contributions and prediction breakdown
    """
    instance_shap = shap_values[instance_idx]
    instance_features = X.iloc[instance_idx]
    
    # Sort by absolute contribution
    sorted_idx = np.argsort(-np.abs(instance_shap))
    
    contributions = []
    for idx in sorted_idx:
        feature_name = X.columns[idx]
        contributions.append({
            "feature": feature_name,
            "value": float(instance_features.iloc[idx]),
            "shap_value": float(instance_shap[idx]),
            "direction": "positive" if instance_shap[idx] > 0 else "negative"
        })
    
    explanation = {
        "instance_idx": instance_idx,
        "expected_value": expected_value,
        "prediction_contribution": float(np.sum(instance_shap)),
        "contributions": contributions
    }
    
    return explanation


def plot_waterfall(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    instance_idx: int,
    expected_value: float,
    max_display: int = 15,
    title: str = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create waterfall plot for instance explanation.
    
    Shows how features contribute to moving from base prediction
    to the final prediction.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        instance_idx: Index of instance
        expected_value: Base prediction value
        max_display: Max features to show
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Create a SHAP Explanation object
    instance_shap = shap_values[instance_idx]
    instance_data = X.iloc[instance_idx].values
    
    explanation = shap.Explanation(
        values=instance_shap,
        base_values=expected_value,
        data=instance_data,
        feature_names=list(X.columns)
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.waterfall_plot(explanation, max_display=max_display, show=False)
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig = plt.gcf()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Waterfall plot saved to {save_path}")
    
    return plt.gcf()


def plot_force(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    instance_idx: int,
    expected_value: float,
    title: str = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create force plot for instance explanation.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        instance_idx: Index of instance
        expected_value: Base prediction value
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    instance_shap = shap_values[instance_idx]
    
    # Force plot returns HTML by default, we need matplotlib
    shap.force_plot(
        expected_value,
        instance_shap,
        X.iloc[instance_idx],
        matplotlib=True,
        show=False
    )
    
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Force plot saved to {save_path}")
    
    return plt.gcf()


def generate_audit_explanation(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    expected_value: float,
    instance_idx: int
) -> str:
    """
    Generate human-readable explanation for audit.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        expected_value: Base prediction value
        instance_idx: Index of instance
        
    Returns:
        Markdown-formatted explanation string
    """
    explanation = explain_instance(shap_values, X, instance_idx, expected_value)
    
    true_label = "Default" if y_true[instance_idx] == 1 else "No Default"
    pred_label = "Default" if y_pred[instance_idx] == 1 else "No Default"
    probability = y_proba[instance_idx]
    
    lines = [
        f"## Instance Explanation (ID: {instance_idx})",
        "",
        f"**True Label:** {true_label}",
        f"**Predicted Label:** {pred_label}",
        f"**Predicted Probability:** {probability:.2%}",
        "",
        "### Top Contributing Features",
        "",
        "| Feature | Value | SHAP Contribution | Direction |",
        "|---------|-------|-------------------|-----------|"
    ]
    
    # Top 10 contributors
    for contrib in explanation["contributions"][:10]:
        direction = "↑ Risk" if contrib["direction"] == "positive" else "↓ Risk"
        lines.append(
            f"| {contrib['feature']} | {contrib['value']:.4f} | "
            f"{contrib['shap_value']:+.4f} | {direction} |"
        )
    
    lines.extend([
        "",
        "### Interpretation",
        ""
    ])
    
    # Generate interpretation
    top_positive = [c for c in explanation["contributions"][:5] if c["direction"] == "positive"]
    top_negative = [c for c in explanation["contributions"][:5] if c["direction"] == "negative"]
    
    if top_positive:
        risk_factors = ", ".join([c["feature"] for c in top_positive[:3]])
        lines.append(f"**Risk Factors:** {risk_factors}")
    
    if top_negative:
        protective = ", ".join([c["feature"] for c in top_negative[:3]])
        lines.append(f"**Protective Factors:** {protective}")
    
    return "\n".join(lines)


def explain_decision_batch(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    y_proba: np.ndarray,
    decision_type: str,
    sample_size: int = 5
) -> list[dict]:
    """
    Explain a batch of decisions of a specific type.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        y_proba: Predicted probabilities
        decision_type: "flag" (high prob), "pass" (low prob), or "review" (medium)
        sample_size: Number of samples to explain
        
    Returns:
        List of explanation dicts
    """
    if decision_type == "flag":
        mask = y_proba >= 0.6
    elif decision_type == "pass":
        mask = y_proba < 0.15
    else:  # review
        mask = (y_proba >= 0.15) & (y_proba < 0.6)
    
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        logger.warning(f"No samples found for decision type: {decision_type}")
        return []
    
    sample_indices = np.random.choice(
        indices,
        size=min(sample_size, len(indices)),
        replace=False
    )
    
    explanations = []
    for idx in sample_indices:
        explanations.append(explain_instance(shap_values, X, idx))
    
    return explanations
