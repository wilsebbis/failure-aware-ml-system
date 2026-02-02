"""
SHAP Global Explainability

Global feature importance and model understanding.
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


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "tree",
    max_samples: int = 1000
) -> tuple[np.ndarray, shap.Explainer]:
    """
    Compute SHAP values for a model.
    
    Args:
        model: Fitted model (must have predict_proba or similar)
        X: Features to explain
        model_type: "tree" for tree-based, "kernel" for model-agnostic
        max_samples: Maximum samples for background (kernel only)
        
    Returns:
        Tuple of (shap_values array, explainer)
    """
    logger.info(f"Computing SHAP values ({model_type}) for {len(X)} samples...")
    
    if model_type == "tree":
        # For XGBoost, Random Forest, etc.
        if hasattr(model, 'get_booster'):
            # XGBoost
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'estimators_'):
            # Random Forest
            explainer = shap.TreeExplainer(model)
        else:
            raise ValueError("Model not recognized as tree-based")
    else:
        # Kernel SHAP for any model
        background = shap.sample(X, min(max_samples, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
    
    shap_values = explainer.shap_values(X)
    
    # Handle multi-output (binary classification returns 2 arrays)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # Positive class
    
    logger.info(f"SHAP values computed: shape {shap_values.shape}")
    
    return shap_values, explainer


def get_feature_importance_from_shap(
    shap_values: np.ndarray,
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Extract feature importance from SHAP values.
    
    Uses mean absolute SHAP value as importance metric.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance, sorted descending
    """
    importance = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
        "mean_shap": shap_values.mean(axis=0),
        "std_shap": shap_values.std(axis=0)
    })
    
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df = importance_df.reset_index(drop=True)
    
    return importance_df


def plot_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 20,
    title: str = "SHAP Feature Importance",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create SHAP summary plot (beeswarm).
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        max_display: Max features to show
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False
    )
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        fig = plt.gcf()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Summary plot saved to {save_path}")
    
    return plt.gcf()


def plot_bar(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 15,
    title: str = "Mean |SHAP| Feature Importance",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create SHAP bar plot.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        max_display: Max features to show
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        plot_type="bar",
        show=False
    )
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        fig = plt.gcf()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Bar plot saved to {save_path}")
    
    return plt.gcf()


def plot_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = None,
    title: str = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create SHAP dependence plot for a feature.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        feature: Feature to plot
        interaction_feature: Feature for color coding
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    shap.dependence_plot(
        feature,
        shap_values,
        X,
        interaction_index=interaction_feature,
        show=False,
        ax=ax
    )
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Dependence plot saved to {save_path}")
    
    return fig


def generate_all_global_plots(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    output_dir: Path = None,
    top_features: int = 5
) -> dict[str, Path]:
    """
    Generate all global explainability plots.
    
    Args:
        shap_values: SHAP values array
        X: Feature DataFrame
        output_dir: Directory to save plots
        top_features: Number of features for dependence plots
        
    Returns:
        Dict mapping plot name to path
    """
    output_dir = output_dir or FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Summary plot
    plt.figure()
    plot_summary(shap_values, X, save_path=output_dir / "shap_summary.png")
    paths["summary"] = output_dir / "shap_summary.png"
    plt.close()
    
    # Bar plot
    plt.figure()
    plot_bar(shap_values, X, save_path=output_dir / "shap_bar.png")
    paths["bar"] = output_dir / "shap_bar.png"
    plt.close()
    
    # Top feature dependence plots
    importance_df = get_feature_importance_from_shap(shap_values, list(X.columns))
    top_feature_names = importance_df["feature"].head(top_features).tolist()
    
    for feature in top_feature_names:
        safe_name = feature.replace("/", "_").replace(" ", "_")
        path = output_dir / f"shap_dependence_{safe_name}.png"
        
        plt.figure()
        plot_dependence(shap_values, X, feature, save_path=path)
        paths[f"dependence_{feature}"] = path
        plt.close()
    
    logger.info(f"Generated {len(paths)} global explainability plots")
    
    return paths
