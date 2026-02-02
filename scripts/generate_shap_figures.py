"""
SHAP Visualization Generator

Creates publication-quality SHAP visualizations for documentation.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load import load_data
from src.data.preprocess import DataPreprocessor
from src.features.build_features import build_features
from src.data.split import stratified_split
from src.models.xgboost_model import XGBoostModel
from src.config import FIGURES_DIR

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def generate_shap_visualizations():
    """Generate all SHAP visualizations for documentation."""
    
    print("=" * 60)
    print("SHAP Visualization Generator")
    print("=" * 60)
    
    # 1. Load and prepare data
    print("\n[1/5] Loading data...")
    df = load_data()
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(df)
    df_features = build_features(df_processed)
    
    X = df_features.drop(columns=["default"])
    y = df_features["default"]
    
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    
    # 2. Train XGBoost
    print("\n[2/5] Training XGBoost model...")
    model = XGBoostModel()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    # 3. Compute SHAP values
    print("\n[3/5] Computing SHAP values...")
    explainer = shap.TreeExplainer(model.model)
    
    # Use a sample for efficiency
    sample_size = min(500, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    
    # 4. Generate Summary Plot
    print("\n[4/5] Generating summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_sample, 
        show=False,
        max_display=15,
        plot_size=(12, 8)
    )
    plt.title("SHAP Feature Impact Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {FIGURES_DIR / 'shap_summary.png'}")
    
    # 5. Generate Waterfall Plot (example case)
    print("\n[5/5] Generating waterfall plot...")
    
    # Find an interesting case (flagged)
    probas = model.predict_proba(X_sample)[:, 1]
    flagged_idx = np.where(probas >= 0.6)[0]
    if len(flagged_idx) > 0:
        example_idx = flagged_idx[0]
    else:
        example_idx = np.argmax(probas)
    
    # Create Explanation object for waterfall
    explanation = shap.Explanation(
        values=shap_values[example_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[example_idx].values,
        feature_names=X_sample.columns.tolist()
    )
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    plt.title(f"Local Explanation: Flagged Case (p={probas[example_idx]:.2f})", 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_waterfall_example.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {FIGURES_DIR / 'shap_waterfall_example.png'}")
    
    # 6. Generate Bar Plot (global importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, 
        X_sample, 
        plot_type="bar",
        show=False,
        max_display=15
    )
    plt.title("Mean |SHAP| Feature Importance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_importance_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {FIGURES_DIR / 'shap_importance_bar.png'}")
    
    # 7. Generate Dependence Plot for top feature
    top_feature = X_sample.columns[np.argmax(np.abs(shap_values).mean(axis=0))]
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        top_feature,
        shap_values,
        X_sample,
        show=False
    )
    plt.title(f"SHAP Dependence: {top_feature}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_dependence_top.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {FIGURES_DIR / 'shap_dependence_top.png'}")
    
    print("\n" + "=" * 60)
    print("SHAP VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print(f"\nGenerated figures in: {FIGURES_DIR}/")
    print("  - shap_summary.png")
    print("  - shap_waterfall_example.png")
    print("  - shap_importance_bar.png")
    print("  - shap_dependence_top.png")


if __name__ == "__main__":
    generate_shap_visualizations()
