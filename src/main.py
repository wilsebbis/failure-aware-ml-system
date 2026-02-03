"""
Main Pipeline

End-to-end execution of the failure-aware ML system.
Supports multiple datasets via the Adapter Pattern.

Usage:
    # Default (UCI Credit)
    uv run python -m src.main
    
    # Specific dataset
    uv run python -m src.main --dataset home_credit
    uv run python -m src.main --dataset ieee_cis
    uv run python -m src.main --dataset lending_club
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.logging import setup_logging
from src.data.factory import get_adapter, list_adapters
from src.data.preprocess import preprocess_data
from src.data.split import stratified_split, create_shifted_test_set
from src.features.build_features import build_features
from src.models.baseline_logistic import LogisticBaseline
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.thresholds import find_abstention_thresholds
from src.decision_policy.triage import TriagePolicy
from src.monitoring.confidence_collapse import detect_confidence_collapse
from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)

# Config directory
CONFIG_DIR = Path(__file__).parent / "config"


def load_config(dataset_name: str) -> dict:
    """Load YAML configuration for a dataset."""
    config_path = CONFIG_DIR / f"{dataset_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}. "
            f"Available datasets: {list_adapters()}"
        )
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(dataset_name: str = "uci_credit"):
    """
    Execute the full ML pipeline with the specified dataset.
    
    Args:
        dataset_name: One of "uci_credit", "home_credit", "ieee_cis", "lending_club"
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("FAILURE-AWARE ML SYSTEM PIPELINE")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("=" * 60)
    
    # 1. Load configuration and data via adapter
    logger.info("\n[1/7] Loading data via adapter...")
    
    try:
        config = load_config(dataset_name)
    except FileNotFoundError:
        # Fallback to default config for backward compatibility
        logger.warning(f"Config not found for {dataset_name}, using defaults")
        config = {
            "dataset": {"name": dataset_name, "path": f"data/raw/{dataset_name}"},
            "triage": {"auto_approve_threshold": 0.05, "auto_decline_threshold": 0.50}
        }
    
    # Initialize adapter
    adapter = get_adapter(
        name=config.get("dataset", {}).get("name", dataset_name),
        config=config.get("dataset", {})
    )
    
    # Load and engineer features using adapter
    try:
        adapter.load_raw()
        adapter.feature_engineer()
        X, y = adapter.get_features_and_target()
        
        logger.info(f"  Loaded {len(X):,} samples with {len(X.columns)} features")
        logger.info(f"  Target rate: {y.mean():.2%}")
        logger.info(f"  Split strategy: {adapter.get_split_strategy()}")
        
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.info(f"\nFalling back to UCI Credit dataset...")
        
        # Fallback to original load for backward compatibility
        from src.data.load import load_data
        df = load_data()
        X, y, _ = preprocess_data(df)
    
    # Build additional features
    X_features = build_features(X)
    
    # 2. Split data (respecting adapter's strategy)
    logger.info("\n[2/7] Splitting data...")
    
    split_strategy = adapter.get_split_strategy() if hasattr(adapter, 'get_split_strategy') else "stratified"
    
    if split_strategy == "temporal" and hasattr(adapter, 'get_temporal_split_indices'):
        # Time-based split for temporal datasets
        train_idx, test_idx = adapter.get_temporal_split_indices()
        val_size = int(len(train_idx) * 0.1)
        
        X_train = X_features.loc[train_idx[:-val_size]]
        X_val = X_features.loc[train_idx[-val_size:]]
        X_test = X_features.loc[test_idx]
        y_train = y.loc[train_idx[:-val_size]]
        y_val = y.loc[train_idx[-val_size:]]
        y_test = y.loc[test_idx]
        
        logger.info(f"  Using temporal split (train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)})")
    else:
        # Standard stratified split
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X_features, y)
        logger.info(f"  Using stratified split (train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)})")
    
    # 3. Train models
    logger.info("\n[3/7] Training models...")
    
    # Logistic baseline
    logistic = LogisticBaseline()
    logistic.fit(X_train, y_train, calibrate=True, X_cal=X_val, y_cal=y_val)
    
    # Random Forest
    rf = RandomForestModel()
    rf.fit(X_train, y_train, calibrate=True, X_cal=X_val, y_cal=y_val)
    
    # XGBoost
    xgb = XGBoostModel()
    xgb.fit(X_train, y_train, X_val=X_val, y_val=y_val, calibrate=True, X_cal=X_val, y_cal=y_val)
    
    # 4. Evaluate models
    logger.info("\n[4/7] Evaluating models...")
    models = {"Logistic": logistic, "RandomForest": rf, "XGBoost": xgb}
    
    best_model_name = None
    best_recall = 0
    
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test, threshold=0.5)
        
        metrics = compute_all_metrics(y_test.values, y_pred, y_proba)
        logger.info(f"\n{name}:")
        logger.info(f"  Recall: {metrics['recall']:.2%}")
        logger.info(f"  FN Rate: {metrics['fn_rate']:.2%}")
        logger.info(f"  ECE: {metrics.get('ece', 'N/A'):.4f}")
        
        if metrics['recall'] > best_recall:
            best_recall = metrics['recall']
            best_model_name = name
    
    logger.info(f"\nBest model by recall: {best_model_name}")
    best_model = models[best_model_name]
    
    # 5. Optimize thresholds from config
    logger.info("\n[5/7] Optimizing thresholds...")
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    
    # Use config thresholds or find optimal
    triage_config = config.get("triage", {})
    t_neg = triage_config.get("auto_approve_threshold", 0.05)
    t_pos = triage_config.get("auto_decline_threshold", 0.50)
    
    logger.info(f"  Pass threshold: p < {t_neg}")
    logger.info(f"  Flag threshold: p >= {t_pos}")
    
    # 6. Apply triage policy
    logger.info("\n[6/7] Applying triage policy...")
    triage = TriagePolicy(threshold_negative=t_neg, threshold_positive=t_pos)
    stats = triage.get_stats(y_proba_test, y_test.values)
    
    logger.info(f"  Pass rate: {stats['pass_rate']:.1%}")
    logger.info(f"  Flag rate: {stats['flag_rate']:.1%}")
    logger.info(f"  Review rate: {stats['review_rate']:.1%}")
    
    # Calculate Pass Queue Defect Rate (the key metric)
    pass_mask = y_proba_test < t_neg
    if pass_mask.sum() > 0:
        pass_defect_rate = y_test.values[pass_mask].mean()
        system_recall = 1 - pass_defect_rate * (pass_mask.sum() / len(y_test))
        logger.info(f"  Pass Queue Defect Rate: {pass_defect_rate:.2%}")
        logger.info(f"  System Recall: {system_recall:.1%}")
    
    # 7. Test distribution shift
    logger.info("\n[7/7] Testing distribution shift...")
    X_shifted, y_shifted = create_shifted_test_set(X_test, y_test)
    y_proba_shifted = best_model.predict_proba(X_shifted)[:, 1]
    
    collapse = detect_confidence_collapse(y_proba_test, y_proba_shifted, y_test.values, y_shifted.values)
    
    if collapse["collapse_detected"]:
        logger.warning(f"⚠ Confidence collapse detected! Drop: {collapse['confidence_drop']:.2%}")
    else:
        logger.info(f"✓ No confidence collapse (drop: {collapse['confidence_drop']:.2%})")
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return {
        "dataset": dataset_name,
        "best_model": best_model_name,
        "triage_stats": stats,
        "collapse_analysis": collapse
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Failure-Aware ML System Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  uci_credit    - UCI Credit Card Default (simple, single CSV)
  home_credit   - Home Credit Default Risk (multi-table joins)
  ieee_cis      - IEEE-CIS Fraud Detection (temporal splits)
  lending_club  - Lending Club (IRR optimization)
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        default="uci_credit",
        choices=list_adapters(),
        help="Dataset to use (default: uci_credit)"
    )
    
    args = parser.parse_args()
    run_pipeline(dataset_name=args.dataset)


if __name__ == "__main__":
    main()
