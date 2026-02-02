"""
Main Pipeline

End-to-end execution of the failure-aware ML system.
"""

import logging
import numpy as np
import pandas as pd

from src.utils.logging import setup_logging
from src.data.load import load_data, get_class_weight
from src.data.preprocess import preprocess_data
from src.data.split import stratified_split, create_shifted_test_set
from src.features.build_features import build_features
from src.models.baseline_logistic import LogisticBaseline
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.evaluation.metrics import compute_all_metrics, print_classification_report
from src.evaluation.thresholds import find_optimal_threshold_for_recall, find_abstention_thresholds
from src.decision_policy.triage import TriagePolicy
from src.monitoring.drift import detect_drift
from src.monitoring.confidence_collapse import detect_confidence_collapse
from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def run_pipeline():
    """Execute the full ML pipeline."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("FAILURE-AWARE ML SYSTEM PIPELINE")
    logger.info("=" * 60)
    
    # 1. Load and preprocess data
    logger.info("\n[1/7] Loading data...")
    df = load_data()
    X_processed, y, preprocessor = preprocess_data(df)
    X_features = build_features(X_processed)
    
    # 2. Split data
    logger.info("\n[2/7] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X_features, y)
    
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
    
    # 5. Optimize thresholds
    logger.info("\n[5/7] Optimizing thresholds...")
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    
    # Find abstention thresholds
    t_neg, t_pos, thresh_metrics = find_abstention_thresholds(y_test.values, y_proba_test)
    
    # 6. Apply triage policy
    logger.info("\n[6/7] Applying triage policy...")
    triage = TriagePolicy(threshold_negative=t_neg, threshold_positive=t_pos)
    stats = triage.get_stats(y_proba_test, y_test.values)
    
    logger.info(f"  Pass rate: {stats['pass_rate']:.1%}")
    logger.info(f"  Flag rate: {stats['flag_rate']:.1%}")
    logger.info(f"  Review rate: {stats['review_rate']:.1%}")
    
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
        "best_model": best_model_name,
        "triage_stats": stats,
        "collapse_analysis": collapse
    }


if __name__ == "__main__":
    run_pipeline()
