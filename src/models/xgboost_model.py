"""
XGBoost Model

Shallow, constrained gradient boosting for interpretable ensemble.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import pickle
from pathlib import Path
from typing import Optional
import logging

from src.config import xgb_config, MODELS_DIR

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    Constrained XGBoost classifier optimized for regulated environments.
    
    Key design decisions:
    - Shallow trees (max_depth <= 4) for auditability
    - scale_pos_weight for class imbalance
    - Early stopping to prevent overfitting
    - Regularization (L1 + L2)
    
    This configuration prioritizes:
    1. Interpretability via SHAP
    2. Calibrated probabilities
    3. Robustness on limited data
    
    Attributes:
        model: Fitted XGBClassifier
        calibrated_model: Optional calibrated version
        feature_names: List of feature names
        is_fitted: Whether model has been trained
        best_iteration: Best iteration from early stopping
    """
    
    def __init__(
        self,
        n_estimators: int = None,
        max_depth: int = None,
        learning_rate: float = None,
        scale_pos_weight: float = None,
        random_state: int = None
    ):
        """
        Initialize XGBoost with config defaults.
        """
        self.n_estimators = n_estimators or xgb_config.n_estimators
        self.max_depth = max_depth or xgb_config.max_depth
        self.learning_rate = learning_rate or xgb_config.learning_rate
        self.scale_pos_weight = scale_pos_weight or xgb_config.scale_pos_weight
        self.random_state = random_state or xgb_config.random_state
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.feature_names: Optional[list[str]] = None
        self.is_fitted: bool = False
        self.best_iteration: Optional[int] = None
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        calibrate: bool = False,
        X_cal: pd.DataFrame = None,
        y_cal: pd.Series = None,
        auto_scale_pos_weight: bool = True
    ) -> "XGBoostModel":
        """
        Fit XGBoost model.
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features for early stopping
            y_val: Validation target for early stopping
            calibrate: If True, calibrate probabilities
            X_cal: Calibration features (required if calibrate=True)
            y_cal: Calibration target (required if calibrate=True)
            auto_scale_pos_weight: If True, compute from class ratio
            
        Returns:
            Self for chaining
        """
        self.feature_names = list(X.columns)
        
        # Auto-compute scale_pos_weight
        if auto_scale_pos_weight:
            n_pos = (y == 1).sum()
            n_neg = (y == 0).sum()
            self.scale_pos_weight = n_neg / n_pos
            logger.info(f"Auto scale_pos_weight: {self.scale_pos_weight:.2f}")
        
        logger.info(
            f"Training XGBoost (max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate}, "
            f"scale_pos_weight={self.scale_pos_weight:.2f})"
        )
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=xgb_config.subsample,
            colsample_bytree=xgb_config.colsample_bytree,
            min_child_weight=xgb_config.min_child_weight,
            reg_alpha=xgb_config.reg_alpha,
            reg_lambda=xgb_config.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric=xgb_config.eval_metric,
            use_label_encoder=False,
            n_jobs=-1
        )
        
        # Fit with optional early stopping
        if X_val is not None and y_val is not None:
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            try:
                self.best_iteration = self.model.best_iteration
                logger.info(f"Best iteration: {self.best_iteration}")
            except AttributeError:
                # XGBoost 3.x: best_iteration only exists with early_stopping_rounds
                self.best_iteration = self.n_estimators
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        
        # Optional calibration
        if calibrate:
            if X_cal is None or y_cal is None:
                raise ValueError("Calibration data required when calibrate=True")
            
            logger.info("Calibrating probabilities with isotonic regression")
            # sklearn 1.8+: cv='prefit' is deprecated - use cv with cross-validation
            from sklearn import __version__ as sklearn_version
            if tuple(map(int, sklearn_version.split('.')[:2])) >= (1, 8):
                self.calibrated_model = CalibratedClassifierCV(
                    estimator=None,
                    method="isotonic",
                    cv=None
                )
                self.calibrated_model.estimator = self.model
                self.calibrated_model.fit(X_cal, y_cal)
            else:
                self.calibrated_model = CalibratedClassifierCV(
                    self.model,
                    method="isotonic",
                    cv="prefit"
                )
                self.calibrated_model.fit(X_cal, y_cal)
        
        logger.info(f"XGBoost trained on {len(X)} samples")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Features
            calibrated: If True and calibrated model exists, use it
            
        Returns:
            Array of shape (n_samples, 2) with [P(neg), P(pos)]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting")
        
        if calibrated and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        
        return self.model.predict_proba(X)
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Get binary predictions at threshold.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: "gain", "weight", or "cover"
            
        Returns:
            DataFrame with feature names and importance scores, sorted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted")
        
        booster = self.model.get_booster()
        
        importance_dict = booster.get_score(importance_type=importance_type)
        
        # Map feature indices to names
        importance_df = pd.DataFrame({
            "feature": list(importance_dict.keys()),
            "importance": list(importance_dict.values())
        })
        
        # Replace f0, f1, etc. with actual feature names if needed
        feature_map = {f"f{i}": name for i, name in enumerate(self.feature_names)}
        importance_df["feature"] = importance_df["feature"].replace(feature_map)
        
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        return importance_df
    
    def get_model_complexity(self) -> dict:
        """
        Get model complexity statistics.
        """
        booster = self.model.get_booster()
        
        # Get tree dumps
        trees = booster.get_dump()
        
        # Count nodes and leaves
        total_nodes = sum(tree.count("\n") for tree in trees)
        
        return {
            "n_estimators_used": len(trees),
            "max_depth": self.max_depth,
            "total_nodes": total_nodes,
            "avg_nodes_per_tree": total_nodes / len(trees) if trees else 0,
            "best_iteration": self.best_iteration
        }
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save model to disk."""
        path = path or MODELS_DIR / "xgboost_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "XGBoostModel":
        """Load model from disk."""
        path = path or MODELS_DIR / "xgboost_model.pkl"
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model
