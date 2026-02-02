"""
Random Forest Classifier

Interpretable ensemble model with built-in feature importance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pickle
from pathlib import Path
from typing import Optional
import logging

from src.config import rf_config, MODELS_DIR

logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    Random Forest classifier with constrained depth for interpretability.
    
    Key properties:
    - Feature importance via impurity decrease
    - Class weighting for imbalance
    - Constrained depth to prevent overfitting
    - Optional probability calibration
    
    Attributes:
        model: Fitted RandomForestClassifier
        calibrated_model: Optional calibrated version
        feature_names: List of feature names
        is_fitted: Whether model has been trained
    """
    
    def __init__(
        self,
        n_estimators: int = None,
        max_depth: int = None,
        min_samples_split: int = None,
        min_samples_leaf: int = None,
        class_weight: str = None,
        random_state: int = None
    ):
        """
        Initialize Random Forest with config defaults.
        """
        self.n_estimators = n_estimators or rf_config.n_estimators
        self.max_depth = max_depth or rf_config.max_depth
        self.min_samples_split = min_samples_split or rf_config.min_samples_split
        self.min_samples_leaf = min_samples_leaf or rf_config.min_samples_leaf
        self.class_weight = class_weight or rf_config.class_weight
        self.random_state = random_state or rf_config.random_state
        
        self.model: Optional[RandomForestClassifier] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.feature_names: Optional[list[str]] = None
        self.is_fitted: bool = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calibrate: bool = False,
        X_cal: pd.DataFrame = None,
        y_cal: pd.Series = None
    ) -> "RandomForestModel":
        """
        Fit Random Forest model.
        
        Args:
            X: Training features
            y: Training target
            calibrate: If True, calibrate probabilities
            X_cal: Calibration features (required if calibrate=True)
            y_cal: Calibration target (required if calibrate=True)
            
        Returns:
            Self for chaining
        """
        self.feature_names = list(X.columns)
        
        logger.info(
            f"Training Random Forest (n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth})"
        )
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=rf_config.n_jobs,
            oob_score=True  # Out-of-bag error estimate
        )
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"OOB Score: {self.model.oob_score_:.4f}")
        
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
        
        logger.info(f"Random Forest trained on {len(X)} samples")
        
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
    
    def get_feature_importance(self, importance_type: str = "impurity") -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: "impurity" (MDI) or "permutation" (placeholder)
            
        Returns:
            DataFrame with feature names and importance scores, sorted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted")
        
        if importance_type == "impurity":
            importances = self.model.feature_importances_
        else:
            # Permutation importance would require additional computation
            importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        })
        
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        return importance_df
    
    def get_tree_stats(self) -> dict:
        """
        Get statistics about the ensemble.
        """
        depths = [tree.get_depth() for tree in self.model.estimators_]
        n_leaves = [tree.get_n_leaves() for tree in self.model.estimators_]
        
        return {
            "n_estimators": self.n_estimators,
            "avg_depth": np.mean(depths),
            "max_depth_actual": max(depths),
            "avg_n_leaves": np.mean(n_leaves),
            "oob_score": self.model.oob_score_ if hasattr(self.model, "oob_score_") else None
        }
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save model to disk."""
        path = path or MODELS_DIR / "random_forest.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RandomForestModel":
        """Load model from disk."""
        path = path or MODELS_DIR / "random_forest.pkl"
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model
