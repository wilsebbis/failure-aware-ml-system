"""
Baseline Logistic Regression Model

Interpretable baseline for comparison and auditability.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path
from typing import Optional
import logging

from src.config import logistic_config, MODELS_DIR

logger = logging.getLogger(__name__)


class LogisticBaseline:
    """
    Regularized logistic regression baseline.
    
    Key properties:
    - Fully interpretable via coefficients
    - Class weighting for imbalance
    - Optional probability calibration
    
    Attributes:
        model: Fitted LogisticRegression
        calibrated_model: Optional calibrated version
        feature_names: List of feature names
        is_fitted: Whether model has been trained
    """
    
    def __init__(
        self,
        C: float = None,
        penalty: str = None,
        class_weight: str = None,
        random_state: int = None
    ):
        """
        Initialize logistic regression with config defaults.
        """
        self.C = C or logistic_config.C
        self.penalty = penalty or logistic_config.penalty
        self.class_weight = class_weight or logistic_config.class_weight
        self.random_state = random_state or logistic_config.random_state
        
        self.model: Optional[LogisticRegression] = None
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
    ) -> "LogisticBaseline":
        """
        Fit logistic regression model.
        
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
        
        logger.info(f"Training Logistic Regression (C={self.C}, penalty={self.penalty})")
        
        # Use Pipeline with StandardScaler for better convergence
        logistic = LogisticRegression(
            C=self.C,
            l1_ratio=0.0 if self.penalty == "l2" else 1.0,  # l1_ratio=0 is L2, l1_ratio=1 is L1
            solver="saga" if self.penalty == "l1" else logistic_config.solver,
            max_iter=logistic_config.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic", logistic)
        ])
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Optional calibration
        if calibrate:
            if X_cal is None or y_cal is None:
                raise ValueError("Calibration data required when calibrate=True")
            
            logger.info("Calibrating probabilities with isotonic regression")
            # sklearn 1.8+: use fitted_estimator parameter instead of cv='prefit'
            from sklearn import __version__ as sklearn_version
            if tuple(map(int, sklearn_version.split('.')[:2])) >= (1, 8):
                self.calibrated_model = CalibratedClassifierCV(
                    estimator=None,
                    method="isotonic",
                    cv=None
                )
                # Manually set the fitted estimator
                self.calibrated_model.estimator = self.model
                self.calibrated_model.fit(X_cal, y_cal)
            else:
                self.calibrated_model = CalibratedClassifierCV(
                    self.model,
                    method="isotonic",
                    cv="prefit"
                )
                self.calibrated_model.fit(X_cal, y_cal)
        
        logger.info(f"Logistic Regression trained on {len(X)} samples")
        
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
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients for interpretability.
        
        Returns:
            DataFrame with feature names and coefficients, sorted by absolute value
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted")
        
        # Access logistic regression inside Pipeline
        logistic = self.model.named_steps["logistic"]
        
        coef_df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": logistic.coef_[0],
            "abs_coefficient": np.abs(logistic.coef_[0])
        })
        
        coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
        coef_df = coef_df.reset_index(drop=True)
        
        return coef_df[["feature", "coefficient"]]
    
    def get_odds_ratios(self) -> pd.DataFrame:
        """
        Get odds ratios for interpretability.
        
        Odds ratio > 1: Feature increases odds of default
        Odds ratio < 1: Feature decreases odds of default
        """
        coef_df = self.get_coefficients()
        coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
        return coef_df
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save model to disk."""
        path = path or MODELS_DIR / "logistic_baseline.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LogisticBaseline":
        """Load model from disk."""
        path = path or MODELS_DIR / "logistic_baseline.pkl"
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return model
