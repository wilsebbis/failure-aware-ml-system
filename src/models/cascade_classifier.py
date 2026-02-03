"""
Two-Stage Cascade Classifier.

Solves the "Review Bottleneck" problem where a single model sends
70%+ of cases to manual review due to low confidence.

Architecture:
    Stage 1 (Filter): High-recall model to identify "Easy Non-Defaults"
    Stage 2 (Focus): Specialized model for "Hard Cases"

The key insight is that a generalist model tries to learn easy and hard
patterns simultaneously, often failing at the hard ones. A cascade
architecture lets each stage specialize.

Example:
    cascade = CascadeClassifier(
        stage1_model=XGBoostModel(config),
        stage2_model=XGBoostModel(config),
        stage1_pass_threshold=0.02
    )
    cascade.fit(X_train, y_train, X_val, y_val)
    probs = cascade.predict_proba(X_test)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class CascadeClassifier:
    """
    Two-stage cascade for reducing Review bottleneck.
    
    Stage 1: Trained on all data, optimized to identify "Easy Non-Defaults"
             with high confidence. Uses a strict threshold to pass only
             cases where we're very confident of non-default.
             
    Stage 2: Trained ONLY on the "Hard Cases" that Stage 1 couldn't
             confidently classify. This specialization often finds
             separation where a generalist model sees noise.
    
    Attributes:
        stage1: First-stage model (filter)
        stage2: Second-stage model (focused)
        stage1_pass_threshold: Probability below which Stage 1 passes directly
        stage1_flag_threshold: Probability above which Stage 1 flags directly
        min_hard_cases: Minimum samples needed to train Stage 2
    """
    
    def __init__(
        self,
        stage1_model: Any,
        stage2_model: Any,
        stage1_pass_threshold: float = 0.02,
        stage1_flag_threshold: float = 0.80,
        min_hard_cases: int = 500,
    ):
        """
        Initialize cascade classifier.
        
        Args:
            stage1_model: Model instance for first stage (filter)
            stage2_model: Model instance for second stage (focus)
            stage1_pass_threshold: Pass threshold for Stage 1 (very confident non-default)
            stage1_flag_threshold: Flag threshold for Stage 1 (very confident default)
            min_hard_cases: Minimum samples to train Stage 2
        """
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.stage1_pass_threshold = stage1_pass_threshold
        self.stage1_flag_threshold = stage1_flag_threshold
        self.min_hard_cases = min_hard_cases
        
        self.stage2_trained = False
        self.stage1_stats = {}
        self.stage2_stats = {}
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "CascadeClassifier":
        """
        Fit the cascade classifier.
        
        Stage 1 trains on all data.
        Stage 2 trains only on "Hard Cases" (uncertain predictions from Stage 1).
        
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        
        Returns:
            self
        """
        logger.info("=== Cascade Classifier Training ===")
        
        # Stage 1: Train on all data (use calibration)
        logger.info("Stage 1: Training Gatekeeper (Logistic) on all data...")
        self._fit_model(self.stage1, X, y, X_val, y_val, calibrate=True)
        
        # Get Stage 1 predictions to identify hard cases
        stage1_probs = self._get_proba(self.stage1, X)
        
        # Hard cases: between pass and flag thresholds (the "gray zone")
        hard_mask = (
            (stage1_probs >= self.stage1_pass_threshold) & 
            (stage1_probs < self.stage1_flag_threshold)
        )
        
        n_easy_pass = (stage1_probs < self.stage1_pass_threshold).sum()
        n_easy_flag = (stage1_probs >= self.stage1_flag_threshold).sum()
        n_hard = hard_mask.sum()
        
        self.stage1_stats = {
            "easy_pass": int(n_easy_pass),
            "easy_flag": int(n_easy_flag),
            "hard_cases": int(n_hard),
            "easy_pass_pct": n_easy_pass / len(y) * 100,
            "hard_pct": n_hard / len(y) * 100,
        }
        
        logger.info(f"  Stage 1 Results:")
        logger.info(f"    Easy PASS: {n_easy_pass:,} ({self.stage1_stats['easy_pass_pct']:.1f}%)")
        logger.info(f"    Easy FLAG: {n_easy_flag:,}")
        logger.info(f"    Hard Cases: {n_hard:,} ({self.stage1_stats['hard_pct']:.1f}%)")
        
        # Stage 2: Train only on hard cases
        if n_hard >= self.min_hard_cases:
            logger.info(f"Stage 2: Training Specialist (XGBoost) on {n_hard:,} hard cases...")
            
            # Filter to hard cases using boolean indexing
            X_hard = X.loc[hard_mask] if isinstance(X, pd.DataFrame) else X[hard_mask]
            y_hard = y.loc[hard_mask] if isinstance(y, pd.Series) else y[hard_mask]
            
            # Validation set for Stage 2 (optional)
            X_val_2, y_val_2 = None, None
            if X_val is not None and y_val is not None:
                val_probs = self._get_proba(self.stage1, X_val)
                val_hard_mask = (
                    (val_probs >= self.stage1_pass_threshold) & 
                    (val_probs < self.stage1_flag_threshold)
                )
                if val_hard_mask.sum() >= 100:
                    X_val_2 = X_val.loc[val_hard_mask] if isinstance(X_val, pd.DataFrame) else X_val[val_hard_mask]
                    y_val_2 = y_val.loc[val_hard_mask] if isinstance(y_val, pd.Series) else y_val[val_hard_mask]
            
            self._fit_model(self.stage2, X_hard, y_hard, X_val_2, y_val_2, calibrate=True)
            self.stage2_trained = True
            
            self.stage2_stats = {
                "train_samples": int(n_hard),
                "positive_rate": float(y_hard.mean() * 100),
            }
            logger.info(f"  Stage 2 trained on {n_hard:,} samples (positive rate: {self.stage2_stats['positive_rate']:.1f}%)")
        else:
            logger.warning(f"  Stage 2 skipped: only {n_hard} hard cases (need {self.min_hard_cases})")
            self.stage2_trained = False
        
        logger.info("=== Cascade Training Complete ===")
        return self
    
    def _fit_model(self, model, X, y, X_val, y_val, calibrate=True):
        """Fit a model handling different API signatures."""
        try:
            # Try XGBoostModel signature first
            model.fit(X, y, X_val=X_val, y_val=y_val, calibrate=calibrate, X_cal=X_val, y_cal=y_val)
        except TypeError:
            try:
                # Try LogisticBaseline signature
                model.fit(X, y, calibrate=calibrate, X_cal=X_val, y_cal=y_val)
            except TypeError:
                # Fallback to basic sklearn signature
                model.fit(X, y)
    
    def _get_proba(self, model, X) -> np.ndarray:
        """Get probabilities handling both 1D and 2D output."""
        probs = model.predict_proba(X)
        if len(probs.shape) > 1:
            return probs[:, 1]
        return probs
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using cascade.
        
        Easy cases (Stage 1 confident) use Stage 1 probabilities.
        Hard cases use Stage 2 probabilities (if trained).
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predicted probabilities
        """
        # Stage 1 predictions
        stage1_probs = self._get_proba(self.stage1, X)
        
        if not self.stage2_trained:
            return stage1_probs
        
        # Identify hard cases
        hard_mask = (
            (stage1_probs >= self.stage1_pass_threshold) & 
            (stage1_probs < self.stage1_flag_threshold)
        )
        
        # Use Stage 2 for hard cases
        final_probs = stage1_probs.copy()
        if hard_mask.sum() > 0:
            X_hard = X.loc[hard_mask] if isinstance(X, pd.DataFrame) else X[hard_mask]
            stage2_probs = self._get_proba(self.stage2, X_hard)
            final_probs[hard_mask] = stage2_probs
        
        return final_probs
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def get_cascade_stats(self) -> dict:
        """Return statistics about cascade performance."""
        return {
            "stage1": self.stage1_stats,
            "stage2": self.stage2_stats if self.stage2_trained else None,
            "stage2_trained": self.stage2_trained,
        }
