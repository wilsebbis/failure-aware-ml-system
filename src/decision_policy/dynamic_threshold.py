"""
Dynamic Rolling Threshold for Drift Adaptation.

Fraud attacks are bursty. A static threshold of p < 0.02 is dangerous
because during an attack, the average probability shifts.

This module implements Rolling Mean + StdDev thresholding:
- IF p > RollingMean(last_1000) + 3*StdDev THEN FLAG
- This automatically "tightens the gates" during a spike

Example:
    from src.decision_policy.dynamic_threshold import DynamicThreshold
    
    threshold = DynamicThreshold(window_size=1000)
    
    for prob in predictions:
        pass_thresh = threshold.get_pass_threshold()
        flag_thresh = threshold.get_flag_threshold()
        threshold.update(prob)
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DynamicThreshold:
    """
    Rolling window adaptive thresholding for fraud/anomaly detection.
    
    Instead of static thresholds, uses rolling statistics to adapt
    to distribution shifts and attack bursts automatically.
    
    Formula:
        FLAG if p > rolling_mean + k * rolling_std
        PASS if p < rolling_mean - k * rolling_std (clamped to base)
    
    During a fraud spike:
        - Rolling mean increases → FLAG threshold stays relative
        - PASS threshold tightens → more cases go to review
        - System automatically "closes the gates"
    
    Attributes:
        window_size: Number of recent predictions to track
        base_pass_threshold: Minimum PASS threshold (floor)
        base_flag_threshold: Default FLAG threshold
        k_sigma: Number of standard deviations for FLAG
        spike_sensitivity: Multiplier for detecting spikes
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        base_pass_threshold: float = 0.02,
        base_flag_threshold: float = 0.50,
        k_sigma: float = 3.0,
        spike_sensitivity: float = 1.5,
    ):
        """
        Initialize dynamic threshold.
        
        Args:
            window_size: Rolling window size (transactions to track)
            base_pass_threshold: Minimum PASS threshold
            base_flag_threshold: Default FLAG threshold before adaptation
            k_sigma: Standard deviations for FLAG threshold
            spike_sensitivity: Ratio above baseline that triggers tightening
        """
        self.window_size = window_size
        self.base_pass_threshold = base_pass_threshold
        self.base_flag_threshold = base_flag_threshold
        self.k_sigma = k_sigma
        self.spike_sensitivity = spike_sensitivity
        
        # Rolling window of recent predictions
        self.history = deque(maxlen=window_size)
        
        # Baseline statistics (set after first full window)
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        
        # Current state
        self.is_spike_mode = False
        self.spike_ratio = 1.0
    
    def update(self, prob: float) -> None:
        """
        Update rolling window with new prediction.
        
        Args:
            prob: Predicted probability for current transaction
        """
        self.history.append(prob)
        
        # Set baseline after first full window
        if len(self.history) == self.window_size and self.baseline_mean is None:
            self.baseline_mean = np.mean(self.history)
            self.baseline_std = np.std(self.history)
            logger.info(f"DynamicThreshold baseline set: mean={self.baseline_mean:.4f}, std={self.baseline_std:.4f}")
        
        # Detect spike mode
        if self.baseline_mean is not None and len(self.history) >= self.window_size // 2:
            current_mean = np.mean(self.history)
            self.spike_ratio = current_mean / max(self.baseline_mean, 1e-6)
            
            was_spike = self.is_spike_mode
            self.is_spike_mode = self.spike_ratio > self.spike_sensitivity
            
            if self.is_spike_mode and not was_spike:
                logger.warning(f"SPIKE DETECTED: rolling mean {current_mean:.4f} is {self.spike_ratio:.1f}x baseline")
            elif not self.is_spike_mode and was_spike:
                logger.info("Spike ended, returning to normal thresholds")
    
    def update_batch(self, probs: np.ndarray) -> None:
        """Update with batch of predictions."""
        for prob in probs:
            self.update(prob)
    
    def get_pass_threshold(self) -> float:
        """
        Get current dynamic PASS threshold.
        
        During spike: Tightens (lower) to send more to review
        Normal: Returns base threshold
        
        Returns:
            Current PASS threshold
        """
        if not self.is_spike_mode or len(self.history) < 100:
            return self.base_pass_threshold
        
        # Tighten threshold proportionally to spike
        # If spike_ratio = 2.0, halve the pass threshold
        tightened = self.base_pass_threshold / self.spike_ratio
        
        # Floor at 0.001 (always allow some auto-pass)
        return max(tightened, 0.001)
    
    def get_flag_threshold(self) -> float:
        """
        Get current dynamic FLAG threshold.
        
        Uses rolling_mean + k*rolling_std formula.
        
        Returns:
            Current FLAG threshold
        """
        if len(self.history) < 100:
            return self.base_flag_threshold
        
        rolling_mean = np.mean(self.history)
        rolling_std = np.std(self.history)
        
        # FLAG if above mean + k*sigma
        dynamic_flag = rolling_mean + self.k_sigma * rolling_std
        
        # Cap between reasonable bounds
        return np.clip(dynamic_flag, 0.3, 0.9)
    
    def get_thresholds(self) -> Tuple[float, float]:
        """Get both thresholds as tuple (pass, flag)."""
        return self.get_pass_threshold(), self.get_flag_threshold()
    
    def get_stats(self) -> dict:
        """Get current threshold statistics."""
        return {
            "window_size": self.window_size,
            "samples_seen": len(self.history),
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "current_mean": np.mean(self.history) if self.history else None,
            "current_std": np.std(self.history) if self.history else None,
            "spike_mode": self.is_spike_mode,
            "spike_ratio": self.spike_ratio,
            "current_pass_threshold": self.get_pass_threshold(),
            "current_flag_threshold": self.get_flag_threshold(),
        }
    
    def reset(self) -> None:
        """Reset threshold state (clear history and baseline)."""
        self.history.clear()
        self.baseline_mean = None
        self.baseline_std = None
        self.is_spike_mode = False
        self.spike_ratio = 1.0


class AdaptiveTriagePolicy:
    """
    Triage policy with dynamic threshold adaptation.
    
    Combines the DynamicThreshold with triage decisions for
    production-grade fraud detection.
    
    Example:
        policy = AdaptiveTriagePolicy()
        
        for X_batch in stream:
            probs = model.predict_proba(X_batch)
            decisions = policy.apply(probs)
            # Returns array of "PASS", "REVIEW", "FLAG"
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        base_pass: float = 0.02,
        base_flag: float = 0.50,
    ):
        self.threshold = DynamicThreshold(
            window_size=window_size,
            base_pass_threshold=base_pass,
            base_flag_threshold=base_flag,
        )
    
    def apply(self, probs: np.ndarray, update_history: bool = True) -> np.ndarray:
        """
        Apply adaptive triage to predictions.
        
        Args:
            probs: Array of predicted probabilities
            update_history: Whether to update rolling window
            
        Returns:
            Array of decisions ("PASS", "REVIEW", "FLAG")
        """
        pass_thresh, flag_thresh = self.threshold.get_thresholds()
        
        decisions = np.full(len(probs), "REVIEW", dtype=object)
        decisions[probs < pass_thresh] = "PASS"
        decisions[probs >= flag_thresh] = "FLAG"
        
        if update_history:
            self.threshold.update_batch(probs)
        
        return decisions
    
    def get_stats(self) -> dict:
        """Get threshold statistics."""
        return self.threshold.get_stats()
