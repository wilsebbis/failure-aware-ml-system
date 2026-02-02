"""
Three-Way Decision Triage

Routes predictions to: auto-pass, auto-flag, or human review.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from src.config import decision_config

logger = logging.getLogger(__name__)


class Decision(Enum):
    PASS = "pass"
    FLAG = "flag"
    REVIEW = "review"


@dataclass
class TriageResult:
    decision: Decision
    probability: float
    confidence: float
    reason: str
    
    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "probability": self.probability,
            "confidence": self.confidence,
            "reason": self.reason
        }


class TriagePolicy:
    """Three-way decision policy: PASS < t_neg | REVIEW | FLAG >= t_pos"""
    
    def __init__(self, threshold_negative: float = None, threshold_positive: float = None):
        self.threshold_negative = threshold_negative or decision_config.threshold_negative
        self.threshold_positive = threshold_positive or decision_config.threshold_positive
        logger.info(f"TriagePolicy: PASS < {self.threshold_negative:.2f} | FLAG >= {self.threshold_positive:.2f}")
    
    def decide(self, probability: float) -> TriageResult:
        confidence = abs(probability - 0.5) * 2
        if probability < self.threshold_negative:
            return TriageResult(Decision.PASS, probability, confidence, f"Low risk (p={probability:.2%})")
        elif probability >= self.threshold_positive:
            return TriageResult(Decision.FLAG, probability, confidence, f"High risk (p={probability:.2%})")
        else:
            return TriageResult(Decision.REVIEW, probability, confidence, f"Uncertain (p={probability:.2%})")
    
    def get_decision_labels(self, probabilities: np.ndarray) -> np.ndarray:
        decisions = np.empty(len(probabilities), dtype=object)
        decisions[probabilities < self.threshold_negative] = "pass"
        decisions[probabilities >= self.threshold_positive] = "flag"
        decisions[(probabilities >= self.threshold_negative) & (probabilities < self.threshold_positive)] = "review"
        return decisions
    
    def get_stats(self, probabilities: np.ndarray, y_true: np.ndarray = None) -> dict:
        decisions = self.get_decision_labels(probabilities)
        n_total = len(probabilities)
        stats = {
            "pass_rate": (decisions == "pass").sum() / n_total,
            "flag_rate": (decisions == "flag").sum() / n_total,
            "review_rate": (decisions == "review").sum() / n_total,
        }
        if y_true is not None:
            pass_mask = decisions == "pass"
            if pass_mask.sum() > 0:
                stats["fn_rate_in_pass"] = (y_true[pass_mask] == 1).sum() / pass_mask.sum()
        return stats
