"""
Failure-Aware ML System Configuration

Central configuration for paths, hyperparameters, thresholds, and model settings.
All magic numbers are documented and justified.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # UCI Credit Card Default dataset ID
    uci_dataset_id: int = 350
    
    # Train/Val/Test split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Random seed for reproducibility
    random_state: int = 42
    
    # Target column name
    target_column: str = "default"
    
    # Columns to drop (non-predictive)
    drop_columns: list[str] = field(default_factory=lambda: ["ID"])


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class LogisticConfig:
    """Configuration for baseline logistic regression."""
    
    C: float = 0.1  # Strong regularization for small data
    penalty: Literal["l1", "l2", "elasticnet"] = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    class_weight: str = "balanced"  # Handle imbalance
    random_state: int = 42


@dataclass
class RandomForestConfig:
    """Configuration for random forest classifier."""
    
    n_estimators: int = 100
    max_depth: int = 8  # Constrained depth for interpretability
    min_samples_split: int = 20  # Prevent overfitting on small data
    min_samples_leaf: int = 10
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class XGBoostConfig:
    """
    Configuration for constrained XGBoost.
    
    Shallow trees (max_depth <= 4) for:
    - Auditability
    - Reduced overfitting on limited data
    - SHAP reliability
    """
    
    n_estimators: int = 100
    max_depth: int = 4  # Shallow for interpretability
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    scale_pos_weight: float = 1.0  # Will be computed from data
    random_state: int = 42
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 10


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics and thresholds."""
    
    # Cost matrix for asymmetric errors
    # FN cost >> FP cost (missing a default is worse than false alarm)
    cost_fn: float = 10.0  # Cost of false negative
    cost_fp: float = 1.0   # Cost of false positive
    
    # Calibration settings
    calibration_method: Literal["isotonic", "sigmoid"] = "isotonic"
    n_calibration_bins: int = 10
    
    # Expected Calibration Error settings
    ece_n_bins: int = 10


# =============================================================================
# DECISION POLICY CONFIGURATION
# =============================================================================

@dataclass
class DecisionPolicyConfig:
    """
    Configuration for three-way decision output.
    
    Decision regions:
    - p < threshold_negative: Auto-PASS (confident negative)
    - p > threshold_positive: Auto-FLAG (confident positive)  
    - Otherwise: HUMAN REVIEW (abstention)
    
    Thresholds are tuned during calibration to minimize FN while
    controlling human review volume.
    """
    
    # Initial thresholds (will be optimized)
    threshold_negative: float = 0.15  # Below this → auto-pass
    threshold_positive: float = 0.60  # Above this → auto-flag
    
    # Target maximum human review rate
    max_review_rate: float = 0.30  # Max 30% to human review
    
    # Minimum recall constraint
    min_recall: float = 0.90  # Must achieve at least 90% recall


# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

@dataclass
class MonitoringConfig:
    """Configuration for drift and failure detection."""
    
    # Population Stability Index thresholds
    psi_warning: float = 0.1   # Warning level
    psi_critical: float = 0.25  # Critical shift detected
    
    # Confidence collapse detection
    # If mean confidence drops by this much, trigger alert
    confidence_drop_threshold: float = 0.15
    
    # Calibration error spike threshold
    ece_spike_threshold: float = 0.10


# =============================================================================
# INSTANTIATE DEFAULT CONFIGS
# =============================================================================

data_config = DataConfig()
logistic_config = LogisticConfig()
rf_config = RandomForestConfig()
xgb_config = XGBoostConfig()
eval_config = EvaluationConfig()
decision_config = DecisionPolicyConfig()
monitoring_config = MonitoringConfig()
