# Evaluation API

Metrics calculation, calibration assessment, and threshold optimization.

---

## MetricsCalculator

```python
class MetricsCalculator:
    """Calculate classification metrics with focus on recall."""
    
    def compute_all(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray
    ) -> dict:
        """
        Compute all metrics.
        
        Returns:
            dict with keys: recall, precision, fnr, fpr, ece, auc
        """
        ...
```

---

## Calibration Functions

```python
def compute_ece(
    y_true: np.ndarray, 
    y_proba: np.ndarray, 
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error."""
    ...

def calibration_curve(
    y_true: np.ndarray, 
    y_proba: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve (fraction of positives per bin)."""
    ...
```

---

## Threshold Optimization

```python
def optimize_thresholds(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    target_fnr: float = 0.005
) -> dict:
    """
    Find optimal thresholds for target FNR.
    
    Returns:
        dict with threshold_negative, threshold_positive
    """
    ...
```

---

## Usage Example

```python
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.calibration import compute_ece
from src.evaluation.thresholds import optimize_thresholds

# Calculate all metrics
calc = MetricsCalculator()
metrics = calc.compute_all(y_true, y_pred, y_proba)

print(f"Recall: {metrics['recall']:.2%}")
print(f"FNR: {metrics['fnr']:.2%}")
print(f"ECE: {metrics['ece']:.4f}")

# Compute ECE manually
ece = compute_ece(y_true, y_proba, n_bins=10)

# Optimize thresholds for target FNR
thresholds = optimize_thresholds(
    y_val, 
    proba_val,
    target_fnr=0.005
)
```

---

## Metrics Reference

| Metric | Formula | Target |
|--------|---------|--------|
| Recall | TP / (TP + FN) | Maximize |
| Precision | TP / (TP + FP) | Secondary |
| FNR | FN / (FN + TP) | < 1% |
| ECE | Σ bin_weight × |acc - conf| | < 0.05 |

---

## Related

- [Confidence & Calibration](../concepts/confidence.md)
- [Threshold Derivation](../concepts/thresholds.md)
