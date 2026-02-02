# Evaluation API

Metrics calculation, calibration assessment, and threshold optimization.

---

## MetricsCalculator

::: src.evaluation.metrics.MetricsCalculator
    options:
      show_root_heading: true
      members_order: source

---

## Calibration Functions

::: src.evaluation.calibration
    options:
      show_root_heading: true
      members_order: source

---

## Threshold Optimization

::: src.evaluation.thresholds
    options:
      show_root_heading: true
      members_order: source

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
