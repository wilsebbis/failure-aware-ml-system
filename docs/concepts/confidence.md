# Confidence & Calibration

This page defines the key metrics used for decision-making and model health monitoring.

---

## Term Definitions

### Calibrated Probability

**Definition**: P(default=1 | features), after post-hoc calibration via isotonic regression.

**How computed**:
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(
    estimator=base_model,
    method="isotonic",
    cv=5
)
calibrated.fit(X_cal, y_cal)
proba = calibrated.predict_proba(X_test)[:, 1]
```

**Operational use**: This is the score used for triage thresholds. It represents the model's estimate of the true default probability.

---

### Confidence

**Definition**: A measure of prediction certainty, calculated as:

```
confidence = 1 - H([p, 1-p]) / log(2)
```

Where H is the binary entropy. This scales to [0, 1] where:
- **1.0** = model is 100% certain (p = 0 or p = 1)
- **0.0** = model is maximally uncertain (p = 0.5)

**Alternative definition** (used in monitoring):
```python
confidence_mean = np.mean(np.maximum(proba, 1 - proba))
```

This is the mean of the "winning class" probability across a batch.

---

### Confidence Collapse

**Definition**: When confidence_mean drops significantly, indicating the model is outputting near-uniform predictions.

**Detection threshold**: Alert when `confidence_mean < 0.70`

**Causes**:
- Distribution shift in input features
- Model degradation
- Adversarial inputs

**Mitigation**: Automatically widen the abstention band (route more to REVIEW).

---

## Expected Calibration Error (ECE)

**Definition**: Weighted average of the gap between accuracy and confidence across bins.

**Formula**:
```
ECE = Σ (n_bin / n_total) × |accuracy(bin) - mean_confidence(bin)|
```

**Implementation**:
```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    y_true, 
    y_proba, 
    n_bins=10, 
    strategy='uniform'
)

ece = np.mean(np.abs(prob_true - prob_pred))
```

**Interpretation**:
| ECE Value | Status |
|-----------|--------|
| < 0.02 | Excellent calibration |
| 0.02 - 0.05 | Good |
| 0.05 - 0.10 | Needs attention |
| > 0.10 | Recalibration required |

---

## Calibration Curve

A well-calibrated model's calibration curve should follow the diagonal:

![Calibration Curve Example](../figures/shap_summary.png)

**How to interpret**:
- Points above diagonal → model is underconfident
- Points below diagonal → model is overconfident

---

## Related

- [Threshold Derivation](thresholds.md)
- [Queue Mechanics](queue.md)
