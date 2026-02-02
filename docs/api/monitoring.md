# Monitoring API

Distribution drift detection and confidence collapse monitoring.

---

## Drift Detection

```python
def compute_psi(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    n_bins: int = 10
) -> dict[str, float]:
    """
    Compute Population Stability Index per feature.
    
    Returns:
        dict mapping feature name to PSI score
    """
    ...

def detect_drift(
    psi_scores: dict[str, float],
    threshold: float = 0.25
) -> list[str]:
    """Return list of features with PSI above threshold."""
    ...
```

---

## Confidence Collapse

```python
def compute_confidence_metrics(
    probabilities: np.ndarray
) -> dict:
    """
    Compute confidence statistics.
    
    Returns:
        dict with keys: mean, std, entropy, abstention_rate
    """
    ...

def detect_collapse(
    metrics: dict,
    threshold: float = 0.70
) -> bool:
    """Return True if confidence collapse detected."""
    ...
```

---

## Usage Example

### Detect Distribution Shift

```python
from src.monitoring.drift import compute_psi, detect_drift

# Population Stability Index per feature
psi_scores = compute_psi(
    reference=X_train,
    current=X_test
)

# Alert if any feature drifted
alerts = detect_drift(psi_scores, threshold=0.25)
if alerts:
    print(f"Drift detected in: {alerts}")
```

### Monitor Confidence

```python
from src.monitoring.confidence_collapse import (
    compute_confidence_metrics,
    detect_collapse
)

metrics = compute_confidence_metrics(probabilities)
print(f"Mean confidence: {metrics['mean']:.2f}")
print(f"Entropy: {metrics['entropy']:.2f}")

if detect_collapse(metrics, threshold=0.70):
    print("⚠️ Confidence collapse detected!")
```

---

## Key Metrics

### Population Stability Index (PSI)

```
PSI = Σ (current_pct - reference_pct) × ln(current_pct / reference_pct)
```

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.10 | No significant shift |
| 0.10 - 0.25 | Minor shift, monitor |
| > 0.25 | Significant shift, investigate |

### Confidence Metrics

| Metric | Formula | Alert Threshold |
|--------|---------|-----------------|
| `mean_confidence` | mean(max(p, 1-p)) | < 0.70 |
| `abstention_rate` | % in REVIEW band | > 60% |
| `entropy` | -Σ p log(p) | < reference - 0.1 |

---

## Monitoring Dashboard

```python
from src.monitoring.drift import generate_monitoring_report

report = generate_monitoring_report(
    reference=X_train,
    current=X_new_batch,
    probas=new_probas
)

# report contains:
# - psi_by_feature: {feature: score}
# - confidence_metrics: {mean, std, entropy}
# - alerts: [list of triggered alerts]
```

---

## Related

- [Confidence & Calibration](../concepts/confidence.md)
- [Queue Mechanics](../concepts/queue.md)
