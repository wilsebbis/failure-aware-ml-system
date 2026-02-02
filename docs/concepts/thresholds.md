# Threshold Derivation

This page explains how the PASS/REVIEW/FLAG thresholds are computed.

---

## The Problem

We need two thresholds:

1. **threshold_negative** (τ⁻): Below this, auto-approve
2. **threshold_positive** (τ⁺): Above this, auto-flag

The region between them is the **abstention band**—cases routed to human review.

---

## Optimization Objective

We solve a **constrained optimization**:

```
minimize: abstention_rate (τ⁻, τ⁺)
subject to:
    FNR(τ⁺) ≤ target_fnr        # catch enough defaults
    FPR(τ⁻) ≤ target_fpr        # don't overload reviewers
```

In practice, we prioritize **recall** (catching defaults) over queue size.

---

## Algorithm

### Step 1: Find τ⁺ (Flag Threshold)

Pick the **minimum threshold** such that False Negative Rate stays below target:

```python
def find_flag_threshold(y_true, y_proba, max_fnr=0.005):
    """
    Find minimum threshold such that FNR <= max_fnr.
    
    Lower threshold = more cases flagged = lower FNR.
    """
    for threshold in np.linspace(0.99, 0.01, 1000):
        y_pred = (y_proba >= threshold).astype(int)
        
        # False negatives: actual=1, predicted=0
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        positives = (y_true == 1).sum()
        fnr = fn / positives
        
        if fnr <= max_fnr:
            return threshold
    
    return 0.5  # fallback if target unreachable
```

### Step 2: Find τ⁻ (Pass Threshold)

Pick the **maximum threshold** such that the miss rate in the "pass" region is acceptable:

```python
def find_pass_threshold(y_true, y_proba, max_miss_rate=0.01):
    """
    Find maximum threshold such that few positives slip through.
    """
    for threshold in np.linspace(0.01, 0.50, 500):
        pass_mask = y_proba < threshold
        
        if pass_mask.sum() == 0:
            continue
            
        miss_rate = y_true[pass_mask].mean()
        
        if miss_rate <= max_miss_rate:
            return threshold
    
    return 0.10  # conservative fallback
```

---

## Current Production Values

| Threshold | Value | Rationale |
|-----------|-------|-----------|
| τ⁻ (pass) | 0.15 | 99% of cases below this are true negatives |
| τ⁺ (flag) | 0.60 | Captures 95%+ of defaults with this ceiling |

These were derived on validation data (4,500 samples) and validated on test holdout.

---

## Sensitivity Analysis

```
τ⁺ = 0.50  →  FNR = 0.3%, Review Rate = 52%
τ⁺ = 0.60  →  FNR = 0.5%, Review Rate = 46%  ← Current
τ⁺ = 0.70  →  FNR = 1.2%, Review Rate = 40%
```

The tradeoff is always: **lower FNR ↔ higher review volume**.

---

## When to Recalibrate Thresholds

- **Quarterly**: As part of model refresh cycle
- **On drift detection**: PSI > 0.25 triggers threshold review
- **On performance degradation**: ECE > 0.10 or recall drop > 5%

---

## Related

- [Confidence & Calibration](confidence.md)
- [Queue Mechanics](queue.md)
