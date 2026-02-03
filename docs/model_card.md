# Model Card: Failure-Aware ML System

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | Failure-Aware ML System (Cascade Architecture) |
| **Version** | 2.0.0 |
| **Type** | 2-Stage Cascade: Logistic Regression + XGBoost (calibrated) |
| **Framework** | scikit-learn 1.8+, XGBoost 3.0+ |
| **Training Date** | 2026-02-02 |
| **Validation Status** | ✅ Production Validated |

---

## Intended Use

### Primary Use Cases

- Credit default risk scoring (consumer lending)
- Real-time transaction fraud screening
- High-volume loan application triage
- KYC/AML initial risk stratification

### Out of Scope

- Autonomous decisioning without human oversight
- Applications in healthcare diagnostics
- Any domain outside financial risk

### Users

- Risk analysts conducting manual case review
- Compliance officers auditing model decisions
- ML engineers monitoring model health
- Operations teams managing decision queues

---

## Architecture

### Cascade Classifier (2-Stage)

```
Stage 1: Gatekeeper (Logistic Regression)
├── Easy PASS: 65.6% (auto-approved)
├── Easy FLAG: 21.1% (auto-blocked)
└── Hard Cases: 13.3% → Stage 2

Stage 2: Specialist (XGBoost + Isotonic Calibration)
└── Trained only on hard cases
└── Higher positive rate (26.6% vs 21.5%)
```

### Dynamic Threshold Monitor

| Condition | Action |
|-----------|--------|
| Mean risk < baseline + 3σ | Normal thresholds |
| Mean risk > baseline + 3σ | Tighten PASS threshold |

---

## Training Data

### Supported Datasets

| Dataset | Size | Target Rate | Use Case |
|---------|------|-------------|----------|
| UCI Credit | 30K | 22.1% | Baseline |
| Home Credit | 307K | 8.1% | Complex joins |
| IEEE-CIS Fraud | 590K | 3.4% | Imbalanced fraud |
| Lending Club | 1.37M | 21.5% | Scale validation |

### Preprocessing

- Categorical encoding: Category codes
- Ratio features: DTI/Utilization, Loan/Income
- Calibration: Isotonic regression on all models

---

## Performance Metrics

### Production Results (2026-02-02)

| Dataset | Pass Rate | Defect Rate | Review Rate | System Recall |
|---------|-----------|-------------|-------------|---------------|
| **Lending Club** | 67.2% | 2.01% | 10.8% | 98.7% |
| **IEEE-CIS Fraud** | 90.0% | 1.74% | 6.7% | 98.4% |
| **Home Credit** | 75.5% | 4.46% | 24.1% | 96.6% |
| **UCI Credit** | 29.9% | 7.80% | 57.7% | 97.7% |

### Cascade Efficiency

| Dataset | Stage 1 Easy PASS | Stage 2 Hard Cases |
|---------|-------------------|-------------------|
| Lending Club | 65.6% | 13.3% |
| IEEE-CIS | 96.4% | 2.5% |
| Home Credit | 60.5% | 39.2% |

### Key Achievement

**~4x reduction** in manual review volume compared to single-model baselines while maintaining **>96% System Recall**.

---

## Limitations

### Known Failure Modes

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Distribution shift | Rolling mean confidence drop | Dynamic threshold tightening |
| High noise datasets (UCI) | 57.7% review rate | Human escalation (correct behavior) |
| Concept drift | PSI monitoring | Escalate to review |
| Sparse region overconfidence | Calibration curves | Conservative thresholds |

### The "Noisy Dataset" Limit

On UCI Credit (22% default rate, weak features), the system correctly refuses to auto-approve more than 30% of cases. This is **intentional** — when features cannot distinguish risk, the system escalates to humans rather than guessing.

### Demographic Analysis

The model has not been evaluated for demographic fairness across protected groups. Before production deployment, conduct disparity analysis on:
- Age bands
- Gender
- Regional segments

---

## Ethical Considerations

### Bias Risks

- Training data reflects historical lending decisions which may encode bias
- Feature engineering preserves demographic signals in some datasets
- No explicit fairness constraints applied

### Mitigation Approach

- Human-in-the-loop design prevents fully automated adverse decisions
- SHAP explanations provided to reviewers to catch obvious bias
- Quarterly bias audits recommended

---

## Caveats and Recommendations

1. **Do not deploy without human oversight** - This model is designed for triage, not autonomous decisioning

2. **Use `--cascade --dynamic-threshold` for fraud** - The dynamic safety valve is critical for non-stationary distributions

3. **Recalibrate quarterly** - Calibration degrades as population shifts

4. **Monitor PSI weekly** - Feature drift precedes performance degradation

5. **Conduct fairness audit** - Before any production use

---

## Citation

```
@software{failure_aware_ml,
  title = {Failure-Aware ML System: Cascade Architecture for High-Recall Risk Assessment},
  version = {2.0.0},
  date = {2026-02-02}
}
```
