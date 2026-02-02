# Model Card: Sentinel Risk Engine

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | Sentinel Default Risk Classifier |
| **Version** | 1.0.0 |
| **Type** | XGBoost Gradient Boosted Trees (calibrated) |
| **Framework** | scikit-learn 1.8+, XGBoost 3.0+ |
| **Training Date** | 2026-02-02 |
| **Authors** | [Your Name] |

---

## Intended Use

### Primary Use Cases

- Credit default risk scoring in consumer lending
- Real-time transaction fraud screening
- KYC/AML initial risk stratification

### Out of Scope

- Autonomous decisioning without human oversight
- Applications in healthcare diagnostics
- Any domain outside financial risk

### Users

- Risk analysts conducting manual case review
- Compliance officers auditing model decisions
- ML engineers monitoring model health

---

## Training Data

### Dataset

| Property | Value |
|----------|-------|
| Source | UCI ML Repository - Default of Credit Card Clients |
| Size | 30,000 records |
| Features | 23 (demographic + payment history) |
| Target | Binary (default next month) |
| Class Balance | 22% positive (default), 78% negative |

### Preprocessing

- Categorical encoding: One-hot (EDUCATION, MARRIAGE)
- Continuous scaling: StandardScaler
- Ordinal preserved: PAY_0 through PAY_6 (delay months)

### Split

| Set | Size | Positive Rate |
|-----|------|---------------|
| Train | 21,000 (70%) | 22.1% |
| Validation | 4,500 (15%) | 22.2% |
| Test | 4,500 (15%) | 22.1% |

---

## Performance Metrics

### Test Set Results

| Model | Recall | FNR | ECE | Calibration |
|-------|--------|-----|-----|-------------|
| **XGBoost** | 31.4% | 68.6% | 0.016 | Isotonic |
| Random Forest | 29.0% | 71.0% | 0.012 | Isotonic |
| Logistic | 23.8% | 76.2% | 0.015 | Isotonic |

### Triage Distribution

| Decision | Rate | Description |
|----------|------|-------------|
| PASS | 46.8% | Auto-approved |
| REVIEW | 46.0% | Human review |
| FLAG | 7.2% | Auto-blocked |

---

## Limitations

### Known Failure Modes

| Failure | Detection | Mitigation |
|---------|-----------|------------|
| Distribution shift | PSI monitoring | Escalate to review |
| Confidence collapse | Mean confidence < 0.70 | Widen abstention band |
| Sparse region overconfidence | Calibration curves | Conservative thresholds |
| Label noise | CV variance | Regularization |

### Demographic Analysis

The model has not been evaluated for demographic fairness across protected groups. Before production deployment, conduct disparity analysis on:
- Age bands
- Gender
- Regional segments

---

## Ethical Considerations

### Bias Risks

- Training data reflects historical lending decisions which may encode bias
- Feature engineering preserves raw demographic signals (SEX, AGE)
- No explicit fairness constraints applied

### Mitigation Approach

- Human-in-the-loop design prevents fully automated adverse decisions
- Explanations provided to reviewers to catch obvious bias
- Quarterly bias audits recommended

---

## Caveats and Recommendations

1. **Do not deploy without human oversight** - This model is designed for triage, not autonomous decisioning

2. **Recalibrate quarterly** - Calibration degrades as population shifts

3. **Monitor PSI weekly** - Feature drift precedes performance degradation

4. **Conduct fairness audit** - Before any production use
