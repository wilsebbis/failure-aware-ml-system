# Model Card: Failure-Aware Classifier

## Model Overview

| Field | Value |
|-------|-------|
| **Model Type** | Shallow Gradient Boosted Trees (XGBoost) |
| **Version** | 1.0.0 |
| **Framework** | scikit-learn, XGBoost |
| **Task** | Binary Classification |

---

## Intended Use

### Primary Use Case
Decision support for high-stakes binary classification in regulated environments.

### Target Users
- Risk analysts
- Compliance officers
- Human reviewers in escalation workflows

### Out-of-Scope Uses
- ❌ Fully automated decision-making
- ❌ Consumer-facing unsupervised deployment
- ❌ Real-time high-throughput inference

---

## Training Data

| Property | Value |
|----------|-------|
| **Dataset** | UCI Credit Card Default |
| **Samples** | ~30,000 |
| **Features** | 23 original + 8 derived |
| **Class Balance** | ~22% positive (imbalanced) |
| **Time Period** | April-September 2005 |

---

## Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Recall (Positive) | High | Primary optimization target |
| False Negative Rate | Minimized | Most critical error |
| Precision | Moderate | Acceptable given FN priority |
| ECE | Low | Post-calibration |

*Exact values depend on threshold configuration.*

---

## Decision Policy

Three-way output:
1. **PASS** (p < 0.15) → Auto-approve, low risk
2. **REVIEW** (0.15 ≤ p < 0.60) → Human escalation
3. **FLAG** (p ≥ 0.60) → Auto-flag, high risk

---

## Explainability

- **Global**: SHAP feature importance (summary plots)
- **Local**: SHAP waterfall plots per instance
- **Audit Trail**: Markdown-formatted explanations

---

## Limitations

| Limitation | Mitigation |
|------------|------------|
| Sensitive to distribution shift | Drift monitoring + abstention |
| Requires periodic recalibration | ECE tracking |
| Performance depends on human review quality | Training + guidelines |
| Taiwan 2005 data may not generalize | Validate on new populations |

---

## Ethical Considerations

- ✅ Human oversight required for all high-stakes decisions
- ✅ No automated adverse actions
- ✅ Audit logs maintained for all predictions
- ⚠️ Demographic features (SEX, AGE) included - monitor for disparate impact

---

## Maintenance

| Requirement | Frequency |
|-------------|-----------|
| Recalibration check | Quarterly |
| Drift monitoring | Continuous |
| Full retrain | Annual or on significant shift |
