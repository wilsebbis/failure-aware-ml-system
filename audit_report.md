# Internal Model Audit Report

## System Information

| Field | Value |
|-------|-------|
| **System Name** | Failure-Aware Credit Risk Classifier |
| **Audit Date** | 2026-01-15 |
| **Auditor** | ML Governance Team |
| **Version** | 1.0.0 |

---

## Executive Summary

**Overall Risk Rating: MODERATE (Controlled)**

The system demonstrates strong alignment with regulated ML best practices. Risks are explicitly documented and mitigated through human review and conservative decision policies.

---

## Model Choice Review

| Criterion | Assessment |
|-----------|------------|
| Interpretability | ✅ Tree-based models with SHAP |
| Deep learning exclusion | ✅ Justified due to data size and audit requirements |
| Feature attribution | ✅ Available for all predictions |

**Assessment: ACCEPTABLE**

---

## Error Asymmetry Handling

| Requirement | Implementation |
|-------------|----------------|
| FN prioritization | ✅ Primary optimization metric |
| Threshold tuning | ✅ Recall-focused optimization |
| FP management | ✅ Routed to human review |

**Assessment: STRONG**

---

## Explainability & Auditability

| Requirement | Implementation |
|-------------|----------------|
| Global explanations | ✅ SHAP summary plots |
| Local explanations | ✅ Instance-level waterfall plots |
| Feature behavior validation | ✅ Domain-aligned checks |
| Reproducibility | ✅ Seeded random states |

**Assessment: STRONG**

---

## Failure Mode Analysis

### Identified Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Distribution shift | Medium | High | PSI monitoring |
| Calibration drift | Medium | Medium | ECE tracking |
| Sparse-region overconfidence | Low | High | Abstention policy |
| Label noise amplification | Low | Medium | Regularization |

### Mitigation Effectiveness

- ✅ Abstention policy implemented
- ✅ Drift monitoring operational
- ✅ Human review escalation defined

**Assessment: ACCEPTABLE with ongoing monitoring**

---

## Human Oversight

| Requirement | Implementation |
|-------------|----------------|
| Clear escalation criteria | ✅ Three-way decision policy |
| Documented review workflows | ✅ Triage documentation |
| Model as decision support | ✅ Not sole decision-maker |
| Override capability | ✅ Human can override any decision |

**Assessment: STRONG**

---

## Data Governance

| Requirement | Status |
|-------------|--------|
| Data lineage documented | ✅ UCI dataset sourced |
| Schema validation | ✅ Automated checks |
| Feature documentation | ✅ Audit justifications provided |
| PII handling | ⚠️ Demographic features present - monitor |

**Assessment: ACCEPTABLE**

---

## Recommendation

**APPROVED for limited-scope deployment** with the following conditions:

1. ☐ Ongoing drift monitoring (weekly PSI reports)
2. ☐ Quarterly recalibration review
3. ☐ Human reviewer training before go-live
4. ☐ Disparate impact analysis on demographic features
5. ☐ 90-day post-deployment review

---

## Sign-off

| Role | Name | Date |
|------|------|------|
| Model Owner | [TBD] | |
| Risk Officer | [TBD] | |
| Compliance | [TBD] | |
