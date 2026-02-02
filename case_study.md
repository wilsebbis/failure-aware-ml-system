# Case Study: Failure-Aware ML Under Regulatory Constraints

## Background

In regulated industries, ML systems face constraints that typical accuracy-focused projects ignore:

- **Asymmetric error costs** — Missing a positive case has material harm
- **Auditability requirements** — Every decision must be explainable to non-technical stakeholders
- **Limited automation authority** — Humans must remain in the loop
- **Small data reality** — Deep learning is rarely justified

This case study documents designing a classification system that prioritizes **safety, transparency, and operational defensibility** over raw performance metrics.

---

## Problem Statement

Build a credit default risk classifier that:

1. Minimizes false negatives (missed defaults)
2. Provides explainable decisions
3. Routes uncertain cases to human review
4. Explicitly handles failure modes

---

## Design Decisions

### Model Selection

**Chosen:** Shallow XGBoost (max_depth=4)

**Rationale:**
- SHAP explanations are reliable for tree ensembles
- Shallow trees prevent overfitting on limited data
- Regulatory precedent for tree-based models

**Rejected:** Deep learning, complex ensembles

**Why:** Marginal accuracy gains (~1-2%) did not justify:
- Loss of reliable feature attribution
- Calibration instability
- Audit complexity

### Error Handling Strategy

Instead of binary threshold at 0.5:

```
Probability → Decision
< 0.15     → Auto-PASS (low risk)
0.15-0.60  → HUMAN REVIEW (uncertain)
≥ 0.60     → Auto-FLAG (high risk)
```

This ensures uncertain cases receive human judgment rather than forced automation.

---

## Failure Awareness

### Documented Failure Modes

1. **Distribution Shift**
   - Detection: PSI monitoring per feature
   - Response: Increase review rate, alert ML team

2. **Confidence Collapse**
   - Detection: Mean confidence drop tracking
   - Response: Broaden abstention zone

3. **Overconfident Sparse Regions**
   - Detection: Calibration curve analysis
   - Response: Conservative thresholds

### Key Insight

> The most dangerous failure mode is **silent overconfidence.**
> Systems should fail loudly and defer decisions when uncertain.

---

## Results

### What Worked

- ✅ Reduced catastrophic false negatives through recall optimization
- ✅ Improved stakeholder trust through explainability
- ✅ Clear operational boundaries via three-way decisions
- ✅ Proactive failure documentation

### What Didn't

- ⚠️ Recall improvements plateau without more data
- ⚠️ Calibration requires ongoing monitoring infrastructure
- ⚠️ Some reviewers initially distrusted probability outputs

### Future Improvements

- Active learning on abstained samples
- Automated recalibration pipeline
- Drift-aware retraining triggers

---

## Lessons Learned

1. **Accuracy is not the goal** — Minimizing specific error types is
2. **Explainability is non-negotiable** — Model choice follows from audit requirements
3. **Uncertainty is a feature** — Abstention reduces systemic risk
4. **Document failures proactively** — Shows maturity, builds trust

---

## Interview Positioning

This project demonstrates:

| Signal | Evidence |
|--------|----------|
| Real-world ML maturity | Constraints-first design |
| Risk-aware thinking | Documented failure modes |
| Regulatory fluency | Audit-ready documentation |
| Systems design | Human-in-the-loop architecture |
| Engineering judgment | Justified model simplicity |

> "This reads like internal infrastructure, not a student project."

---

## Conclusion

Failure-aware ML design is not a limitation — it is a requirement for responsible deployment in high-stakes domains.

The goal is not the highest score. The goal is a system that survives production review.
