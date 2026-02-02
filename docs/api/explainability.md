# Explainability API

SHAP-based global and local explanations for audit workflows.

---

## Global Explanations

::: src.explainability.shap_global
    options:
      show_root_heading: true
      members_order: source

---

## Local Explanations

::: src.explainability.shap_local
    options:
      show_root_heading: true
      members_order: source

---

## Usage Example

### Global Summary

```python
from src.explainability.shap_global import generate_global_explanations

results = generate_global_explanations(
    model=xgb_model.model,  # The underlying XGBoost model
    X_sample=X_test.sample(500, random_state=42),
    output_dir="figures/"
)

print(f"Generated: {results['summary_plot']}")
print(f"Top features: {results['top_features']}")
```

### Local Explanation

```python
from src.explainability.shap_local import (
    generate_audit_explanation,
    explain_single_prediction
)

# Markdown format for audit
explanation_md = explain_single_prediction(
    model=xgb_model.model,
    X=X_test,
    instance_idx=42,
    output_path="outputs/case_42.md"
)

# Structured format for integration
audit = generate_audit_explanation(
    shap_values=shap_values,
    X=X_test,
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    expected_value=explainer.expected_value,
    instance_idx=42
)
print(audit)
```

---

## Output Formats

### Markdown (for human reviewers)

```markdown
# Case #42 Explanation

**Decision**: FLAG
**Probability**: 0.73

## Contributing Factors

| Feature | Value | Impact |
|---------|-------|--------|
| PAY_0 | 2 | +0.23 |
| utilization_ratio | 0.96 | +0.12 |
```

### JSON (for integration)

```json
{
  "case_id": 42,
  "decision": "FLAG",
  "probability": 0.73,
  "features": [
    {"name": "PAY_0", "value": 2, "shap": 0.23},
    {"name": "utilization_ratio", "value": 0.96, "shap": 0.12}
  ]
}
```

---

## Related

- [Generating Explanations Guide](../guides/explanations.md)
- [Queue Mechanics](../concepts/queue.md)
