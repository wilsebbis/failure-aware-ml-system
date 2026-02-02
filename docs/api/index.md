# API Reference

This section documents the public API for Sentinel's major modules.

---

## Modules

### Core Pipeline

| Module | Purpose |
|--------|---------|
| [Decision Policy](decision_policy.md) | Three-way triage logic |
| [Evaluation](evaluation.md) | Metrics, calibration, thresholds |
| [Explainability](explainability.md) | SHAP global & local |
| [Monitoring](monitoring.md) | Drift & confidence detection |

### Data Layer

- `src.data.load` - Data loading and schema validation
- `src.data.preprocess` - Feature scaling and encoding
- `src.data.split` - Stratified train/val/test splits
- `src.features.build_features` - Derived feature engineering

### Model Layer

- `src.models.baseline_logistic` - Calibrated logistic regression
- `src.models.random_forest` - Calibrated random forest
- `src.models.xgboost_model` - Calibrated XGBoost

---

## Quick Reference

### TriagePolicy

```python
from src.decision_policy.triage import TriagePolicy

policy = TriagePolicy(
    threshold_negative=0.15,
    threshold_positive=0.60
)

decisions = policy.decide(probabilities)
# Returns: Series with values PASS, REVIEW, FLAG
```

### MetricsCalculator

```python
from src.evaluation.metrics import MetricsCalculator

calc = MetricsCalculator()
metrics = calc.compute_all(y_true, y_pred, y_proba)
# Returns: dict with recall, fnr, ece, etc.
```

### Explainer

```python
from src.explainability.shap_local import explain_single_prediction

md = explain_single_prediction(model, X, idx=42)
# Returns: Markdown string with SHAP explanation
```

---

## Using mkdocstrings

Each API page uses mkdocstrings to render docstrings automatically:

```markdown
::: src.decision_policy.triage.TriagePolicy
```

This pulls the class docstring, methods, and type hints directly from the source code.
