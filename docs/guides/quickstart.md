# Quick Start Guide

Get Sentinel running in under 5 minutes.

---

## Prerequisites

- Python 3.11-3.13
- [uv](https://github.com/astral-sh/uv) package manager

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sentinel-risk-engine.git
cd sentinel-risk-engine

# Install dependencies
uv sync

# Verify installation
uv run python -c "import src; print('✓ Installation successful')"
```

---

## Run the Pipeline

```bash
uv run python -m src.main
```

Expected output:
```
============================================================
FAILURE-AWARE ML SYSTEM PIPELINE
============================================================

[1/7] Loading data...
  Loaded data: 30000 rows, 24 columns
  Target distribution: {0: 0.78, 1: 0.22}

[2/7] Splitting data...
  Train: 21000 samples, 22.12% positive

[3/7] Training models...
  Logistic Regression trained
  Random Forest trained (OOB: 0.77)
  XGBoost trained

[4/7] Evaluating models...
  Best model by recall: XGBoost (31.4%)

[5/7] Optimizing thresholds...

[6/7] Applying triage policy...
  Pass rate: 46.8%
  Flag rate: 7.2%
  Review rate: 46.0%

[7/7] Testing distribution shift...
  ✓ No confidence collapse

============================================================
PIPELINE COMPLETE
============================================================
```

---

## Generate SHAP Visualizations

```bash
uv run python scripts/generate_shap_figures.py
```

Output saved to `figures/`:
- `shap_summary.png`
- `shap_waterfall_example.png`
- `shap_importance_bar.png`
- `shap_dependence_top.png`

---

## Score a Single Case

```python
from src.data.load import load_data
from src.data.preprocess import DataPreprocessor
from src.features.build_features import build_features
from src.models.xgboost_model import XGBoostModel
from src.decision_policy.triage import TriagePolicy

# Load pre-trained model (or train new)
model = XGBoostModel.load()

# Prepare sample
sample = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 32,
    "PAY_0": 2,
    # ... rest of features
}

# Preprocess and predict
preprocessor = DataPreprocessor.load()
X = preprocessor.transform(pd.DataFrame([sample]))
X = build_features(X)

proba = model.predict_proba(X)[:, 1]

# Apply triage
policy = TriagePolicy()
decision = policy.decide_single(proba[0])

print(f"Decision: {decision.decision}")  # PASS, REVIEW, or FLAG
print(f"Probability: {proba[0]:.2f}")
```

---

## Next Steps

- [Running the Full Pipeline](pipeline.md)
- [Generating Explanations](explanations.md)
- [API Reference](../api/index.md)
