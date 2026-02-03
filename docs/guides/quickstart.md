# Quick Start Guide

Get Failure-Aware ML System running in under 5 minutes.

---

## Prerequisites

- Python 3.11-3.13
- [uv](https://github.com/astral-sh/uv) package manager

---

## Installation

```bash
# Clone the repository
git clone https://github.com/wilsebbis/failure-aware-ml-system.git
cd failure-aware-ml-system

# Install dependencies
uv sync

# Verify installation
uv run python -c "import src; print('✓ Installation successful')"
```

---

## Run the Pipeline

### Default (UCI Credit)

```bash
uv run python -m src.main
```

### With Other Datasets

```bash
# Download professional datasets (requires Kaggle API)
python scripts/download_data.py --dataset all

# Run with Home Credit (7-table joins)
uv run python -m src.main --dataset home_credit

# Run with IEEE-CIS (temporal splits)
uv run python -m src.main --dataset ieee_cis

# Run with Lending Club (IRR optimization)
uv run python -m src.main --dataset lending_club
```

### Expected Output

```
============================================================
FAILURE-AWARE ML SYSTEM PIPELINE
Dataset: uci_credit
============================================================

[1/7] Loading data via adapter...
  Loaded 30,000 samples with 23 features
  Target rate: 22.12%
  Split strategy: stratified

[2/7] Splitting data...
  Using stratified split (train: 21000, val: 3000, test: 6000)

[3/7] Training models...
  Logistic Regression trained
  Random Forest trained
  XGBoost trained

[4/7] Evaluating models...
  Best model by recall: XGBoost

[5/7] Optimizing thresholds...
  Pass threshold: p < 0.05
  Flag threshold: p >= 0.50

[6/7] Applying triage policy...
  Pass rate: 18.0%
  Flag rate: 14.0%
  Review rate: 68.0%
  Pass Queue Defect Rate: 1.80%
  System Recall: 98.2%

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
