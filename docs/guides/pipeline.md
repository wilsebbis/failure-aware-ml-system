# Running the Pipeline

Detailed guide to the end-to-end training and evaluation pipeline.

---

## Pipeline Stages

The main pipeline (`src/main.py`) executes 7 stages:

| Stage | Description | Output |
|-------|-------------|--------|
| 1. Load Data | Fetch UCI dataset, validate schema | DataFrame |
| 2. Split | Stratified train/val/test | 70/15/15 split |
| 3. Train Models | Logistic, RF, XGBoost with calibration | Fitted models |
| 4. Evaluate | Recall, FNR, ECE per model | Metrics dict |
| 5. Optimize Thresholds | Find τ⁻, τ⁺ for target FNR | Thresholds |
| 6. Apply Triage | Compute pass/review/flag rates | Statistics |
| 7. Test Drift | Synthetic distribution shift | Collapse check |

---

## Running with Options

```bash
# Default run
uv run python -m src.main

# Skip validation (faster, dev mode)
uv run python -c "
from src.main import run_pipeline
run_pipeline(validate_data=False)
"
```

---

## Configuration

Edit `src/config.py` to adjust:

```python
@dataclass
class DataConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42

@dataclass 
class DecisionConfig:
    threshold_negative: float = 0.15
    threshold_positive: float = 0.60
```

---

## Saved Artifacts

After running, artifacts are saved to:

```
data/
├── raw/credit_default_raw.csv
├── processed/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
models/
├── logistic_baseline.pkl
├── random_forest.pkl
└── xgboost_model.pkl
```

---

## Common Issues

### Schema Validation Error

```
ValueError: Schema validation failed: Missing expected columns
```

**Fix**: Delete cached data and re-run:
```bash
rm data/raw/credit_default_raw.csv
uv run python -m src.main
```

### UV Cache Permission Error

```
error: failed to open file `/Users/.../.cache/uv/sdists-v9/.git`
```

**Fix**: Clear UV cache:
```bash
sudo rm -rf ~/.cache/uv/sdists-v9/.git
```

---

## Next Steps

- [Generating Explanations](explanations.md)
- [API Reference](../api/index.md)
