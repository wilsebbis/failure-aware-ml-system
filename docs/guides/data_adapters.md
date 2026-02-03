# Data Adapters

The Failure-Aware ML System uses an **Adapter Pattern** to support multiple professional datasets with a unified training pipeline.

---

## Why Adapters?

Each dataset has unique:

- **Schema**: Different column names and types
- **Feature Engineering**: Domain-specific transformations
- **Split Strategy**: Random vs temporal
- **Target Definition**: Binary vs continuous

The adapter pattern abstracts these differences, letting the pipeline operate identically regardless of data source.

---

## Available Adapters

### UCI Credit Card Default

| Property | Value |
|----------|-------|
| **Size** | 30,000 samples, 24 features |
| **Target** | Binary (default next month) |
| **Split** | Stratified random |
| **Skill** | Baseline |

```bash
uv run python -m src.main --dataset uci_credit
```

---

### Home Credit Default Risk

| Property | Value |
|----------|-------|
| **Size** | 300,000+ samples, 100+ features |
| **Target** | Binary (loan default) |
| **Split** | Stratified random |
| **Skill** | Data Engineering |

**Data engineering flex**: Joins 7 relational tables into a single feature vector:

```
application_train.csv   # Main table
    ├── bureau.csv           # Credit bureau records
    │   └── bureau_balance.csv   # Monthly balances
    ├── previous_application.csv # Past loan apps
    ├── POS_CASH_balance.csv     # POS loan history
    ├── credit_card_balance.csv  # CC history
    └── installments_payments.csv # Payment records
```

```python
from src.data import get_adapter

adapter = get_adapter("home_credit", {"path": "data/raw/home_credit"})
adapter.load_raw()       # Loads and joins all 7 tables
adapter.feature_engineer()  # Aggregations, ratios, etc.
X, y = adapter.get_features_and_target()
```

---

### IEEE-CIS Fraud Detection

| Property | Value |
|----------|-------|
| **Size** | 500,000+ transactions |
| **Target** | Binary (is fraud) |
| **Split** | Temporal (by TransactionDT) |
| **Skill** | ML Ops |

**ML Ops flex**: Uses temporal splits to prevent data leakage.

```python
adapter = get_adapter("ieee_cis", {"path": "data/raw/ieee_cis"})
adapter.load_raw()
adapter.feature_engineer()

# Get time-based split indices
train_idx, test_idx = adapter.get_temporal_split_indices()
```

Features 339 anonymized "V" features mimicking real bank data. The adapter creates V-feature group aggregations automatically.

---

### Lending Club

| Property | Value |
|----------|-------|
| **Size** | 2M+ loans |
| **Target** | Continuous (IRR) or Binary (profitable) |
| **Split** | Temporal (by issue date) |
| **Skill** | Business Value |

**Business value flex**: Optimizes for profit, not just accuracy.

```python
# Continuous IRR target (default)
adapter = get_adapter("lending_club", {
    "path": "data/raw/lending_club",
    "target_mode": "irr"  # or "binary" or "profitable"
})

adapter.load_raw()
adapter.feature_engineer()
X, y = adapter.get_features_and_target()

print(f"IRR range: {y.min():.2%} to {y.max():.2%}")
```

---

## Adding a New Adapter

1. **Create adapter class** in `src/data/adapters/`:

```python
from .base import RiskDataAdapter

class MyDataAdapter(RiskDataAdapter):
    def load_raw(self) -> None:
        self.df = pd.read_csv(self.config["path"])
        
    def feature_engineer(self) -> None:
        # Domain-specific transformations
        pass
        
    def get_features_and_target(self):
        return self.df.drop("target"), self.df["target"]
        
    def get_split_strategy(self) -> str:
        return "stratified"  # or "temporal"
```

2. **Register in factory** (`src/data/factory.py`):

```python
from .adapters.my_data import MyDataAdapter

ADAPTER_REGISTRY["my_data"] = MyDataAdapter
```

3. **Create config** (`src/config/my_data.yaml`):

```yaml
dataset:
  name: "my_data"
  path: "data/raw/my_data.csv"
  
triage:
  auto_approve_threshold: 0.05
  auto_decline_threshold: 0.50
```

---

## Configuration Files

Each adapter has a YAML config in `src/config/`:

```yaml
# Example: src/config/ieee_cis.yaml
dataset:
  name: "ieee_cis"
  path: "data/raw/ieee_cis"
  temporal_col: "TransactionDT"
  train_time_ratio: 0.8

triage:
  auto_approve_threshold: 0.02
  auto_decline_threshold: 0.30

split:
  strategy: "temporal"

model:
  hyperparameters:
    n_estimators: 300
    scale_pos_weight: 20  # Handle class imbalance
```
