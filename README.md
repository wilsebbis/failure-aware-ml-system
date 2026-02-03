# Failure-Aware ML System

<div align="center">

**Production-Validated Risk Engine | Cascade Architecture | 98%+ System Recall**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://wilsebbis.github.io/failure-aware-ml-system)

[**Documentation**](https://wilsebbis.github.io/failure-aware-ml-system) ·
[Quick Start](#60-second-demo) ·
[Architecture](#cascade-architecture) ·
[Results](#production-results)

</div>

---

## What Is This?

A **high-recall, failure-aware classification system** for regulated environments (Credit Risk, Fraud Detection). Unlike standard ML solutions that maximize ROC-AUC, this system minimizes **catastrophic failures (False Negatives)** while solving the operational bottleneck of manual review.

**Key Achievement:**
Successfully validated on **1.3 Million records** (Lending Club) and complex fraud datasets (IEEE-CIS), achieving **98%+ System Recall** while automating **67-90% of decisions** via a novel Cascade Architecture.

| Decision | Trigger | Outcome |
|----------|---------|---------|
| ✅ **PASS** | Calibrated probability < 0.10 | Auto-approved |
| ⚠️ **REVIEW** | 0.10 ≤ probability < 0.50 | Routed to human reviewer |
| 🚨 **FLAG** | Probability ≥ 0.50 | Auto-blocked |

---

## Production Results

*Verified on Feb 2, 2026 using `--cascade --dynamic-threshold` across 4 datasets.*

| Dataset | Scale | Automation | Defect Rate | Review Load | System Recall |
|---------|-------|------------|-------------|-------------|---------------|
| **Lending Club** | 1.3M rows | **67.2%** 🟢 | 2.01% | 10.8% | **98.7%** |
| **IEEE-CIS Fraud** | 590K rows | **90.0%** 🟢 | 1.74% | 6.7% | **98.4%** |
| **Home Credit** | 307K rows | **75.5%** 🟡 | 4.46% | 24.1% | **96.6%** |
| **UCI Credit** | 30K rows | **29.9%** 🟠 | 7.80% | 57.7% | **97.7%** |

> **Operational Impact:** The Cascade architecture reduces Manual Review workload by **~4x** compared to single-model baselines, making the system operationally viable.

---

## 60-Second Demo

```bash
git clone https://github.com/wilsebbis/failure-aware-ml-system.git
cd failure-aware-ml-system
uv sync
uv run python -m src.main --dataset lending_club --cascade --dynamic-threshold
```

**Expected Output:**
```
[6/7] Applying triage policy...
  Pass rate: 67.2%
  Flag rate: 22.0%
  Review rate: 10.8%
  Pass Queue Defect Rate: 2.01%
  System Recall: 98.7%

[7/7] Testing distribution shift...
✓ No confidence collapse (drop: 0.29%)
```

---

## Cascade Architecture

The system uses a **two-stage Gatekeeper + Specialist** architecture:

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Gatekeeper (Logistic Regression)                   │
│                                                             │
│   65.6% ──► Easy PASS (auto-approved instantly)             │
│   21.1% ──► Easy FLAG (auto-blocked)                        │
│   13.3% ──► Hard Cases (sent to Stage 2)                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Specialist (XGBoost)                               │
│                                                             │
│   Trained ONLY on hard cases (higher positive rate)         │
│   Better calibration for edge cases                         │
│   Isotonic calibration for reliable probabilities           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ DYNAMIC THRESHOLD MONITOR (Safety Valve)                    │
│                                                             │
│   Monitors rolling mean of pass queue risk scores           │
│   If drift detected (mean > baseline + 3σ) → tightens gate  │
│   Maintains safety during fraud bursts / concept drift      │
└─────────────────────────────────────────────────────────────┘
```

### Why Cascade?

| Problem | Single Model | Cascade |
|---------|-------------|---------|
| Home Credit flagged 76% for review | ❌ Undeployable | ✅ Reduced to 24% |
| Fraud scoring 590K transactions | ❌ Heavy XGBoost on all | ✅ 96.4% cleared by fast Logistic |
| Concept drift in IEEE-CIS | ❌ Silent failure | ✅ Dynamic threshold detected & adapted |

---

## Mode Selection

```bash
# Standard (all datasets)
uv run python -m src.main --dataset <name>

# Cascade (recommended)
uv run python -m src.main --dataset <name> --cascade

# Cascade + Dynamic (fraud/drift scenarios)
uv run python -m src.main --dataset <name> --cascade --dynamic-threshold
```

| Dataset | Recommended | Rationale |
|---------|-------------|-----------|
| **UCI Credit** | `--cascade` | Breaks review bottleneck |
| **Home Credit** | `--cascade` | Reduces 76% → 24% review |
| **IEEE-CIS Fraud** | `--cascade --dynamic-threshold` | Handles fraud bursts |
| **Lending Club** | `--cascade` | Scale efficiency (1.3M rows) |

**Tip:** You can always use `--cascade --dynamic-threshold` — the dynamic logic only activates when drift is detected.

---

## Supported Datasets

| Dataset | Skill Demonstrated | Size |
|---------|-------------------|------|
| **UCI Credit** | Baseline (single CSV) | 3MB |
| **Home Credit** | Data engineering (7-table joins) | 800MB |
| **IEEE-CIS Fraud** | ML Ops (temporal splits, 339 features) | 1.2GB |
| **Lending Club** | Business value (IRR optimization) | 1.5GB |

```bash
# Download all datasets
python scripts/download_data.py --dataset all

# Run all with full pipeline
for ds in uci_credit home_credit ieee_cis lending_club; do
  uv run python -m src.main --dataset $ds --cascade --dynamic-threshold
done
```

---

## Project Structure

```
failure-aware-ml-system/
├── src/
│   ├── data/adapters/     # Dataset-specific loaders (Home Credit, IEEE-CIS, etc.)
│   ├── models/            # Cascade classifier, XGBoost, Logistic
│   ├── decision_policy/   # Triage policy, dynamic thresholds
│   ├── evaluation/        # Metrics, calibration
│   └── explainability/    # SHAP explanations
├── docs/                  # MkDocs documentation
├── scripts/               # Data download, SHAP generation
└── data/raw/              # Downloaded datasets
```

---

## Documentation

📖 **[Full Documentation](https://wilsebbis.github.io/failure-aware-ml-system)**

| Section | Description |
|---------|-------------|
| [Mode Selection](docs/guides/mode-selection.md) | When to use Cascade vs Dynamic |
| [Threshold Derivation](docs/concepts/thresholds.md) | How we compute thresholds |
| [Model Card](docs/model_card.md) | Limitations & ethics |
| [Audit Report](docs/audit_report.md) | Production validation results |

---

## License

MIT License - See [LICENSE](LICENSE) for details.
