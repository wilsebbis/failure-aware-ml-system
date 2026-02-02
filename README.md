# Sentinel: High-Recall Risk Assessment Engine

<div align="center">

**Audit-First Classification | Human-in-the-Loop | Asymmetric Error Optimization**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://yourusername.github.io/sentinel-risk-engine)

[**Documentation**](https://yourusername.github.io/sentinel-risk-engine) ·
[Quick Start](#60-second-demo) ·
[Architecture](#architecture)

</div>

---

## What Is This?

Sentinel is a **high-recall risk scoring engine** for regulated environments. It takes tabular input data and outputs a three-way decision:

| Decision | Trigger | Outcome |
|----------|---------|---------|
| ✅ **PASS** | Calibrated probability < 0.15 | Auto-approved |
| ⚠️ **REVIEW** | 0.15 ≤ probability < 0.60 | Routed to human reviewer |
| 🚨 **FLAG** | Probability ≥ 0.60 | Auto-blocked |

This is **not** a fully automated decision system. It routes uncertain cases to humans rather than forcing a bad automated decision.

---

## 60-Second Demo

### Sample Payload

```json
{
  "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 32,
  "PAY_0": 2, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
  "BILL_AMT1": 48000, "BILL_AMT2": 45000, "BILL_AMT3": 42000,
  "BILL_AMT4": 40000, "BILL_AMT5": 38000, "BILL_AMT6": 36000,
  "PAY_AMT1": 2000, "PAY_AMT2": 1500, "PAY_AMT3": 1000,
  "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000
}
```

### Expected Output

```
Decision: FLAG
Calibrated Probability: 0.73

Top Contributing Features:
  +0.23  PAY_0 = 2 (payment delay)
  +0.12  utilization_ratio = 0.96
  +0.08  max_delay = 2

Audit Artifact: /outputs/case_12345_explanation.md
```

### Run It

```bash
git clone https://github.com/yourusername/sentinel-risk-engine.git
cd sentinel-risk-engine
uv sync
uv run python -m src.main
```

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Data Ingestion"]
        A[Payload] -->|Schema Check| B{Valid?}
        B -->|Yes| C[Features]
    end
    
    subgraph Model["Inference"]
        C --> D[XGBoost]
        D --> E[Isotonic Calibration]
        E --> F["P(default)"]
    end
    
    subgraph Policy["Triage"]
        F --> G{Thresholds}
        G -->|"p < 0.15"| H[PASS]
        G -->|"0.15–0.60"| I[REVIEW]
        G -->|"p >= 0.60"| J[FLAG]
    end
    
    subgraph Audit["Audit"]
        I --> K[SHAP Explanation]
        J --> K
    end
```

---

## Key Metrics

| Model | Recall | FNR | ECE |
|-------|--------|-----|-----|
| XGBoost | 31.4% | 68.6% | 0.016 |
| Random Forest | 29.0% | 71.0% | 0.012 |
| Logistic | 23.8% | 76.2% | 0.015 |

| Triage | Rate |
|--------|------|
| PASS | 46.8% |
| REVIEW | 46.0% |
| FLAG | 7.2% |

---

## Explainability

Sentinel generates explainability artifacts that **can support compliance workflows** (GDPR, FCRA). These are audit aids, not legal compliance by themselves.

![SHAP Summary](figures/shap_summary.png)

See [Generating Explanations](docs/guides/explanations.md) for details.

---

## Documentation

📖 **[Full Documentation](https://yourusername.github.io/sentinel-risk-engine)**

| Section | Description |
|---------|-------------|
| [Confidence & Calibration](docs/concepts/confidence.md) | Precise metric definitions |
| [Threshold Derivation](docs/concepts/thresholds.md) | How we compute thresholds |
| [Queue Mechanics](docs/concepts/queue.md) | Feedback loops & retraining |
| [API Reference](docs/api/index.md) | Module documentation |
| [Model Card](docs/model_card.md) | Limitations & ethics |

---

## Project Structure

```
sentinel-risk-engine/
├── src/                    # Source code
│   ├── data/               # Loading, preprocessing
│   ├── models/             # Logistic, RF, XGBoost
│   ├── evaluation/         # Metrics, calibration
│   ├── explainability/     # SHAP explanations
│   ├── decision_policy/    # Three-way triage
│   └── monitoring/         # Drift detection
├── docs/                   # MkDocs documentation
├── figures/                # SHAP visualizations
└── tests/                  # Unit tests
```

---

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Generate SHAP figures
uv run python scripts/generate_shap_figures.py

# Build docs locally
mkdocs serve
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.
