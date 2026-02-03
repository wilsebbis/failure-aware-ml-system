# Mode Selection Guide

This guide helps you choose the right operational mode for your deployment.

## Quick Reference

*Verified on Feb 2, 2026 using `--cascade --dynamic-threshold`*

| Mode | Best For | Pass Rate | Defect Rate | Review Rate | System Recall |
|------|----------|-----------|-------------|-------------|---------------|
| **Standard** | Max recall, high review capacity | 7.5% | 5.65% | 81.4% | 99.6% |
| **Cascade** | Break review bottleneck | 29.9% - 75.5% | 2.01% - 7.80% | 10.8% - 57.7% | 96.6% - 98.7% |
| **Cascade + Dynamic** | Fraud detection, concept drift | 67.2% - 90.0% | 1.74% - 2.01% | 6.7% - 10.8% | 98.4% - 98.7% |

---

## Production Results by Dataset

| Dataset | Scale | Pass Rate | Defect Rate | Review Rate | System Recall |
|---------|-------|-----------|-------------|-------------|---------------|
| **Lending Club** | 1.3M | 67.2% 🟢 | 2.01% | 10.8% | 98.7% |
| **IEEE-CIS Fraud** | 590K | 90.0% 🟢 | 1.74% | 6.7% | 98.4% |
| **Home Credit** | 307K | 75.5% 🟡 | 4.46% | 24.1% | 96.6% |
| **UCI Credit** | 30K | 29.9% 🟠 | 7.80% | 57.7% | 97.7% |

---

## Standard Mode (Baseline)

```bash
python -m src.main --dataset uci_credit
```

### When to Use
- Simple datasets where model performance is already strong
- Maximum recall is critical and you have ample human reviewers
- Small volume (< 10K decisions/day)

### Limitations
- Sends **70-80% of cases to manual review**
- Inefficient at scale
- No protection against concept drift

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Gatekeeper (Logistic Regression)                   │
│                                                             │
│   96.4% ──► Easy PASS (auto-approved)                       │
│    2.5% ──► Hard Cases (sent to Stage 2)                    │
│    1.1% ──► Easy FLAG (auto-blocked)                        │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Specialist (XGBoost)                               │
│                                                             │
│   Trained ONLY on hard cases from Stage 1                   │
│   Higher positive rate (30% vs 3.5% in full data)           │
│   Better calibration on edge cases                          │
└─────────────────────────────────────────────────────────────┘
```

### Why It Works

1. **Speed**: The Gatekeeper (Logistic Regression) is 100x faster than XGBoost
2. **Focus**: The Specialist only sees the "gray zone" where decisions actually matter
3. **Efficiency**: Reduces manual review by **50-70%** compared to baseline

### Benchmark Results (IEEE-CIS Fraud)

| Metric | Value |
|--------|-------|
| Stage 1 Easy PASS | 96.4% |
| Hard Cases to Stage 2 | 2.5% |
| Pass Rate | 90.0% |
| Review Rate | 6.7% |
| System Recall | 98.4% |

---

## Cascade + Dynamic Mode (Fraud Defense)

```bash
python -m src.main --dataset ieee_cis --cascade --dynamic-threshold
```

### When to Use
- Fraud detection at scale
- Time-series data with concept drift
- Potential for attack scenarios (botnets, coordinated fraud)
- Need to maintain safety during distribution shifts

### How It Works

The Dynamic Threshold adds a **safety layer** on top of Cascade:

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Gatekeeper (Logistic)                              │
│ STAGE 2: Specialist (XGBoost)                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ DYNAMIC THRESHOLD MONITOR                                   │
│                                                             │
│   Monitors rolling mean of PASS queue risk scores           │
│   If mean > baseline + 3σ → SPIKE DETECTED                  │
│   Action: Tightens PASS threshold, forces more to REVIEW    │
└─────────────────────────────────────────────────────────────┘
```

### Why You Need Both

| Component | Provides | Without It |
|-----------|----------|------------|
| **Cascade** | Speed (process millions cheaply) | Every transaction hits expensive XGBoost |
| **Dynamic** | Resilience (survive fraud spikes) | Vulnerable to coordinated attacks |

### Attack Scenario

1. A botnet attack starts
2. The Gatekeeper might still see bots as "safe" (surface-level features look normal)
3. The Dynamic Monitor notices the average risk in the PASS queue is creeping up
4. It automatically **locks the door**, forcing more traffic to REVIEW until the attack subsides

### Benchmark Results (IEEE-CIS Fraud)

| Metric | Value |
|--------|-------|
| Pass Rate | 90.0% |
| Review Rate | 6.7% |
| Flag Rate | 3.3% |
| System Recall | 98.4% |
| Pass Queue Defect Rate | 1.74% |

---

## Dataset-Specific Recommendations

| Dataset | Recommended Mode | Why |
|---------|-----------------|-----|
| **UCI Credit** | Cascade | Breaks the Review Bottleneck |
| **Home Credit** | Cascade | High volume, imbalanced data |
| **IEEE-CIS Fraud** | **Cascade + Dynamic** | Fraud detection, concept drift |
| **Lending Club** | Standard or Cascade | Lower volume, IRR optimization |

---

## Command Reference

```bash
# Standard mode (all datasets)
python -m src.main --dataset <dataset_name>

# Cascade mode
python -m src.main --dataset <dataset_name> --cascade

# Dynamic thresholds only
python -m src.main --dataset <dataset_name> --dynamic-threshold

# Cascade + Dynamic (recommended for fraud)
python -m src.main --dataset <dataset_name> --cascade --dynamic-threshold
```

## Available Datasets

- `uci_credit` - UCI Credit Card Default (default)
- `home_credit` - Home Credit Default Risk
- `ieee_cis` - IEEE-CIS Fraud Detection
- `lending_club` - Lending Club Loan Data
