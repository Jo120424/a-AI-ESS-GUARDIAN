# AI-ESS GUARDIAN
## Predictive AI-Driven Anomaly Detection in Component Burn-In & Environmental Stress Screening

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-100%20passed-success.svg)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Prototype Status](https://img.shields.io/badge/status-Research%20Prototype-orange.svg)](#17-limitations--domain-boundaries)

---

## 1. Project Title
**AI-ESS GUARDIAN:** AI-Driven Dynamic Anomaly Detection and Degradation Drift Forecasting in Electronic Component Environmental Stress Screening (ESS) and Burn-In Testing.

---

## 2. Problem Statement
High-reliability electronic components (microcontrollers, ASICs, power semiconductors, capacitors, and RF hybrids) intended for aerospace, satellite, and mission-critical systems undergo Environmental Stress Screening (ESS) and Burn-In testing at elevated temperatures (85°C to 125°C) and voltage overstress.

Traditional screening relies almost universally on **static parametric go/no-go datasheet limits** (MIL-PRF-62F, MIL-STD-883, JEDEC JESD22). Under this static paradigm, a latent-defect component that remains within datasheet limits during early testing is declared qualified:

```text
Traditional Screening Example:
Manufacturing Lot Median Leakage:    10.0 µA
Component Measured Leakage:          45.0 µA
Datasheet Maximum Allowable Limit:   50.0 µA

Evaluation: 45.0 µA <= 50.0 µA  ==>  PASS (Flight Qualified)
```
**The Failure Mode:** The component is significantly abnormal relative to its manufacturing cohort (+4.5x lot median, robust Z > +6.0) and exhibits steep kinetic drift. Once installed into a flight vehicle or satellite payload, continued operational stress causes premature dielectric breakdown, thermal runaway, or catastrophic mission loss.

---

## 3. Why Traditional Screening is Insufficient
1. **Zero Cohort Context:** Static screening evaluates each device in complete isolation, oblivious to lot-to-lot baseline shifts or extreme intra-lot outlier dispersion.
2. **Blind to Degradation Kinetics:** Static limits check single instantaneous snapshots ($t_{\text{screen}}$) rather than time-series degradation velocity ($\frac{d\,\text{Param}}{dt}$).
3. **High Latent Defect Escape Rate:** In our empirical benchmarks, traditional static screening exhibited an **85.7% to 100.0% false pass rate (defect escape)** during early burn-in horizons.

---

## 4. Proposed AI Solution
**AI-ESS GUARDIAN** transforms screening from a reactive post-failure limit-check into an active, lot-aware predictive filter. The system operates on two core machine learning modules coupled with dynamic safety envelopes and a low-false-negative fusion engine:

```text
RAW BURN-IN TELEMETRY (0h, 24h, 96h, 168h)
                  │
                  ▼
   ROBUST DATA PIPELINE & PREPROCESSING
   (Schema validation, missing imputation, impossible-value clipping)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│     MODULE A     │ │     MODULE B     │
│ Dynamic Outlier  │ │ Time-Series Drift│
│    Detection     │ │   Prediction     │
│ (Median/MAD,     │ │ (XGBoost / HistGB│
│  iForest, LOF)   │ │  Quantile CIs)   │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         └─────────┬──────────┘
                   ▼
       DYNAMIC SAFETY ENVELOPE
       (Data-driven prototype threshold vs. engineering limits)
                   │
                   ▼
     UNIFIED DECISION FUSION ENGINE
     (PASS / REVIEW / REJECT — Low False Negative Priority)
                   │
                   ▼
    EXPLAINABLE QA INSPECTION REPORT
    (Why? Structured reason codes, deviation multipliers, CSV export)
```

---

## 5. Architecture
The project follows a clean, modular, Windows-friendly architecture:

```text
isro/
├── app/
│   └── dashboard.py               # Streamlit Aerospace QA Screening Application
├── backend/
│   ├── data_pipeline/
│   │   ├── sih_pipeline.py        # Robust CSV/Excel ingestion & validation
│   │   ├── preprocess.py          # NASA physical dataset preprocessor
│   │   └── profile_dataset.py     # Dataset quality profiler
│   ├── screening/
│   │   ├── module_a.py            # Dynamic outlier detection (MAD, iForest, LOF)
│   │   ├── module_b.py            # 168h time-series drift forecaster (XGBoost)
│   │   ├── safety_envelope.py     # Dynamic slope & prototype safety envelope
│   │   ├── decision_engine.py     # Unified low-false-negative fusion engine
│   │   ├── explainer.py           # Reusable QA explanation generator
│   │   ├── artifact_manager.py    # Model persistence & loading (models/)
│   │   └── train_and_persist.py   # Training & benchmark evaluation orchestrator
│   ├── experiments/               # NASA physical comparative framework
│   ├── validation/                # Forensic error audit & ablation layer
│   └── server.py                  # Zero-dependency HTTP/REST API server
├── data/
│   ├── raw/                       # NASA Ames PCoE raw datasets
│   ├── processed/                 # Tidy relational panel datasets
│   └── synthetic/                 # Multi-checkpoint semiconductor burn-in dataset
├── models/                        # Serialized model artifacts (joblib, json)
├── tests/                         # 100 passing automated unit tests
├── requirements.txt               # Verified dependency configuration
├── PROJECT_STATUS.md              # Complete audit and continuation record
└── README.md                      # Comprehensive project documentation
```

---

## 6. Module A: Dynamic Outlier Detection
* **Purpose:** Detects components that deviate significantly from their manufacturing lot or healthy reference peers, even while remaining within datasheet boundaries.
* **Methods:**
  - Robust Univariate Statistics: Median Absolute Deviation (MAD) robust Z-scores:
    $$Z_{\text{MAD}} = \frac{x_i - \text{Median}_{\text{lot}}(X)}{1.4826 \times \text{MAD}_{\text{lot}}(X)}$$
  - Unsupervised Multivariate Partitioning: **Isolation Forest (iForest)** isolating non-linear geometric anomalies across parameter sub-spaces.
  - Density-Based Screening: **Local Outlier Factor (LOF)** identifying sparse local neighborhood outliers.
* **Features:** Raw telemetry, lot-relative multipliers ($\frac{x}{\text{Median}}$), backward drift velocities ($\frac{\Delta x}{\Delta t}$), cross-parameter interaction ratios ($\frac{\text{Leakage}}{\text{Iddq}}$).
* **Leak-Free Guarantee:** Lot statistics and scalers are fitted strictly on reference training partitions without test contamination.

---

## 7. Module B: Time-Series Drift Prediction
* **Purpose:** Uses early checkpoints (Value_0h, Value_24h) to forecast long-term degradation (Value_168h).
* **Model:** Gradient Boosted Decision Trees (**XGBoost** / `HistGradientBoostingRegressor`) with dual **Quantile Regressors** ($\alpha=0.10, \alpha=0.90$) outputting an 80% confidence interval.
* **Outputs:**
  - Predicted 168h parametric value ($\hat{y}_{168\text{h}}$)
  - 80% Prediction Interval ($\text{CI}_{\text{lower}}, \text{CI}_{\text{upper}}$)
  - Total projected drift amount and drift percentage
  - Continuous drift risk score $[0.0, 1.0]$.

---

## 8. Dynamic Safety Slope & Safety Envelope (Phase 4)
Rather than solely checking static datasheet ceilings, the system learns dynamic degradation kinetic boundaries from reference populations:

$$\text{early\_slope} = \frac{\text{Value}_{24\text{h}} - \text{Value}_{0\text{h}}}{24.0}, \quad \text{predicted\_slope} = \frac{\hat{\text{Value}}_{168\text{h}} - \text{Value}_{24\text{h}}}{144.0}$$

### Clear Distinction:
* **DATA-DRIVEN PROTOTYPE THRESHOLD:** 95th percentile upper confidence bound estimated empirically from healthy reference cohorts (e.g. 16.56 µA). Provides early operational margin.
* **ENGINEERING / CERTIFICATION LIMIT:** Absolute maximum manufacturer specification limit from datasheet (e.g. 50.00 µA). Non-negotiable hard ceiling.

---

## 9. Decision Engine & Low-False-Negative Priority
Fuses multi-paradigm evidence into exactly one final disposition:
* **PASS (Flight Qualified):** Normal relative behavior ($< 2.0\text{x}$ median) and safe future drift trajectory.
* **REVIEW (Hold / Extended Burn-In):** Moderate lot anomaly ($2.0\text{x} \dots 3.5\text{x}$ median), borderline drift, or prototype envelope breach.
* **REJECT (Mandatory Scrap):** Severe anomaly ($> 3.5\text{x}$ lot median, Robust $Z > +4.5$), forecasted spec breach, or absolute datasheet violation.

**Safety Mandate:** Prioritizes **LOW FALSE NEGATIVES** (zero defect escape into flight vehicles) over nominal accuracy.

---

## 10. Explainability Engine (Phase 6)
Every disposition is accompanied by a transparent QA inspection audit card:

```text
FINAL DECISION: REJECT
Why?
✓ Within absolute datasheet specification (45.00 µA <= 50.00 µA)
⚠ 4.5x deviation from manufacturing lot median (Lot Median: 10.00 µA)
⚠ Early leakage drift slope (0.208 µA/h) exceeds healthy-lot envelope (0.050 µA/h)
⚠ Predicted 168h value (48.50 µA) approaches/exceeds prototype safety threshold (16.56 µA)
```

---

## 11. Datasets
1. **NASA Capacitor Electrical Stress Degradation Dataset (`data/raw/`, `data/processed/`):**
   - Source: NASA Ames Research Center Prognostics Center of Excellence (PCoE).
   - Wet Tantalum Electrolytic Capacitors under 10V continuous electrical overstress across 11 inspection points up to 194 hours.
   - Verified SHA-256: `9db651a10f92...`
2. **Semiconductor Burn-In Benchmark Dataset (`data/synthetic/`):**
   - 90 components across 3 manufacturing lots (`LOT_A`, `LOT_B`, `LOT_C`).
   - Checkpoints: 0h, 24h, 96h, 168h.
   - Parameters: `leakage_current_ua`, `iddq_ma`, `propagation_delay_ns`, `chamber_temp_c`.
   - Labeled archetypes: Healthy stable, mild drift, latent lot outliers (SIH case), accelerating drift, step jumps, and absolute violations.

---

## 12. Installation
The environment runs on standard Python 3.10+ (tested on Python 3.13 and 3.14 on Windows):

```powershell
# Navigate to repository root
cd C:\Users\opppo\Downloads\isro

# Install exact verified dependencies
pip install -r requirements.txt
```

---

## 13. How to Train Models & Persist Artifacts
To execute the leak-free training pipeline and serialize all models into `models/`:

```powershell
python backend/screening/train_and_persist.py
```

Outputs serialized artifacts into `models/`:
- `module_a_detector.joblib`
- `module_b_predictor.joblib`
- `safety_envelope.json`
- `model_metadata.json`
- `comparison_benchmark.csv`

---

## 14. How to Launch Dashboards
The project provides two complementary user interfaces:

### A. Primary Interactive Streamlit QA Application (Recommended)
```powershell
streamlit run app/dashboard.py
```
Open browser at: `http://localhost:8501`

### B. Lightweight Zero-Dependency REST API & Research Server
```powershell
python backend/server.py 8000
```
Open browser at: `http://localhost:8000`

---

## 15. How to Run Automated Unit Tests
To run all 100 passing automated tests covering all 5 mandatory test cases (A–E), leakage prevention, and data pipeline robustness:

```powershell
python -m unittest discover tests/
```

---

## 16. Example 2-Minute SIH Hackathon Demo
1. Launch the Streamlit dashboard: `streamlit run app/dashboard.py`.
2. Toggle the **"🚀 2-Minute SIH Hackathon Demo Mode"** checkbox in the sidebar.
3. Observe the side-by-side comparison:
   - **Traditional Screening Card:** Shows measured leakage = 45 µA vs 50 µA limit $\rightarrow$ **PASS** (Defect escapes!).
   - **AI-ESS Guardian Card:** Shows 4.5x lot median, steep early slope, forecasted 168h drift $\rightarrow$ **REJECT** (Defect intercepted 144 hours early!).
4. Switch to **Section 4 (Trajectory & Envelopes)** to show the dynamic safety envelope and 80% prediction interval.
5. Click **"📥 Export Screening Report (CSV)"** to demonstrate operational compliance reporting.

---

## 17. Validation Methodology & Comparative Results
All paradigms were evaluated out-of-fold on unseen test components:

| Screening Paradigm | Defect Recall | Escape Rate (FNR) | Precision | F1-Score | Yield Loss (FPR) | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE 1: Traditional Static Limits** | 14.3% | 85.7% | 100.0% | 0.2500 | **0.0%** | 0.8635 |
| **BASELINE 2: Dynamic Statistical (MAD)** | **100.0%** | **0.0%** | 100.0% | 1.0000 | **0.0%** | 1.0000 |
| **MODEL: ML Anomaly (Isolation Forest)** | 76.2% | 23.8% | 100.0% | 0.8649 | **0.0%** | 1.0000 |
| **HYBRID: AI-ESS GUARDIAN (Unified)** | **100.0%** | **0.0%** | **100.0%** | **1.0000** | **0.0%** | **1.0000** |

*Key Finding:* AI-ESS GUARDIAN completely eliminates defect escape (**FNR drops from 85.7% down to 0.0%**) while preserving 100% precision.

---

## 18. Limitations & Domain Boundaries

> [!IMPORTANT]
> **RESEARCH & HACKATHON PROTOTYPE BOUNDARY DISCLAIMER**
> 1. This system is a **research and engineering prototype** developed for comparative algorithm evaluation and the Smart India Hackathon.
> 2. It does **NOT** constitute space agency (ISRO / NASA) flight qualification or production aerospace certification.
> 3. Production flight deployment requires certified automated test equipment (ATE), physical environmental chamber qualification (MIL-STD-883 / MIL-STD-202), radiation hardness assurance (RHA), complete traceability, and domain approval by designated Quality Assurance Review Boards.
> 4. Synthetic demonstration datasets model standard industrial failure dynamics for stress-testing and demonstration; they do not replace qualification test data on physical flight lots.

---

## 19. Future Improvements
1. **Directional Anomaly Bounding:** Enhancing unsupervised kernels with one-sided directional constraints to eliminate survivor yield loss.
2. **Physics-Informed Neural Networks (PINNs):** Embedding Arrhenius and Eyring thermal acceleration models into drift regression loss functions.
3. **Automated Test Equipment (ATE) Live Ingestion:** Real-time GPIB / VISA instrument streaming during active thermal chamber cycling.

---

## 20. Authors & Provenance
* **Project:** AI-ESS GUARDIAN (ISRO SIH Problem Statement)
* **Location:** `C:\Users\opppo\Downloads\isro`
* **Status:** 100% Working, Demo-Ready, Fully Tested.
