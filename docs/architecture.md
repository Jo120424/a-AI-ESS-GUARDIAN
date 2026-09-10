# System Architecture & Pipeline Design

## 1. Overview

This document specifies the end-to-end system architecture for the predictive AI-based Environmental Stress Screening (ESS) research system. The architecture guarantees scientific reproducibility, component-isolated evaluation, and zero future information leakage.

---

## 2. Conceptual Pipeline Architecture

```text
                                  +-----------------------+
                                  |     PUBLIC DATASET    |
                                  |   (NASA PCoE / Open)  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |    DATA VALIDATION    |
                                  | Schema, Units, Ranges |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |     PREPROCESSING     |
                                  | Feature Extraction,   |
                                  | Normalization, Split  |
                                  +-----------+-----------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
        +-------------------------+                       +-------------------------+
        |  TRADITIONAL SCREENING  |                       |   DYNAMIC STATISTICAL   |
        |  * Static Spec Limits   |                       |  * Robust Z-score (MAD) |
        |  * MIL-PRF-62F Rules    |                       |  * Mahalanobis Distance |
        |  * Component Isolated   |                       |  * PCA Residual Energy  |
        +------------+------------+                       +------------+------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                                  +-----------------------+
                                  |    AI/ML SCREENING    |
                                  |  * Isolation Forest   |
                                  |  * One-Class SVM      |
                                  |  * Local Outlier Fact |
                                  |  * Degradation Model  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  SAME UNSEEN TEST SET |
                                  | Component-Isolated CV |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  GROUND TRUTH / PROXY |
                                  | Latent Failure Event  |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  PERFORMANCE ANALYSIS |
                                  | Recall, FNR, Precision|
                                  | AUC, Yield Loss, Cost |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |  EXPLAINABLE RESULTS  |
                                  | Feature Attributions  |
                                  | Anomaly Diagnostics   |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  |    EVIDENCE-BASED     |
                                  |      CONCLUSION       |
                                  +-----------------------+
```

---

## 3. Component Modules

### A. Data Layer (`data/`)
* **`data/raw/`**: Unmodified raw source files downloaded from verified public repositories.
* **`data/processed/`**: Harmonized tabular representations with validated units ($C$ in $\mu\text{F}$, $\text{ESR}$ in $\Omega$, $V$ in $\text{V}$, $t$ in $\text{s}$).
* **`data/metadata/`**: Machine-readable JSON specifications of dataset schema, limits, and environmental conditions.
* **`data/sample/`**: Verified, representative component time-series used for rapid test execution and offline UI exploration.

### B. Preprocessing & Partitioning Layer (`backend/preprocessing/`)
* **Component Grouping:** Explicit partitioning by discrete `component_id` ensures that training and test sets share zero common physical units.
* **Temporal Truncation:** For an experiment evaluated at screening horizon $t_{\text{screen}}$, all measurements where $t > t_{\text{screen}}$ are strictly masked out of the screening inputs.
* **Feature Engineering:** Computes initial baseline offsets ($\Delta C / C_0$), rate-of-change ($\frac{d\,\text{ESR}}{dt}$), and multivariate impedance phase metrics.

### C. Screening Engines (`backend/screening/`)
* **Level 1 (Traditional):** Evaluates static scalar limits ($\text{LSL} \le x \le \text{USL}$). Independent of lot distribution.
* **Level 2 (Dynamic Statistical):** Fits robust covariance estimators and population medians on the lot at $t_{\text{screen}}$ to compute distance metrics.
* **Level 3 (AI/ML):** Trains unsupervised estimators on nominal reference distributions or supervised classifiers on training components to produce continuous anomaly scores.

### D. Evaluation & Metrics Engine (`backend/evaluation/`)
* Consumes predictions from all three screening levels evaluated on the **exact same test partition**.
* Evaluates decisions against the ground-truth latent defect proxy.
* Generates confusion matrices, ROC/PR curves, escape rates, and yield loss figures.

### E. Frontend Research Dashboard (`frontend/`)
* Lightweight, zero-dependency browser application built with standard HTML5, CSS3, and modern JavaScript.
* Communicates with backend REST endpoints to display dataset catalogs, feasibility tables, screening simulation, and model comparisons.
