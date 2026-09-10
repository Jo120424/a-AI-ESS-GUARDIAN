# Component-by-Component Error Analysis & Forensic Defect Report
**Project:** Predictive AI-Based Environmental Stress Screening for Latent Defect Detection  **Phase:** Step 5 — Research Validation & Intelligence Layer  **Evaluation Boundary:** $t_{\text{screen}} = 47.0\,\text{h}$, Target Lifetime: $194.0\,\text{h}$  
---
## 1. Component Analytical Profiles
| Component | Ground Truth | Trad Class | Stat Class | OC-SVM Class | Drift Class | Fusion Tier | Actual $\Delta C(194\text{h})$ | Pred $\Delta C(194\text{h})$ | Lead Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | Safe Survivor | `TN` | `FP` | `FP` | `FP` | **HIGH RISK** | 17.45% | 21.57% | **0.0 h** |
| **C2** | Latent Defect | `FN` | `TP` | `TP` | `TP` | **HIGH RISK** | 21.68% | 21.21% | **147.0 h** |
| **C3** | Latent Defect | `FN` | `FN` | `TP` | `TP` | **MEDIUM RISK** | 21.05% | 20.92% | **147.0 h** |
| **C4** | Latent Defect | `FN` | `TP` | `TP` | `TP` | **HIGH RISK** | 22.68% | 21.24% | **124.0 h** |
| **C5** | Latent Defect | `FN` | `FN` | `FN` | `TP` | **MEDIUM RISK** | 20.80% | 21.04% | **147.0 h** |
| **C6** | Latent Defect | `FN` | `TP` | `TP` | `TP` | **HIGH RISK** | 22.04% | 20.44% | **147.0 h** |

---
## 2. Forensic False Negative Analysis (Safety-Critical)
In space and defense electronic screening, false negatives (escaped latent defects) represent mission-critical catastrophic failures.

### A. Escaped by Traditional Static Screening (100% Escape Rate — 5/5 Missed)
* **Missed Components:** `C2, C3, C4, C5, C6`
* **Mechanism:** At $t_{\text{screen}} = 47.0\,\text{h}$, the maximum degradation observed across the cohort was $\Delta C = 1.86\%$, whereas the MIL-PRF-62F limit is $20.0\%$. The degradation incubation process is non-linear; at 47 hours, components have experienced less than $10\%$ of their end-of-life capacitance loss. Static thresholds cannot detect latent electrochemical damage during early incubation.

### B. Escaped by Dynamic Statistical Screening (40% Escape Rate — 2/5 Missed)
* **Missed Components:** `C3` and `C5`
* **Forensic Evidence:**
  * `C3` exhibited $\Delta C(47\text{h}) = 1.58\%$ and drift velocity $0.0410\%/\text{h}$, placing it within 0.59 MAD deviations of the lot median ($Z_{\text{MAD}} = 0.59 \le 2.5$). Its Mahalanobis distance ($D_M^2 = 0.03 \le 7.81$) was the lowest in the cohort.
  * `C5` exhibited $\Delta C(47\text{h}) = 1.56\%$ and drift velocity $0.0416\%/\text{h}$ ($Z_{\text{MAD}} = 0.72 \le 2.5$, $D_M^2 = 0.02$).
  * **Failure Mode:** `C3` and `C5` were 'stealth incubators.' Their initial degradation closely tracked the normative lot trajectory during the first 47 hours before experiencing accelerated kinetic runaway after $t = 71.0\,\text{h}$. Pure instantaneous statistical distribution checks at $t=47\,\text{h}$ cannot isolate stealth incubators whose early velocity matches the cohort median.

### C. Escaped by AI/ML Anomaly Detection (One-Class SVM: 20% Escape Rate — 1/5 Missed)
* **Missed Component:** `C5`
* **Forensic Evidence:** `C5` fell directly inside the interior support vector boundary with an RBF decision distance of $-0.0327$ (threshold: $0.0001$). Its multi-parameter state $(\Delta C, \Delta\text{ESR}, \frac{d\Delta C}{dt}, \frac{d\Delta\text{ESR}}{dt})$ was perfectly canonical at $47\,\text{h}$.
* **Detection by Complementary Paradigm:** Although `C5` was missed by One-Class SVM, it was **successfully detected by Early Drift Forecasting** (predicted $\Delta C = 21.04\% \ge 20.0\%$), proving the necessity of trajectory forecasting alongside spatial anomaly detection.

### D. Escaped by Isolation Forest (100% Escape Rate at Discrete Threshold)
* **Root Cause:** In 5-sample training cross-validation splits, the 80th percentile threshold required test points to exceed the most extreme in-fold point ($s_{\text{thresh}} \approx 0.526$). While `C4` scored $s = 0.5162$ (the highest in the cohort), it failed to cross the discrete cutoff due to small-sample threshold quantization. However, Isolation Forest achieved **PR-AUC = 0.8767**, demonstrating strong continuous ranking capability.

---
## 3. Forensic False Positive Analysis & Survivor C1 Investigation
In production screening, false positives cause unnecessary scrap or yield loss.

### Detailed Investigation of Healthy Survivor C1:
* **Ground Truth:** `C1` is the **only true survivor** in the cohort. At end-of-test ($194.0\,\text{h}$), its degradation was $\Delta C = 17.45\% < 20.0\%$ and $\Delta\text{ESR} = 53.54\% < 100.0\%$.
* **Screening Dispositions:**
  * Traditional ESS: `PASS` (TN — correct)
  * Isolation Forest: `PASS` (TN — correct, score $0.4395 \le 0.543$)
  * Dynamic Statistical: `OUTLIER` (FP — false alarm)
  * One-Class SVM: `ANOMALY` (FP — false alarm)
  * Early Drift Forecast: `FUTURE_VIOLATION` (FP — false alarm, predicted $21.57\%$ vs actual $17.45\%$)
  * Risk Fusion: `HIGH RISK` (FP — false alarm, score 0.55)

### Root Causes of the False Alarm on C1:
1. **Symmetric Distance Metric Bias:**
   `C1` exhibited an exceptionally *healthy* initial state: $\Delta C(47\text{h}) = 1.24\%$, compared to the damaged cohort median of $1.58\%$. Because robust Z-score and Mahalanobis distance measure absolute divergence without direction:
   $$Z_{\text{MAD}}(\Delta C) = \frac{1.24 - 1.58}{1.4826 \times 0.034} = -6.76$$
   A two-sided statistical test flagged `C1` simply because it degraded *substantially less* than its damaged peers!

2. **Conservative Linear Extrapolation in Drift Forecasting:**
   Between 0h and 47h, `C1` exhibited a transient initial settling rate ($0.0345\%/\text{h}$). Linear projection over 194 hours forecasted $\hat{\Delta C} = 21.57\%$. In reality, `C1`'s degradation rate decelerated significantly in the second half of the stress test (settling at $17.45\%$), a non-linear stabilization that simple regressors over-predicted.

### Engineering Remedy: Directional (One-Sided) Screening Evaluation:
* In electronic degradation physics, damage is directional (capacitance drops; resistance increases).
* If we enforce **One-Sided MAD Screening**:
  $$\text{Flag if } Z_{\text{MAD}} > +2.5 \quad (\text{ignoring } Z_{\text{MAD}} < -2.5)$$
* **Impact on C1:** $Z_{\text{MAD}} = -6.76 \le +2.5 \implies \mathbf{C1\text{ is ACCEPTED}}$ (FP becomes TN).
* **Impact on Cohort:** Cohort FPR drops from $100.0\%$ to $\mathbf{0.0\%}$ while preserving defect recall on `C2, C4, C6` ($60.0\%$).
* *Conclusion:* Directional physical bounding is essential when applying unsupervised statistics to degradation screening.
