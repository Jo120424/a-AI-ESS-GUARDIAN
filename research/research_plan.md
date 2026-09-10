# Research Plan & Experimental Protocol

## 1. Title & Research Scope
**Title:** Predictive AI-Based Environmental Stress Screening for Latent Defect Detection in Electronic Components: A Comparative Study Against Traditional Screening

**Core Research Question:**
> *"Can dynamic, data-driven screening detect electronic components harboring latent defects that conventional static specification screening allows to pass into critical assemblies?"*

---

## 2. Formal Hypotheses

* **Null Hypothesis ($H_0$):**
  Dynamic statistical screening (Level 2) and machine learning anomaly detection (Level 3) yield no statistically significant improvement in the detection of latent-risk components over conventional static limit screening (Level 1) when evaluated at the screening cutoff horizon on unseen test components, or they incur an unacceptable false positive rate (unnecessary rejection rate $> 15\%$).
  $$\text{Recall}_{\text{AI}} \le \text{Recall}_{\text{Traditional}} \quad \text{or} \quad \text{FPR}_{\text{AI}} - \text{FPR}_{\text{Traditional}} > \Delta_{\text{acceptable}}$$

* **Alternative Hypothesis ($H_1$):**
  Dynamic statistical methods and AI/ML models can detect subtle multi-parameter drift signatures and population-relative anomalies during early screening, achieving a significantly higher recall of latent failures ($\text{Recall}_{\text{AI}} > \text{Recall}_{\text{Traditional}}$) while maintaining a false negative escape rate strictly lower than traditional static thresholding.

* **Research Neutrality Commitment:** The experimental framework is constructed impartially. We do not assume AI will outperform traditional methods; should traditional or simple robust statistical screening prove superior in accuracy, stability, or interpretability, the research conclusion will unequivocally report that finding.

---

## 3. The Three Screening Paradigms Under Comparison

The study systematically implements and benchmarks three operational screening levels:

```text
                                 RAW SENSOR / MEASUREMENT DATA (at t_screen)
                                                      |
                   +----------------------------------+----------------------------------+
                   |                                  |                                  |
                   v                                  v                                  v
         [ LEVEL 1: TRADITIONAL ]           [ LEVEL 2: DYNAMIC STATISTICAL ]      [ LEVEL 3: AI / ML ]
         * Static Datasheet Limits          * Population Robust Z-Score           * Isolation Forest
         * MIL-PRF-62F Threshold            * Mahalanobis Distance (DM)           * One-Class SVM
         * Upper/Lower Spec Limits (USL/LSL)* PCA Residual / Hotelling's T^2      * Local Outlier Factor (LOF)
                   |                                  |                           * Degradation Predictor
                   |                                  |                                  |
                   +----------------------------------+----------------------------------+
                                                      |
                                                      v
                                        EVALUATION ON THE EXACT SAME
                                         UNSEEN TEST COMPONENTS
                                                      |
                                                      v
                                      GROUND TRUTH LATENT DEFECT PROXY
                                                      |
                                                      v
                                          COMPARATIVE METRIC SUITE
```

### Level 1 — Traditional Screening (Static Specification)
* Operates on absolute thresholds derived from component specifications:
  $$\text{Decision}(i) = \begin{cases} \text{FAIL}, & \text{if } C_i(t_{\text{screen}}) < C_{\text{min}} \text{ or } \text{ESR}_i(t_{\text{screen}}) > \text{ESR}_{\text{max}} \\ \text{PASS}, & \text{otherwise} \end{cases}$$
* Representative of standard aerospace, defense, and industrial screening practices where parts are evaluated independently without reference to lot context.

### Level 2 — Dynamic Statistical Screening (Population-Relative)
* Evaluates each component relative to its production lot or stress cohort distribution at screening time $t_{\text{screen}}$:
  * **Robust Z-Score (MAD):** Identifies units whose parameter drift exceeds $k \times \text{MAD}$ of the cohort.
  * **Mahalanobis Distance ($D_M$):** Accounts for multivariate covariance across parameters $[C, \text{ESR}, |Z|, \theta]$.
  * **PCA Residual Energy ($Q$ / SPE):** Flags breakdown in expected linear parameter correlation.

### Level 3 — AI/ML Predictive Screening
* Learns complex, non-linear degradation manifolds:
  * **Unsupervised Anomaly Detectors:** Isolation Forest, One-Class SVM, and Local Outlier Factor trained on early nominal operational baselines.
  * **Degradation Trajectory Forecasters:** Regression / sequence models mapping early drift rates $[t_0 \dots t_{\text{screen}}]$ to predicted state at future horizon $t_{\text{future}}$.

---

## 4. Non-Negotiable Core Principle: Identical Test Sets & Zero Leakage

To prevent methodological bias and ensure reproducibility:
1. **Component-Isolated Partitioning:** Train and test splits are strictly partitioned by discrete **Component ID** (e.g., Leave-One-Component-Out or K-Fold Group Cross-Validation). Observations from the same component will NEVER appear in both training and test sets.
2. **Identical Test Set Evaluation:** Every screening method (Traditional, Dynamic Statistical, AI/ML) must be evaluated on the **exact same test components** at the **exact same screening cutoff time step** ($t_{\text{screen}}$).
3. **Strict Temporal Isolation:** For screening at cycle $t_{\text{screen}}$, models are strictly restricted to features recorded at or before $t_{\text{screen}}$. No future information from $t > t_{\text{screen}}$ is accessible to any screening algorithm.

---

## 5. Ground Truth & Latent Defect Definition

In accordance with empirical reliability physics and the primary dataset metadata ([`dataset_metadata.json`](file:///C:/Users/user/Desktop/isro/data/metadata/dataset_metadata.json)):
* A component is verified as having a **Latent Defect** if and only if:
  1. At screening cycle $t_{\text{screen}}$, its physical parameters meet Level 1 specification limits ($\Delta C > -20\%$ and $\text{ESR} < 2\times \text{ESR}_0$), meaning traditional screening would allow it to PASS.
  2. Under continued stress exposure ($t_{\text{screen}} < t \le t_{\text{EOL}}$), the component experiences premature parametric or catastrophic failure prior to the design reliability threshold.
* This proxy definition represents the quintessential objective of Environmental Stress Screening: precipitating and detecting latent defects before components enter critical service.

---

## 6. Planned Evaluation Metric Suite

The comparative evaluation will compute the following metrics across all three screening paradigms:

| Category | Metric | Mathematical Definition | Operational Meaning in ESS |
| :--- | :--- | :--- | :--- |
| **Defect Detection** | **Recall (Sensitivity)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Percentage of latent defective units successfully detected. |
| **Escape Risk** | **False Negative Rate (FNR)** | $\frac{\text{FN}}{\text{TP} + \text{FN}} = 1 - \text{Recall}$ | **Most critical metric**: Latent defects passing into assemblies. |
| **Escape Count** | **False Negatives (FN)** | Integer Count | Absolute number of hazardous components escaping screening. |
| **Inspection Purity** | **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Proportion of flagged components that are genuinely defective. |
| **Harmonic Balance**| **F1 Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Balance between escape reduction and false yield loss. |
| **Threshold Invariance** | **PR-AUC & ROC-AUC** | Area under PR and ROC curves | Model discriminatory power under extreme class imbalance. |
| **Yield Penalty** | **False Positive Rate (FPR)** | $\frac{\text{FP}}{\text{FP} + \text{TN}}$ | Healthy components incorrectly rejected (yield loss). |
| **Yield Scrap** | **Unnecessary Rejection Rate** | $\frac{\text{FP}}{\text{Total Screened}}$ | Economic cost of over-screening. |
| **Prognostic Error**| **MAE & RMSE** | $\frac{1}{N}\sum \|y - \hat{y}\|$, $\sqrt{\frac{1}{N}\sum (y - \hat{y})^2}$ | Degradation trajectory forecasting precision (Level 3 models). |
| **Lead Time** | **Early Detection Margin** | $t_{\text{EOL}} - t_{\text{flagged}}$ | Screening cycles gained before catastrophic failure occurs. |
