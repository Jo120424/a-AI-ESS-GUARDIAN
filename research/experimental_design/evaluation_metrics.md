# Comparative Evaluation Metrics Specification

## 1. The Asymmetric Cost of Screening Decisions

In high-reliability screening for aerospace, satellite, and mission-critical electronics, decision errors have **highly asymmetric costs**:

$$\text{Cost}(\text{False Negative: Missed Latent Defect}) \gg \text{Cost}(\text{False Positive: Yield Scrap})$$

* A **False Negative (Test Escape)** means a component harboring latent incubation damage passes screening, gets soldered onto an integrated satellite flight computer or launch vehicle controller, and fails catastrophic during mission operation.
* A **False Positive (Over-Rejection)** incurs only the marginal material cost of replacing a raw component during factory testing.

Therefore, standard accuracy is **scientifically inappropriate** as an evaluation metric. The evaluation framework prioritizes **Recall, False Negative Rate (FNR), and Precision-Recall Area Under the Curve (PR-AUC)**.

---

## 2. Formal Metric Suite

```text
                                        EVALUATION METRIC SUITE
                                                   │
         ┌─────────────────────────┬───────────────┴───────────────┬─────────────────────────┐
         ▼                         ▼                               ▼                         ▼
   [ DEFECT DETECTION ]      [ YIELD & COST ]               [ DISCRIMINATION ]        [ LEAD TIME ]
   • Recall (Sensitivity)    • False Positive Rate (FPR)    • PR-AUC                  • Warning Margin (h)
   • False Negative Rate     • Unnecessary Rejection Rate   • ROC-AUC                 • Early Detection %
   • Missed Escapes (FN)     • Precision                    • Balanced Accuracy
```

### Primary Detection Metrics (Latent Defect Interception)
1. **Recall / Sensitivity ($R$):**
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   Measures the proportion of genuine latent defect components successfully intercepted before leaving the factory.
2. **False Negative Rate ($\text{FNR}$):**
   $$\text{FNR} = \frac{\text{FN}}{\text{TP} + \text{FN}} = 1 - \text{Recall}$$
   **The single most critical reliability metric**: the fraction of dangerous components escaping screening into assembly.
3. **Absolute Test Escapes ($\text{FN}$):**
   $$\text{Escapes} = \sum_{i} \mathbb{I}(y_i = 1 \land \hat{y}_i = 0)$$
   Integer count of hazardous defective units passing screening.
4. **Precision ($P$):**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   Proportion of rejected components that were genuinely defective.
5. **Harmonic Balance (F1-Score):**
   $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
6. **PR-AUC (Precision-Recall Area Under the Curve):**
   Evaluates threshold-invariant ranking performance under class imbalance.

### Secondary & Operational Yield Metrics
1. **False Positive Rate ($\text{FPR}$ / Alpha Risk):**
   $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
   Proportion of healthy survivor components incorrectly flagged as anomalies.
2. **Unnecessary Rejection Rate ($\text{URR}$ / Producer's Yield Loss):**
   $$\text{URR} = \frac{\text{FP}}{N_{\text{total}}}$$
   Economic scrap penalty incurred by over-aggressive screening.
3. **Specificity (True Negative Rate):**
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} = 1 - \text{FPR}$$
4. **Early Detection Lead Time ($\Delta t_{\text{lead}}$):**
   $$\Delta t_{\text{lead}}(i) = t_{\text{failure}}(i) - t_{\text{screen}}$$
   Hours of warning gained prior to catastrophic in-service breakdown.

---

## 3. Metric Evaluation Matrix

| Metric | Level 1: Traditional | Level 2: Dynamic Statistical | Level 3: AI/ML | Operational Threshold Goal |
| :--- | :---: | :---: | :---: | :--- |
| **Recall** | Expected: Low | Expected: Moderate–High | Expected: High | $\ge 80.0\%$ |
| **False Negative Rate** | Expected: High | Expected: Moderate | Expected: Low | $\le 20.0\%$ |
| **False Positive Rate** | Expected: 0.0% | Expected: Low | Expected: Low–Moderate | $\le 15.0\%$ (Yield ceiling) |
| **PR-AUC** | Baseline scalar | Calculated over $\tau_{\text{stat}}$ | Calculated over $\tau_{\text{ML}}$ | Maximize |
