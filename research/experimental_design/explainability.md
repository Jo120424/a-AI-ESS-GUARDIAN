# Threshold Selection Protocol and Model Explainability Design

## 1. Frozen Threshold Selection Protocol

To prevent threshold selection leakage and post-hoc metric tuning:

```text
======================= THRESHOLD CALIBRATION PROTOCOL =======================
  1. TRAINING PARTITION (Fold k)
     • Fit model parameters (medians, covariances, isolation trees)
     • Compute training anomaly score distribution S_train
     ▼
  2. THRESHOLD CALIBRATION (Training Only)
     • Statistical: Set θ_RZ = 2.5 (98.7% confidence)
     • Mahalanobis: Set θ_MD = sqrt(chi2(0.95, df=p))
     • Isolation Forest: Calibrate tau_ML at 90th percentile of S_train
     ▼
  3. THRESHOLD LOCKING
     • Freeze all decision boundaries and parameters
     ▼
  4. BLIND TEST INFERENCE
     • Evaluate held-out test component k through frozen decision boundary
=============================================================================
```
**Mandatory Rule:** The test component's telemetry and ground truth are never examined during threshold calibration.

---

## 2. Multi-Level Explainability Framework

In mission-critical electronics, black-box declarations (*"AI says anomaly"*) are unacceptable. Spacecraft quality engineers require physical traceability to authorize scrap or redesign. The system provides layered explainability across all three paradigms:

```text
                                 EXPLAINABILITY ENGINE
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
   [ LEVEL 1: TRADITIONAL ]       [ LEVEL 2: STATISTICAL ]          [ LEVEL 3: AI / ML ]
   Threshold Delta Breakdown      Partial Mahalanobis Contribution  Tree Path & SHAP Attribution
   • Which spec was breached?     • Which channel drove distance?   • Which nonlinear feature
   • Breach margin (Δ - θ)        • Robust Z-score vector           • contributed to short path?
```

### Level 1 Explainability: Specification Deviation Margin
* **Explanation Output:** Exact parameter violation and delta margin:
  $$\text{Margin}_C = \Delta C_i(t_{\text{screen}}) - \theta_C \quad \text{and} \quad \text{Margin}_{\text{ESR}} = \Delta\text{ESR}_i(t_{\text{screen}}) - \theta_{\text{ESR}}$$
* **Diagnostic Message:** *"Component C4 rejected: Capacitance loss (20.04%) exceeded MIL-PRF-62F ceiling (20.00%) by +0.04%."*

### Level 2 Explainability: Mahalanobis Partial Distance Decomposition
For multivariate distance $D_M^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$, the contribution $C_j$ of feature $j$ is decomposed as:

$$C_j(\mathbf{x}_i) = (x_{i,j} - \mu_j) \sum_{m=1}^p (\boldsymbol{\Sigma}^{-1})_{j,m} (x_{i,m} - \mu_m)$$

* **Diagnostic Message:** *"Component C4 flagged as statistical anomaly ($D_M = 3.82 > 2.79$): Drift velocity $\frac{d\,\Delta C}{dt}$ contributed $68.4\%$ of total anomaly distance."*

### Level 3 Explainability: Isolation Tree Path Feature Attribution & SHAP
For Isolation Forest:
1. **Tree Split Depth Attribution:** Tracks which feature splits occurred closest to the root for the test instance, identifying the dominant dimension isolating the component.
2. **Kernel SHAP Value Decomposition:**
   $$s(\mathbf{x}_i) = \phi_0 + \sum_{j=1}^p \phi_j(\mathbf{x}_i)$$
   Where $\phi_j$ represents the additive attribution of feature $j$ pushing the anomaly score towards rejection.
* **Diagnostic Message:** *"Component C4 flagged by Isolation Forest ($s = 0.68 > 0.55$): $\phi(\text{drift\_velocity\_c}) = +0.11$, $\phi(\text{c\_to\_esr\_ratio}) = +0.04$."*
