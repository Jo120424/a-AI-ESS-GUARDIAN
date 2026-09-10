# Feature Engineering Specification

## 1. Feature Hierarchy & Pipeline Architecture

All features engineered for the comparative screening study are categorized into a structured hierarchy. Every feature is strictly causal, computed using observations at or prior to screening cutoff $t_{\text{screen}}$ ($t \le t_{\text{screen}}$):

```text
                                  FEATURE HIERARCHY (Pre-Screening Only)
                                                    │
        ┌───────────────────┬───────────────────────┼───────────────────────┬───────────────────┐
        ▼                   ▼                       ▼                       ▼                   ▼
   [ RAW VALUES ]     [ DIFFERENCES ]         [ KINETIC / RATES ]     [ POPULATION ]      [ MULTIVARIATE ]
   • delta_C_pct      • delta_C_minus_t0      • velocity_C (dC/dt)    • robust_z_C        • mahalanobis_d
   • delta_ESR_pct    • delta_ESR_minus_t0    • velocity_ESR          • robust_z_vC       • pca_pc1_score
   • capacitance_uf                           • accel_C (d2C/dt2)     • lot_pctile_C      • pca_residual_q
   • esr_ohms                                 • drift_ratio_C_ESR
```

---

## 2. Mathematical Formulations

### Category A: Raw Baseline Features
Extracted directly from calibrated test instrumentation at $t = t_{\text{screen}}$:
* $\Delta C(t_{\text{screen}}) = \frac{C_0 - C(t_{\text{screen}})}{C_0} \times 100\%$
* $\Delta\text{ESR}(t_{\text{screen}}) = \frac{\text{ESR}(t_{\text{screen}}) - \text{ESR}_0}{\text{ESR}_0} \times 100\%$

### Category B: Total Initial Offsets (Difference Features)
Measures cumulative deviation accumulated since the pre-stress baseline:
$$\Delta_{\text{offset},C} = \Delta C(t_{\text{screen}}) - \Delta C(t_0)$$
$$\Delta_{\text{offset},\text{ESR}} = \Delta\text{ESR}(t_{\text{screen}}) - \Delta\text{ESR}(t_0)$$

### Category C: Causal Kinetic & Drift Rate Features
First- and second-order backward discrete derivatives approximating instantaneous drift dynamics:
1. **Drift Velocity ($\text{Velocity}_C$):**
   $$\text{Velocity}_C(t) = \frac{\Delta C(t) - \Delta C(t - \Delta t)}{\Delta t} \quad [\% / \text{hour}]$$
2. **Drift Acceleration ($\text{Accel}_C$):**
   $$\text{Accel}_C(t) = \frac{\text{Velocity}_C(t) - \text{Velocity}_C(t - \Delta t)}{\Delta t} \quad [\% / \text{hour}^2]$$
3. **Kinetic Dissipation Drift Ratio:**
   $$\text{Ratio}_{C/\text{ESR}}(t) = \frac{\Delta C(t) + \epsilon}{\Delta\text{ESR}(t) + \epsilon}$$

### Category D: Population-Relative Features (Lot-Adaptive)
Computed relative to the training component lot distribution at $t_{\text{screen}}$:
1. **Robust Z-Score of Degradation Level:**
   $$\text{RZ}_C(i) = \frac{\Delta C_i(t_{\text{screen}}) - \text{Median}_{\text{train}}(\Delta C)}{1.4826 \times \text{MAD}_{\text{train}}(\Delta C)}$$
2. **Robust Z-Score of Drift Velocity:**
   $$\text{RZ}_{v,C}(i) = \frac{\text{Velocity}_{C,i}(t_{\text{screen}}) - \text{Median}_{\text{train}}(\text{Velocity}_C)}{1.4826 \times \text{MAD}_{\text{train}}(\text{Velocity}_C)}$$
3. **Cohort Percentile Rank:**
   $$\text{Rank}_{\text{pct}}(i) = \frac{\text{Rank}(\Delta C_i)}{N_{\text{train}}} \times 100\%$$

### Category E: Multivariate Coupling Features
1. **Regularized Mahalanobis Distance ($D_M$):**
   $$D_M(\mathbf{x}_i) = \sqrt{(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{\text{train}})^T \hat{\boldsymbol{\Sigma}}_{\text{reg}}^{-1} (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{\text{train}})}$$
2. **PCA Residual Error ($Q$-Statistic):**
   $$Q_i = \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 = \|\mathbf{x}_i - \mathbf{P}_k \mathbf{P}_k^T \mathbf{x}_i\|^2$$
   Where $\mathbf{P}_k$ is the projection matrix containing the first $k$ principal eigenvectors of the training covariance matrix.

---

## 3. Strict Non-Leakage Execution Contract
* All backward derivatives require $t \ge t_1$. At $t_0 = 0$, velocity is formally defined as $0.0$.
* No rolling windows or centered averages using future measurements are permitted.
* Population parameters ($\text{Median}_{\text{train}}$, $\text{MAD}_{\text{train}}$, $\hat{\boldsymbol{\mu}}_{\text{train}}$, $\hat{\boldsymbol{\Sigma}}_{\text{reg}}$) are fitted **strictly on the training partition**.
