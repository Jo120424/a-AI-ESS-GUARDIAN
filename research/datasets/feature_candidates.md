# Candidate Features for Predictive Environmental Stress Screening

## 1. Overview
This document categorizes all direct, derived, temporal, and population-relative features designed for the three screening paradigms (Traditional, Dynamic Statistical, AI/ML). In accordance with Step 2 constraints, features are formulated and documented here prior to model training in subsequent steps.

---

## 2. Feature Taxonomy

```text
                                         CANDIDATE FEATURES
                                                 |
         +-------------------+-------------------+-------------------+-------------------+
         |                   |                   |                   |                   |
         v                   v                   v                   v                   v
     [ DIRECT ]       [ ENVIRONMENTAL ]     [ TEMPORAL ]        [ POPULATION ]      [ INTERACTIONS ]
   * delta_C_pct      * stress_voltage_v    * dC_dt (velocity)  * robust_z_dC       * C_to_ESR_ratio
   * delta_ESR_pct    * aging_time_hours    * dESR_dt           * robust_z_dESR     * dissipation_factor
   * capacitance_uf   * test_step           * d2C_dt2 (accel)   * lot_mahalanobis   * power_loss_proxy
   * esr_ohms                               * d2ESR_dt2         * pca_q_residual
```

---

## 3. Detailed Feature Specifications

### A. Direct Empirical Measurements
Directly extracted and calibrated from raw telemetry:
1. `delta_capacitance_pct`: Percentage drop in capacitance ($\Delta C = \frac{C_0 - C(t)}{C_0} \times 100\%$). Primary aging health indicator.
2. `delta_esr_pct`: Percentage increase in Equivalent Series Resistance ($\Delta\text{ESR} = \frac{\text{ESR}(t) - \text{ESR}_0}{\text{ESR}_0} \times 100\%$). Indicator of electrolyte evaporation.
3. `capacitance_uf`: Reconstructed physical capacitance ($C(t) = 2200 \times (1 - \Delta C / 100)$).
4. `esr_ohms`: Reconstructed physical resistance ($\text{ESR}(t) = 0.045 \times (1 + \Delta\text{ESR} / 100)$).

### B. Environmental & Test Context Variables
1. `stress_voltage_v`: Constant electrical stress bias ($10.0\,\text{V}$).
2. `aging_time_hours`: Elapsed burn-in duration ($t \in [0, 194]\,\text{h}$).
3. `test_step`: Chronological inspection cycle ($0 \dots 10$).

### C. Temporal Drift & Kinetic Features (Pre-$t_{\text{screen}}$ Only)
Features capturing the dynamics and velocity of degradation up to the screening cutoff horizon:
1. `drift_velocity_c` ($\frac{d\,\Delta C}{dt}$):
   $$\text{Velocity}_C(t) = \frac{\Delta C(t) - \Delta C(t - \Delta t)}{\Delta t} \quad [\% / \text{hour}]$$
2. `drift_velocity_esr` ($\frac{d\,\Delta\text{ESR}}{dt}$):
   $$\text{Velocity}_{\text{ESR}}(t) = \frac{\Delta\text{ESR}(t) - \Delta\text{ESR}(t - \Delta t)}{\Delta t} \quad [\% / \text{hour}]$$
3. `drift_acceleration_c` ($\frac{d^2\,\Delta C}{dt^2}$):
   $$\text{Accel}_C(t) = \frac{\text{Velocity}_C(t) - \text{Velocity}_C(t - \Delta t)}{\Delta t} \quad [\% / \text{hour}^2]$$
4. `cumulative_drift_ratio` ($\frac{\Delta C(t)}{\Delta\text{ESR}(t)}$): Captures relative kinetics between dielectric degradation and electrolyte drying.

### D. Dynamic Population-Relative Features (Lot-Relative)
Computed relative to peer component distribution at screening cycle $t_{\text{screen}}$:
1. `robust_z_delta_c`: Robust Z-score using Median Absolute Deviation (MAD):
   $$\text{RZ}_C(i) = \frac{\Delta C_i(t) - \text{Median}(\Delta C)}{\text{MAD}(\Delta C) \times 1.4826}$$
2. `robust_z_velocity_c`: Robust Z-score of drift rate relative to cohort peers.
3. `mahalanobis_distance`: Multivariate distance from lot centroid in $[\Delta C, \Delta\text{ESR}]$ feature space:
   $$D_M(\mathbf{x}_i) = \sqrt{(\mathbf{x}_i - \boldsymbol{\mu}_{\text{lot}})^T \boldsymbol{\Sigma}_{\text{robust}}^{-1} (\mathbf{x}_i - \boldsymbol{\mu}_{\text{lot}})}$$
4. `pca_residual_q`: Reconstruction error from principal component projection, flagging breakdown in normal parameter correlation.

---

## 4. Screening Paradigm Mapping

| Feature Set | Level 1: Traditional | Level 2: Dynamic Statistical | Level 3: AI/ML |
| :--- | :---: | :---: | :---: |
| Absolute $\Delta C$, $\Delta\text{ESR}$ | **YES** (Static threshold) | YES (Input to stats) | YES (Feature) |
| Temporal Velocities ($\frac{dC}{dt}$) | NO | **YES** (Rate limits) | **YES** (Feature) |
| Robust Z-scores & MAD | NO | **YES** (Core detector) | YES (Feature) |
| Mahalanobis Distance $D_M$ | NO | **YES** (Core detector) | YES (Feature) |
| Non-linear Unsupervised Scores | NO | NO | **YES** (Isolation Forest / LOF) |
