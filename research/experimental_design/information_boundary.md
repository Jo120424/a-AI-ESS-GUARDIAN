# Information Boundary & Temporal Quarantine Protocol

## 1. The Temporal Quarantine Principle

In Environmental Stress Screening, the screening decision occurs at an early operational cutoff milestone ($t_{\text{screen}}$). Any algorithm evaluated at $t_{\text{screen}}$ must operate under strict **temporal isolation**:

```text
======================= INFORMATION BOUNDARY TIMELINE =======================
  t = 0h (Baseline)                      t_screen (Cutoff)               t = 194h (EOL)
    │                                          │                                │
    ├──────────────────────────────────────────┤                                │
    │      PRE-SCREENING OBSERVATION WINDOW    │                                │
    │  • Pristine initial measurements (t=0)   │                                │
    │  • Telemetry up to t_screen              │                                │
    │  • Backward drift velocity (dC/dt)       │                                │
    │  • Lot-relative robust statistics        │                                │
    └──────────────────┬───────────────────────┘                                │
                       │                                                        │
                       ▼                                                        │
         [ SCREENING DECISION POINT ]                                           │
         Output: PASS or REJECT                                                 │
                       │                                                        │
                       └────────────────────────────────────────────────────────┤
                                                FUTURE UNSEEN HORIZON          │
                                                • Telemetry for t > t_screen   │
                                                • Time to EOL (t_fail)          │
                                                • Final degradation severity    │
                                                ─────────────────────────────   │
                                                STRICTLY QUARANTINED            │
                                                ONLY USED TO EVALUATE           │
                                                GROUND TRUTH LABELS!            │
=============================================================================
```

---

## 2. Feature Classification Under Information Boundary

Every variable in the dataset is classified into one of four operational categories relative to screening cutoff $t_{\text{screen}}$:

| Feature / Variable | Classification | Rationale & Guard Mechanism |
| :--- | :---: | :--- |
| `component_id` | **AVAILABLE AT SCREENING** | Used solely for component isolation and grouping; never as a numerical feature. |
| `lot_id` | **AVAILABLE AT SCREENING** | Identifies cohort membership (`LOT_10V_EOS`) for lot-relative statistical baselines. |
| `stress_voltage_v` | **AVAILABLE AT SCREENING** | Constant known environmental stress parameter ($10.0\,\text{V}$). |
| `aging_time_hours` ($t \le t_{\text{screen}}$) | **AVAILABLE AT SCREENING** | Elapsed screening duration. |
| `delta_capacitance_pct` ($t \le t_{\text{screen}}$) | **AVAILABLE AT SCREENING** | Primary empirical degradation telemetry recorded up to cutoff. |
| `delta_esr_pct` ($t \le t_{\text{screen}}$) | **AVAILABLE AT SCREENING** | Internal series resistance telemetry recorded up to cutoff. |
| `drift_velocity_c` ($t \le t_{\text{screen}}$) | **AVAILABLE AT SCREENING** | Backward difference rate ($\frac{\Delta C(t) - \Delta C(t - \Delta t)}{\Delta t}$) using only historical time steps. |
| `drift_velocity_esr` ($t \le t_{\text{screen}}$) | **AVAILABLE AT SCREENING** | Backward difference rate using only historical time steps. |
| `robust_z_delta_c` ($t \le t_{\text{screen}}$) | **AVAILABLE AT SCREENING** | Computed relative to peer components present in the training fold at $t_{\text{screen}}$. |
| `delta_capacitance_pct` ($t > t_{\text{screen}}$) | **AVAILABLE ONLY IN FUTURE** | **Strictly Quarantined**: Future trajectory data used solely to establish downstream ground-truth labels. |
| `delta_esr_pct` ($t > t_{\text{screen}}$) | **AVAILABLE ONLY IN FUTURE** | **Strictly Quarantined**: Future series resistance data. |
| `first_failure_step` / `t_fail` | **AVAILABLE ONLY IN FUTURE** | Downstream failure timestamp; target variable for evaluation. |
| `component_ever_fails` | **AVAILABLE ONLY IN FUTURE** | Downstream ground-truth target; models must never access this column during screening. |
| Forward Differences ($\frac{\Delta C(t + \Delta t) - \Delta C(t)}{\Delta t}$) | **POTENTIAL LEAKAGE (FORBIDDEN)** | Would introduce future trend knowledge into the past; strictly barred by causal backward calculation. |
| Global Feature Scalers Fitted on Full Panel | **POTENTIAL LEAKAGE (FORBIDDEN)** | Scalers fitted on test folds violate non-leakage; all transformers must be fitted strictly inside cross-validation training folds. |
| Synthetic Post-Test Ground Truth | **NEVER AVAILABLE** | Non-existent; only empirical physical measurements are used. |

---

## 3. Automated Quarantine Implementation
The preprocessing and experimental execution scripts enforce this boundary through physical array slicing:
```python
# Input feature extraction strictly filters by t <= t_screen
X_screening = df[df["aging_time_hours"] <= t_screen].copy()

# Target ground-truth evaluation inspects only t > t_screen
future_trajectory = df[df["aging_time_hours"] > t_screen].copy()
```
Any screening pipeline attempting to pass a feature indexed at $t > t_{\text{screen}}$ will raise an immediate automated assertion error.
