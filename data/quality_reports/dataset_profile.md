# Dataset Quality Profile: NASA Capacitor Electrical Stress

## 1. Executive Summary
* **Dataset Name:** NASA Capacitor Electrical Stress Degradation Dataset
* **Source Archive:** `EOS_DataSet.mat` (1090 bytes)
* **Total Observations:** 66 records (6 discrete physical components × 11 time steps)
* **Features:** 10 variables
* **Missing Data:** **0.00%** (100.0% data completeness across all fields)
* **Exact Duplicate Rows:** 0
* **Time Range:** 0.0 h to 194.0 h across 11 inspection intervals

---

## 2. Dataset-Level Metrics

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Component Count** | 6 units (`C1` to `C6`) | Sufficient for group-isolated leave-one-out evaluation. |
| **Observations / Unit** | 11 longitudinal cycles | Enables trajectory slope & curvature modeling. |
| **Total Normal Records** | 60 (90.9%) | Observations satisfying MIL-PRF-62F static limits. |
| **Total Fail Records** | 6 (9.1%) | Observations exceeding static limits ($\Delta C \ge 20\%$). |
| **Duplicate Component-Time** | 0 records | Strict 1-to-1 temporal uniqueness verified. |

---

## 3. Column Statistics Table

| Column | Type | Count | Missing | Mean | Median | Std | Min | Max | Skewness |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `component_id` | categorical | 66 | 0.0% | — | — | — | — | — | — |
| `lot_id` | categorical | 66 | 0.0% | — | — | — | — | — | — |
| `test_step` | numeric | 66 | 0.0% | 5.0 | 5.0 | 3.1865 | 0.0 | 10.0 | 0.0 |
| `aging_time_hours` | numeric | 66 | 0.0% | 106.0 | 116.0 | 61.4006 | 0.0 | 194.0 | -0.3098 |
| `delta_capacitance_pct` | numeric | 66 | 0.0% | 7.8412 | 6.1138 | 7.0777 | 0.0 | 22.675 | 0.661 |
| `delta_esr_pct` | numeric | 66 | 0.0% | 31.8279 | 33.6988 | 16.8894 | 0.0 | 53.54 | -0.4571 |
| `capacitance_uf` | numeric | 66 | 0.0% | 2027.4929 | 2065.4955 | 155.71 | 1701.15 | 2200.0 | -0.661 |
| `esr_ohms` | numeric | 66 | 0.0% | 0.0593 | 0.0602 | 0.0076 | 0.045 | 0.0691 | -0.4573 |
| `stress_voltage_v` | numeric | 66 | 0.0% | 10.0 | 10.0 | 0.0 | 10.0 | 10.0 | 0.0 |
| `static_spec_fail` | numeric | 66 | 0.0% | 0.0909 | 0.0 | 0.2897 | 0.0 | 1.0 | 2.9127 |

---

## 4. Component Degradation Summary

| Component | Test Steps | Initial ΔC (%) | Final ΔC (%) | Initial ΔESR (%) | Final ΔESR (%) | Spec Violated? | First Failure (h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C1` | 11 | 0.00% | 17.45% | 0.00% | 53.54% | NO | nan |
| `C2` | 11 | 0.00% | 21.68% | 0.00% | 51.04% | **YES** (194.0h) | 194.0 |
| `C3` | 11 | 0.00% | 21.05% | 0.00% | 52.39% | **YES** (194.0h) | 194.0 |
| `C4` | 11 | 0.00% | 22.68% | 0.00% | 53.30% | **YES** (171.0h) | 171.0 |
| `C5` | 11 | 0.00% | 20.80% | 0.00% | 52.53% | **YES** (194.0h) | 194.0 |
| `C6` | 11 | 0.00% | 22.04% | 0.00% | 52.03% | **YES** (194.0h) | 194.0 |

---

## 5. Key Empirical Observations
1. **Pristine Initial State:** At $t = 0\,\text{h}$, all 6 units show $0.00\%$ drift in both Capacitance and ESR, confirming perfect baseline synchronization.
2. **Latent Incubation Period:** Between $0\,\text{h}$ and $71\,\text{h}$, capacitance drop remains below $2.5\%$ across all components (well below the $20\%$ traditional failure threshold).
3. **Divergent Downstream Outcomes:** Units `C2`, `C4`, and `C6` degrade faster and cross the $20\%$ end-of-life threshold at $t = 194\,\text{h}$, while units `C1` and `C5` remain within specification ($17.45\%$ and $20.8\%$), demonstrating natural latent variance under identical stress.
