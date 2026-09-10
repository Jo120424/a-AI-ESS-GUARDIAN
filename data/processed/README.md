# Processed Dataset Documentation: NASA Capacitor Degradation

## 1. File Specification
* **Target File:** `data/processed/dataset_processed.csv`
* **Observation Count:** 66 rows
* **Column Count:** 14 columns
* **Source:** Raw MATLAB file `data/raw/nasa_capacitor_electrical_stress/EOS_DataSet.mat`

---

## 2. Transformations Applied
1. **Matrix Reshaping:** Extracted 11x6 matrices of Capacitance drop (Delta C) and ESR rise (Delta ESR) and unpivoted them into a standardized tidy format where each record represents a single component at a specific test step.
2. **Physical Parameter Reconstruction:** Calibrated physical capacitance (C in uF) and resistance (ESR in Ohms) using nominal datasheet values (C0 = 2200 uF, ESR0 = 0.045 Ohms) from Celaya et al. (2012).
3. **Temporal Feature Engineering:** Computed causal backward drift velocity (d(Delta C)/dt and d(Delta ESR)/dt) using backward differences to prevent future time leakage.
4. **Baseline Label Creation:** Flagged `static_spec_fail` using the standard military threshold (Delta C >= 20%).
5. **Ground-Truth Target:** Identified components that experience downstream failure (`component_ever_fails`) for comparative screening validation.

---

## 3. Data Integrity & Safeguards
* **Missing Values:** Zero missing values.
* **Duplicates:** Zero duplicate component-time entries.
* **Leakage Guard:** Normalization/scaling parameters are **NOT** applied to the complete CSV; they must be fitted strictly inside training folds during cross-validation.
