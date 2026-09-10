# Data Dictionary: NASA Capacitor Electrical Stress Degradation Dataset

## 1. Overview & Dataset Provenance
* **Source:** NASA Ames Research Center, Prognostics Center of Excellence (PCoE) / NASA Open Data Portal
* **Raw Archive File:** `EOS_DataSet.mat` within `EOS_DataSet.zip`
* **Original Study:** Celaya, J. R., Kulkarni, C., Biswas, G., & Goebel, K. (2012). *"Towards A Model-based Prognostics Methodology for Electrolytic Capacitors: A Case Study Based on Electrical Overstress Accelerated Aging."* Annual Conference of the PHM Society 2012.
* **Component Type:** Commercial Aluminum Electrolytic Capacitors (2200 µF nominal, 10V rated).
* **Stress Protocol:** Accelerated electrical overstress at 10V with continuous environmental and electrical parameter tracking across 11 discrete measurement intervals up to 194 hours.

---

## 2. Feature Specification Table

| Field | Meaning | Unit | Data Type | Role | Missing % | Notes |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| `component_id` | Discrete physical capacitor unit identifier (`C1` to `C6`) | Categorical / String | `string` | Component Identifier | 0.0% | Maps to matrix column index in `EOS_DataSet.mat`. |
| `lot_id` | Manufacturing / stress cohort identifier (`LOT_10V_EOS`) | Categorical / String | `string` | Lot / Batch Identifier | 0.0% | Denotes the 10V electrical overstress test cohort. |
| `test_step` | Sequential measurement inspection index ($0 \dots 10$) | Discrete Integer | `int64` | Test Cycle | 0.0% | Chronological index of test inspections. |
| `aging_time_hours` | Cumulative stress exposure duration since test start | Hours ($\text{h}$) | `float64` | Timestamp / Elapsed Time | 0.0% | Discrete time points: 0, 24, 47, 71, 94, 116, 139, 149, 161, 171, 194. |
| `delta_capacitance_pct` | Percentage decrease in capacitance from pristine nominal state ($\frac{C_0 - C(t)}{C_0} \times 100\%$) | Percent ($\%$) | `float64` | Degradation Indicator | 0.0% | Monotonically increasing loss; MIL-PRF-62F threshold is $\ge 20\%$. |
| `delta_esr_pct` | Percentage increase in Equivalent Series Resistance ($\frac{\text{ESR}(t) - \text{ESR}_0}{\text{ESR}_0} \times 100\%$) | Percent ($\%$) | `float64` | Degradation Indicator | 0.0% | Electrolyte evaporation indicator; failure criterion is $\ge 100\%$ (doubling). |
| `capacitance_uf` | Estimated physical capacitance reconstructed from nominal $C_0 = 2200\,\mu\text{F}$ | Microfarads ($\mu\text{F}$) | `float64` | Electrical Measurement | 0.0% | Derived via $C_0 \times (1 - \Delta C / 100)$; initial value $2200\,\mu\text{F}$. |
| `esr_ohms` | Estimated physical ESR reconstructed from baseline $\text{ESR}_0 = 0.045\,\Omega$ | Ohms ($\Omega$) | `float64` | Electrical Measurement | 0.0% | Derived via $\text{ESR}_0 \times (1 + \Delta\text{ESR} / 100)$; initial value $0.045\,\Omega$. |
| `stress_voltage_v` | Constant applied electrical overstress bias | Volts ($\text{V}$) | `float64` | Environmental Variable | 0.0% | Constant 10.0 V stress bias during accelerated testing. |
| `static_spec_fail` | Traditional screening static threshold violation flag | Binary Flag | `int32` | Target / Baseline Indicator | 0.0% | 1 if $\Delta C \ge 20\%$ or $\Delta\text{ESR} \ge 100\%$, 0 otherwise. |

---

## 3. Verified Field Interpretations & Boundaries

1. **Matrix Mapping in Raw File:**
   * In raw `EOS_DataSet.mat`, array `aging_time` has dimension `(11, 1)`.
   * Arrays `C` and `ESR` have dimensions `(11, 6)`, where row $j \in [0 \dots 10]$ corresponds to `aging_time[j]` and column $i \in [0 \dots 5]$ corresponds to physical device `C{i+1}`.
2. **Degradation Semantics:**
   * The matrix `C` contains non-negative percentage drop values ($0.0 \dots 22.68\%$).
   * The matrix `ESR` contains non-negative percentage rise values ($0.0 \dots 53.54\%$).
   * At $t = 0$, both drift matrices are exactly $0.00\%$, establishing the pristine pre-stress baseline.
3. **Absence of Invented Data:**
   * Every recorded numerical measurement is extracted strictly from the uncompressed raw binary MAT file (`sha256: 9db651a10f92d2046a477838c08fe1cdbaf27d7bc4062a856a373721400cb4a3`).
   * No synthetic noise or artificial records have been added.
