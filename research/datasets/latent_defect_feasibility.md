# Latent-Defect Feasibility Analysis

## 1. Context & Scientific Feasibility Objective
A primary prerequisite of this research is determining whether the selected empirical dataset can legitimately support the concept of a **latent defect** without violating scientific integrity or fabricating data.

---

## 2. Evaluation of Candidate Definitions

| Definition Candidate | Feasibility on NASA Capacitor Dataset | Supporting Empirical Evidence | Limitations & Caveats |
| :--- | :---: | :--- | :--- |
| **Definition A:** Component initially appears normal under static limits but later experiences failure. | **FULLY SUPPORTED** | At $t \le 71\,\text{h}$, all units have $\Delta C < 2.5\%$ (well within the $20\%$ spec limit). At $t = 171\text{h}-194\text{h}$, 5 out of 6 units cross the $20\%$ EOL threshold while 1 unit survives. | 6 total components in cohort; requires leave-one-out validation. |
| **Definition B:** Component initially passes static threshold but later violates threshold. | **FULLY SUPPORTED** | Units `C2, C3, C4, C5, C6` satisfy Level 1 static screening at early cycles ($t_{\text{screen}} = 24\text{h}, 47\text{h}, 71\text{h}$) but later fail. Unit `C1` never violates the threshold throughout 194 hours. | Threshold is fixed by MIL-PRF-62F industry specification. |
| **Definition C:** Component exhibits abnormal degradation trajectory slope before failure. | **FULLY SUPPORTED** | Unit `C4` exhibits an accelerated drift velocity ($\frac{d\,\Delta C}{dt} = 0.125\%/\text{h}$ vs cohort average $0.108\%/\text{h}$) observable as early as $t = 47\,\text{h}-71\,\text{h}$. | Small cohort sample size requires robust rank-based metrics. |
| **Definition D:** Component has an anomalous trajectory relative to healthy lot peers. | **FULLY SUPPORTED** | Robust Z-score (MAD) and Mahalanobis distance on $[\Delta C, \Delta\text{ESR}]$ divergence at $t_{\text{screen}} = 47\,\text{h}$ distinguish early divergent units from peer median. | Requires lot-relative normalization to avoid single-unit dominance. |

---

## 3. Recommended Primary Proxy Definition
We select **Definition A combined with Definition B** as the primary scientific proxy:
$$\text{Latent Defect Component} \iff \begin{cases} \text{At early screening horizon } t_{\text{screen}} = 47\,\text{h}: \Delta C_i(t_{\text{screen}}) < 20\% \text{ (Passes Traditional Screening)} \\ \text{AND} \\ \text{At final stress horizon } t_{\text{final}} = 194\,\text{h}: \Delta C_i(t_{\text{final}}) \ge 20\% \text{ (Premature EOL Failure)} \end{cases}$$

### Advantages
1. **Physical Grounding:** Directly matches aerospace Environmental Stress Screening (ESS) goals: intercepting parts during early burn-in that pass simple DC/parametric limits but carry internal incubation damage causing premature failure in mission service.
2. **Defensible Threshold:** The $20\%$ capacitance degradation cutoff is universally codified in military capacitor reliability standards (MIL-PRF-62F), avoiding arbitrary threshold tuning.
3. **Reproducibility:** Transparent mathematical formulation verifiable by any third-party researcher.

### Limitations
1. **Sample Size:** The dataset cohort consists of 6 discrete units tracked longitudinally across 11 cycles. While time-series resolution is high, population cross-validation must use **Leave-One-Component-Out (LOCO)** to ensure statistical rigor.
2. **Stress Mode:** Accelerated electrical overstress was applied continuously rather than mixed-mode thermal cycling or vibration.

---

## 4. Leakage Risk & Safeguards

| Leakage Threat | Severity | Safeguard Implementation |
| :--- | :--- | :--- |
| **Temporal Leakage** | Critical | Models evaluated at $t_{\text{screen}}$ are strictly prohibited from consuming observations where $t > t_{\text{screen}}$. The feature matrix contains only historical slices $[t_0 \dots t_{\text{screen}}]$. |
| **Component Identity Leakage** | Critical | Train and test partitions are partitioned strictly by `component_id`. All 11 time steps of a test component are held out from training. |
| **Global Scaling Leakage** | Moderate | Preprocessing scalers (StandardScaler, RobustScaler, min/max) are fitted **only** on the training component folds, then applied to the held-out test components. |
