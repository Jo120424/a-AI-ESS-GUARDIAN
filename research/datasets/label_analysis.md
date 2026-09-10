# Failure and Degradation Analysis: Ground Truth vs. Research Proxy Labels

## 1. Overview & Research Integrity Mandate
In accordance with Section 10 and Non-Negotiable Research Rule #8:
> *"Never call proxy labels ground truth. Clearly distinguish between ground-truth labels directly provided by the original dataset and research-defined proxy labels."*

This document provides a rigorous breakdown of labels and physical state indicators present in the NASA Capacitor Electrical Stress Degradation dataset.

---

## 2. Ground-Truth Data Provided by Original Dataset

The raw file (`EOS_DataSet.mat`) originates from controlled accelerated electrical overstress experiments published by NASA Ames PCoE (Celaya et al., 2012). It provides the following **empirical ground-truth physical measurements**:

1. **Longitudinal Degradation Trajectories:**
   * Measured percentage capacitance drop ($\Delta C(t)$) from baseline $t=0$.
   * Measured percentage Equivalent Series Resistance increase ($\Delta\text{ESR}(t)$) from baseline $t=0$.
   * Discrete time sequence ($t \in \{0, 24, 47, 71, 94, 116, 139, 149, 161, 171, 194\}\,\text{hours}$).
2. **Physical Failure Threshold Violations:**
   * Under military specification **MIL-PRF-62F** and EIA-395 standards, an aluminum electrolytic capacitor is classified as having reached End-of-Life (EOL) when:
     $$\Delta C(t) \ge 20.0\% \quad \text{or} \quad \Delta\text{ESR}(t) \ge 100.0\% \quad (\text{doubling of nominal ESR})$$
   * In the raw dataset:
     * Component `C4` crosses $\Delta C = 20.04\%$ at $t = 171\,\text{h}$.
     * Components `C2`, `C3`, `C5`, and `C6` cross $20.0\%$ at $t = 194\,\text{h}$.
     * Component `C1` reaches $\Delta C = 17.45\%$ at $t = 194\,\text{h}$ (survives without static spec violation).

---

## 3. Absence of Native "Latent Defect" Labels in Raw Data

* **Critical Fact:** The raw NASA dataset does **NOT** contain a categorical column named `latent_defect`.
* **Reason:** In physical reliability engineering, a "latent defect" is not a static property etched on a device at manufacture; it is an internal flaw (e.g. microscopic dielectric thinning, electrolyte seal flaw, or interfacial contamination) that remains invisible under initial static tests but accelerates degradation under operational stress.
* **Non-Fabrication Policy:** We will **never** claim that the original NASA authors provided binary latent defect labels. Instead, any latent defect classification must be formally defined as a **scientifically justified research proxy**.

---

## 4. Research-Defined Proxy Labels for Comparative Screening

To answer the core research question:
> *"Can dynamic, data-driven screening detect components at risk of future/latent failure that conventional static screening allows to pass?"*

We formulate two mathematically transparent proxy label candidates for evaluation:

### Proxy Label A: End-of-Life Cutoff Proxy (`label_latent_eol`)
* **Definition:**
  $$\text{Latent Risk}(i, t_{\text{screen}}) = \begin{cases} 1, & \text{if } \Delta C_i(t_{\text{screen}}) < 20\% \text{ (Passes Static Spec at } t_{\text{screen}}\text{)} \\ & \text{AND } \Delta C_i(t_{\text{final}}) \ge 20\% \text{ (Violates Spec at or before EOL Horizon)} \\ 0, & \text{otherwise} \end{cases}$$
* **At Early Screening Horizon $t_{\text{screen}} = 47\,\text{h}$:**
  * All 6 components have $\Delta C < 1.9\%$ (100% Traditional PASS).
  * Units `C2, C3, C4, C5, C6` fail before or at $194\,\text{h}$ ($\text{Label} = 1$).
  * Unit `C1` does not fail ($\text{Label} = 0$).

### Proxy Label B: Accelerated Degradation Drift Rate Proxy (`label_fast_degrader`)
* **Definition:**
  $$\text{Fast Degrader}(i) = \begin{cases} 1, & \text{if } \left.\frac{d\,\Delta C_i}{dt}\right|_{t > t_{\text{screen}}} > \text{Cohort Median Slope} \\ 0, & \text{otherwise} \end{cases}$$
* Identifies components whose latent internal degradation rate diverges significantly from the robust peer population under continued stress.

---

## 5. Temporal Leakage Prevention Mandate

To prevent target leakage:
1. When screening is performed at cutoff $t_{\text{screen}}$, screening models are strictly quarantined from all observations recorded at $t > t_{\text{screen}}$.
2. Proxy labels are computed **exclusively** on the future trajectory ($t > t_{\text{screen}}$) to serve as the unseen ground-truth evaluation benchmark.
3. Feature engineering at $t_{\text{screen}}$ uses only historical measurements $[t_0 \dots t_{\text{screen}}]$.
