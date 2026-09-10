# Level 1: Traditional Screening Design

## 1. Concept & Operational Principle

Traditional screening represents conventional industrial and aerospace screening methodology:
* Evaluates components **statically and independently** against absolute specification thresholds.
* Ignores the distribution or variance of the peer production lot.
* Ignores the temporal drift velocity or acceleration of degradation.
* Flow:
  $$\text{Component Measurement at } t_{\text{screen}} \longrightarrow \text{Static Thresholds } [\theta_C, \theta_{\text{ESR}}] \longrightarrow \text{PASS or REJECT}$$

---

## 2. Specification Grounding (MIL-PRF-62F & EIA-395)

In accordance with Option 1 of the research protocol:
* **Primary Specification Baseline (Military Standard):**
  Under **MIL-PRF-62F** (Performance Specification for Capacitors, Fixed, Electrolytic, Aluminum Oxide) and Vishay/Nichicon commercial screening datasheets:
  1. **Capacitance Degradation Threshold ($\theta_C$):**
     $$\Delta C(t_{\text{screen}}) = \frac{C_0 - C(t_{\text{screen}})}{C_0} \times 100\% \ge 20.0\% \implies \text{REJECT}$$
  2. **ESR Degradation Threshold ($\theta_{\text{ESR}}$):**
     $$\Delta\text{ESR}(t_{\text{screen}}) = \frac{\text{ESR}(t_{\text{screen}}) - \text{ESR}_0}{\text{ESR}_0} \times 100\% \ge 100.0\% \quad (\text{Doubling of nominal ESR}) \implies \text{REJECT}$$

* **Tightened ESS Specification Baseline (Experimental Variant):**
  Some high-reliability screening programs specify an interim post-burn-in drift limit (e.g., maximum permissible burn-in drift $\Delta C \le 5.0\%$ or $10.0\%$):
  $$\theta_{C,\text{tight}} = 5.0\% \quad \text{and} \quad \theta_{\text{ESR},\text{tight}} = 30.0\%$$
  This tightened variant will be evaluated alongside the standard 20% spec to ensure an exhaustive comparison.

---

## 3. Decision Rule Formulation

For any test component $i$ evaluated at screening cutoff horizon $t_{\text{screen}}$:
$$\hat{y}_{i,\text{Trad}} = \begin{cases} 1 \text{ (REJECT / Anomaly)}, & \text{if } \Delta C_i(t_{\text{screen}}) \ge \theta_C \quad \text{OR} \quad \Delta\text{ESR}_i(t_{\text{screen}}) \ge \theta_{\text{ESR}} \\ 0 \text{ (PASS / Accept)}, & \text{otherwise} \end{cases}$$

### Multi-Parameter Union Rule
The rejection criteria uses a logical **OR** combination: a component is rejected if *any* critical parameter breaches its specification ceiling, reflecting conservative flight-hardware screening policy.

---

## 4. Empirical Behavior at Early Screening Windows

Applying this rule to the empirical NASA Capacitor dataset reveals the fundamental limitation of traditional screening:

| Screening Horizon | Max Observed $\Delta C$ in Lot | Max Observed $\Delta\text{ESR}$ in Lot | Traditional Decision ($\theta_C=20\%$) | Traditional Escape Rate |
| :---: | :---: | :---: | :---: | :---: |
| **$t_{\text{screen}} = 24\,\text{h}$** | $0.83\%$ (Unit C4) | $9.89\%$ (Unit C6) | **PASS for all 6 units** (0% Rejected) | **100% Escape of Latent Defects** |
| **$t_{\text{screen}} = 47\,\text{h}$** | $1.86\%$ (Unit C4) | $17.46\%$ (Unit C2) | **PASS for all 6 units** (0% Rejected) | **100% Escape of Latent Defects** |
| **$t_{\text{screen}} = 71\,\text{h}$** | $2.48\%$ (Unit C4) | $25.24\%$ (Unit C4) | **PASS for all 6 units** (0% Rejected) | **100% Escape of Latent Defects** |

### Scientific Conclusion for Traditional Screening
Because latent defects incubate with subtle early parameter changes well below catastrophic failure limits, **traditional static screening allows 100% of future-failing components to pass undetected into service**. This empirical finding precisely establishes the research baseline.
