# Research Hypotheses and Scientific Questions

## 1. Primary Research Question
> **“Can a dynamic AI/ML-based screening system detect electronic components at risk of latent failure that traditional static specification screening allows to pass?”**

---

## 2. Secondary Research Questions

1. **Comparative Statistical Superiority:**
   Does dynamic statistical screening (Level 2: lot-relative robust Z-scores, Mahalanobis distance) yield a statistically significant reduction in false negatives (test escapes) compared to traditional static specification screening (Level 1: MIL-PRF-62F)?
2. **AI/ML Incremental Utility:**
   Does unsupervised machine learning (Level 3: Isolation Forest, One-Class SVM) improve detection sensitivity or early-detection lead time over both static thresholding and dynamic statistical techniques?
3. **Prognostic Foresight of Early Telemetry:**
   Do early stress measurements (e.g. at $t \le 47\,\text{h}$, representing ~24% of stress life) contain sufficient physical and statistical precursor signal to forecast downstream parametric degradation ($t = 194\,\text{h}$)?
4. **Multivariate Synergy:**
   Does joint modeling of multi-parameter dynamics (Capacitance drop $\Delta C$ and ESR growth $\Delta\text{ESR}$ combined with drift velocity $\frac{d\,\Delta C}{dt}$) identify latent defect risks more accurately than isolated, univariate thresholding?
5. **Detection-Yield Trade-Off:**
   Can an AI/ML screening engine detect latent failures without triggering an unacceptable increase in False Positive Rate (unnecessary scrap/yield loss $\text{FPR} > 15\%$)?
6. **Cross-Component Generalization:**
   Do the learned anomaly bounds generalize across unseen discrete physical components under rigorous Leave-One-Component-Out (LOCO) cross-validation?

---

## 3. Formal Statistical Hypotheses

### Primary Hypothesis Pair

* **Null Hypothesis ($H_{0,1}$):**
  There is no statistically significant improvement in the detection recall of latent failure components between the dynamic/ML screening approaches and conventional static screening when evaluated on the exact same unseen test components at the early screening cutoff ($t_{\text{screen}} = 47\,\text{h}$):
  $$H_{0,1}: \text{Recall}_{\text{ML}} = \text{Recall}_{\text{Traditional}} = \text{Recall}_{\text{Statistical}}$$

* **Alternative Hypothesis ($H_{1,1}$):**
  At least one dynamic or machine learning screening method achieves a statistically significant improvement in detection recall ($\text{Recall} > \text{Recall}_{\text{Traditional}}$) without exceeding the maximum allowable false positive rate ceiling ($\text{FPR} \le 15\%$):
  $$H_{1,1}: \text{Recall}_{\text{ML}} > \text{Recall}_{\text{Traditional}} \quad \text{subject to } \text{FPR}_{\text{ML}} \le 0.15$$

### Secondary Hypothesis Pair (Early Kinetic Precursor)

* **Null Hypothesis ($H_{0,2}$):**
  The initial degradation velocities ($\frac{d\,\Delta C}{dt}$ and $\frac{d\,\Delta\text{ESR}}{dt}$) observed during early burn-in ($t \le 47\,\text{h}$) share zero correlation or predictive mutual information with final failure times ($t_{\text{EOL}}$):
  $$H_{0,2}: \rho\left(\left.\frac{d\,\Delta C}{dt}\right|_{t=47\text{h}}, t_{\text{EOL}}\right) = 0$$

* **Alternative Hypothesis ($H_{1,2}$):**
  Early kinetic drift rates are statistically correlated ($\rho < 0, p < 0.05$) with downstream failure latency, providing an empirical basis for dynamic screening.

---

## 4. Scientific Neutrality Commitment
This research adheres to strict scientific objectivity:
* The alternative hypotheses ($H_1$) are **not assumed to be true** prior to experimental execution.
* If empirical evaluation shows that simple robust statistics or traditional static rules outperform complex AI algorithms, or if AI exhibits excessive false alarm penalties, the final research conclusion will unequivocally report that finding.
