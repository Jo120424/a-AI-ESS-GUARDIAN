# Statistical Significance and Hypothesis Testing Plan

## 1. Beyond Raw Point Estimates

Reporting isolated percentage differences (e.g. *"AI detected 80% vs Traditional 0%"*) is insufficient for scientific credibility. The research design incorporates formal statistical significance testing to evaluate whether observed performance differentials are statistically meaningful or attributable to sampling variance:

$$\text{Differential Effect} = \text{Metric}_{\text{ML}} - \text{Metric}_{\text{Traditional}}$$

---

## 2. Planned Statistical Tests

### Test 1: McNemar's Exact Test for Paired Classifications
Because Level 1, Level 2, and Level 3 screening models evaluate the **exact same test components**, their binary decisions ($\hat{y}_{\text{Trad},i}$ vs. $\hat{y}_{\text{ML},i}$) are naturally paired.

We construct the $2 \times 2$ contingency table of paired model outcomes:
```text
                       ML / Dynamic Screening
                       Correct (1)     Incorrect (0)
Traditional   Correct       a               b
Screening     Incorrect     c               d
```
* $a$: Both models correct.
* $b$: Traditional correct, ML incorrect.
* $c$: Traditional incorrect, ML correct (**Discordant pairs favoring ML**).
* $d$: Both models incorrect.

Under the primary null hypothesis ($H_0: p_b = p_c$), the two screening methods have identical error rates.
For small sample sizes ($b + c < 25$), the **Exact Binomial Test** is used:
$$p = 2 \sum_{k=c}^{b+c} \binom{b+c}{k} 0.5^{b+c}$$

### Test 2: Non-Parametric Paired Bootstrap Confidence Intervals
To quantify uncertainty around performance metrics (Recall, Precision, F1, FNR):
1. Resample test component predictions with replacement for $B = 2,000$ bootstrap iterations.
2. For each bootstrap replicate $b$, compute the paired differential:
   $$\Delta \text{Recall}^{(b)} = \text{Recall}_{\text{ML}}^{(b)} - \text{Recall}_{\text{Trad}}^{(b)}$$
3. Construct the two-sided $95\%$ percentile confidence interval:
   $$\text{CI}_{95\%} = \left[\Delta^{(0.025)}, \Delta^{(0.975)}\right]$$
4. If $0 \notin \text{CI}_{95\%}$, the null hypothesis of equal performance is rejected at $\alpha = 0.05$.

### Test 3: Permutation Testing for Feature Precursors
To test secondary hypothesis $H_{0,2}$ (zero correlation between early velocity $\frac{d\,\Delta C}{dt}$ and failure latency $t_{\text{fail}}$):
* Compute the empirical Spearman rank correlation $\rho_{\text{emp}}$.
* Permute failure times across components $M = 10,000$ times to establish the empirical null distribution.
* Compute exact permutation p-value:
  $$p_{\text{perm}} = \frac{1}{M} \sum_{m=1}^M \mathbb{I}(|\rho^{(m)}| \ge |\rho_{\text{emp}}|)$$

---

## 3. Transparency on Sample Size & Power
* **Sample Size Transparency:** The test cohort contains 6 physical units across 11 longitudinal cycles.
* While longitudinal depth per component is high, the sample size along the component dimension ($N = 6$) has limited asymptotic statistical power.
* Therefore, the study will explicitly report **exact non-parametric p-values and bootstrap confidence intervals**, avoiding inflated claims of significance where statistical power is constrained.
