# Level 2: Dynamic Statistical Screening Design

## 1. Concept & Operational Principle

Dynamic Statistical Screening addresses the blind spot of static thresholding by evaluating each component **relative to its peer production lot / stress cohort**:
* Instead of asking *"Is the parameter below the absolute military ceiling?"*, it asks:
  > *"Is this component's drift or degradation trajectory an anomalous outlier compared to the normal population of its production cohort?"*
* Establishes dynamic statistical process control (SPC) bounds adapted to the specific lot's empirical distribution at screening time $t_{\text{screen}}$.

---

## 2. Selected Statistical Screening Methods

Based on the empirical properties of the NASA Capacitor dataset ($N = 6$ discrete units, continuous time-series, multivariate drift), two primary statistical screening formulations are designed:

### Method 2A: Univariate Robust Z-Score via Median Absolute Deviation (MAD)
Standard Z-scores based on mean and standard deviation are notoriously susceptible to masking when outliers pull the sample mean and inflate the variance. We utilize the **Hampel / Rousseeuw Robust Z-score**:

$$\text{RZ}_{i,j}(t_{\text{screen}}) = \frac{x_{i,j}(t_{\text{screen}}) - \text{Median}_k(x_{k,j}(t_{\text{screen}}))}{1.4826 \times \text{MAD}_k(x_{k,j}(t_{\text{screen}}))}$$

Where:
* $x_{i,j}(t_{\text{screen}})$ is feature $j$ (e.g., $\Delta C$ or drift velocity $\frac{d\,\Delta C}{dt}$) for test component $i$.
* $\text{MAD}(X) = \text{Median}(|X - \text{Median}(X)|)$.
* The constant $1.4826$ normalizes the MAD to equal the standard deviation for normally distributed data.
* **Decision Rule:**
  $$\hat{y}_{i,\text{RZ}} = \begin{cases} 1 \text{ (REJECT / Outlier)}, & \text{if } \max_j |\text{RZ}_{i,j}(t_{\text{screen}})| \ge \theta_{\text{RZ}} \\ 0 \text{ (PASS)}, & \text{otherwise} \end{cases}$$
  Where $\theta_{\text{RZ}} = 2.5$ represents the $98.7\%$ empirical confidence threshold.

---

### Method 2B: Multivariate Mahalanobis Distance ($D_M$)
Capacitance loss and ESR increase are physically coupled degradation mechanisms. A component may exhibit a subtle shift in both parameters that appears marginally normal univariately, but represents a severe breakdown in bivariate correlation.

The Mahalanobis distance in feature space $\mathbf{x} = [\Delta C, \Delta\text{ESR}, \frac{d\,\Delta C}{dt}]$ is formulated as:

$$D_M(\mathbf{x}_i) = \sqrt{(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{\text{train}})^T \hat{\boldsymbol{\Sigma}}_{\text{robust}}^{-1} (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{\text{train}})}$$

Where:
* $\hat{\boldsymbol{\mu}}_{\text{train}}$ is the robust multivariate location estimator (Minimum Covariance Determinant or coordinate median) fitted **strictly on the training fold**.
* $\hat{\boldsymbol{\Sigma}}_{\text{robust}}$ is the regularized covariance matrix (using Ledoit-Wolf shrinkage to ensure well-conditioned inversion for small cohorts).
* **Decision Rule:**
  $$\hat{y}_{i,\text{MD}} = \begin{cases} 1 \text{ (REJECT / Multivariate Anomaly)}, & \text{if } D_M^2(\mathbf{x}_i) \ge \chi^2_{p}(1 - \alpha) \\ 0 \text{ (PASS)}, & \text{otherwise} \end{cases}$$
  Where $p$ is feature dimensionality and $\alpha = 0.05$ or $0.10$ is the significance cutoff.

---

## 3. Small-Cohort Adaptation & Regularization Safeguards

Because the cohort consists of 6 components:
1. **Shrinkage Regularization:** The covariance matrix $\boldsymbol{\Sigma}$ is regularized via Ledoit-Wolf shrinkage:
   $$\boldsymbol{\Sigma}_{\text{reg}} = (1 - \lambda)\boldsymbol{\Sigma}_{\text{emp}} + \lambda \nu \mathbf{I}$$
   Preventing singular matrix inversion when $p \approx N$.
2. **Leave-One-Component-Out (LOCO) Parameter Estimation:**
   When evaluating test component $i$, the median, MAD, and covariance parameters are fitted **exclusively on the remaining $N - 1$ training components**. Component $i$ is never included in the calculation of its own baseline distribution.
3. **Strict Pre-$t_{\text{screen}}$ Calculation:**
   All statistical moments are computed using observations strictly at or before $t_{\text{screen}}$.
