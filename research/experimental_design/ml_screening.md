# Level 3: Machine Learning Screening Design

## 1. Concept & AI Screening Philosophy

Machine Learning Screening aims to learn complex, non-linear geometric boundaries and multivariate interactions that simple linear statistics or single thresholds cannot capture:
* **Unsupervised Anomaly Paradigm:** Because true infant mortality and latent defect occurrences are rare in real production lines, unsupervised and semi-supervised anomaly detection is the most operationally realistic ML paradigm.
* The system learns what constitutes **nominal baseline behavior** during early burn-in, and flags any component whose trajectory occupies low-density or anomalous regions of the learned manifold.

---

## 2. Primary Model: Isolation Forest

In accordance with Section 9, the primary AI model is **Isolation Forest (iForest)**.

### Mathematical Formulation
Isolation Forest exploits two quantitative properties of anomalous components:
1. They are few in number.
2. They possess attribute values markedly divergent from nominal points.

An ensemble of $T$ isolation trees is constructed by recursively partitioning the feature space with random axis-aligned splits. The anomaly score $s(\mathbf{x}, n)$ for test component representation $\mathbf{x}$ is defined as:

$$s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}(h(\mathbf{x}))}{c(n)}}$$

Where:
* $h(\mathbf{x})$ is the path length (number of edges traversed from the root to the terminating leaf node).
* $\mathbb{E}(h(\mathbf{x}))$ is the average path length across all $T$ trees in the forest.
* $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree constructed with $n$ samples:
  $$c(n) = 2\left(\ln(n - 1) + 0.5772156649\right) - \frac{2(n - 1)}{n}$$
* **Score Interpretation:**
  * When $\mathbb{E}(h(\mathbf{x})) \to 0$, path length is short $\implies s \to 1$ (High Anomaly / Latent Defect Risk).
  * When $\mathbb{E}(h(\mathbf{x})) \to c(n)$, path length is average $\implies s \approx 0.5$ (Nominal Behavior).
  * When $\mathbb{E}(h(\mathbf{x})) \to n - 1$, path length is long $\implies s \to 0$ (Firmly Normal Cluster).

### Decision Rule
$$\hat{y}_{i,\text{iForest}} = \begin{cases} 1 \text{ (REJECT / Anomaly)}, & \text{if } s(\mathbf{x}_i) \ge \tau_{\text{iForest}} \\ 0 \text{ (PASS)}, & \text{otherwise} \end{cases}$$
Where threshold $\tau_{\text{iForest}}$ is calibrated on the training fold validation boundary (e.g. 90th percentile).

---

## 3. Secondary Model: One-Class Support Vector Machine (OC-SVM)

To cross-verify non-linear boundary estimation, a secondary kernelized estimator is designed:
* Uses a Radial Basis Function (RBF) kernel: $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$.
* Maps training observations into a reproducing kernel Hilbert space and finds the maximal margin hyperplane separating the nominal data from the origin:
  $$\min_{\mathbf{w}, \boldsymbol{\xi}, \rho} \frac{1}{2}\|\mathbf{w}\|^2 + \frac{1}{\nu n}\sum_{k=1}^n \xi_k - \rho \quad \text{subject to } \langle \mathbf{w}, \Phi(\mathbf{x}_k)\rangle \ge \rho - \xi_k, \xi_k \ge 0$$
* **Decision Function:**
  $$f(\mathbf{x}_i) = \text{sign}\left(\sum_{k=1}^n \alpha_k K(\mathbf{x}_k, \mathbf{x}_i) - \rho\right)$$
  Negative values indicate rejection.

---

## 4. Input Feature Vectors at Screening Horizon $t_{\text{screen}}$

The feature vector $\mathbf{x}_i(t_{\text{screen}})$ presented to the ML model strictly encapsulates historical measurements:
$$\mathbf{x}_i = \left[\Delta C_i(t_{\text{screen}}), \Delta\text{ESR}_i(t_{\text{screen}}), \left.\frac{d\,\Delta C_i}{dt}\right|_{t_{\text{screen}}}, \left.\frac{d\,\Delta\text{ESR}_i}{dt}\right|_{t_{\text{screen}}}, \Delta C_i(t_{\text{screen}}) - \Delta C_i(t_0)\right]$$

### Non-Leakage Pipeline Integration
1. The ML models are trained **only on training fold components** using observations up to $t_{\text{screen}}$.
2. Feature scalers (`RobustScaler`) are fit exclusively on the training component folds.
3. Unseen test components are transformed and evaluated using the frozen model.
