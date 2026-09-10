# Comparative Methodological Analysis: Level 1 vs Level 2 vs Level 3

## 1. Paradigm Comparison Matrix

The table below contrasts the theoretical and operational properties of the three screening paradigms evaluated in this research:

| Property | Level 1: Traditional Screening | Level 2: Dynamic Statistical Screening | Level 3: AI / ML Screening (Isolation Forest) |
| :--- | :--- | :--- | :--- |
| **Fixed Thresholds** | **YES** (Static datasheet limits: $\Delta C \ge 20\%$) | **NO** (Adapts to training lot distribution) | **NO** (Learns geometric density envelope) |
| **Population Adaptive** | **NO** (Evaluates units in complete isolation) | **YES** (Normalizes by median & MAD of cohort) | **YES** (Learns distribution manifold from lot) |
| **Multivariate Modeling** | **NO** (Univariate scalar checks combined via OR) | **YES** (Mahalanobis distance $D_M$ on covariance) | **YES** (Joint non-linear multi-attribute isolation) |
| **Temporal Dynamic Rates** | **NO** (Static snapshot value only) | **YES** (Evaluates drift velocity $\frac{d\,\Delta C}{dt}$) | **YES** (Consumes multi-step drift vectors) |
| **Requires Failure Labels?** | **NO** (Uses specification limits) | **NO** (Unsupervised outlier detection) | **NO** (Unsupervised anomaly detection) |
| **Handles Unknown Anomaly Types?** | **NO** (Catches only pre-defined threshold breaches) | **MODERATE** (Detects linear covariance outliers) | **HIGH** (Isolates arbitrary non-linear cluster deviations) |
| **Explainability Traceability** | **DIRECT** (Scalar violation margin $\Delta - \theta$) | **EXCELLENT** (Partial Mahalanobis contribution) | **GOOD** (Tree-depth split & SHAP values) |
| **Computational Complexity** | Minimal ($\mathcal{O}(1)$ comparisons) | Low ($\mathcal{O}(N p^2)$ matrix inversion) | Moderate ($\mathcal{O}(T \cdot n \log n)$ ensemble trees) |
| **Implementation Footprint** | Extremely lightweight; standard test equipment | Simple vector routines; deployable on microcontrollers | Python runtime / scikit-learn model serialization |
| **Main Advantage** | Highly standardized; universally understood in contracts | Detects lot outliers without complex hyperparameter tuning | Captures subtle non-linear interactions & early kinetic drift |
| **Main Limitation** | Completely blind to latent incubation defects | Assumes unimodal/elliptical outlier distributions | Sensitive to sample size; requires careful validation tuning |

---

## 2. Complementary Hybrid Screening Vision

Rather than viewing the three paradigms as mutually exclusive adversaries, this research also lays the groundwork for an operational **hierarchical defense-in-depth pipeline**:
1. **Stage 1 (Traditional Filter):** Rapidly screens out gross functional opens/shorts and catastrophic manufacturing flaws at zero computational overhead.
2. **Stage 2 (Dynamic Statistical Monitor):** Flags clear population outliers and lot variance shifts using robust Z-scores and Mahalanobis distances.
3. **Stage 3 (AI/ML Predictive Guard):** Evaluates remaining borderline parts to detect subtle multi-parameter degradation trajectories and latent infant mortality incubation.
