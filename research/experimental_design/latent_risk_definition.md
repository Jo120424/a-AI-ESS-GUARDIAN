# Latent Risk and Failure Definitions

## 1. Ground Truth vs. Research-Defined Proxy

To maintain absolute scientific transparency:
* **Ground-Truth Physical Measurement:** The empirical time series of capacitance drop ($\Delta C$) and ESR rise ($\Delta\text{ESR}$) recorded in NASA's `EOS_DataSet.mat`, and the subsequent crossing of the military standard **MIL-PRF-62F** End-of-Life specification threshold ($\Delta C \ge 20.0\%$).
* **Research-Defined Proxy Label:** The binary classification label ($y_i \in \{0, 1\}$) constructed by the researcher to benchmark whether a component passing early screening will fail during subsequent service life.

---

## 2. Evaluation and Ranking of Candidate Definitions

| Definition Option | Formulation | Empirical Feasibility in Dataset | Selected Role |
| :--- | :--- | :---: | :---: |
| **Definition B: Future Threshold Violation** | Component satisfies static limits at early screening cutoff $t_{\text{screen}}$ ($\Delta C < 20\%$) but subsequently violates the MIL-PRF-62F specification threshold ($\Delta C \ge 20\%$) during post-screening operation ($t_{\text{screen}} < t \le 194\,\text{h}$). | **100% Feasible** (Components C2, C3, C4, C5, C6 cross 20% spec; C1 passes throughout). | **PRIMARY DEFINITION (Rank 1)** |
| **Definition A: Future Premature Failure** | Component passes early screening but experiences catastrophic or parametric failure prior to design endurance milestone ($t_{\text{EOL}} \le 171\,\text{h}$). | **Feasible** (Component C4 fails early at 171h; C2, C3, C5, C6 fail at 194h). | **SECONDARY BENCHMARK (Rank 2)** |
| **Definition C: Abnormal Degradation Velocity** | Component passes static limits at $t_{\text{screen}}$ but exhibits a post-screening drift rate $\left.\frac{d\,\Delta C}{dt}\right|_{t > t_{\text{screen}}}$ exceeding the 75th percentile of the population. | **Feasible** (Drift velocity trajectories diverge sharply between C1 and C4). | **SENSITIVITY PROXY (Rank 3)** |
| **Definition D: Unsupervised Trajectory Outlier** | Future degradation path deviates by $> 2.5\times\text{MAD}$ from the lot median trajectory. | **Feasible**, but less directly tied to formal qualification specifications than Definition B. | **EXPLORATORY PROXY (Rank 4)** |

---

## 3. Mathematical Formulation of Primary Definition (Definition B)

Let $t_{\text{screen}}$ be the early screening cutoff horizon (e.g., $t_{\text{screen}} = 47\,\text{h}$ or $t_{\text{screen}} = 71\,\text{h}$).
Let $t_{\text{final}} = 194\,\text{h}$ be the end of the test duration.
Let $\theta_{\text{spec}} = 20.0\%$ be the standard MIL-PRF-62F capacitance degradation limit.

For any discrete physical component $i$:
$$y_i = \begin{cases} 1 \text{ (Latent Defect / Future Risk)}, & \text{if } \Delta C_i(t_{\text{screen}}) < \theta_{\text{spec}} \text{ AND } \max_{t_{\text{screen}} < t \le t_{\text{final}}} \Delta C_i(t) \ge \theta_{\text{spec}} \\ 0 \text{ (Healthy Survivor)}, & \text{if } \max_{0 \le t \le t_{\text{final}}} \Delta C_i(t) < \theta_{\text{spec}} \end{cases}$$

### Physical Grounding in Environmental Stress Screening
In aerospace electronics, the purpose of ESS is to precipitate and capture infant mortality defects:
1. At early burn-in ($t_{\text{screen}} = 47\,\text{h}$), latent defective parts show subtle incubation symptoms (e.g. $\Delta C \approx 1.7\%$) that easily pass static DC inspection limits.
2. If shipped to flight assemblies, these units experience accelerated parametric runaway and fail in the field.
3. This mathematical definition perfectly models the true operational problem without relying on synthetic or artificial label injections.
