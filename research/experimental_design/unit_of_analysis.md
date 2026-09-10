# Unit of Analysis Definition

## 1. Primary Decision Unit: The Physical Component (`component_id`)

The primary unit of analysis for this research is the **individual physical electronic component** ($i \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}\}$).

### Operational Rationale
In real-world manufacturing and high-reliability screening (aerospace, satellite, defense, medical), the fundamental engineering decision is binary at the part level:
$$\text{Decision}(i) \in \{\text{ACCEPT (Pass to Assembly)}, \text{REJECT (Scrap / Disposition)}\}$$

The screening facility does not accept or reject individual time-slice measurements; it must make a definitive disposition on whether the **physical component as an integrated hardware unit** is safe for integration into a flight subsystem.

---

## 2. Rejection of Measurement-Level Pooling

Treating individual measurement rows ($N = 66$) as independent samples for classification evaluation is **scientifically invalid** in this context:

| Evaluation Level | Scientific Validity | Why It Fails or Succeeds |
| :--- | :---: | :--- |
| **Measurement-Level (Row Pooling)** | **INVALID** | Highly autocorrelated repeated measures from the same physical device would span both train and test splits, causing catastrophic **component data leakage** and falsely inflated performance claims. |
| **Test-Cycle Level (Per-Step Slicing)** | Secondary / Diagnostic | Slices observations at discrete inspection milestones ($t = 24\text{h}, 47\text{h}, 71\text{h}$) to evaluate how screening accuracy evolves as a function of burn-in duration. |
| **Component-Level (Part Decision)** | **PRIMARY VALID** | Every physical unit is evaluated as an indivisible entity. All historical telemetry up to screening cutoff $t_{\text{screen}}$ forms the input vector, and the decision applies to that entire unit. |
| **Lot-Level (Batch Acceptance)** | Contextual / Grouping | Provides the statistical reference population against which individual component outlier distances are normalized. |

---

## 3. Formal Data Structure for Evaluation

For an experiment evaluated at screening cutoff horizon $t_{\text{screen}}$:
1. Each test sample is an unseen discrete component $i$.
2. The input feature representation $\mathbf{X}_i(t_{\text{screen}})$ aggregates historical telemetry up to $t_{\text{screen}}$:
   $$\mathbf{X}_i(t_{\text{screen}}) = \left[\Delta C_i(t_{\text{screen}}), \Delta\text{ESR}_i(t_{\text{screen}}), \left.\frac{d\,\Delta C_i}{dt}\right|_{t_{\text{screen}}}, \left.\frac{d\,\Delta\text{ESR}_i}{dt}\right|_{t_{\text{screen}}}, \dots\right]$$
3. The screening engine outputs a component-level prediction:
   $$\hat{y}_i \in \{0 \text{ (Pass)}, 1 \text{ (Reject / Latent Risk)}\}$$
4. The ground truth $y_i$ is evaluated on the component's future operational trajectory:
   $$y_i \in \{0 \text{ (Safe Survivor)}, 1 \text{ (Downstream Failure)}\}$$
