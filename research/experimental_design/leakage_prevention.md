# Data Leakage Prevention and Verification Protocol

## 1. The Six Vulnerabilities of Screening AI Experiments

In predictive reliability screening, standard ML practices frequently introduce subtle forms of data leakage that invalidate research conclusions. This document defines the **six leakage vulnerabilities** and the mandatory programmatic safeguards implemented in our experimental engine.

---

## 2. Leakage Vulnerability Taxonomy & Architectural Safeguards

```text
                                       SIX LEAKAGE THREATS
                                                │
         ┌───────────────────┬──────────────────┼──────────────────┬──────────────────┐
         ▼                   ▼                  ▼                  ▼                  ▼
   [ COMPONENT ]       [ TEMPORAL ]       [ PREPROCESSING ]  [ THRESHOLD ]      [ LABEL / TARGET ]
   Same unit in        Future data        Global scaling     Limits tuned       Ground truth used
   train & test        in early step      across all folds   on test splits     as input feature
```

### Threat 1: Component Leakage
* **Threat:** Repeated measures from component $C_k$ appear in both training and test partitions.
* **Impact:** Model memorizes component-specific random intercepts rather than generalizable degradation physics.
* **Safeguard:** Group-based splitting using `GroupKFold` / LOCO strictly grouped on `component_id`. An automated assertion validates:
  $$\text{Set}(\text{train\_components}) \cap \text{Set}(\text{test\_components}) = \emptyset$$

### Threat 2: Temporal Leakage (Lookahead Bias)
* **Threat:** Features at screening cutoff $t_{\text{screen}}$ utilize forward-looking statistics, centered moving averages, or post-$t_{\text{screen}}$ measurements.
* **Impact:** The algorithm predicts the future using knowledge of the future.
* **Safeguard:** Feature matrices are constructed strictly from time slices where $t \le t_{\text{screen}}$. Derivatives use purely causal backward differences ($\frac{x(t) - x(t - \Delta t)}{\Delta t}$).

### Threat 3: Preprocessing & Scaling Leakage
* **Threat:** `StandardScaler`, `MinMaxScaler`, or `RobustScaler` is fitted on the entire dataset before train/test splitting.
* **Impact:** Test set distribution moments leak into training representations.
* **Safeguard:** All transformer classes are instantiated inside `sklearn.pipeline.Pipeline` or fitted **strictly within the training fold loop**.

### Threat 4: Threshold Selection Leakage
* **Threat:** Anomaly rejection thresholds $\tau$ are tuned to maximize test set F1 or accuracy.
* **Impact:** The reported metric reflects overfitted post-hoc threshold alignment rather than prospective screening utility.
* **Safeguard:** Thresholds are calibrated strictly on training folds (e.g. 95th percentile of training anomaly scores or validation ROC curve) and frozen before application to test partitions.

### Threat 5: Label / Target Leakage
* **Threat:** Derived ground-truth target columns (`component_ever_fails`, `first_failure_step`, `static_spec_fail` at $t_{\text{final}}$) are inadvertently included in the feature set.
* **Impact:** Trivial 100% classification accuracy with zero real-world predictive validity.
* **Safeguard:** Feature input column list is explicitly whitelisted; all columns matching target patterns are programmatically stripped from feature matrices $\mathbf{X}$.

### Threat 6: Lot Leakage
* **Threat:** When testing unseen-lot generalization, batch statistics from the test lot are incorporated into the training reference baseline.
* **Impact:** Dynamic screening appears to generalize across foundries when it is merely memorizing batch variance.
* **Safeguard:** Test lot data is quarantined completely until final frozen evaluation.

---

## 3. Automated Leakage Verification Test Suite
All six safeguards are programmatically verified by automated assertions embedded within `backend/experiments/ground_truth.py` and `tests/test_data_pipeline.py`. Any leakage event raises a fatal `DataLeakageError`, aborting execution immediately.
