# Data Split and Cross-Validation Strategy

## 1. Non-Negotiable Anti-Leakage Partitioning Rule

In reliability and screening research:
> **NEVER randomly split individual measurement rows across train and test partitions when repeated measurements belong to the same physical hardware unit.**

Random row-level splitting causes catastrophic data leakage: the model memorizes the specific physical unit's unique baseline bias during training, producing artificially inflated metrics that fail completely when deployed on genuinely unseen production parts.

---

## 2. Primary Evaluation Strategy: Leave-One-Component-Out (LOCO) Cross-Validation

Given the cohort sample size of $N = 6$ discrete components (`C1` through `C6`), the gold-standard evaluation protocol is **Leave-One-Component-Out (LOCO)** cross-validation:

```text
======================= LEAVE-ONE-COMPONENT-OUT (LOCO) =======================
  Fold 1:  Train on [C2, C3, C4, C5, C6] (t ≤ t_screen)  │  Test on [C1] (t ≤ t_screen)
  Fold 2:  Train on [C1, C3, C4, C5, C6] (t ≤ t_screen)  │  Test on [C2] (t ≤ t_screen)
  Fold 3:  Train on [C1, C2, C4, C5, C6] (t ≤ t_screen)  │  Test on [C3] (t ≤ t_screen)
  Fold 4:  Train on [C1, C2, C3, C5, C6] (t ≤ t_screen)  │  Test on [C4] (t ≤ t_screen)
  Fold 5:  Train on [C1, C2, C3, C4, C6] (t ≤ t_screen)  │  Test on [C5] (t ≤ t_screen)
  Fold 6:  Train on [C1, C2, C3, C4, C5] (t ≤ t_screen)  │  Test on [C6] (t ≤ t_screen)
=============================================================================
```

### Execution Protocol for Each Fold $k \in \{1 \dots 6\}$:
1. **Component Isolation:** Component $k$ is quarantined completely as the **unseen test unit**.
2. **Temporal Slicing:** Only training component records where $t \le t_{\text{screen}}$ are passed to model fitting.
3. **Scaler & Model Fitting:**
   * Feature scalers (`RobustScaler`) fit strictly on the 5 training units.
   * Statistical baselines (Median, MAD, Covariance $\boldsymbol{\Sigma}_{\text{reg}}$) fit strictly on the 5 training units.
   * AI/ML estimators (Isolation Forest, OC-SVM) fit strictly on the 5 training units.
4. **Frozen Inference:** Test component $k$'s pre-$t_{\text{screen}}$ telemetry is projected through the frozen pipeline to generate screening prediction $\hat{y}_k$.
5. **Ground Truth Comparison:** $\hat{y}_k$ is compared against $y_k$ (derived from component $k$'s future trajectory $t > t_{\text{screen}}$).

---

## 3. Generalization to Unseen Production Lots

To evaluate whether models generalize beyond the training lot:
* When multi-lot datasets are available (e.g. 10V, 12V, 14V stress cohorts), the pipeline supports **Leave-One-Lot-Out (LOLO)** cross-validation:
  $$\text{Train on Lot } A \text{ and } B \longrightarrow \text{Evaluate on completely unseen Lot } C$$
* This tests whether dynamic thresholds adapt to different manufacturing batches and stress acceleration factors.

---

## 4. Chronological Rolling-Origin Validation

For temporal sequence evaluations:
* Training observations are restricted to $[t_0 \dots t_{\text{screen}}]$.
* Testing evaluates the component's state at $t_{\text{screen}}$.
* Future observations ($t > t_{\text{screen}}$) are **never** presented during training or inference; they remain strictly quarantined for ground-truth verification.
