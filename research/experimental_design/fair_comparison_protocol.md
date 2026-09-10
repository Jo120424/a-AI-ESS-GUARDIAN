# Fair Comparison Protocol & Identical Test Partitioning

## 1. The Core Scientific Imperative

A comparative study of screening paradigms is valid if and only if:
> **All competing screening approaches (Traditional, Dynamic Statistical, AI/ML) are evaluated on the EXACT SAME unseen test components, under the EXACT SAME screening observation window ($t_{\text{screen}}$), with identical information boundaries and identical ground-truth targets.**

Evaluating models on differing test sets, or granting AI models access to features denied to statistical benchmarks, introduces confounding variables that invalidate experimental conclusions.

---

## 2. Fair Comparison Evaluation Flow

```text
                                 HELD-OUT TEST COMPONENT i
                               (Quarantined from Training Folds)
                                             │
                        ┌────────────────────┼────────────────────┐
                        ▼                    ▼                    ▼
               [ LEVEL 1: TRAD ]    [ LEVEL 2: STAT ]    [ LEVEL 3: ML ]
               Static Limits        Robust Z-score /     Isolation Forest /
               (MIL-PRF-62F)        Mahalanobis Distance One-Class SVM
                        │                    │                    │
                        ▼                    ▼                    ▼
                   Decision ŷ_trad      Decision ŷ_stat      Decision ŷ_ml
                   ∈ {PASS, REJECT}     ∈ {PASS, REJECT}     ∈ {PASS, REJECT}
                        │                    │                    │
                        └────────────────────┼────────────────────┘
                                             │
                                             ▼
                                [ SAME GROUND TRUTH y_i ]
                                Evaluated on post-t_screen
                                operational trajectory
                                             │
                                             ▼
                               [ PAIRED METRIC COMPUTATION ]
                               • Paired ΔRecall
                               • Paired ΔFalse Negative Rate
                               • McNemar's Contingency Test
                               • Paired Bootstrap Confidence Intervals
```

---

## 3. Strict Experimental Controls

| Experimental Condition | Traditional (Level 1) | Dynamic Statistical (Level 2) | AI/ML (Level 3) | Research Variable Tested |
| :--- | :---: | :---: | :---: | :--- |
| **Test Component Identity** | Component $k$ | Component $k$ | Component $k$ | **FIXED CONTROL** |
| **Screening Window ($t_{\text{screen}}$)** | $47.0\,\text{h}$ | $47.0\,\text{h}$ | $47.0\,\text{h}$ | **FIXED CONTROL** |
| **Permitted Telemetry** | $t \le 47.0\,\text{h}$ | $t \le 47.0\,\text{h}$ | $t \le 47.0\,\text{h}$ | **FIXED CONTROL** |
| **Future Lookahead** | Quarantined | Quarantined | Quarantined | **FIXED CONTROL** |
| **Ground Truth Label $y_i$** | $y_i(\text{EOL})$ | $y_i(\text{EOL})$ | $y_i(\text{EOL})$ | **FIXED CONTROL** |
| **Screening Methodology** | Static Spec Limits | Population Outlier Distance | Learned Non-linear Manifold | **TEST VARIABLE** |

---

## 4. Rejection of Asymmetric Tuning
* Traditional screening thresholds are fixed by MIL-PRF-62F military standards ($\Delta C = 20\%$) and tightened industrial specs ($\Delta C = 5\%$).
* Dynamic statistical thresholds are calibrated strictly on training fold bounds ($\theta_{\text{RZ}} = 2.5$).
* AI/ML anomaly thresholds are calibrated on the 90th/95th training percentile without peeking at test component labels.
* No method is manually retuned post-evaluation.
