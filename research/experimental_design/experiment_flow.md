# End-to-End Experimental Execution Workflow

## 1. Systematic Execution Pipeline

The complete comparative screening experiment proceeds through 12 rigorous stages, designed to guarantee zero data leakage and identical evaluation conditions:

```text
===================================================================================
                                1. PUBLIC RAW DATASET
                     NASA Ames PCoE Capacitor Overstress (EOS)
                                         │
                                         ▼
                              2. QUALITY & SCHEMA AUDIT
                      Zero Missing Values, Monotonic Timestamps
                                         │
                                         ▼
                          3. COMPONENT / LOT IDENTIFICATION
                   Discrete Units C1–C6 in Cohort LOT_10V_EOS
                                         │
                                         ▼
                       4. LEAVE-ONE-COMPONENT-OUT (LOCO) SPLIT
                      Fold k: 5 Units Training │ 1 Unit Testing
                                         │
                                         ▼
                        5. EARLY SCREENING CUTOFF (t_screen)
                      Temporal Quarantine: Telemetry for t ≤ 47h
                                         │
                                         ▼
         ┌───────────────────────────────┴───────────────────────────────┐
         ▼                               ▼                               ▼
 [ 6. LEVEL 1: TRAD ]            [ 7. LEVEL 2: STAT ]            [ 8. LEVEL 3: ML ]
 Static Limits                   Robust Z-Score &                Isolation Forest
 MIL-PRF-62F 20%                 Mahalanobis Distance            Trained on 5 Units
         │                               │                               │
         ▼                               ▼                               ▼
    Decision ŷ_trad                 Decision ŷ_stat                 Decision ŷ_ml
    ∈ {PASS, REJECT}                ∈ {PASS, REJECT}                ∈ {PASS, REJECT}
         │                               │                               │
         └───────────────────────────────┼───────────────────────────────┘
                                         │
                                         ▼
                       9. FUTURE OBSERVATION & GROUND TRUTH
                 Quarantine Lifted for t > 47h to Evaluate EOL Failure
                                         │
                                         ▼
                           10. PAIRED CONFUSION MATRICES
                     TP, FP, TN, FN Computed on Identical Units
                                         │
                                         ▼
                         11. COMPARATIVE METRIC COMPUTATION
                       Recall, FNR, FPR, PR-AUC, Early Warning
                                         │
                                         ▼
                        12. STATISTICAL INFERENCE & EXPLAIN
                     McNemar's Test, Bootstrap CIs, Tree SHAP Values
                                         │
                                         ▼
                            13. EVIDENCE-BASED CONCLUSION
===================================================================================
```

---

## 2. Detailed Execution Stage Protocol

1. **Stage 1 (Raw Ingestion):** Ingest `data/raw/nasa_capacitor_electrical_stress/EOS_DataSet.mat`. Checksum verified against acquisition record.
2. **Stage 2 (Quality Control):** Validate absence of nulls, negative parameters, or non-monotonic timestamps.
3. **Stage 3 (Tidy Transformation):** Convert 2D arrays into panel DataFrame `data/processed/dataset_processed.csv`.
4. **Stage 4 (LOCO Partitioning):** For fold $k \in \{1 \dots 6\}$, test unit is $C_k$; training units are $\{C_j\}_{j \neq k}$.
5. **Stage 5 (Information Quarantine):** Quarantined feature extractor slices data at $t \le t_{\text{screen}} = 47\,\text{h}$.
6. **Stage 6 (Traditional Baseline):** Checks if $\Delta C \ge 20\%$ or $\Delta\text{ESR} \ge 100\%$.
7. **Stage 7 (Statistical Engine):** Computes robust median, MAD, and regularized covariance on training fold; evaluates test unit Mahalanobis distance.
8. **Stage 8 (ML Engine):** Fits Isolation Forest on training fold pre-$t_{\text{screen}}$ feature space; evaluates test unit anomaly score.
9. **Stage 9 (Target Generation):** Inspects future trajectory ($47\,\text{h} < t \le 194\,\text{h}$) to establish true latent defect status $y_k$.
10. **Stage 10 (Paired Matrix):** Tabulates paired decisions across all 6 LOCO test folds.
11. **Stage 11 (Metric Evaluation):** Calculates paired Recall, FNR, FPR, Yield Scrap, and Lead Time.
12. **Stage 12 (Significance Testing):** Runs McNemar's exact test and 2,000-sample bootstrap confidence intervals.
13. **Stage 13 (Report Generation):** Renders explainability breakdowns and final empirical research conclusions.
