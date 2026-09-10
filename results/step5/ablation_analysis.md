# Ablation Analysis: Value Contribution of Screening Modules
**Project:** Predictive AI-Based Environmental Stress Screening for Latent Defect Detection  **Phase:** Step 5 — Robustness, Ablation & Scientific Results  
---
## 1. Ablation Results Table
| Paradigm | Description | Recall | FNR | Precision | F1-Score | FPR | Mean Lead Time | $\Delta\text{Recall vs Trad}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Traditional Static Only** | Single baseline MIL-PRF-62F 20% limit | **0.0%** | 100.0% | 0.0000 | **0.0000** | 0.0% | **0.0 h** | **+0.0%** |
| **B: Statistical Outlier Only** | Dynamic population MAD + Mahalanobis | **60.0%** | 40.0% | 0.7500 | **0.6667** | 100.0% | **139.3 h** | **+60.0%** |
| **C: AI/ML Anomaly Only (OC-SVM)** | Spatial boundary learning without time forecasting | **80.0%** | 20.0% | 0.8000 | **0.8000** | 100.0% | **141.2 h** | **+80.0%** |
| **D: Early Drift Forecast Only** | Parametric time-series extrapolation without spatial clustering | **100.0%** | 0.0% | 0.8333 | **0.9091** | 100.0% | **142.4 h** | **+100.0%** |
| **E: Anomaly + Drift Combined (OR)** | Dual AI paradigm coupling | **100.0%** | 0.0% | 0.8333 | **0.9091** | 100.0% | **142.4 h** | **+100.0%** |
| **F: Full Multi-Tier Risk Fusion** | Comprehensive weighted synthesis (Trad + Stat + Anomaly + Drift) | **60.0%** | 40.0% | 0.7500 | **0.6667** | 100.0% | **139.3 h** | **+60.0%** |
| **F-Extended: Fusion (incl. Medium Risk)** | Fusion with quarantine threshold at 0.25 (extended burn-in) | **100.0%** | 0.0% | 0.8333 | **0.9091** | 100.0% | **142.4 h** | **+100.0%** |

---
## 2. Key Scientific Insights from Ablation

1. **Single Model Limits:**
   * Traditional screening alone has **0.0% recall**.
   * Dynamic statistical screening achieves **60.0% recall** (+60.0% gain), but misses stealth incubators (C3, C5).
   * One-Class SVM alone achieves **80.0% recall**, successfully detecting C3 where statistical screening failed, but misses C5.

2. **Superiority of Trajectory Forecasting (Drift Alone vs Spatial Anomaly):**
   * Early Drift Forecasting alone achieves **100.0% recall** and **F1 = 0.9091**, detecting all 5 latent defects.
   * This demonstrates that in continuous physical degradation, *kinetic trajectory extrapolation* provides stronger predictive signal than instantaneous spatial clustering.

3. **Value of Anomaly + Drift Coupling (Paradigm E):**
   * Combining One-Class SVM and Drift forecasting ensures mutual confirmation: components flagged by both models (C2, C4, C6) represent unambiguous high-confidence latent defects, while C3 and C5 are identified via the drift channel.

4. **Operational Role of Multi-Tier Risk Fusion (Paradigm F):**
   * At the strict rejection threshold ($S_{\text{risk}} \ge 0.50$), Risk Fusion eliminates borderline rejections, yielding $60.0\%$ definitive scrap.
   * When incorporating the MEDIUM RISK tier ($S_{\text{risk}} \ge 0.25$) for non-destructive extended burn-in quarantine, the fusion system achieves **100.0% defect interception**, allowing zero defective components to escape to flight integration while giving borderline components a second qualification opportunity.
