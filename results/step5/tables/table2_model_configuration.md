# Table 2: Screening Paradigms & Estimator Configurations

| Paradigm | Algorithm / Framework | Hyperparameters | Information Input |
| --- | --- | --- | --- |
| Level 1: Traditional Static | Threshold Comparator | Delta C >= 20.0% (MIL-PRF-62F), Delta ESR >= 100.0% | Single static telemetry point at t = 47.0h |
| Level 1-Tightened | Tightened Comparator | Delta C >= 5.0%, Delta ESR >= 25.0% | Single static telemetry point at t = 47.0h |
| Level 2: Dynamic Statistical | Robust MAD Z-score + Ledoit-Wolf Mahalanobis | Z_crit = 2.5, alpha = 0.05 (chi2_cutoff = 7.815, df=3) | Multivariate lot relative distribution at t <= 47.0h |
| Level 3A: AI/ML Anomaly (iForest) | Isolation Forest (100 trees) | n_estimators = 100, max_samples = 5, contamination = 0.20 | Standardized spatial feature vector (5 variables) |
| Level 3A: AI/ML Anomaly (OC-SVM) | One-Class SVM (RBF Kernel) | kernel = 'rbf', nu = 0.20, gamma = 'scale' | Standardized spatial feature vector (5 variables) |
| Level 3B: Early Drift Forecaster | Ridge Regression + Gradient Boosting Ensemble | Ridge alpha = 1.0; GBR n_est = 30, depth = 2, lr = 0.1 | Early trajectory sequence (t = 0h, 24h, 47h) |
| Level 4: Risk Fusion Engine | Multi-Paradigm Weighted Linear Ensemble | Weights: Trad=0.20, Stat=0.30, Anomaly=0.25, Drift=0.25; Rejection >= 0.50 | Probabilistic / discrete outputs of Levels 1, 2, 3A, 3B |
