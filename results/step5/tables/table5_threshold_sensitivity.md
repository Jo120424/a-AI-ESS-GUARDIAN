# Table 5: Controlled Threshold Sensitivity Analysis

| model_name | parameter_name | threshold_value | true_positives | false_positives | false_negatives | true_negatives | recall | false_negative_rate | precision | false_positive_rate | f1_score | pr_auc | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dynamic Statistical | robust_z_threshold | 1.5 | 3 | 1 | 2 | 0 | 0.6 | 0.4 | 0.75 | 1.0 | 0.6667 | 0.81 | Sensitivity variant |
| Dynamic Statistical | robust_z_threshold | 2.0 | 3 | 1 | 2 | 0 | 0.6 | 0.4 | 0.75 | 1.0 | 0.6667 | 0.81 | Sensitivity variant |
| Dynamic Statistical | robust_z_threshold | 2.5 | 3 | 1 | 2 | 0 | 0.6 | 0.4 | 0.75 | 1.0 | 0.6667 | 0.81 | Baseline step 4 configuration |
| Dynamic Statistical | robust_z_threshold | 3.0 | 2 | 1 | 3 | 0 | 0.4 | 0.6 | 0.6667 | 1.0 | 0.5 | 0.81 | Sensitivity variant |
| Dynamic Statistical | robust_z_threshold | 3.5 | 2 | 1 | 3 | 0 | 0.4 | 0.6 | 0.6667 | 1.0 | 0.5 | 0.81 | Sensitivity variant |
| Dynamic Statistical | robust_z_threshold | 4.5 | 2 | 1 | 3 | 0 | 0.4 | 0.6 | 0.6667 | 1.0 | 0.5 | 0.81 | Sensitivity variant |
| AI/ML One-Class SVM | decision_offset | -0.05 | 5 | 1 | 0 | 0 | 1.0 | 0.0 | 0.8333 | 1.0 | 0.9091 | 0.81 | Sensitivity variant |
| AI/ML One-Class SVM | decision_offset | 0.0 | 4 | 1 | 1 | 0 | 0.8 | 0.2 | 0.8 | 1.0 | 0.8 | 0.81 | Baseline step 4 configuration |
| AI/ML One-Class SVM | decision_offset | 0.05 | 3 | 1 | 2 | 0 | 0.6 | 0.4 | 0.75 | 1.0 | 0.6667 | 0.81 | Sensitivity variant |
| AI/ML One-Class SVM | decision_offset | 0.1 | 3 | 1 | 2 | 0 | 0.6 | 0.4 | 0.75 | 1.0 | 0.6667 | 0.81 | Sensitivity variant |
| AI/ML One-Class SVM | decision_offset | 0.2 | 2 | 1 | 3 | 0 | 0.4 | 0.6 | 0.6667 | 1.0 | 0.5 | 0.81 | Sensitivity variant |
| AI/ML One-Class SVM | decision_offset | 0.3 | 1 | 1 | 4 | 0 | 0.2 | 0.8 | 0.5 | 1.0 | 0.2857 | 0.81 | Sensitivity variant |
| Early Drift Forecaster | spec_limit_delta_c_pct | 18.0 | 5 | 1 | 0 | 0 | 1.0 | 0.0 | 0.8333 | 1.0 | 0.9091 | 0.71 | Sensitivity variant |
| Early Drift Forecaster | spec_limit_delta_c_pct | 19.0 | 5 | 1 | 0 | 0 | 1.0 | 0.0 | 0.8333 | 1.0 | 0.9091 | 0.71 | Sensitivity variant |
| Early Drift Forecaster | spec_limit_delta_c_pct | 20.0 | 5 | 1 | 0 | 0 | 1.0 | 0.0 | 0.8333 | 1.0 | 0.9091 | 0.71 | Baseline step 4 configuration |
| Early Drift Forecaster | spec_limit_delta_c_pct | 21.0 | 3 | 1 | 2 | 0 | 0.6 | 0.4 | 0.75 | 1.0 | 0.6667 | 0.71 | Sensitivity variant |
| Early Drift Forecaster | spec_limit_delta_c_pct | 21.5 | 0 | 1 | 5 | 0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 0.71 | Sensitivity variant |
| Early Drift Forecaster | spec_limit_delta_c_pct | 22.0 | 0 | 0 | 5 | 1 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.71 | Sensitivity variant |
| Risk Fusion Engine | rejection_score_threshold | 0.2 | 5 | 1 | 0 | 0 | 1.0 | 0.0 | 0.8333 | 1.0 | 0.9091 | 0.7833 | Sensitivity variant |
| Risk Fusion Engine | rejection_score_threshold | 0.25 | 5 | 1 | 0 | 0 | 1.0 | 0.0 | 0.8333 | 1.0 | 0.9091 | 0.7833 | Sensitivity variant |
| Risk Fusion Engine | rejection_score_threshold | 0.35 | 3 | 1 | 2 | 0 | 0.6 | 0.4 | 0.75 | 1.0 | 0.6667 | 0.7833 | Sensitivity variant |
| Risk Fusion Engine | rejection_score_threshold | 0.5 | 3 | 1 | 2 | 0 | 0.6 | 0.4 | 0.75 | 1.0 | 0.6667 | 0.7833 | Baseline step 4 configuration |
| Risk Fusion Engine | rejection_score_threshold | 0.6 | 0 | 0 | 5 | 1 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.7833 | Sensitivity variant |
| Risk Fusion Engine | rejection_score_threshold | 0.7 | 0 | 0 | 5 | 1 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.7833 | Sensitivity variant |
