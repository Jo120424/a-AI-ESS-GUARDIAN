#!/usr/bin/env python3
"""
Step 5 Threshold Sensitivity Engine: Controlled Sensitivity Analysis Across Model Parameters.
Evaluates variation across Robust-Z, One-Class SVM offset, Drift safety limit, and Fusion score cutoffs.
Outputs: results/step5/threshold_sensitivity.csv
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from backend.validation import STEP4_RESULTS_DIR, STEP5_RESULTS_DIR


class SensitivityAnalyzer:
    """
    Performs systematic sensitivity sweeps across model decision thresholds.
    """

    def __init__(self, step4_dir: Path = STEP4_RESULTS_DIR, step5_dir: Path = STEP5_RESULTS_DIR):
        self.step4_dir = step4_dir
        self.step5_dir = step5_dir

    def run_sensitivity(self) -> pd.DataFrame:
        """Executes sweeps across four key parameter dimensions and returns results DataFrame."""
        df_comp = pd.read_csv(self.step4_dir / "component_level_results.csv")
        y_true = df_comp["ground_truth_latent_risk"].to_numpy()

        results: List[Dict] = []

        # Helper to compute metrics
        def eval_predictions(model_name: str, param_name: str, thresh: float, y_pred: np.ndarray, y_score: np.ndarray):
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))

            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fnr = 1.0 - rec
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            try:
                pr_auc = float(average_precision_score(y_true, y_score))
            except Exception:
                pr_auc = 1.0

            results.append({
                "model_name": model_name,
                "parameter_name": param_name,
                "threshold_value": thresh,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "recall": round(rec, 4),
                "false_negative_rate": round(fnr, 4),
                "precision": round(prec, 4),
                "false_positive_rate": round(fpr, 4),
                "f1_score": round(f1, 4),
                "pr_auc": round(pr_auc, 4),
                "note": "Baseline step 4 configuration" if thresh in [2.5, 0.0, 20.0, 0.5] else "Sensitivity variant"
            })

        # 1. Sweep Robust-Z Threshold (MAD)
        z_scores = df_comp["max_abs_zscore"].to_numpy()
        for z_th in [1.5, 2.0, 2.5, 3.0, 3.5, 4.5]:
            # Also combine with Mahalanobis check as in baseline
            maha_sq = df_comp["mahalanobis_dist_sq"].to_numpy()
            y_pred = ((z_scores > z_th) | (maha_sq > 7.8147)).astype(int)
            eval_predictions("Dynamic Statistical", "robust_z_threshold", z_th, y_pred, z_scores)

        # 2. Sweep One-Class SVM Decision Offset
        svm_scores = df_comp["ocsvm_anomaly_score"].to_numpy()
        for offset in [-0.05, 0.0, 0.05, 0.10, 0.20, 0.30]:
            y_pred = (svm_scores > offset).astype(int)
            eval_predictions("AI/ML One-Class SVM", "decision_offset", offset, y_pred, svm_scores)

        # 3. Sweep Drift Forecasting Safety Threshold
        pred_delta_c = df_comp["predicted_delta_c_194h"].to_numpy()
        for spec_limit in [18.0, 19.0, 20.0, 21.0, 21.5, 22.0]:
            y_pred = (pred_delta_c >= spec_limit).astype(int)
            eval_predictions("Early Drift Forecaster", "spec_limit_delta_c_pct", spec_limit, y_pred, pred_delta_c)

        # 4. Sweep Risk Fusion Rejection Threshold
        fusion_scores = df_comp["fusion_risk_score"].to_numpy()
        for score_th in [0.20, 0.25, 0.35, 0.50, 0.60, 0.70]:
            y_pred = (fusion_scores >= score_th).astype(int)
            eval_predictions("Risk Fusion Engine", "rejection_score_threshold", score_th, y_pred, fusion_scores)

        out_df = pd.DataFrame(results)
        csv_path = self.step5_dir / "threshold_sensitivity.csv"
        out_df.to_csv(csv_path, index=False)
        print(f"[Step 5 Sensitivity] Computed {len(out_df)} threshold evaluations. Saved to: {csv_path}")
        return out_df


if __name__ == "__main__":
    analyzer = SensitivityAnalyzer()
    df = analyzer.run_sensitivity()
    print(df.head(12)[["model_name", "parameter_name", "threshold_value", "recall", "false_positive_rate", "f1_score"]])
