#!/usr/bin/env python3
"""
Step 5 Explainability Summary Generator: Consolidates Multi-Paradigm Rationales.
Outputs: results/step5/explainability_summary.csv
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from backend.validation import STEP4_RESULTS_DIR, STEP5_RESULTS_DIR


class ExplainabilityConsolidator:
    """
    Consolidates explainability features across all screening models into a single matrix.
    """

    def __init__(self, step4_dir: Path = STEP4_RESULTS_DIR, step5_dir: Path = STEP5_RESULTS_DIR):
        self.step4_dir = step4_dir
        self.step5_dir = step5_dir

    def run_consolidation(self) -> pd.DataFrame:
        """Loads component results and extracts transparent multi-model rationales."""
        df_comp = pd.read_csv(self.step4_dir / "component_level_results.csv")

        records: List[Dict] = []
        for _, row in df_comp.iterrows():
            cid = row["component_id"]

            records.append({
                "component_id": cid,
                "ground_truth": "Latent Defect" if row["ground_truth_latent_risk"] == 1 else "Safe Survivor",
                # Level 1 Traditional Static
                "trad_decision": row["traditional_status"],
                "trad_violating_parameters": str(row["violating_parameters"]),
                "trad_safety_margin_c_pct": round(row["spec_margin_c"], 2),
                # Level 2 Dynamic Statistical
                "stat_decision": row["statistical_status"],
                "max_mad_zscore": round(row["max_abs_zscore"], 2),
                "mahalanobis_dist_sq": round(row["mahalanobis_dist_sq"], 2),
                "stat_contributing_parameters": str(row["contributing_parameters"]),
                # Level 3A AI Anomaly
                "iforest_anomaly_score": round(row["iforest_anomaly_score"], 4),
                "iforest_flag": int(row["iforest_flag"]),
                "ocsvm_anomaly_score": round(row["ocsvm_anomaly_score"], 4),
                "ocsvm_flag": int(row["ocsvm_flag"]),
                "primary_deviating_feature": row["primary_contributing_feature"],
                "max_feature_deviation_z": round(row["max_feature_deviation_z"], 2),
                # Level 3B Drift Prediction
                "drift_decision": row["drift_status"],
                "predicted_delta_c_194h": round(row["predicted_delta_c_194h"], 2),
                "actual_delta_c_194h": round(row["actual_delta_c_194h"], 2),
                "prediction_error_signed": round(row["prediction_signed_error"], 2),
                "drift_safety_margin": round(row["forecast_safety_margin"], 2),
                # Multi-Tier Risk Fusion
                "fusion_risk_score": round(row["fusion_risk_score"], 2),
                "risk_tier": row["risk_tier"],
                "fusion_decision": row["fusion_status"],
                "flagged_systems_count": int(row["flagged_systems_count"]),
                "flagged_systems": str(row["flagged_systems"]),
                "plain_language_explanation": row["fusion_explanation"]
            })

        df_out = pd.DataFrame(records)
        csv_path = self.step5_dir / "explainability_summary.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"[Step 5 Explainability] Consolidated matrix saved to: {csv_path}")
        return df_out


if __name__ == "__main__":
    consolidator = ExplainabilityConsolidator()
    df = consolidator.run_consolidation()
    print(df[["component_id", "risk_tier", "max_mad_zscore", "predicted_delta_c_194h", "plain_language_explanation"]])
