#!/usr/bin/env python3
"""
Step 5 Cost-Sensitive Analysis & Engineering Trade-Off Engine.
Models aerospace risk economics where defect escapes (FN) incur catastrophic mission losses,
while false alarms (FP) incur component replacement costs.
Outputs:
- results/step5/cost_sensitivity.csv
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.validation import STEP4_RESULTS_DIR, STEP5_RESULTS_DIR


class CostAnalyzer:
    """
    Evaluates expected screening cost as a function of the critical cost ratio C_FN / C_FP.
    """

    COST_RATIOS = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0]

    def __init__(self, step4_dir: Path = STEP4_RESULTS_DIR, step5_dir: Path = STEP5_RESULTS_DIR):
        self.step4_dir = step4_dir
        self.step5_dir = step5_dir

    def run_cost_analysis(self) -> pd.DataFrame:
        """Calculates total and normalized screening costs across all cost ratios."""
        df_comp = pd.read_csv(self.step4_dir / "component_level_results.csv")
        y_true = df_comp["ground_truth_latent_risk"].to_numpy()
        N = len(y_true)

        # Extract predictions
        methods = {
            "Traditional (MIL-PRF-62F 20%)": df_comp["traditional_decision"].to_numpy(),
            "Dynamic Statistical (MAD/Maha)": df_comp["statistical_decision"].to_numpy(),
            "AI/ML Anomaly (One-Class SVM)": df_comp["ocsvm_flag"].to_numpy(),
            "AI/ML Anomaly (Isolation Forest)": df_comp["iforest_flag"].to_numpy(),
            "Early Drift Forecast (194h)": df_comp["drift_decision"].to_numpy(),
            "Risk Fusion Engine (Multi-Tier)": df_comp["fusion_decision"].to_numpy(),
            "Risk Fusion (incl. Quarantine Medium)": (df_comp["fusion_risk_score"].to_numpy() >= 0.25).astype(int),
        }

        # Calculate FN and FP counts for each method
        confusion_counts = {}
        for m_name, y_pred in methods.items():
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))
            confusion_counts[m_name] = {"FN": fn, "FP": fp, "TP": tp, "TN": tn}

        results: List[Dict] = []
        c_fp = 1.0  # Normalized component scrap/replacement cost

        for ratio in self.COST_RATIOS:
            c_fn = ratio * c_fp
            trad_cost = confusion_counts["Traditional (MIL-PRF-62F 20%)"]["FN"] * c_fn + confusion_counts["Traditional (MIL-PRF-62F 20%)"]["FP"] * c_fp

            for m_name, counts in confusion_counts.items():
                total_cost = counts["FN"] * c_fn + counts["FP"] * c_fp
                cost_per_comp = total_cost / N
                cost_savings = trad_cost - total_cost
                savings_pct = (cost_savings / trad_cost * 100.0) if trad_cost > 0 else 0.0

                results.append({
                    "cost_ratio_fn_to_fp": ratio,
                    "c_fn": c_fn,
                    "c_fp": c_fp,
                    "method": m_name,
                    "fn_count": counts["FN"],
                    "fp_count": counts["FP"],
                    "total_expected_cost": round(total_cost, 4),
                    "cost_per_component": round(cost_per_comp, 4),
                    "cost_savings_vs_traditional": round(cost_savings, 4),
                    "cost_savings_pct": round(savings_pct, 2),
                    "is_superior_to_traditional": total_cost < trad_cost
                })

        df_cost = pd.DataFrame(results)
        csv_path = self.step5_dir / "cost_sensitivity.csv"
        df_cost.to_csv(csv_path, index=False)
        print(f"[Step 5 Cost Analysis] Generated {len(df_cost)} cost evaluations. Saved to: {csv_path}")
        return df_cost


if __name__ == "__main__":
    analyzer = CostAnalyzer()
    df = analyzer.run_cost_analysis()
    # Print comparison at ratio = 10.0 (mission-critical scenario)
    sub = df[df["cost_ratio_fn_to_fp"] == 10.0]
    print(sub[["method", "fn_count", "fp_count", "total_expected_cost", "cost_savings_pct", "is_superior_to_traditional"]])
