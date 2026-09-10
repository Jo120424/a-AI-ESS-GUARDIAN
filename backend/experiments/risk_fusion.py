#!/usr/bin/env python3
"""
Model E: Multi-Paradigm Risk Fusion Engine.
Synthesizes decisions from Traditional, Statistical, AI Anomaly, and Drift Prediction
into a calibrated, explainable multi-tier risk classification:
- LOW RISK
- MEDIUM RISK
- HIGH RISK
Provides transparent audit trails for aerospace qualification engineers.
"""

from typing import Dict, List, Optional
import pandas as pd


class RiskFusionEngine:
    """
    Fuses multi-level screening indicators into unified component risk dispositions.
    """

    def __init__(
        self,
        weight_traditional: float = 0.15,
        weight_statistical: float = 0.30,
        weight_aiml: float = 0.30,
        weight_drift: float = 0.25,
        high_risk_threshold: float = 0.50,
        medium_risk_threshold: float = 0.20
    ):
        self.w_trad = weight_traditional
        self.w_stat = weight_statistical
        self.w_ml = weight_aiml
        self.w_drift = weight_drift
        self.high_thresh = high_risk_threshold
        self.med_thresh = medium_risk_threshold

    def fuse_component(
        self,
        cid: str,
        trad_res: Dict,
        stat_res: Dict,
        ml_res: Dict,
        drift_res: Dict
    ) -> Dict:
        """
        Synthesizes the 4 screening assessments for a single component.
        """
        y_trad = int(trad_res.get("traditional_decision", 0))
        y_stat = int(stat_res.get("statistical_decision", 0))
        y_ml = int(ml_res.get("aiml_decision", 0))
        y_drift = int(drift_res.get("drift_decision", 0))

        # 1. Compute Continuous Multi-Factor Risk Score [0, 1]
        risk_score = (
            self.w_trad * y_trad +
            self.w_stat * y_stat +
            self.w_ml * y_ml +
            self.w_drift * y_drift
        )
        risk_score = min(max(float(risk_score), 0.0), 1.0)

        # 2. Determine Risk Category
        # High Risk if:
        # - Traditional limit breached (immediate hard fail), OR
        # - Risk score >= 0.50 (at least two dynamic systems flag failure), OR
        # - Drift forecasts spec violation AND an anomaly model triggers
        is_high = (
            y_trad == 1 or
            risk_score >= self.high_thresh or
            (y_drift == 1 and (y_ml == 1 or y_stat == 1))
        )

        if is_high:
            risk_tier = "HIGH RISK"
            final_decision = 1
        elif risk_score >= self.med_thresh or (y_stat == 1 or y_ml == 1 or y_drift == 1):
            risk_tier = "MEDIUM RISK"
            final_decision = 0  # Review or conditional pass in standard operations
        else:
            risk_tier = "LOW RISK"
            final_decision = 0

        # 3. Construct Transparent Engineering Explanation
        flagged_systems = []
        if y_trad == 1:
            flagged_systems.append("Traditional Static Spec")
        if y_stat == 1:
            flagged_systems.append("Dynamic Statistical Outlier")
        if y_ml == 1:
            flagged_systems.append("AI/ML Anomaly Detector")
        if y_drift == 1:
            flagged_systems.append("Early Drift Trajectory Forecast")

        if is_high:
            reasons = []
            if y_stat == 1:
                reasons.append(f"population deviation (Max MAD Z={stat_res.get('max_abs_zscore', 0):.2f})")
            if y_ml == 1:
                reasons.append(f"AI isolation score ({ml_res.get('iforest_anomaly_score', 0):.3f})")
            if y_drift == 1:
                reasons.append(f"forecasted Delta C={drift_res.get('predicted_delta_c_194h', 0):.2f}% exceeds 20% limit")
            explanation = (
                f"FLAGGED HIGH RISK (Score: {risk_score:.2f}): Component flagged by {len(flagged_systems)}/4 screening systems "
                f"[{', '.join(flagged_systems)}]. Critical factors: {'; '.join(reasons)}."
            )
        elif risk_tier == "MEDIUM RISK":
            explanation = (
                f"MONITOR / MEDIUM RISK (Score: {risk_score:.2f}): Isolated screening flag from "
                f"[{', '.join(flagged_systems)}]. Does not meet full rejection threshold, recommended for extended burn-in."
            )
        else:
            explanation = (
                f"CLEARED / LOW RISK (Score: {risk_score:.2f}): All screening paradigms confirm nominal health. "
                f"Forecasted degradation remains well within specification."
            )

        return {
            "component_id": cid,
            "fusion_risk_score": round(risk_score, 4),
            "risk_tier": risk_tier,
            "fusion_decision": final_decision,
            "fusion_status": "REJECT" if final_decision == 1 else "ACCEPT",
            "y_traditional": y_trad,
            "y_statistical": y_stat,
            "y_aiml": y_ml,
            "y_drift": y_drift,
            "flagged_systems_count": len(flagged_systems),
            "flagged_systems": flagged_systems,
            "explanation": explanation
        }

    def fuse_batch(
        self,
        trad_df: pd.DataFrame,
        stat_df: pd.DataFrame,
        ml_df: pd.DataFrame,
        drift_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fuses predictions across all components.
        """
        components = trad_df["component_id"].tolist()
        results = []
        for cid in components:
            t_row = trad_df[trad_df["component_id"] == cid].iloc[0].to_dict()
            s_row = stat_df[stat_df["component_id"] == cid].iloc[0].to_dict()
            m_row = ml_df[ml_df["component_id"] == cid].iloc[0].to_dict()
            d_row = drift_df[drift_df["component_id"] == cid].iloc[0].to_dict()

            res = self.fuse_component(cid, t_row, s_row, m_row, d_row)
            results.append(res)
        return pd.DataFrame(results)
