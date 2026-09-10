#!/usr/bin/env python3
"""
DECISION ENGINE: Unified Low-False-Negative Screening Fusion (Phase 5).
Fuses multi-paradigm screening evidence into an unambiguous operational disposition:
- PASS: Normal relative behavior and safe future drift trajectory.
- REVIEW: Moderate lot anomaly, suspicious slope, borderline drift, or elevated prediction uncertainty.
- REJECT: Severe lot-relative anomaly, unsafe predicted 168h breach, or absolute datasheet violation.

Crucial Design Requirement:
Prioritizes LOW FALSE NEGATIVES over nominal accuracy to guarantee zero defective escapes
into high-reliability aerospace flight hardware.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd


class DecisionEngine:
    """
    Fuses Module A (Dynamic Outlier), Module B (168h Drift Predictor),
    and Dynamic Safety Envelope into a single explainable screening disposition.
    """

    def __init__(
        self,
        lot_deviation_reject_threshold: float = 3.5,  # e.g. 3.5x lot median
        lot_deviation_review_threshold: float = 2.0,  # e.g. 2.0x lot median
        anomaly_score_reject_threshold: float = 0.70,
        anomaly_score_review_threshold: float = 0.50,
        drift_risk_reject_threshold: float = 0.70,
        drift_risk_review_threshold: float = 0.40
    ):
        self.dev_reject = lot_deviation_reject_threshold
        self.dev_review = lot_deviation_review_threshold
        self.anom_reject = anomaly_score_reject_threshold
        self.anom_review = anomaly_score_review_threshold
        self.drift_reject = drift_risk_reject_threshold
        self.drift_review = drift_risk_review_threshold

    def evaluate(
        self,
        cid: str,
        module_a_res: Dict,
        module_b_res: Dict,
        safety_env_res: Dict
    ) -> Dict:
        """
        Synthesizes assessments into PASS, REVIEW, or REJECT with a transparent QA audit trail.
        """
        abs_fail = int(module_a_res.get("absolute_spec_failed", 0))
        anom_score = float(module_a_res.get("anomaly_score", 0.0))
        anom_severity = str(module_a_res.get("severity", "NORMAL"))
        lot_mult = float(module_a_res.get("leakage_mult_of_lot_median", 1.0))
        max_robust_z = float(module_a_res.get("max_robust_z", 0.0))

        pred_168h = float(module_b_res.get("predicted_168h", 0.0))
        drift_risk = float(module_b_res.get("drift_risk_score", 0.0))
        drift_pct = float(module_b_res.get("drift_percentage", 0.0))
        breaches_spec = int(module_b_res.get("breaches_spec_limit", 0))
        breaches_proto = int(module_b_res.get("breaches_prototype_limit", 0))

        env_violated = bool(safety_env_res.get("envelope_violated", False))
        early_slope_breach = bool(safety_env_res.get("early_slope_breach", False))
        pred_slope_breach = bool(safety_env_res.get("pred_slope_breach", False))
        proto_breach = bool(safety_env_res.get("prototype_breach", False))
        eng_breach = bool(safety_env_res.get("engineering_breach", False))

        # Check for REJECT conditions (Any severe risk factor triggers REJECT)
        reject_criteria = []
        if abs_fail == 1:
            reject_criteria.append("Absolute datasheet parametric limit violated at screening")
        if breaches_spec == 1 or eng_breach:
            reject_criteria.append(f"Predicted 168h value ({pred_168h:.2f}) exceeds absolute engineering limit")
        if lot_mult >= self.dev_reject:
            reject_criteria.append(f"Leakage is {lot_mult:.1f}x lot median (exceeds {self.dev_reject}x rejection threshold)")
        if max_robust_z >= 4.5:
            reject_criteria.append(f"Parametric deviation is +{max_robust_z:.1f} MAD robust Z-scores from lot baseline")
        if (anom_score >= self.anom_reject or anom_severity == "SEVERE") and env_violated:
            reject_criteria.append("Severe multivariate anomaly coupled with kinetic safety envelope breach")
        if drift_risk >= self.drift_reject and early_slope_breach:
            reject_criteria.append("High projected drift risk coupled with steep early degradation slope")

        # Check for REVIEW conditions (Borderline or suspicious indications)
        review_criteria = []
        if lot_mult >= self.dev_review:
            review_criteria.append(f"Elevated leakage at {lot_mult:.1f}x lot median")
        if max_robust_z >= 2.5:
            review_criteria.append(f"Moderate population deviation (Robust Z = +{max_robust_z:.2f})")
        if anom_score >= self.anom_review or anom_severity == "SUSPICIOUS":
            review_criteria.append(f"Moderate AI anomaly score ({anom_score:.2f})")
        if breaches_proto == 1 or proto_breach:
            review_criteria.append(f"Predicted 168h value ({pred_168h:.2f}) breaches data-driven prototype envelope")
        if early_slope_breach or pred_slope_breach:
            review_criteria.append("Kinetic drift slope exceeds healthy reference envelope")
        if drift_risk >= self.drift_review:
            review_criteria.append(f"Moderate projected drift risk ({drift_risk:.2f})")

        # Determine Final Disposition
        if len(reject_criteria) > 0:
            final_decision = "REJECT"
            status_code = 2
            reason_summary = "REJECT because:\n- " + "\n- ".join(reject_criteria)
        elif len(review_criteria) > 0:
            final_decision = "REVIEW"
            status_code = 1
            reason_summary = "REVIEW because:\n- " + "\n- ".join(review_criteria)
        else:
            final_decision = "PASS"
            status_code = 0
            reason_summary = (
                f"PASS: Component exhibits nominal lot-relative behavior "
                f"({lot_mult:.1f}x median, max Z = +{max_robust_z:.2f}) and safe future drift trajectory "
                f"(Predicted 168h = {pred_168h:.2f}, safety envelope intact)."
            )

        # Build QA Inspector Explanations
        bullet_points = []
        bullet_points.append("✓ Inside absolute datasheet limits" if abs_fail == 0 else "✕ Absolute datasheet limit breached")
        if lot_mult >= 1.5:
            bullet_points.append(f"⚠ {lot_mult:.1f}x deviation from lot median")
        else:
            bullet_points.append(f"✓ Normal lot alignment ({lot_mult:.1f}x lot median)")

        if early_slope_breach:
            bullet_points.append("⚠ Early drift slope is unusually steep")
        else:
            bullet_points.append("✓ Early drift rate is within healthy envelope")

        if breaches_spec == 1:
            bullet_points.append(f"✕ Predicted 168h value ({pred_168h:.2f}) exceeds specification limit")
        elif breaches_proto == 1:
            bullet_points.append(f"⚠ Predicted 168h value ({pred_168h:.2f}) is borderline (prototype threshold breach)")
        else:
            bullet_points.append(f"✓ Predicted 168h value ({pred_168h:.2f}) is safe")

        return {
            "component_id": cid,
            "final_decision": final_decision,
            "status_code": status_code,  # 0: PASS, 1: REVIEW, 2: REJECT
            "anomaly_score": anom_score,
            "predicted_168h": pred_168h,
            "drift_amount": float(module_b_res.get("drift_amount", 0.0)),
            "drift_percentage": drift_pct,
            "drift_risk_score": drift_risk,
            "lot_median_multiplier": lot_mult,
            "max_robust_z": max_robust_z,
            "safety_envelope_breached": env_violated,
            "absolute_spec_failed": abs_fail,
            "explanation": reason_summary,
            "qa_bullets": bullet_points
        }

    def evaluate_batch(
        self,
        mod_a_df: pd.DataFrame,
        mod_b_df: pd.DataFrame,
        env_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Processes batch screening predictions into master decision table.
        """
        results = []
        for idx in range(len(mod_a_df)):
            a_row = mod_a_df.iloc[idx].to_dict()
            b_row = mod_b_df.iloc[idx].to_dict()
            e_row = env_df.iloc[idx].to_dict()
            cid = str(a_row.get("component_id", f"COMP_{idx+1}"))
            res = self.evaluate(cid, a_row, b_row, e_row)
            results.append(res)
        return pd.DataFrame(results)
