#!/usr/bin/env python3
"""
ADAPTIVE SCREENING INTELLIGENCE (ASI)
Software-Based Decision-Support Engine for Component Burn-In & Screening.

Transforms linear one-way screening:
  DATA -> ANOMALY -> PREDICTION -> SAFETY -> DECISION
into closed-loop intelligent screening:
  DATA -> ANOMALY -> PREDICTION -> SAFETY -> DECISION -> ADAPTIVE SCREENING -> RECOMMENDED NEXT ACTION -> NEW MEASUREMENT -> RE-ANALYSIS

Adheres strictly to aerospace reliability safety:
- Software-based prototype decision-support tool.
- Does NOT claim autonomous control of real aerospace/ISRO test equipment.
- All recommendations are advisory; final authority remains with qualified QA/reliability engineers.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd


class AdaptiveAction:
    """Standardized action taxonomy for Adaptive Screening Intelligence."""
    CONTINUE_STANDARD_SCREENING = "CONTINUE_STANDARD_SCREENING"
    ADDITIONAL_MEASUREMENT = "ADDITIONAL_MEASUREMENT"
    PRIORITY_SCREENING = "PRIORITY_SCREENING"
    QA_REVIEW = "QA_REVIEW"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class AdaptiveScreeningEngine:
    """
    Evaluates multi-signal screening evidence to determine whether current data
    is sufficient for disposition, and recommends the optimal next screening action.
    """

    VALID_CHECKPOINTS = ["0h", "24h", "96h", "168h"]

    def __init__(
        self,
        engineering_limit: float = 50.0,
        prototype_limit: float = 45.0,
        high_lot_deviation_threshold: float = 3.0,
        moderate_lot_deviation_threshold: float = 2.0,
        high_z_threshold: float = 3.5,
        moderate_z_threshold: float = 2.2,
        high_drift_risk_threshold: float = 0.70,
        moderate_drift_risk_threshold: float = 0.40
    ):
        self.engineering_limit = float(engineering_limit)
        self.prototype_limit = float(prototype_limit)
        self.high_lot_dev = float(high_lot_deviation_threshold)
        self.mod_lot_dev = float(moderate_lot_deviation_threshold)
        self.high_z = float(high_z_threshold)
        self.mod_z = float(moderate_z_threshold)
        self.high_drift_risk = float(high_drift_risk_threshold)
        self.mod_drift_risk = float(moderate_drift_risk_threshold)

    def evaluate(
        self,
        component_id: str,
        lot_id: str = "UNKNOWN_LOT",
        current_checkpoint: str = "24h",
        raw_measurements: Optional[Dict[str, float]] = None,
        module_a_res: Optional[Dict[str, Any]] = None,
        module_b_res: Optional[Dict[str, Any]] = None,
        safety_env_res: Optional[Dict[str, Any]] = None,
        decision_res: Optional[Dict[str, Any]] = None,
        data_quality: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Determines the AI-recommended next screening action from synthesized evidence.
        """
        raw_measurements = raw_measurements or {}
        module_a_res = module_a_res or {}
        module_b_res = module_b_res or {}
        safety_env_res = safety_env_res or {}
        decision_res = decision_res or {}
        data_quality = data_quality or {"completeness": 1.0, "status": "VALID"}

        # Normalize checkpoint string
        cp_str = str(current_checkpoint).lower().strip()
        if not cp_str.endswith("h"):
            cp_str = f"{cp_str}h"

        # 1. Validation & Data Completeness Guard
        if data_quality.get("status") == "INVALID" or data_quality.get("completeness", 1.0) < 0.5:
            return self._build_insufficient_data_response(
                component_id, lot_id, cp_str, "Incomplete or corrupted telemetry records."
            )

        # Extract primary evidence signals from existing sub-modules
        abs_failed = int(module_a_res.get("absolute_spec_failed", 0))
        anom_score = float(module_a_res.get("anomaly_score", 0.0))
        anom_severity = str(module_a_res.get("severity", "NORMAL"))
        lot_mult = float(module_a_res.get("leakage_mult_of_lot_median", 1.0))
        max_z = float(module_a_res.get("max_robust_z", 0.0))

        pred_168 = float(module_b_res.get("predicted_168h", 0.0))
        drift_risk = float(module_b_res.get("drift_risk_score", 0.0))
        drift_pct = float(module_b_res.get("drift_percentage", 0.0))
        ci_lower = float(module_b_res.get("ci_lower_80", pred_168 * 0.90))
        ci_upper = float(module_b_res.get("ci_upper_80", pred_168 * 1.10))
        ci_width = max(0.1, ci_upper - ci_lower)

        proto_limit = float(safety_env_res.get("data_driven_prototype_limit", self.prototype_limit))
        eng_limit = float(safety_env_res.get("engineering_limit", self.engineering_limit))
        early_slope = float(safety_env_res.get("early_slope", 0.0))
        max_early_slope = float(safety_env_res.get("healthy_envelope_max_early_slope", 0.05))
        safety_margin = max(0.0, eng_limit - pred_168)
        proto_margin = max(0.0, proto_limit - pred_168)

        current_decision = str(decision_res.get("final_decision", "PASS"))

        # Evidence dictionary for auditability
        evidence_used = {
            "component_id": component_id,
            "lot_id": lot_id,
            "checkpoint": cp_str,
            "absolute_spec_failed": abs_failed,
            "anomaly_score": round(anom_score, 4),
            "anomaly_severity": anom_severity,
            "lot_median_multiplier": round(lot_mult, 2),
            "max_robust_z": round(max_z, 3),
            "predicted_168h": round(pred_168, 2),
            "drift_risk_score": round(drift_risk, 4),
            "drift_percentage": round(drift_pct, 2),
            "confidence_interval_80": [round(ci_lower, 2), round(ci_upper, 2)],
            "confidence_interval_width": round(ci_width, 2),
            "early_slope": round(early_slope, 4),
            "healthy_envelope_max_early_slope": round(max_early_slope, 4),
            "safety_margin_to_datasheet": round(safety_margin, 2),
            "margin_to_prototype_limit": round(proto_margin, 2),
            "current_screening_decision": current_decision
        }

        # 2. Checkpoint-Specific Adaptive Reasoning
        reason_codes = []

        # --- A. CHECKPOINT 168h: TERMINAL ESS POINT ---
        if cp_str == "168h":
            recommended_cp = None
            if abs_failed == 1 or current_decision == "REJECT" or pred_168 >= eng_limit:
                action = AdaptiveAction.QA_REVIEW
                priority = "CRITICAL"
                risk_level = "CRITICAL"
                confidence = 0.98
                reason_codes.append("TERMINAL_SPECIFICATION_OR_LOT_REJECTION")
                reason_codes.append("FORMAL_MRB_ESCALATION_REQUIRED")
                explanation = (
                    f"Full 168-hour flight burn-in cycle completed. Component {component_id} "
                    f"exceeds screening criteria (Disposition: {current_decision}). "
                    f"Recommend formal QA Engineering / Material Review Board review."
                )
                obs_text = f"Terminal 168h screening concluded with disposition: {current_decision}."
                why_text = "Burn-in screening period is complete; component does not meet acceptance baseline."
                rec_text = "Escalate component to QA Engineering Review Board."
                ben_text = "Prevents defective component escape into flight hardware."
            elif current_decision == "REVIEW" or proto_margin < 2.0:
                action = AdaptiveAction.QA_REVIEW
                priority = "HIGH"
                risk_level = "ELEVATED"
                confidence = 0.92
                reason_codes.append("BORDERLINE_168H_DEGRADATION")
                reason_codes.append("ENGINEERING_SIGN_OFF_REQUIRED")
                explanation = (
                    f"Full 168-hour flight burn-in completed. Component {component_id} "
                    f"shows borderline drift ({pred_168:.2f} µA). Recommend qualified reliability engineer review."
                )
                obs_text = f"168h endpoint reached with borderline parametric drift ({pred_168:.2f} µA)."
                why_text = "Latent degradation rate requires sign-off before flight installation."
                rec_text = "Conduct detailed QA reliability review."
                ben_text = "Ensures mission risk is formally documented and assessed."
            else:
                action = AdaptiveAction.CONTINUE_STANDARD_SCREENING
                priority = "LOW"
                risk_level = "NOMINAL"
                confidence = 0.99
                reason_codes.append("BURNIN_COMPLETE_PASS")
                reason_codes.append("PEER_LOT_CONFORMANCE")
                explanation = (
                    f"Full 168-hour burn-in completed successfully. Component {component_id} "
                    f"conforms to lot baseline and absolute specifications. Cleared for flight integration."
                )
                obs_text = "All scheduled burn-in hours completed within nominal specifications."
                why_text = "Telemetry proves component stability across entire stress profile."
                rec_text = "Conclude ESS screening; release for flight assembly."
                ben_text = "Validates component flight-readiness."

        # --- B. CHECKPOINT 96h: INTERMEDIATE RE-ANALYSIS ---
        elif cp_str == "96h":
            recommended_cp = "168h"
            if abs_failed == 1 or pred_168 >= eng_limit:
                action = AdaptiveAction.QA_REVIEW
                priority = "CRITICAL"
                risk_level = "CRITICAL"
                confidence = 0.95
                reason_codes.append("SPECIFICATION_LIMIT_BREACH")
                reason_codes.append("UNSAFE_168H_TRAJECTORY")
                explanation = (
                    f"Intermediate 96h measurement reveals severe acceleration. Forecasted 168h "
                    f"value ({pred_168:.2f} µA) exceeds datasheet limit ({eng_limit:.1f} µA). Immediate review required."
                )
                obs_text = f"96h intermediate telemetry indicates unsafe projected 168h endpoint ({pred_168:.2f} µA)."
                why_text = "Degradation is accelerating beyond acceptable limits."
                rec_text = "Immediately halt stress and escalate for engineering review."
                ben_text = "Prevents catastrophic thermal runaway during burn-in."
            elif current_decision == "REJECT" or drift_risk >= self.high_drift_risk or proto_margin < 2.0:
                action = AdaptiveAction.PRIORITY_SCREENING
                priority = "HIGH"
                risk_level = "HIGH"
                confidence = 0.90
                reason_codes.append("ACCELERATING_DRIFT")
                reason_codes.append("LOW_SAFETY_MARGIN")
                reason_codes.append("PRIORITY_168H_MONITORING")
                explanation = (
                    f"96h re-analysis indicates high drift risk ({drift_risk:.2f}) approaching safety ceiling. "
                    f"Recommend priority screening toward 168h with heightened sensor frequency."
                )
                obs_text = f"Persistent drift through 96h ({lot_mult:.1f}x lot baseline, risk {drift_risk:.2f})."
                why_text = "Component continues to diverge from peer cohort."
                rec_text = "Priority 168h final measurement with continuous thermal monitoring."
                ben_text = "Captures exact terminal degradation without risking undetected failure."
            elif current_decision == "REVIEW" or lot_mult >= self.mod_lot_dev:
                action = AdaptiveAction.ADDITIONAL_MEASUREMENT
                priority = "MEDIUM"
                risk_level = "ELEVATED"
                confidence = 0.85
                reason_codes.append("ELEVATED_LOT_DEVIATION")
                reason_codes.append("INTERMEDIATE_DRIFT_CONFIRMED")
                explanation = (
                    f"96h telemetry confirms elevated lot-relative behavior. Continued measurement "
                    f"at 168h is recommended to verify long-term kinetic asymptote."
                )
                obs_text = f"Moderate elevation persists at 96h ({lot_mult:.1f}x lot median)."
                why_text = "Kinetics show slight recovery or linear creep needing final verification."
                rec_text = "Proceed to full 168h burn-in endpoint."
                ben_text = "Confirms whether drift plateaus or continues toward limit."
            else:
                action = AdaptiveAction.CONTINUE_STANDARD_SCREENING
                priority = "LOW"
                risk_level = "NOMINAL"
                confidence = 0.94
                reason_codes.append("STABLE_INTERMEDIATE_KINETICS")
                reason_codes.append("NOMINAL_PEER_ALIGNMENT")
                explanation = (
                    f"96h checkpoint confirms nominal degradation rate and safe trajectory. "
                    f"Continue standard screening toward 168h."
                )
                obs_text = "Component stable at 96h with low anomaly score and safe margin."
                why_text = "Degradation follows healthy reference population distribution."
                rec_text = "Continue standard burn-in to 168h."
                ben_text = "Completes standard screening protocol with high confidence."

        # --- C. CHECKPOINT 24h: EARLY SCREENING (CORE ASI DEMO) ---
        elif cp_str == "24h":
            recommended_cp = "96h"
            if abs_failed == 1:
                action = AdaptiveAction.QA_REVIEW
                priority = "CRITICAL"
                risk_level = "CRITICAL"
                confidence = 0.99
                reason_codes.append("IMMEDIATE_SPECIFICATION_VIOLATION")
                explanation = (
                    f"Component {component_id} breached absolute datasheet limit at 24h. "
                    f"Halt standard burn-in and initiate immediate QA engineering review."
                )
                obs_text = "Parametric measurement exceeds absolute manufacturer limits at 24h."
                why_text = "Component is definitively defective and will not recover."
                rec_text = "Immediate QA engineering review and lot containment."
                ben_text = "Avoids wasting burn-in chamber capacity on confirmed rejects."
            elif lot_mult >= self.high_lot_dev or max_z >= self.high_z or anom_severity == "SEVERE":
                # Classic IC_A_002 morphology: inside datasheet limit, but 3.7x lot median!
                # Evidence indicates high risk, but future trajectory is uncertain at only 24h.
                action = AdaptiveAction.ADDITIONAL_MEASUREMENT
                priority = "HIGH"
                risk_level = "HIGH"
                confidence = 0.88
                reason_codes.append("HIGH_LOT_DEVIATION")
                reason_codes.append("LATENT_KINETIC_ANOMALY")
                reason_codes.append("PREDICTION_UNCERTAINTY")
                if early_slope > max_early_slope:
                    reason_codes.append("ACCELERATING_EARLY_DRIFT")
                explanation = (
                    f"Component {component_id} exhibits significant deviation from its lot baseline "
                    f"({lot_mult:.1f}x lot median, Robust Z = +{max_z:.2f}) despite remaining inside absolute limits. "
                    f"Additional intermediate measurement at 96h is recommended to confirm trajectory before final disposition."
                )
                obs_text = f"Component is {lot_mult:.1f}x above lot median at 24h with an elevated anomaly score ({anom_score:.2f})."
                why_text = "Lot-relative outlier behavior indicates latent manufacturing defects that may drift catastrophically."
                rec_text = "Perform additional intermediate measurement at 96h."
                ben_text = "Empirical 96h measurement will resolve prediction uncertainty and determine whether drift is accelerating."
            elif current_decision == "REVIEW" or lot_mult >= self.mod_lot_dev or drift_risk >= self.mod_drift_risk:
                action = AdaptiveAction.ADDITIONAL_MEASUREMENT
                priority = "MEDIUM"
                risk_level = "ELEVATED"
                confidence = 0.82
                reason_codes.append("ELEVATED_LOT_DEVIATION")
                reason_codes.append("BORDERLINE_DRIFT")
                reason_codes.append("PREDICTION_UNCERTAINTY")
                explanation = (
                    f"Telemetry at 24h shows moderate deviation from lot median ({lot_mult:.1f}x). "
                    f"Intermediate measurement at 96h recommended to verify degradation rate."
                )
                obs_text = f"Moderate elevation relative to lot baseline ({lot_mult:.1f}x lot median)."
                why_text = "Early 24h baseline has insufficient elapsed time to guarantee 168h stability."
                rec_text = "Schedule intermediate measurement at 96h."
                ben_text = "Significantly narrows confidence interval before final screening."
            else:
                action = AdaptiveAction.CONTINUE_STANDARD_SCREENING
                priority = "LOW"
                risk_level = "NOMINAL"
                confidence = 0.92
                reason_codes.append("NOMINAL_PEER_ALIGNMENT")
                reason_codes.append("STABLE_EARLY_KINETICS")
                explanation = (
                    f"Component {component_id} closely tracks its peer cohort distribution "
                    f"({lot_mult:.1f}x lot median). Continue standard screening schedule toward 96h."
                )
                obs_text = f"Early 24h measurement aligns with lot median ({lot_mult:.1f}x)."
                why_text = "No signs of latent kinetic outlier behavior."
                rec_text = "Continue standard ESS screening toward 96h."
                ben_text = "Standard screening protocol suffices with high confidence."

        # --- D. CHECKPOINT 0h: PRE-BURNIN BASELINE ---
        else:
            recommended_cp = "24h"
            if abs_failed == 1:
                action = AdaptiveAction.QA_REVIEW
                priority = "CRITICAL"
                risk_level = "CRITICAL"
                confidence = 0.99
                reason_codes.append("PRE_STRESS_SPECIFICATION_FAILURE")
                explanation = f"Pre-stress 0h measurement violates specification limit. Escalate for QA review."
                obs_text = "Out of spec before burn-in begins."
                why_text = "Immediate defect."
                rec_text = "Halt testing; reject component."
                ben_text = "Saves testing resources."
            else:
                action = AdaptiveAction.CONTINUE_STANDARD_SCREENING
                priority = "LOW"
                risk_level = "NOMINAL"
                confidence = 0.85
                reason_codes.append("PRE_STRESS_BASELINE_ACCEPTABLE")
                explanation = f"Pre-stress baseline established. Proceed to 24h initial burn-in checkpoint."
                obs_text = "0h baseline established."
                why_text = "Ready for thermal electrical screening."
                rec_text = "Proceed to 24h checkpoint."
                ben_text = "Establishes baseline for drift calculation."

        structured_justification = {
            "observed": obs_text,
            "why_it_matters": why_text,
            "recommendation": rec_text,
            "expected_benefit": ben_text
        }

        return {
            "action": action,
            "priority": priority,
            "risk_level": risk_level,
            "confidence": round(float(confidence), 3),
            "recommended_checkpoint": recommended_cp,
            "reason_codes": reason_codes,
            "explanation": explanation,
            "structured_justification": structured_justification,
            "evidence_used": evidence_used
        }

    def _build_insufficient_data_response(
        self,
        component_id: str,
        lot_id: str,
        checkpoint: str,
        detail: str
    ) -> Dict[str, Any]:
        return {
            "action": AdaptiveAction.INSUFFICIENT_DATA,
            "priority": "HIGH",
            "risk_level": "UNKNOWN",
            "confidence": 0.0,
            "recommended_checkpoint": checkpoint,
            "reason_codes": ["INSUFFICIENT_TELEMETRY", "MISSING_BASELINE_DATA"],
            "explanation": f"INSUFFICIENT DATA for {component_id}: {detail} Cannot reliably establish screening recommendation.",
            "structured_justification": {
                "observed": "Telemetry data missing, truncated, or physically impossible.",
                "why_it_matters": "Decision engine requires valid baseline measurements to compute lot relative kinetics.",
                "recommendation": "Re-measure component channel and verify instrumentation connection.",
                "expected_benefit": "Restores data completeness required for AI screening."
            },
            "evidence_used": {
                "component_id": component_id,
                "lot_id": lot_id,
                "checkpoint": checkpoint,
                "error": detail
            }
        }


def recommend_next_screening_action(
    component_id: str,
    lot_id: str = "UNKNOWN_LOT",
    current_checkpoint: str = "24h",
    raw_measurements: Optional[Dict[str, float]] = None,
    module_a_res: Optional[Dict[str, Any]] = None,
    module_b_res: Optional[Dict[str, Any]] = None,
    safety_env_res: Optional[Dict[str, Any]] = None,
    decision_res: Optional[Dict[str, Any]] = None,
    data_quality: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convenience top-level functional interface for Adaptive Screening Intelligence."""
    engine = AdaptiveScreeningEngine()
    return engine.evaluate(
        component_id=component_id,
        lot_id=lot_id,
        current_checkpoint=current_checkpoint,
        raw_measurements=raw_measurements,
        module_a_res=module_a_res,
        module_b_res=module_b_res,
        safety_env_res=safety_env_res,
        decision_res=decision_res,
        data_quality=data_quality
    )
