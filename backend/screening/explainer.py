#!/usr/bin/env python3
"""
EXPLAINABILITY ENGINE (Phase 6).
Generates transparent, physics-based explanations, feature attributions,
and lot baseline comparisons for aerospace QA screening inspectors.
Never presents an unexplained "AI says REJECT" black box.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class ScreeningExplainer:
    """
    Produces structured, human-readable QA audit cards and feature attributions.
    """

    @staticmethod
    def generate_component_card(
        cid: str,
        lot_id: str,
        raw_values: Dict[str, float],
        lot_medians: Dict[str, float],
        module_a_res: Dict,
        module_b_res: Dict,
        envelope_res: Dict,
        decision_res: Dict
    ) -> Dict:
        """
        Creates an explainable diagnostic card for a specific component.
        """
        final_decision = decision_res.get("final_decision", "PASS")
        status_color = {
            "PASS": "#10b981",    # Emerald green
            "REVIEW": "#f59e0b",  # Amber / orange
            "REJECT": "#ef4444"   # Crimson red
        }.get(final_decision, "#64748b")

        # Parametric deviations relative to lot baseline
        comparisons = []
        for param, val in raw_values.items():
            if param in lot_medians:
                med = lot_medians[param]
                ratio = val / max(med, 1e-4) if med > 0 else 1.0
                comparisons.append({
                    "parameter": param,
                    "measured_value": round(val, 3),
                    "lot_median": round(med, 3),
                    "ratio_to_median": round(ratio, 2),
                    "status": "ELEVATED" if ratio >= 2.0 else ("BORDERLINE" if ratio >= 1.5 else "NOMINAL")
                })

        # Structured Why? breakdown
        why_bullets = decision_res.get("qa_bullets", [])

        # Lead time calculation
        lead_time_h = 168.0 - 24.0  # Early warning advance notice

        return {
            "component_id": cid,
            "lot_id": lot_id,
            "final_decision": final_decision,
            "status_color": status_color,
            "decision_headline": f"FINAL DECISION: {final_decision}",
            "why_bullets": why_bullets,
            "anomaly_score": module_a_res.get("anomaly_score", 0.0),
            "primary_abnormal_parameter": module_a_res.get("primary_abnormal_parameter", "None"),
            "lot_deviation_multiplier": decision_res.get("lot_median_multiplier", 1.0),
            "early_slope": envelope_res.get("early_slope", 0.0),
            "healthy_envelope_max_early_slope": envelope_res.get("healthy_envelope_max_early_slope", 0.05),
            "predicted_168h": module_b_res.get("predicted_168h", 0.0),
            "actual_168h": module_b_res.get("actual_168h", None),
            "safety_prototype_threshold": envelope_res.get("data_driven_prototype_limit", 45.0),
            "engineering_datasheet_limit": envelope_res.get("engineering_limit", 50.0),
            "drift_percentage": module_b_res.get("drift_percentage", 0.0),
            "confidence_interval_80": [
                module_b_res.get("ci_lower_80", 0.0),
                module_b_res.get("ci_upper_80", 0.0)
            ],
            "lead_time_advance_notice_hours": lead_time_h,
            "parameter_comparisons": comparisons,
            "full_explanation_text": decision_res.get("explanation", "")
        }

    @staticmethod
    def format_text_report(card: Dict) -> str:
        """
        Formats a clean ASCII report suitable for QA logging.
        """
        lines = [
            "=" * 60,
            f" AI-ESS GUARDIAN: SCREENING AUDIT REPORT",
            f" Component ID: {card['component_id']} | Lot: {card['lot_id']}",
            "=" * 60,
            f" {card['decision_headline']}",
            "-" * 60,
            " Screening Factors:",
        ]
        for b in card["why_bullets"]:
            lines.append(f"   {b}")
        lines.extend([
            "-" * 60,
            f" Anomaly Score:               {card['anomaly_score']:.3f} / 1.000",
            f" Lot Baseline Multiplier:     {card['lot_deviation_multiplier']:.1f}x Lot Median",
            f" Early Drift Slope:           {card['early_slope']:.4f} /h (Envelope Limit: {card['healthy_envelope_max_early_slope']:.4f} /h)",
            f" Predicted 168h Value:        {card['predicted_168h']:.2f} (Datasheet Limit: {card['engineering_datasheet_limit']:.2f})",
            f" Data-Driven Prototype Limit: {card['safety_prototype_threshold']:.2f}",
            f" Predictive Advance Warning:  {card['lead_time_advance_notice_hours']:.1f} hours",
            "=" * 60
        ])
        return "\n".join(lines)
