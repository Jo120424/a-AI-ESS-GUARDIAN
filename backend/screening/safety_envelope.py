#!/usr/bin/env python3
"""
Dynamic Safety Slope & Safety Envelope Module (Phase 4).
Estimates data-driven acceptable drift behavior from reference/healthy populations.
Clearly distinguishes between DATA-DRIVEN PROTOTYPE THRESHOLD and ENGINEERING/CERTIFICATION LIMIT.

Formulations:
- early_slope = (Value_24h - Value_0h) / 24.0
- predicted_slope = (Value_168h - Value_24h) / 144.0
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class DynamicSafetyEnvelope:
    """
    Learns robust kinetic drift bounds from reference component populations
    to construct dynamic, data-driven screening safety envelopes.
    """

    def __init__(
        self,
        target_parameter: str = "leakage_current_ua",
        engineering_limit: float = 50.0,
        prototype_percentile: float = 95.0,
        mad_multiplier: float = 3.0
    ):
        self.target_parameter = target_parameter
        self.engineering_limit = float(engineering_limit)
        self.prototype_percentile = float(prototype_percentile)
        self.mad_multiplier = float(mad_multiplier)

        # Learned envelope parameters
        self.healthy_median_early_slope_: float = 0.0
        self.healthy_mad_early_slope_: float = 0.0
        self.healthy_max_early_slope_: float = 0.0
        self.healthy_median_pred_slope_: float = 0.0
        self.healthy_mad_pred_slope_: float = 0.0
        self.healthy_max_pred_slope_: float = 0.0
        self.data_driven_prototype_limit_: float = engineering_limit * 0.90
        self.is_fitted_: bool = False

    def fit(
        self,
        reference_df: pd.DataFrame,
        val_0h_col: str = "leakage_0h",
        val_24h_col: str = "leakage_24h",
        val_168h_col: str = "leakage_168h"
    ) -> "DynamicSafetyEnvelope":
        """
        Learns the normative drift envelope from reference training components.
        To avoid data leakage, only reference/training components are used.
        """
        if len(reference_df) == 0:
            raise ValueError("Reference dataset for safety envelope cannot be empty.")

        v0 = reference_df[val_0h_col].astype(float).values
        v24 = reference_df[val_24h_col].astype(float).values
        v168 = reference_df[val_168h_col].astype(float).values

        early_slopes = (v24 - v0) / 24.0
        pred_slopes = (v168 - v24) / 144.0

        # Calculate robust median and MAD for early slopes
        med_early = float(np.median(early_slopes))
        mad_early = float(np.median(np.abs(early_slopes - med_early)))
        effective_mad_early = max(mad_early, 1e-4)

        # Upper percentile / robust bound for acceptable early slope
        p_early = float(np.percentile(early_slopes, self.prototype_percentile))
        max_early = med_early + self.mad_multiplier * effective_mad_early * 1.4826
        envelope_max_early = max(p_early, max_early)

        # Calculate robust median and MAD for long-term slopes
        med_pred = float(np.median(pred_slopes))
        mad_pred = float(np.median(np.abs(pred_slopes - med_pred)))
        effective_mad_pred = max(mad_pred, 1e-4)

        p_pred = float(np.percentile(pred_slopes, self.prototype_percentile))
        max_pred = med_pred + self.mad_multiplier * effective_mad_pred * 1.4826
        envelope_max_pred = max(p_pred, max_pred)

        # Data-driven prototype limit: 95th percentile of 168h reference population,
        # capped conservatively at 90% of the absolute engineering limit.
        ref_168h_p95 = float(np.percentile(v168, self.prototype_percentile))
        prototype_thresh = min(ref_168h_p95 * 1.15, self.engineering_limit * 0.90)

        self.healthy_median_early_slope_ = round(med_early, 5)
        self.healthy_mad_early_slope_ = round(effective_mad_early, 5)
        self.healthy_max_early_slope_ = round(float(envelope_max_early), 5)

        self.healthy_median_pred_slope_ = round(med_pred, 5)
        self.healthy_mad_pred_slope_ = round(effective_mad_pred, 5)
        self.healthy_max_pred_slope_ = round(float(envelope_max_pred), 5)

        self.data_driven_prototype_limit_ = round(float(prototype_thresh), 3)
        self.is_fitted_ = True
        return self

    def evaluate(
        self,
        val_0h: float,
        val_24h: float,
        predicted_168h: float
    ) -> Dict:
        """
        Evaluates a single component's early and forecasted slope against the safety envelope.
        """
        if not self.is_fitted_:
            # Default envelope heuristic if fit was not yet called
            self.healthy_max_early_slope_ = 0.05
            self.healthy_max_pred_slope_ = 0.05
            self.data_driven_prototype_limit_ = self.engineering_limit * 0.90

        early_slope = (float(val_24h) - float(val_0h)) / 24.0
        predicted_slope = (float(predicted_168h) - float(val_24h)) / 144.0

        # Check conditions
        early_slope_breach = early_slope > self.healthy_max_early_slope_
        pred_slope_breach = predicted_slope > self.healthy_max_pred_slope_
        prototype_breach = float(predicted_168h) >= self.data_driven_prototype_limit_
        engineering_breach = float(predicted_168h) >= self.engineering_limit

        envelope_violated = early_slope_breach or pred_slope_breach or prototype_breach or engineering_breach

        # Margins
        margin_to_prototype = self.data_driven_prototype_limit_ - float(predicted_168h)
        margin_to_engineering = self.engineering_limit - float(predicted_168h)

        # Plain language explanation
        reasons = []
        if early_slope_breach:
            reasons.append(
                f"Early drift rate ({early_slope:.4f}/h) exceeds healthy-lot envelope max ({self.healthy_max_early_slope_:.4f}/h)"
            )
        if pred_slope_breach:
            reasons.append(
                f"Forecasted future slope ({predicted_slope:.4f}/h) exceeds normative envelope ({self.healthy_max_pred_slope_:.4f}/h)"
            )
        if prototype_breach and not engineering_breach:
            reasons.append(
                f"Predicted 168h ({predicted_168h:.2f}) approaches/exceeds data-driven prototype threshold ({self.data_driven_prototype_limit_:.2f})"
            )
        if engineering_breach:
            reasons.append(
                f"Predicted 168h ({predicted_168h:.2f}) violates absolute engineering datasheet limit ({self.engineering_limit:.2f})"
            )

        if envelope_violated:
            explanation = "SAFETY ENVELOPE BREACH: " + "; ".join(reasons)
        else:
            explanation = (
                f"WITHIN SAFETY ENVELOPE: Early slope {early_slope:.4f}/h and forecasted slope {predicted_slope:.4f}/h "
                f"are within healthy envelope. Forecast margin to prototype threshold: {margin_to_prototype:.2f}."
            )

        return {
            "early_slope": round(float(early_slope), 5),
            "predicted_slope": round(float(predicted_slope), 5),
            "healthy_envelope_max_early_slope": self.healthy_max_early_slope_,
            "healthy_envelope_max_pred_slope": self.healthy_max_pred_slope_,
            "data_driven_prototype_limit": self.data_driven_prototype_limit_,
            "engineering_limit": self.engineering_limit,
            "margin_to_prototype": round(float(margin_to_prototype), 4),
            "margin_to_engineering": round(float(margin_to_engineering), 4),
            "early_slope_breach": bool(early_slope_breach),
            "pred_slope_breach": bool(pred_slope_breach),
            "prototype_breach": bool(prototype_breach),
            "engineering_breach": bool(engineering_breach),
            "envelope_violated": bool(envelope_violated),
            "envelope_status": "BREACH" if envelope_violated else "SAFE",
            "threshold_classification": "DATA-DRIVEN PROTOTYPE THRESHOLD (Research/Screening)",
            "certification_classification": "ENGINEERING/CERTIFICATION LIMIT (Datasheet)",
            "explanation": explanation
        }

    def to_dict(self) -> Dict:
        return {
            "target_parameter": self.target_parameter,
            "engineering_limit": self.engineering_limit,
            "data_driven_prototype_limit": self.data_driven_prototype_limit_,
            "healthy_median_early_slope": self.healthy_median_early_slope_,
            "healthy_mad_early_slope": self.healthy_mad_early_slope_,
            "healthy_max_early_slope": self.healthy_max_early_slope_,
            "healthy_median_pred_slope": self.healthy_median_pred_slope_,
            "healthy_mad_pred_slope": self.healthy_mad_pred_slope_,
            "healthy_max_pred_slope": self.healthy_max_pred_slope_,
            "is_fitted": self.is_fitted_
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DynamicSafetyEnvelope":
        inst = cls(
            target_parameter=d.get("target_parameter", "leakage_current_ua"),
            engineering_limit=d.get("engineering_limit", 50.0)
        )
        inst.data_driven_prototype_limit_ = d.get("data_driven_prototype_limit", 45.0)
        inst.healthy_median_early_slope_ = d.get("healthy_median_early_slope", 0.0)
        inst.healthy_mad_early_slope_ = d.get("healthy_mad_early_slope", 0.0)
        inst.healthy_max_early_slope_ = d.get("healthy_max_early_slope", 0.05)
        inst.healthy_median_pred_slope_ = d.get("healthy_median_pred_slope", 0.0)
        inst.healthy_mad_pred_slope_ = d.get("healthy_mad_pred_slope", 0.0)
        inst.healthy_max_pred_slope_ = d.get("healthy_max_pred_slope", 0.05)
        inst.is_fitted_ = d.get("is_fitted", True)
        return inst
