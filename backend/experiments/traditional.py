#!/usr/bin/env python3
"""
Model A: Traditional Static Environmental Stress Screening (ESS) Baseline.
Evaluates components deterministically against invariant datasheet/specification
limits according to MIL-PRF-62F space qualification guidelines.
"""

from typing import Dict, List, Optional
import pandas as pd


class TraditionalScreening:
    """
    Traditional Static Screening Baseline according to MIL-PRF-62F specification.
    Operates without population context or trajectory forecasting.
    """

    def __init__(
        self,
        spec_limit_delta_c: float = 20.0,
        spec_limit_delta_esr: float = 100.0,
        tightened_limit_delta_c: float = 5.0,
        tightened_limit_delta_esr: float = 30.0
    ):
        self.spec_limit_delta_c = spec_limit_delta_c
        self.spec_limit_delta_esr = spec_limit_delta_esr
        self.tightened_limit_delta_c = tightened_limit_delta_c
        self.tightened_limit_delta_esr = tightened_limit_delta_esr

    def evaluate_component(self, comp_features: Dict[str, float]) -> Dict:
        """
        Evaluates a single component based on its pre-screening feature dictionary.
        """
        cid = comp_features.get("component_id", "UNKNOWN")
        delta_c = float(comp_features.get("delta_capacitance_pct", 0.0))
        delta_esr = float(comp_features.get("delta_esr_pct", 0.0))

        # Standard MIL-PRF-62F check
        violating_params = []
        if delta_c >= self.spec_limit_delta_c:
            violating_params.append(f"delta_capacitance_pct ({delta_c:.2f}% >= {self.spec_limit_delta_c}%)")
        if delta_esr >= self.spec_limit_delta_esr:
            violating_params.append(f"delta_esr_pct ({delta_esr:.2f}% >= {self.spec_limit_delta_esr}%)")

        decision = 1 if len(violating_params) > 0 else 0
        margin_c = self.spec_limit_delta_c - delta_c
        margin_esr = self.spec_limit_delta_esr - delta_esr

        # Tightened baseline check
        violating_tightened = []
        if delta_c >= self.tightened_limit_delta_c:
            violating_tightened.append(f"delta_capacitance_pct ({delta_c:.2f}% >= {self.tightened_limit_delta_c}%)")
        if delta_esr >= self.tightened_limit_delta_esr:
            violating_tightened.append(f"delta_esr_pct ({delta_esr:.2f}% >= {self.tightened_limit_delta_esr}%)")

        decision_tightened = 1 if len(violating_tightened) > 0 else 0

        # Formulate plain-language explanation
        if decision == 1:
            explanation = (
                f"REJECTED: Static MIL-PRF-62F violation detected at screening cutoff: "
                f"{', '.join(violating_params)}."
            )
        else:
            explanation = (
                f"PASSED: Telemetry within static limits (Delta C = {delta_c:.2f}% < {self.spec_limit_delta_c}%, "
                f"Delta ESR = {delta_esr:.2f}% < {self.spec_limit_delta_esr}%). "
                f"Safety margin to limit: {margin_c:.2f}%."
            )

        return {
            "component_id": cid,
            "traditional_decision": decision,
            "traditional_status": "FAIL" if decision == 1 else "PASS",
            "traditional_decision_tightened": decision_tightened,
            "traditional_status_tightened": "FAIL" if decision_tightened == 1 else "PASS",
            "delta_c_at_screen": delta_c,
            "delta_esr_at_screen": delta_esr,
            "spec_margin_c": round(margin_c, 4),
            "spec_margin_esr": round(margin_esr, 4),
            "violating_parameters": violating_params,
            "num_violations": len(violating_params),
            "explanation": explanation
        }

    def evaluate_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates a batch of components.
        """
        results = []
        for _, row in features_df.iterrows():
            res = self.evaluate_component(row.to_dict())
            results.append(res)
        return pd.DataFrame(results)
