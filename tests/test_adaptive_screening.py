#!/usr/bin/env python3
"""
Unit Test Suite for Adaptive Screening Intelligence (ASI).
Validates:
1. Normal component -> CONTINUE_STANDARD_SCREENING
2. Suspicious component (lot deviation + uncertainty) -> ADDITIONAL_MEASUREMENT
3. High-risk component (steep drift, high anomaly) -> PRIORITY_SCREENING / QA_REVIEW
4. Insufficient / incomplete data -> INSUFFICIENT_DATA
5. Near safety envelope -> escalation / review behavior
6. 168h final checkpoint -> final disposition, no unnecessary next measurement
7. Invalid inputs / malformed values -> graceful fallback
8. Checkpoint transitions (24h -> 96h -> 168h)
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from backend.screening.adaptive_screening import AdaptiveScreeningEngine, AdaptiveAction, recommend_next_screening_action


class TestAdaptiveScreeningEngine(unittest.TestCase):

    def setUp(self):
        self.engine = AdaptiveScreeningEngine(
            engineering_limit=50.0,
            prototype_limit=45.0,
            high_lot_deviation_threshold=3.0,
            moderate_lot_deviation_threshold=2.0
        )

    def test_normal_component_continues_standard_screening(self):
        """Case 1: Normal component aligned with lot median at 24h -> CONTINUE_STANDARD_SCREENING."""
        res = self.engine.evaluate(
            component_id="IC_NORM_001",
            lot_id="LOT_A_2026",
            current_checkpoint="24h",
            raw_measurements={"0h": 10.0, "24h": 10.2},
            module_a_res={
                "anomaly_score": 0.15,
                "severity": "NORMAL",
                "leakage_mult_of_lot_median": 1.02,
                "max_robust_z": 0.25,
                "absolute_spec_failed": 0
            },
            module_b_res={
                "predicted_168h": 11.5,
                "drift_risk_score": 0.12,
                "drift_percentage": 15.0,
                "ci_lower_80": 10.5,
                "ci_upper_80": 12.5
            },
            safety_env_res={
                "early_slope": 0.0083,
                "healthy_envelope_max_early_slope": 0.05,
                "data_driven_prototype_limit": 45.0,
                "engineering_limit": 50.0
            },
            decision_res={"final_decision": "PASS"}
        )
        self.assertEqual(res["action"], AdaptiveAction.CONTINUE_STANDARD_SCREENING)
        self.assertEqual(res["recommended_checkpoint"], "96h")
        self.assertEqual(res["priority"], "LOW")
        self.assertIn("NOMINAL_PEER_ALIGNMENT", res["reason_codes"])
        self.assertGreater(res["confidence"], 0.8)

    def test_suspicious_component_recommends_additional_measurement(self):
        """Case 2: Elevated lot deviation (e.g. IC_A_002 3.69x) at 24h -> ADDITIONAL_MEASUREMENT at 96h."""
        res = self.engine.evaluate(
            component_id="IC_A_002",
            lot_id="LOT_A_2026",
            current_checkpoint="24h",
            raw_measurements={"0h": 39.35, "24h": 38.61},
            module_a_res={
                "anomaly_score": 0.832,
                "severity": "SEVERE",
                "leakage_mult_of_lot_median": 3.69,
                "max_robust_z": 4.12,
                "absolute_spec_failed": 0
            },
            module_b_res={
                "predicted_168h": 44.60,
                "drift_risk_score": 0.88,
                "drift_percentage": 13.3,
                "ci_lower_80": 39.1,
                "ci_upper_80": 48.2
            },
            safety_env_res={
                "early_slope": -0.0308,
                "healthy_envelope_max_early_slope": 0.05,
                "data_driven_prototype_limit": 45.0,
                "engineering_limit": 50.0
            },
            decision_res={"final_decision": "REVIEW"}
        )
        self.assertEqual(res["action"], AdaptiveAction.ADDITIONAL_MEASUREMENT)
        self.assertEqual(res["recommended_checkpoint"], "96h")
        self.assertEqual(res["priority"], "HIGH")
        self.assertIn("HIGH_LOT_DEVIATION", res["reason_codes"])

    def test_high_risk_trajectory_at_96h_triggers_priority_or_qa(self):
        """Case 3: Accelerating drift approaching limits at 96h -> PRIORITY_SCREENING or QA_REVIEW."""
        res = self.engine.evaluate(
            component_id="IC_DRIFT_HIGH",
            lot_id="LOT_A_2026",
            current_checkpoint="96h",
            raw_measurements={"0h": 10.0, "24h": 18.0, "96h": 35.0},
            module_a_res={
                "anomaly_score": 0.85,
                "severity": "SEVERE",
                "leakage_mult_of_lot_median": 3.4,
                "max_robust_z": 4.5,
                "absolute_spec_failed": 0
            },
            module_b_res={
                "predicted_168h": 48.5,
                "drift_risk_score": 0.92,
                "drift_percentage": 250.0,
                "ci_lower_80": 44.0,
                "ci_upper_80": 53.0
            },
            safety_env_res={
                "early_slope": 0.33,
                "healthy_envelope_max_early_slope": 0.05,
                "data_driven_prototype_limit": 45.0,
                "engineering_limit": 50.0
            },
            decision_res={"final_decision": "REJECT"}
        )
        self.assertIn(res["action"], [AdaptiveAction.PRIORITY_SCREENING, AdaptiveAction.QA_REVIEW])
        self.assertEqual(res["recommended_checkpoint"], "168h")
        self.assertIn("ACCELERATING_DRIFT", res["reason_codes"])

    def test_insufficient_or_corrupted_data(self):
        """Case 4: Missing or invalid telemetry records -> INSUFFICIENT_DATA."""
        res = self.engine.evaluate(
            component_id="IC_CORRUPT_001",
            current_checkpoint="24h",
            data_quality={"completeness": 0.2, "status": "INVALID"}
        )
        self.assertEqual(res["action"], AdaptiveAction.INSUFFICIENT_DATA)
        self.assertIn("INSUFFICIENT_TELEMETRY", res["reason_codes"])
        self.assertEqual(res["confidence"], 0.0)

    def test_component_near_safety_envelope_at_96h(self):
        """Case 5: Predicted value approaching safety ceiling -> low safety margin recognized."""
        res = self.engine.evaluate(
            component_id="IC_SAFETY_MARGIN",
            current_checkpoint="96h",
            module_b_res={"predicted_168h": 44.8, "drift_risk_score": 0.85},
            safety_env_res={"data_driven_prototype_limit": 45.0, "engineering_limit": 50.0},
            decision_res={"final_decision": "REVIEW"}
        )
        self.assertIn(res["action"], [AdaptiveAction.PRIORITY_SCREENING, AdaptiveAction.QA_REVIEW])
        self.assertIn("LOW_SAFETY_MARGIN", res["reason_codes"])

    def test_168h_final_checkpoint_concludes_screening(self):
        """Case 6: 168h terminal screening -> no next checkpoint, final disposition action."""
        # A: Normal pass at 168h
        res_pass = self.engine.evaluate(
            component_id="IC_NORM_FINAL",
            current_checkpoint="168h",
            module_a_res={"absolute_spec_failed": 0, "severity": "NORMAL"},
            module_b_res={"predicted_168h": 12.0},
            decision_res={"final_decision": "PASS"}
        )
        self.assertEqual(res_pass["action"], AdaptiveAction.CONTINUE_STANDARD_SCREENING)
        self.assertIsNone(res_pass["recommended_checkpoint"])
        self.assertIn("BURNIN_COMPLETE_PASS", res_pass["reason_codes"])

        # B: Reject at 168h
        res_reject = self.engine.evaluate(
            component_id="IC_FAIL_FINAL",
            current_checkpoint="168h",
            module_a_res={"absolute_spec_failed": 1},
            decision_res={"final_decision": "REJECT"}
        )
        self.assertEqual(res_reject["action"], AdaptiveAction.QA_REVIEW)
        self.assertIsNone(res_reject["recommended_checkpoint"])
        self.assertEqual(res_reject["priority"], "CRITICAL")

    def test_invalid_input_handling(self):
        """Case 7: Handled gracefully when inputs are empty dictionaries or non-standard types."""
        res = recommend_next_screening_action(
            component_id="IC_EMPTY",
            current_checkpoint="invalid_checkpoint",
            raw_measurements={}
        )
        self.assertIn("action", res)
        self.assertIn("priority", res)
        self.assertIn("confidence", res)
        self.assertIn("explanation", res)

    def test_checkpoint_transitions_sequence(self):
        """Case 8: Verify recommendations across progressive 0h -> 24h -> 96h -> 168h screening."""
        # 0h pre-stress baseline
        res_0 = self.engine.evaluate("IC_SEQ", current_checkpoint="0h")
        self.assertEqual(res_0["recommended_checkpoint"], "24h")

        # 24h normal
        res_24 = self.engine.evaluate("IC_SEQ", current_checkpoint="24h")
        self.assertEqual(res_24["recommended_checkpoint"], "96h")

        # 96h normal
        res_96 = self.engine.evaluate("IC_SEQ", current_checkpoint="96h")
        self.assertEqual(res_96["recommended_checkpoint"], "168h")

        # 168h final
        res_168 = self.engine.evaluate("IC_SEQ", current_checkpoint="168h")
        self.assertIsNone(res_168["recommended_checkpoint"])


if __name__ == "__main__":
    unittest.main()
