#!/usr/bin/env python3
"""
AI-ESS GUARDIAN: Master Automated Test Suite (Phase 13).
Validates:
1. Data loading and ingestion (CSV / Excel / DataFrame)
2. Preprocessing & feature extraction
3. Lot-aware normalization without data leakage
4. Anomaly scoring (Module A)
5. Time-series drift prediction (Module B)
6. Dynamic safety slope & envelope calculations (Phase 4)
7. Unified decision engine (Phase 5)
8. Missing data handling (missing 24h, 96h, 168h)
9. Malformed input & impossible physical values
10. Model artifact serialization & loading (Phase 12)

Mandatory Test Cases:
- CASE A: Healthy component -> PASS
- CASE B: Within datasheet limit but strongly abnormal relative to lot (Classic SIH case) -> REJECT or REVIEW
- CASE C: Increasing early drift predicts unsafe 168h behavior -> REJECT or REVIEW
- CASE D: Absolute specification violation -> REJECT
- CASE E: Missing 168h during inference -> early screening fully supported
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from backend.data_pipeline.sih_pipeline import RobustDataPipeline
from backend.screening.artifact_manager import ModelArtifactManager
from backend.screening.decision_engine import DecisionEngine
from backend.screening.explainer import ScreeningExplainer
from backend.screening.module_a import DynamicOutlierDetector
from backend.screening.module_b import TimeSeriesDriftPredictor
from backend.screening.safety_envelope import DynamicSafetyEnvelope


class TestAIESSGuardian(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_path = PROJECT_ROOT / "data" / "synthetic" / "burnin_semiconductor_screening.csv"
        if not data_path.exists():
            from data.synthetic.generate_synthetic_data import generate_burnin_dataset
            cls.df = generate_burnin_dataset(seed=42)
        else:
            cls.df = pd.read_csv(data_path)

        cls.pipeline = RobustDataPipeline()
        cls.mod_a = DynamicOutlierDetector(random_state=42)
        cls.feat_df = DynamicOutlierDetector.extract_features_from_panel(cls.df)
        cls.mod_a.fit(cls.feat_df)

        cls.mod_b = TimeSeriesDriftPredictor(random_state=42)
        cls.wide_df, cls.y_targets = TimeSeriesDriftPredictor.prepare_training_features(cls.df)
        cls.mod_b.fit(cls.wide_df, cls.y_targets)

        cls.safety_env = DynamicSafetyEnvelope(engineering_limit=50.0)
        cls.safety_env.fit(cls.wide_df, val_0h_col="val_0h", val_24h_col="val_24h", val_168h_col="val_168h_target")

        cls.engine = DecisionEngine()

    # --------------------------------------------------------------------------
    # 1. DATA PIPELINE TESTS
    # --------------------------------------------------------------------------
    def test_01_data_loading_and_standardization(self):
        """Pipeline must standardize column names and retain all components."""
        cleaned, summary = self.pipeline.validate_and_clean(self.df)
        self.assertTrue(summary["is_valid"])
        self.assertGreater(summary["components_count"], 0)
        self.assertIn("component_id", cleaned.columns)
        self.assertIn("leakage_current_ua", cleaned.columns)

    def test_02_impossible_value_detection(self):
        """Pipeline must detect and clip physically impossible values (e.g. negative leakage)."""
        bad_df = pd.DataFrame({
            "component_id": ["BAD_01", "BAD_02"],
            "lot_id": ["LOT_X", "LOT_X"],
            "leakage_current_ua": [-45.0, 99999.0],  # Negative and extreme values
            "burnin_hours": [0.0, 24.0]
        })
        cleaned, summary = self.pipeline.validate_and_clean(bad_df)
        self.assertGreaterEqual(cleaned["leakage_current_ua"].min(), 0.0)
        self.assertLessEqual(cleaned["leakage_current_ua"].max(), 1000.0)
        self.assertTrue(any("impossible" in iss.lower() for iss in summary["issues_resolved"]))

    def test_03_missing_data_imputation(self):
        """Pipeline must impute missing values using lot medians without crashing."""
        missing_df = pd.DataFrame({
            "component_id": ["C1", "C2", "C3"],
            "lot_id": ["LOT_A", "LOT_A", "LOT_A"],
            "leakage_current_ua": [10.0, np.nan, 12.0],
            "burnin_hours": [0.0, 24.0, 96.0]
        })
        cleaned, summary = self.pipeline.validate_and_clean(missing_df)
        self.assertEqual(cleaned["leakage_current_ua"].isnull().sum(), 0)
        self.assertTrue(any("missing" in iss.lower() for iss in summary["issues_resolved"]))

    # --------------------------------------------------------------------------
    # 2. MODULE A & LOT NORMALIZATION TESTS
    # --------------------------------------------------------------------------
    def test_04_lot_normalization_and_outlier_scoring(self):
        """Module A must produce bounded anomaly scores [0, 1] and robust Z-scores."""
        sample_comp = self.feat_df.iloc[0].to_dict()
        res = self.mod_a.evaluate_component(sample_comp)
        self.assertTrue(0.0 <= res["anomaly_score"] <= 1.0)
        self.assertIn(res["severity"], ["NORMAL", "SUSPICIOUS", "SEVERE"])
        self.assertIn("leakage_mult_of_lot_median", res)
        self.assertIn("explanation", res)

    # --------------------------------------------------------------------------
    # 3. MODULE B & DRIFT PREDICTION TESTS
    # --------------------------------------------------------------------------
    def test_05_drift_prediction_and_confidence_intervals(self):
        """Module B must predict 168h degradation with valid confidence intervals."""
        sample_comp = self.wide_df.iloc[0].to_dict()
        res = self.mod_b.predict_component(sample_comp)
        self.assertGreater(res["predicted_168h"], 0.0)
        self.assertLessEqual(res["ci_lower_80"], res["predicted_168h"])
        self.assertGreaterEqual(res["ci_upper_80"], res["predicted_168h"])
        self.assertTrue(0.0 <= res["drift_risk_score"] <= 1.0)

    # --------------------------------------------------------------------------
    # 4. SAFETY ENVELOPE TESTS
    # --------------------------------------------------------------------------
    def test_06_safety_envelope_evaluation(self):
        """Dynamic Safety Envelope must correctly calculate early slope and distinguish limits."""
        res = self.safety_env.evaluate(val_0h=10.0, val_24h=10.2, predicted_168h=11.5)
        self.assertAlmostEqual(res["early_slope"], (10.2 - 10.0) / 24.0, places=4)
        self.assertFalse(res["envelope_violated"])
        self.assertIn("DATA-DRIVEN PROTOTYPE THRESHOLD", res["threshold_classification"])
        self.assertIn("ENGINEERING/CERTIFICATION LIMIT", res["certification_classification"])

    # --------------------------------------------------------------------------
    # 5. MANDATORY TEST CASES (A through E)
    # --------------------------------------------------------------------------
    def test_case_a_healthy_component_passes(self):
        """CASE A: Healthy stable component must receive PASS disposition."""
        healthy_comp = {
            "component_id": "IC_HEALTHY_001",
            "lot_id": "LOT_A_2026",
            "leakage_0h": 10.0,
            "leakage_24h": 10.1,
            "leakage_current_ua": 10.1,
            "iddq_ma": 1.45,
            "propagation_delay_ns": 8.4,
            "early_slope_leakage": (10.1 - 10.0) / 24.0
        }
        res_a = self.mod_a.evaluate_component(healthy_comp)
        res_b = self.mod_b.predict_component(healthy_comp)
        res_env = self.safety_env.evaluate(healthy_comp["leakage_0h"], healthy_comp["leakage_24h"], res_b["predicted_168h"])
        dec = self.engine.evaluate("IC_HEALTHY_001", res_a, res_b, res_env)

        self.assertEqual(dec["final_decision"], "PASS", f"Healthy component should PASS, got {dec['final_decision']}: {dec['explanation']}")
        self.assertEqual(dec["status_code"], 0)

    def test_case_b_within_spec_but_lot_outlier_rejected_or_reviewed(self):
        """CASE B: Classic SIH Case (45 uA <= 50 uA limit, but 4.5x lot median) -> REJECT or REVIEW."""
        sih_latent_comp = {
            "component_id": "IC_SIH_LATENT_001",
            "lot_id": "LOT_A_2026",
            "leakage_0h": 40.0,
            "leakage_24h": 45.0,  # <= 50.0 uA datasheet limit! Passes static traditional limits!
            "leakage_current_ua": 45.0,
            "iddq_ma": 2.5,
            "propagation_delay_ns": 10.0,
            "early_slope_leakage": (45.0 - 40.0) / 24.0  # Steep slope
        }
        res_a = self.mod_a.evaluate_component(sih_latent_comp)
        res_b = self.mod_b.predict_component(sih_latent_comp)
        res_env = self.safety_env.evaluate(sih_latent_comp["leakage_0h"], sih_latent_comp["leakage_24h"], res_b["predicted_168h"])
        dec = self.engine.evaluate("IC_SIH_LATENT_001", res_a, res_b, res_env)

        # Traditional would PASS (45 <= 50)
        self.assertEqual(res_a["absolute_spec_failed"], 0)
        # AI-ESS GUARDIAN must intercept via REVIEW or REJECT
        self.assertIn(dec["final_decision"], ["REVIEW", "REJECT"])
        self.assertGreater(dec["lot_median_multiplier"], 3.0)
        self.assertTrue(any("median" in b.lower() for b in dec["qa_bullets"]))

    def test_case_c_accelerating_early_drift_predicts_unsafe_168h(self):
        """CASE C: Accelerating early drift predicting future breach must be flagged REVIEW or REJECT."""
        accelerating_comp = {
            "component_id": "IC_ACCEL_001",
            "lot_id": "LOT_A_2026",
            "leakage_0h": 10.0,
            "leakage_24h": 22.0,  # Rapid jump from 10 to 22 uA in 24h
            "leakage_current_ua": 22.0,
            "iddq_ma": 1.6,
            "propagation_delay_ns": 8.8,
            "early_slope_leakage": (22.0 - 10.0) / 24.0  # 0.50 uA/h (10x healthy slope)
        }
        res_a = self.mod_a.evaluate_component(accelerating_comp)
        res_b = self.mod_b.predict_component(accelerating_comp)
        res_env = self.safety_env.evaluate(accelerating_comp["leakage_0h"], accelerating_comp["leakage_24h"], res_b["predicted_168h"])
        dec = self.engine.evaluate("IC_ACCEL_001", res_a, res_b, res_env)

        self.assertIn(dec["final_decision"], ["REVIEW", "REJECT"])
        self.assertTrue(res_env["early_slope_breach"])

    def test_case_d_absolute_specification_violation_rejected(self):
        """CASE D: Gross breach of datasheet limit (> 50.0 uA) must be immediately REJECTED."""
        gross_defect = {
            "component_id": "IC_GROSS_001",
            "lot_id": "LOT_A_2026",
            "leakage_0h": 48.0,
            "leakage_24h": 58.0,  # Violates 50.0 uA limit
            "leakage_current_ua": 58.0,
            "iddq_ma": 5.5,
            "propagation_delay_ns": 16.0,
            "early_slope_leakage": (58.0 - 48.0) / 24.0
        }
        res_a = self.mod_a.evaluate_component(gross_defect)
        res_b = self.mod_b.predict_component(gross_defect)
        res_env = self.safety_env.evaluate(gross_defect["leakage_0h"], gross_defect["leakage_24h"], res_b["predicted_168h"])
        dec = self.engine.evaluate("IC_GROSS_001", res_a, res_b, res_env)

        self.assertEqual(dec["final_decision"], "REJECT")
        self.assertEqual(res_a["absolute_spec_failed"], 1)

    def test_case_e_missing_168h_early_screening_supported(self):
        """CASE E: System must execute complete screening even when 168h is unmeasured (inference mode)."""
        early_data = {
            "component_id": "IC_INFERENCE_001",
            "lot_id": "LOT_B_2026",
            "leakage_0h": 12.0,
            "leakage_24h": 12.2,
            "leakage_current_ua": 12.2,
            "iddq_ma": 1.6,
            "propagation_delay_ns": 8.7,
            "early_slope_leakage": (12.2 - 12.0) / 24.0
            # Note: No 96h or 168h readings present!
        }
        res_a = self.mod_a.evaluate_component(early_data)
        res_b = self.mod_b.predict_component(early_data)
        res_env = self.safety_env.evaluate(early_data["leakage_0h"], early_data["leakage_24h"], res_b["predicted_168h"])
        dec = self.engine.evaluate("IC_INFERENCE_001", res_a, res_b, res_env)

        # Must generate valid forecast and screening disposition
        self.assertIsNotNone(res_b["predicted_168h"])
        self.assertIn(dec["final_decision"], ["PASS", "REVIEW", "REJECT"])

    # --------------------------------------------------------------------------
    # 6. MODEL ARTIFACT PERSISTENCE
    # --------------------------------------------------------------------------
    def test_07_artifact_serialization_and_loading(self):
        """Artifact manager must serialize and load working models without state loss."""
        mgr = ModelArtifactManager(PROJECT_ROOT / "models")
        self.assertTrue(mgr.artifacts_exist())
        loaded_a, loaded_b, loaded_env = mgr.load_artifacts()

        self.assertTrue(loaded_a.is_fitted_)
        self.assertTrue(loaded_b.is_fitted_)
        self.assertTrue(loaded_env.is_fitted_)

        # Verify predictions match
        sample = self.feat_df.iloc[0].to_dict()
        res_orig = self.mod_a.evaluate_component(sample)
        res_load = loaded_a.evaluate_component(sample)
        self.assertEqual(res_orig["final_decision"] if "final_decision" in res_orig else res_orig["severity"],
                         res_load["final_decision"] if "final_decision" in res_load else res_load["severity"])


if __name__ == "__main__":
    unittest.main()
