#!/usr/bin/env python3
"""
Step 4 Automated Unit Test Suite: Model Screening, Predictions & Evaluation.
Validates:
- Traditional screening decision logic (MIL-PRF-62F).
- Dynamic statistical screening (MAD & Mahalanobis) out-of-fold calibration.
- AI/ML anomaly detection (Isolation Forest & One-Class SVM).
- Early drift trajectory prediction and regression errors.
- Risk fusion multi-tier classification.
- Evaluation metrics and exact McNemar hypothesis testing.
- Reproducibility and persistence of all Step 4 artifacts.
"""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from backend.experiments.anomaly import MLAnomalyScreening
from backend.experiments.drift import DriftPredictor
from backend.experiments.evaluation import (
    calculate_screening_metrics,
    mcnemar_exact_test,
    paired_bootstrap_ci
)
from backend.experiments.ground_truth import (
    extract_screening_features,
    generate_ground_truth_labels,
    load_processed_data,
    verify_anti_leakage
)
from backend.experiments.risk_fusion import RiskFusionEngine
from backend.experiments.statistical import StatisticalScreening
from backend.experiments.traditional import TraditionalScreening


class TestStep4ModelScreening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_processed = load_processed_data()
        cls.t_screen = 47.0
        cls.gt_df = generate_ground_truth_labels(cls.df_processed, t_screen=cls.t_screen)
        cls.feats_df = extract_screening_features(cls.df_processed, t_screen=cls.t_screen)
        cls.drift_df = DriftPredictor.extract_drift_features(cls.df_processed, t_screen=cls.t_screen)

    def test_traditional_screening_decisions(self):
        """Traditional screening must produce PASS (0) at 47h because all Delta C < 20%."""
        screener = TraditionalScreening(spec_limit_delta_c=20.0, tightened_limit_delta_c=5.0)
        batch_res = screener.evaluate_batch(self.feats_df)
        self.assertEqual(len(batch_res), 6)
        # All 6 components must pass traditional static limits at 47h
        self.assertTrue((batch_res["traditional_decision"] == 0).all())
        self.assertTrue((batch_res["traditional_status"] == "PASS").all())
        # All 6 components also pass 5% tightened limit
        self.assertTrue((batch_res["traditional_decision_tightened"] == 0).all())

    def test_statistical_screening_fit_and_evaluate(self):
        """Dynamic statistical screening must fit out-of-fold and generate finite scores."""
        train_c = ["C2", "C3", "C4", "C5", "C6"]
        test_c = "C1"
        train_feats = self.feats_df[self.feats_df["component_id"].isin(train_c)]
        test_row = self.feats_df[self.feats_df["component_id"] == test_c].iloc[0].to_dict()

        screener = StatisticalScreening(mad_threshold=2.5, alpha_significance=0.05)
        screener.fit(train_feats)
        res = screener.evaluate_component(test_row)

        self.assertIn("statistical_decision", res)
        self.assertIn(res["statistical_decision"], [0, 1])
        self.assertTrue(np.isfinite(res["mahalanobis_dist_sq"]))
        self.assertGreaterEqual(res["mahalanobis_dist_sq"], 0.0)
        self.assertIn("explanation", res)

    def test_aiml_anomaly_models(self):
        """AI/ML anomaly models must produce bounded anomaly scores and binary flags."""
        train_c = ["C1", "C2", "C3", "C5", "C6"]
        test_c = "C4"
        train_feats = self.feats_df[self.feats_df["component_id"].isin(train_c)]
        test_row = self.feats_df[self.feats_df["component_id"] == test_c].iloc[0].to_dict()

        ml_screener = MLAnomalyScreening(contamination=0.20, random_state=42)
        ml_screener.fit(train_feats)
        res = ml_screener.evaluate_component(test_row)

        self.assertIn("aiml_decision", res)
        self.assertIn(res["aiml_decision"], [0, 1])
        self.assertIn("iforest_anomaly_score", res)
        self.assertTrue(0.0 <= res["iforest_score_normalized"] <= 1.0)
        self.assertIn("primary_contributing_feature", res)

    def test_drift_prediction_accuracy(self):
        """Early drift prediction must predict 194h degradation with plausible error."""
        train_c = ["C1", "C2", "C3", "C5", "C6"]
        test_c = "C4"
        train_drift = self.drift_df[self.drift_df["component_id"].isin(train_c)]
        test_row = self.drift_df[self.drift_df["component_id"] == test_c].iloc[0].to_dict()

        predictor = DriftPredictor(target_horizon_hours=194.0, spec_limit=20.0, random_state=42)
        predictor.fit(train_drift, train_drift["target_delta_c_194h"].values)
        res = predictor.predict_component(test_row)

        self.assertIn("predicted_delta_c_194h", res)
        # Predicted value must be in reasonable range (e.g. 15% to 30%)
        self.assertTrue(15.0 <= res["predicted_delta_c_194h"] <= 30.0)
        self.assertIn(res["drift_decision"], [0, 1])
        self.assertIsNotNone(res["prediction_abs_error"])
        self.assertLess(res["prediction_abs_error"], 5.0)  # MAE reasonable

    def test_risk_fusion_engine(self):
        """Risk fusion engine must categorize into LOW, MEDIUM, or HIGH risk with valid weights."""
        fusion = RiskFusionEngine()
        cid = "C4"
        trad_res = {"traditional_decision": 0}
        stat_res = {"statistical_decision": 1, "max_abs_zscore": 3.5}
        ml_res = {"aiml_decision": 1, "iforest_anomaly_score": 0.52}
        drift_res = {"drift_decision": 1, "predicted_delta_c_194h": 21.5}

        res = fusion.fuse_component(cid, trad_res, stat_res, ml_res, drift_res)
        self.assertIn(res["risk_tier"], ["LOW RISK", "MEDIUM RISK", "HIGH RISK"])
        self.assertEqual(res["risk_tier"], "HIGH RISK")
        self.assertEqual(res["fusion_decision"], 1)
        self.assertGreaterEqual(res["fusion_risk_score"], 0.50)
        self.assertIn("explanation", res)

    def test_evaluation_metrics_calculation(self):
        """Metrics calculation must correctly compute Recall, FNR, Precision, F1, FPR."""
        y_true = np.array([0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 1])  # 4 TP, 0 FP, 1 FN, 1 TN
        failure_times = [None, 194.0, 194.0, 171.0, 194.0, 194.0]

        metrics = calculate_screening_metrics(y_true, y_pred, failure_times=failure_times, t_screen=47.0)
        self.assertEqual(metrics["true_positives"], 4)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["true_negatives"], 1)
        self.assertAlmostEqual(metrics["recall"], 4.0 / 5.0, places=3)
        self.assertAlmostEqual(metrics["false_negative_rate"], 1.0 / 5.0, places=3)
        self.assertAlmostEqual(metrics["precision"], 1.0, places=3)
        self.assertEqual(metrics["false_positive_rate"], 0.0)
        self.assertGreater(metrics["mean_lead_time_hours"], 120.0)

    def test_mcnemar_and_bootstrap_tests(self):
        """Exact McNemar and bootstrap tests must return valid probability values."""
        y_true = np.array([0, 1, 1, 1, 1, 1])
        y_trad = np.array([0, 0, 0, 0, 0, 0])
        y_cand = np.array([0, 1, 1, 1, 0, 1])

        mcnemar_res = mcnemar_exact_test(y_true, y_trad, y_cand)
        self.assertIn("exact_p_value", mcnemar_res)
        self.assertTrue(0.0 <= mcnemar_res["exact_p_value"] <= 1.0)

        boot_res = paired_bootstrap_ci(y_true, y_trad, y_cand, n_iterations=200, seed=42)
        self.assertIn("delta_recall_mean", boot_res)
        self.assertIn("delta_recall_ci_95", boot_res)
        self.assertEqual(len(boot_res["delta_recall_ci_95"]), 2)
        self.assertLessEqual(boot_res["delta_recall_ci_95"][0], boot_res["delta_recall_ci_95"][1])

    def test_step4_persisted_artifacts(self):
        """All required Step 4 results and figures must exist and be non-empty."""
        results_dir = PROJECT_ROOT / "results" / "step4"
        required_csvs = [
            "traditional_predictions.csv",
            "statistical_predictions.csv",
            "anomaly_predictions.csv",
            "drift_predictions.csv",
            "risk_predictions.csv",
            "component_level_results.csv",
            "comparison.csv",
            "metrics.csv",
            "feature_importance.csv"
        ]
        for fname in required_csvs:
            p = results_dir / fname
            self.assertTrue(p.exists(), f"Missing required result file: {fname}")
            self.assertGreater(p.stat().st_size, 0, f"File is empty: {fname}")

        meta_p = results_dir / "experiment_metadata.json"
        self.assertTrue(meta_p.exists())
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertIn("dataset_sha256", meta)
        self.assertIn("statistical_hypothesis_tests", meta)

        figures_dir = results_dir / "figures"
        figs = list(figures_dir.glob("*.png"))
        self.assertGreaterEqual(len(figs), 12, "Must generate at least 12 publication figures")


if __name__ == "__main__":
    unittest.main()
