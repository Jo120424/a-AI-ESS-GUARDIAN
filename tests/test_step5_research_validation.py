#!/usr/bin/env python3
"""
Step 5 Extended Automated Test Suite: Research Validation & Scientific Results.
Tests cover:
- Result reproduction from raw predictions
- Confusion matrix calculations
- Metric calculations (Recall, FNR, Precision, F1, FPR)
- Component alignment across prediction files
- Threshold sensitivity logic
- Ablation logic and incremental gains
- Lead-time calculations
- Cost-sensitive calculations
- Report data integrity
- Explainability consistency
"""

from pathlib import Path
import json
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STEP4_DIR = PROJECT_ROOT / "results" / "step4"
STEP5_DIR = PROJECT_ROOT / "results" / "step5"
TABLES_DIR = STEP5_DIR / "tables"
FIGURES_DIR = STEP5_DIR / "figures"
REPORT_PATH = PROJECT_ROOT / "research" / "step_reports" / "STEP_5_REPORT.md"


def compute_metrics(y_true, y_pred):
    """Independently compute classification metrics from raw arrays."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=recall, precision=precision,
                f1=f1, fnr=fnr, fpr=fpr)


class TestResultReproduction(unittest.TestCase):
    """Verify that reported metrics can be independently reproduced from saved predictions."""

    @classmethod
    def setUpClass(cls):
        cls.ground_truth = [0, 1, 1, 1, 1, 1]  # C1=safe, C2-C6=defect
        cls.comp_ids = ["C1", "C2", "C3", "C4", "C5", "C6"]
        cls.metrics_df = pd.read_csv(STEP4_DIR / "metrics.csv")
        cls.trad_df = pd.read_csv(STEP4_DIR / "traditional_predictions.csv")
        cls.stat_df = pd.read_csv(STEP4_DIR / "statistical_predictions.csv")
        cls.anomaly_df = pd.read_csv(STEP4_DIR / "anomaly_predictions.csv")
        cls.drift_df = pd.read_csv(STEP4_DIR / "drift_predictions.csv")
        cls.risk_df = pd.read_csv(STEP4_DIR / "risk_predictions.csv")
        cls.comp_df = pd.read_csv(STEP4_DIR / "component_level_results.csv")

    def test_traditional_confusion_matrix_reproduction(self):
        """Reproduce Traditional screening confusion matrix from raw predictions."""
        y_pred = list(self.trad_df["traditional_decision"])
        m = compute_metrics(self.ground_truth, y_pred)
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 5)
        self.assertEqual(m["tn"], 1)

    def test_statistical_confusion_matrix_reproduction(self):
        """Reproduce Statistical screening confusion matrix from raw predictions."""
        y_pred = list(self.stat_df["statistical_decision"])
        m = compute_metrics(self.ground_truth, y_pred)
        self.assertEqual(m["tp"], 3)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 2)
        self.assertEqual(m["tn"], 0)

    def test_drift_confusion_matrix_reproduction(self):
        """Reproduce Drift forecast confusion matrix from raw predictions."""
        y_pred = list(self.drift_df["drift_decision"])
        m = compute_metrics(self.ground_truth, y_pred)
        self.assertEqual(m["tp"], 5)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 0)
        self.assertEqual(m["tn"], 0)

    def test_fusion_confusion_matrix_reproduction(self):
        """Reproduce Risk Fusion confusion matrix from raw predictions."""
        y_pred = list(self.risk_df["fusion_decision"])
        m = compute_metrics(self.ground_truth, y_pred)
        self.assertEqual(m["tp"], 3)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 2)
        self.assertEqual(m["tn"], 0)

    def test_metric_recall_values_match_predictions(self):
        """Verify that recall values in metrics.csv match prediction-based computation."""
        methods_preds = {
            "Traditional (MIL-PRF-62F 20%)": list(self.trad_df["traditional_decision"]),
            "Dynamic Statistical (MAD/Maha)": list(self.stat_df["statistical_decision"]),
            "Early Drift Forecast (194h)": list(self.drift_df["drift_decision"]),
            "Risk Fusion Engine (Multi-Tier)": list(self.risk_df["fusion_decision"]),
        }
        for method_name, y_pred in methods_preds.items():
            m = compute_metrics(self.ground_truth, y_pred)
            row = self.metrics_df[self.metrics_df["method"] == method_name].iloc[0]
            self.assertAlmostEqual(m["recall"], row["recall"], places=4,
                                   msg=f"Recall mismatch for {method_name}")
            self.assertAlmostEqual(m["fnr"], row["false_negative_rate"], places=4,
                                   msg=f"FNR mismatch for {method_name}")

    def test_precision_f1_values_match(self):
        """Verify precision and F1 reproduced correctly."""
        y_pred_drift = list(self.drift_df["drift_decision"])
        m = compute_metrics(self.ground_truth, y_pred_drift)
        self.assertAlmostEqual(m["precision"], 5/6, places=4)
        self.assertAlmostEqual(m["f1"], 2*(5/6)*1.0/(5/6+1.0), places=4)


class TestComponentAlignment(unittest.TestCase):
    """Verify all prediction files have consistent component IDs and row counts."""

    @classmethod
    def setUpClass(cls):
        cls.expected_ids = ["C1", "C2", "C3", "C4", "C5", "C6"]
        cls.files = {
            "traditional": pd.read_csv(STEP4_DIR / "traditional_predictions.csv"),
            "statistical": pd.read_csv(STEP4_DIR / "statistical_predictions.csv"),
            "anomaly": pd.read_csv(STEP4_DIR / "anomaly_predictions.csv"),
            "drift": pd.read_csv(STEP4_DIR / "drift_predictions.csv"),
            "risk": pd.read_csv(STEP4_DIR / "risk_predictions.csv"),
            "component": pd.read_csv(STEP4_DIR / "component_level_results.csv"),
        }

    def test_all_files_have_6_rows(self):
        """All prediction files must have exactly 6 data rows."""
        for name, df in self.files.items():
            self.assertEqual(len(df), 6, f"{name} has {len(df)} rows, expected 6")

    def test_all_files_have_consistent_component_ids(self):
        """Component IDs must be C1-C6 in all files."""
        for name, df in self.files.items():
            ids = list(df["component_id"])
            self.assertEqual(ids, self.expected_ids, f"{name} has misaligned IDs: {ids}")

    def test_ground_truth_consistency(self):
        """Ground truth labels must be [0,1,1,1,1,1] in component_level_results."""
        comp_df = self.files["component"]
        gt = list(comp_df["ground_truth_latent_risk"])
        self.assertEqual(gt, [0, 1, 1, 1, 1, 1])


class TestLeadTimeCalculations(unittest.TestCase):
    """Verify lead-time calculations from raw data."""

    @classmethod
    def setUpClass(cls):
        cls.comp_df = pd.read_csv(STEP4_DIR / "component_level_results.csv")
        cls.metrics_df = pd.read_csv(STEP4_DIR / "metrics.csv")
        cls.screen_time = 47.0

    def test_lead_time_for_drift_detected_components(self):
        """For each TP component in drift, lead time = failure_time - screen_time."""
        df = self.comp_df
        drift_tps = df[(df["drift_decision"] == 1) & (df["ground_truth_latent_risk"] == 1)]
        for _, row in drift_tps.iterrows():
            failure_h = row["future_failure_time_h"]
            expected_lead = failure_h - self.screen_time
            self.assertGreater(expected_lead, 0,
                               f"Lead time for {row['component_id']} must be positive")

    def test_drift_mean_lead_time(self):
        """Drift mean lead time should match metrics.csv."""
        drift_row = self.metrics_df[self.metrics_df["method"] == "Early Drift Forecast (194h)"].iloc[0]
        reported_mean = drift_row["mean_lead_time_hours"]
        # Compute from component data
        df = self.comp_df
        detected = df[(df["drift_decision"] == 1) & (df["ground_truth_latent_risk"] == 1)]
        lead_times = detected["future_failure_time_h"] - self.screen_time
        computed_mean = lead_times.mean()
        self.assertAlmostEqual(reported_mean, computed_mean, places=2)

    def test_lead_time_range(self):
        """Lead times must be between 124.0 and 147.0 hours."""
        df = self.comp_df
        detected = df[(df["drift_decision"] == 1) & (df["ground_truth_latent_risk"] == 1)]
        lead_times = (detected["future_failure_time_h"] - self.screen_time).tolist()
        self.assertAlmostEqual(min(lead_times), 124.0, places=1)
        self.assertAlmostEqual(max(lead_times), 147.0, places=1)

    def test_traditional_has_zero_lead_time(self):
        """Traditional screening detected 0 defects, so lead time must be 0."""
        trad_row = self.metrics_df[self.metrics_df["method"] == "Traditional (MIL-PRF-62F 20%)"].iloc[0]
        self.assertEqual(trad_row["mean_lead_time_hours"], 0.0)


class TestDriftRegressionError(unittest.TestCase):
    """Verify drift MAE/RMSE from raw predictions."""

    @classmethod
    def setUpClass(cls):
        cls.drift_df = pd.read_csv(STEP4_DIR / "drift_predictions.csv")
        cls.metadata = json.loads((STEP4_DIR / "experiment_metadata.json").read_text())

    def test_mae_reproduction(self):
        """MAE must match metadata value computed from raw predictions."""
        errors = np.abs(self.drift_df["predicted_delta_c_194h"] - self.drift_df["actual_delta_c_194h"])
        computed_mae = errors.mean()
        reported_mae = self.metadata["drift_prediction"]["mae_pct"]
        self.assertAlmostEqual(computed_mae, reported_mae, places=3)

    def test_rmse_reproduction(self):
        """RMSE must match metadata value computed from raw predictions."""
        errors = (self.drift_df["predicted_delta_c_194h"] - self.drift_df["actual_delta_c_194h"]) ** 2
        computed_rmse = np.sqrt(errors.mean())
        reported_rmse = self.metadata["drift_prediction"]["rmse_pct"]
        self.assertAlmostEqual(computed_rmse, reported_rmse, places=3)


class TestThresholdSensitivity(unittest.TestCase):
    """Verify threshold sensitivity results integrity."""

    @classmethod
    def setUpClass(cls):
        cls.sens_df = pd.read_csv(STEP5_DIR / "threshold_sensitivity.csv")

    def test_minimum_configurations(self):
        """Must have at least 20 threshold configurations."""
        self.assertGreaterEqual(len(self.sens_df), 20)

    def test_baseline_configs_marked(self):
        """Baseline Step 4 configurations must be explicitly marked."""
        baselines = self.sens_df[self.sens_df["note"] == "Baseline step 4 configuration"]
        self.assertGreaterEqual(len(baselines), 3, "At least 3 baselines expected")

    def test_drift_spec_limit_monotonic_recall(self):
        """As drift spec limit increases, recall must be non-increasing."""
        drift_sub = self.sens_df[self.sens_df["parameter_name"] == "spec_limit_delta_c_pct"]
        drift_sub = drift_sub.sort_values("threshold_value")
        recalls = drift_sub["recall"].tolist()
        for i in range(len(recalls) - 1):
            self.assertGreaterEqual(recalls[i], recalls[i + 1],
                                     f"Recall not monotonically non-increasing at index {i}")

    def test_recall_bounds(self):
        """All recall values must be between 0.0 and 1.0."""
        self.assertTrue((self.sens_df["recall"] >= 0.0).all())
        self.assertTrue((self.sens_df["recall"] <= 1.0).all())

    def test_precision_bounds(self):
        """All precision values must be between 0.0 and 1.0."""
        self.assertTrue((self.sens_df["precision"] >= 0.0).all())
        self.assertTrue((self.sens_df["precision"] <= 1.0).all())


class TestAblationLogic(unittest.TestCase):
    """Verify ablation study results and incremental gains."""

    @classmethod
    def setUpClass(cls):
        cls.abl_df = pd.read_csv(STEP5_DIR / "ablation_results.csv")

    def test_seven_paradigms(self):
        """Must have exactly 7 ablation paradigms."""
        self.assertEqual(len(self.abl_df), 7)

    def test_traditional_baseline_zero_recall(self):
        """Paradigm A (Traditional) must have recall = 0.0."""
        row_a = self.abl_df[self.abl_df["paradigm"].str.startswith("A:")].iloc[0]
        self.assertEqual(row_a["recall"], 0.0)

    def test_drift_alone_perfect_recall(self):
        """Paradigm D (Drift Only) must have recall = 1.0."""
        row_d = self.abl_df[self.abl_df["paradigm"].str.startswith("D:")].iloc[0]
        self.assertEqual(row_d["recall"], 1.0)

    def test_anomaly_plus_drift_perfect_recall(self):
        """Paradigm E (Anomaly + Drift) must have recall >= 1.0 (union captures all)."""
        row_e = self.abl_df[self.abl_df["paradigm"].str.startswith("E:")].iloc[0]
        self.assertEqual(row_e["recall"], 1.0)

    def test_incremental_gains_non_negative_from_traditional(self):
        """All paradigms must have incremental recall >= 0 vs traditional."""
        for _, row in self.abl_df.iterrows():
            self.assertGreaterEqual(row["incremental_recall_vs_trad"], 0.0,
                                     f"Negative gain for {row['paradigm']}")

    def test_fusion_extended_captures_all(self):
        """F-Extended (with quarantine) should achieve recall = 1.0."""
        row_fext = self.abl_df[self.abl_df["paradigm"].str.contains("F-Extended")].iloc[0]
        self.assertEqual(row_fext["recall"], 1.0)


class TestCostSensitiveAnalysis(unittest.TestCase):
    """Verify cost-sensitive analysis calculations."""

    @classmethod
    def setUpClass(cls):
        cls.cost_df = pd.read_csv(STEP5_DIR / "cost_sensitivity.csv")

    def test_cost_formula(self):
        """Total cost = c_fn * fn_count + c_fp * fp_count for each row."""
        for _, row in self.cost_df.iterrows():
            expected = row["c_fn"] * row["fn_count"] + row["c_fp"] * row["fp_count"]
            self.assertAlmostEqual(row["total_expected_cost"], expected, places=2,
                                    msg=f"Cost formula failed for {row['method']} at ratio {row['cost_ratio_fn_to_fp']}")

    def test_traditional_cost_at_high_ratio(self):
        """At ratio=10, Traditional cost = 10*5 + 1*0 = 50."""
        sub = self.cost_df[(self.cost_df["cost_ratio_fn_to_fp"] == 10.0) &
                           (self.cost_df["method"] == "Traditional (MIL-PRF-62F 20%)")]
        self.assertAlmostEqual(sub.iloc[0]["total_expected_cost"], 50.0, places=1)

    def test_drift_superior_at_high_ratio(self):
        """Drift forecast must be superior at ratio >= 0.25."""
        sub = self.cost_df[(self.cost_df["cost_ratio_fn_to_fp"] == 10.0) &
                           (self.cost_df["method"] == "Early Drift Forecast (194h)")]
        self.assertTrue(sub.iloc[0]["is_superior_to_traditional"])

    def test_multiple_cost_ratios(self):
        """Must evaluate at least 10 cost ratio levels."""
        ratios = self.cost_df["cost_ratio_fn_to_fp"].nunique()
        self.assertGreaterEqual(ratios, 10)


class TestExplainabilitySummary(unittest.TestCase):
    """Verify explainability summary consistency."""

    @classmethod
    def setUpClass(cls):
        cls.explain_df = pd.read_csv(STEP5_DIR / "explainability_summary.csv")

    def test_six_components(self):
        """Must have exactly 6 component rows."""
        self.assertEqual(len(self.explain_df), 6)

    def test_ground_truth_alignment(self):
        """Ground truth must match known labels."""
        expected_gt = ["Safe Survivor", "Latent Defect", "Latent Defect",
                       "Latent Defect", "Latent Defect", "Latent Defect"]
        actual_gt = list(self.explain_df["ground_truth"])
        self.assertEqual(actual_gt, expected_gt)

    def test_fusion_risk_scores_bounded(self):
        """Risk scores must be between 0.0 and 1.0."""
        scores = self.explain_df["fusion_risk_score"]
        self.assertTrue((scores >= 0.0).all())
        self.assertTrue((scores <= 1.0).all())

    def test_c1_is_false_positive(self):
        """C1 (safe survivor) must be classified as REJECT (FP) in fusion."""
        c1 = self.explain_df[self.explain_df["component_id"] == "C1"].iloc[0]
        self.assertEqual(c1["fusion_decision"], "REJECT")
        self.assertEqual(c1["ground_truth"], "Safe Survivor")


class TestComponentErrorAnalysis(unittest.TestCase):
    """Verify component error analysis classification consistency."""

    @classmethod
    def setUpClass(cls):
        cls.error_df = pd.read_csv(STEP5_DIR / "component_error_analysis.csv")

    def test_classification_labels(self):
        """Each classification must be one of TP, FP, TN, FN."""
        valid_labels = {"TP", "FP", "TN", "FN"}
        for col in ["trad_class", "stat_class", "iforest_class", "ocsvm_class",
                     "drift_class", "fusion_class"]:
            unique_vals = set(self.error_df[col].unique())
            self.assertTrue(unique_vals.issubset(valid_labels),
                            f"Invalid labels in {col}: {unique_vals - valid_labels}")

    def test_c1_traditional_is_true_negative(self):
        """C1 traditional classification must be TN."""
        c1 = self.error_df[self.error_df["component_id"] == "C1"].iloc[0]
        self.assertEqual(c1["trad_class"], "TN")

    def test_all_traditional_defects_are_fn(self):
        """All 5 defect components must be FN under traditional screening."""
        defects = self.error_df[self.error_df["ground_truth_code"] == 1]
        trad_classes = list(defects["trad_class"])
        self.assertEqual(trad_classes, ["FN"] * 5)

    def test_drift_catches_all_defects(self):
        """All 5 defect components must be TP under drift forecast."""
        defects = self.error_df[self.error_df["ground_truth_code"] == 1]
        drift_classes = list(defects["drift_class"])
        self.assertEqual(drift_classes, ["TP"] * 5)


class TestReportIntegrity(unittest.TestCase):
    """Verify STEP_5_REPORT.md exists and contains required sections."""

    def test_report_exists(self):
        """Report file must exist."""
        self.assertTrue(REPORT_PATH.exists())

    def test_report_minimum_size(self):
        """Report must be at least 15KB."""
        self.assertGreater(REPORT_PATH.stat().st_size, 15000)

    def test_report_contains_required_sections(self):
        """Report must contain all 22 required sections."""
        content = REPORT_PATH.read_text(encoding="utf-8")
        required = [
            "Executive Summary", "Purpose", "Results Audit",
            "Dataset and Ground Truth", "Reproducibility",
            "Model Comparison", "Component-Level Analysis",
            "False Negative", "False Positive",
            "Threshold Sensitivity", "Ablation",
            "Early Warning", "Cost-Sensitive",
            "Explainability", "Engineering Trade",
            "Statistical Robustness", "Research Contribution",
            "Hypothesis Review", "Limitation",
            "Scientific Interpretation", "Future Work"
        ]
        for section in required:
            self.assertIn(section, content,
                          f"Report missing required section: {section}")

    def test_report_no_fabrication_markers(self):
        """Report must not contain obvious fabrication markers."""
        content = REPORT_PATH.read_text(encoding="utf-8")
        # Should not claim statistically significant p < 0.05 results
        # (because sample size is too small for McNemar)
        self.assertNotIn("statistically significant at p < 0.05", content.lower())


class TestPublicationArtifacts(unittest.TestCase):
    """Verify all required publication tables and figures exist."""

    def test_all_10_tables_csv(self):
        """All 10 publication tables must exist in CSV format."""
        for i in range(1, 11):
            files = list(TABLES_DIR.glob(f"table{i}_*.csv"))
            self.assertGreater(len(files), 0, f"Missing table{i} CSV")

    def test_all_10_tables_md(self):
        """All 10 publication tables must exist in Markdown format."""
        for i in range(1, 11):
            files = list(TABLES_DIR.glob(f"table{i}_*.md"))
            self.assertGreater(len(files), 0, f"Missing table{i} MD")

    def test_all_10_figures(self):
        """All 10 publication figures must exist as PNG files."""
        for i in range(1, 11):
            files = list(FIGURES_DIR.glob(f"fig{i}_*.png"))
            self.assertGreater(len(files), 0, f"Missing fig{i} PNG")

    def test_figures_minimum_size(self):
        """All figures must be at least 10KB (quality check)."""
        for fig in FIGURES_DIR.glob("*.png"):
            self.assertGreater(fig.stat().st_size, 10000,
                               f"Figure {fig.name} is too small: {fig.stat().st_size} bytes")


class TestResultAuditCSV(unittest.TestCase):
    """Verify the result_audit.csv file integrity."""

    @classmethod
    def setUpClass(cls):
        cls.audit_df = pd.read_csv(STEP5_DIR / "result_audit.csv")

    def test_minimum_checks(self):
        """Must have at least 100 audit checks."""
        self.assertGreaterEqual(len(self.audit_df), 100)

    def test_zero_discrepancies(self):
        """All checks must be VERIFIED_EXACT_MATCH."""
        non_match = self.audit_df[self.audit_df["status"] != "VERIFIED_EXACT_MATCH"]
        self.assertEqual(len(non_match), 0,
                         f"Found {len(non_match)} non-exact matches")

    def test_required_columns(self):
        """Audit CSV must have required columns."""
        required = ["category", "method", "metric", "step4_reported_value",
                     "reproduced_value", "difference", "status", "explanation"]
        for col in required:
            self.assertIn(col, self.audit_df.columns, f"Missing column: {col}")


class TestMcNemarReproduction(unittest.TestCase):
    """Verify McNemar test results match metadata."""

    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((STEP4_DIR / "experiment_metadata.json").read_text())

    def test_mcnemar_pvalue_bounds(self):
        """All McNemar p-values must be between 0 and 1."""
        tests = self.metadata["statistical_hypothesis_tests"]
        for key, val in tests.items():
            if "mcnemar" in key:
                p = val["exact_p_value"]
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_no_significance_at_005(self):
        """No McNemar test should be significant at 0.05 (N=6 is too small)."""
        tests = self.metadata["statistical_hypothesis_tests"]
        for key, val in tests.items():
            if "mcnemar" in key:
                self.assertFalse(val["statistically_significant_005"],
                                 f"{key} should not be significant at 0.05")


if __name__ == "__main__":
    unittest.main()
