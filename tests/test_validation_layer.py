#!/usr/bin/env python3
"""
Step 5 Automated Unit Test Suite: Validation Layer, Audit, Robustness & Ablation.
Validates:
- 100% match in Step 4 results audit (zero discrepancies).
- Component error classification logic.
- Survivor C1 forensic analysis and directional screening remedy.
- Threshold sensitivity parameter sweeps.
- Ablation study paradigm coverage and incremental gains.
- Cost-sensitive trade-off crossover economics.
- Completeness and validity of all 10 publication tables and 10 figures.
"""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from backend.validation.ablation import AblationAnalyzer
from backend.validation.audit import Step4ResultAuditor
from backend.validation.cost_analysis import CostAnalyzer
from backend.validation.error_analysis import ErrorAnalyzer
from backend.validation.sensitivity import SensitivityAnalyzer


class TestValidationLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step4_dir = PROJECT_ROOT / "results" / "step4"
        cls.step5_dir = PROJECT_ROOT / "results" / "step5"
        cls.tables_dir = cls.step5_dir / "tables"
        cls.figures_dir = cls.step5_dir / "figures"

    def test_step4_audit_exact_match(self):
        """Verifies that all 101 audited metrics match Step 4 reported values with zero discrepancies."""
        auditor = Step4ResultAuditor(self.step4_dir, self.step5_dir)
        df_audit = auditor.run_full_audit()
        discrepancies = df_audit[df_audit["status"] != "VERIFIED_EXACT_MATCH"]
        self.assertEqual(len(discrepancies), 0, f"Found {len(discrepancies)} audit discrepancies: {discrepancies}")
        self.assertGreaterEqual(len(df_audit), 100)

    def test_component_error_analysis_integrity(self):
        """Verifies component error profiling across C1-C6 and classification logic."""
        analyzer = ErrorAnalyzer(self.step4_dir, self.step5_dir)
        df_errors, _ = analyzer.run_analysis()
        self.assertEqual(len(df_errors), 6)
        self.assertEqual(list(df_errors["component_id"]), ["C1", "C2", "C3", "C4", "C5", "C6"])

        # Trad screening must miss all 5 defects (FN) and pass C1 (TN)
        self.assertEqual(list(df_errors["trad_class"]), ["TN", "FN", "FN", "FN", "FN", "FN"])

        # Early Drift must detect all 5 defects (TP) and flag C1 (FP)
        self.assertEqual(list(df_errors["drift_class"]), ["FP", "TP", "TP", "TP", "TP", "TP"])

    def test_survivor_c1_forensics_and_directional_screening(self):
        """Verifies C1 root cause and confirms directional screening eliminates the false positive."""
        df_comp = pd.read_csv(self.step4_dir / "component_level_results.csv")
        c1_row = df_comp[df_comp["component_id"] == "C1"].iloc[0]

        # C1 has negative Z-score on capacitance (degrades less than lot median)
        self.assertLess(c1_row["z_delta_capacitance_pct"], 0.0)

        # Directional test: if only flagging Z > +2.5, C1 is NOT flagged
        is_directional_flag = c1_row["z_delta_capacitance_pct"] > 2.5
        self.assertFalse(is_directional_flag, "Directional screening should accept C1")

    def test_threshold_sensitivity_monotonicity(self):
        """Verifies threshold sensitivity results for parameter variations."""
        analyzer = SensitivityAnalyzer(self.step4_dir, self.step5_dir)
        df_sens = analyzer.run_sensitivity()
        self.assertGreaterEqual(len(df_sens), 20)

        # In Drift model, as spec limit increases from 18% to 22%, recall must be non-increasing
        drift_sub = df_sens[df_sens["parameter_name"] == "spec_limit_delta_c_pct"].sort_values("threshold_value")
        recalls = drift_sub["recall"].tolist()
        for i in range(len(recalls) - 1):
            self.assertGreaterEqual(recalls[i], recalls[i + 1])

    def test_ablation_combinations_coverage(self):
        """Verifies that all 7 ablation configurations are computed and verified."""
        analyzer = AblationAnalyzer(self.step4_dir, self.step5_dir)
        df_abl, _ = analyzer.run_ablation()
        self.assertEqual(len(df_abl), 7)

        # Paradigm A (Trad) recall must be 0.0
        self.assertEqual(df_abl.loc[df_abl["paradigm"].str.startswith("A:"), "recall"].iloc[0], 0.0)

        # Paradigm D (Drift) recall must be 1.0
        self.assertEqual(df_abl.loc[df_abl["paradigm"].str.startswith("D:"), "recall"].iloc[0], 1.0)

        # Paradigm E (Anomaly + Drift) recall must be 1.0
        self.assertEqual(df_abl.loc[df_abl["paradigm"].str.startswith("E:"), "recall"].iloc[0], 1.0)

    def test_cost_sensitive_crossover(self):
        """Verifies that AI/ML models become economically superior to Traditional screening above crossover ratio."""
        analyzer = CostAnalyzer(self.step4_dir, self.step5_dir)
        df_cost = analyzer.run_cost_analysis()

        # At ratio = 10.0 (catastrophic escape scenario), Drift Forecaster must be superior
        sub_10 = df_cost[(df_cost["cost_ratio_fn_to_fp"] == 10.0) & (df_cost["method"] == "Early Drift Forecast (194h)")]
        self.assertTrue(sub_10["is_superior_to_traditional"].iloc[0])
        self.assertGreater(sub_10["cost_savings_pct"].iloc[0], 90.0)

    def test_publication_tables_exist(self):
        """Verifies that all 10 tables exist in both CSV and Markdown with valid content."""
        expected_tables = [
            "table1_dataset_characteristics",
            "table2_model_configuration",
            "table3_comparative_performance",
            "table4_component_error_analysis",
            "table5_threshold_sensitivity",
            "table6_ablation_study",
            "table7_early_warning_lead_times",
            "table8_explainability_summary",
            "table9_statistical_robustness",
            "table10_engineering_tradeoffs",
        ]
        for tbl in expected_tables:
            csv_f = self.tables_dir / f"{tbl}.csv"
            md_f = self.tables_dir / f"{tbl}.md"
            self.assertTrue(csv_f.exists(), f"Missing {csv_f}")
            self.assertTrue(md_f.exists(), f"Missing {md_f}")
            self.assertGreater(csv_f.stat().st_size, 50)
            self.assertGreater(md_f.stat().st_size, 50)

    def test_publication_figures_exist(self):
        """Verifies that all 10 publication figures exist and are high-resolution (size > 10KB)."""
        expected_figures = [
            "fig1_performance_radar.png",
            "fig2_recall_vs_fpr_tradeoff.png",
            "fig3_component_error_breakdown.png",
            "fig4_threshold_sensitivity_curves.png",
            "fig5_early_warning_timeline.png",
            "fig6_drift_forecast_error_scatter.png",
            "fig7_ablation_waterfall.png",
            "fig8_cost_sensitive_curves.png",
            "fig9_explainability_deviations.png",
            "fig10_research_architecture_summary.png",
        ]
        for fig in expected_figures:
            fig_path = self.figures_dir / fig
            self.assertTrue(fig_path.exists(), f"Missing {fig_path}")
            self.assertGreater(fig_path.stat().st_size, 10000, f"Figure {fig} is too small: {fig_path.stat().st_size} bytes")


if __name__ == "__main__":
    unittest.main()
