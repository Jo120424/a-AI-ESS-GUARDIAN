#!/usr/bin/env python3
"""
Automated Test Suite for Step 3 Experimental Design & Ground Truth Engine.
"""

import json
import unittest
from pathlib import Path

import pandas as pd

from backend.experiments.ground_truth import (
    DataLeakageError,
    extract_screening_features,
    generate_ground_truth_labels,
    load_processed_data,
    verify_anti_leakage,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "experiments" / "configs" / "experiment_config.yaml"


class TestExperimentalDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_processed_data()
        cls.t_screen = 47.0

    # Test 1: Ground Truth Label Correctness
    def test_01_ground_truth_labels_at_47h(self):
        gt_df = generate_ground_truth_labels(self.df, t_screen=self.t_screen)
        self.assertEqual(len(gt_df), 6, "Expected 6 component labels")

        # Verify all components pass early static limits at 47h
        self.assertTrue(gt_df["passes_early_static"].all(), "Expected all units to pass static spec at t_screen=47h")

        # Verify C1 survives (0) and C2-C6 fail downstream (1)
        c1_row = gt_df[gt_df["component_id"] == "C1"].iloc[0]
        self.assertEqual(c1_row["ground_truth_latent_risk"], 0, "Component C1 should be 0 (Survivor)")

        failing_units = ["C2", "C3", "C4", "C5", "C6"]
        for uid in failing_units:
            row = gt_df[gt_df["component_id"] == uid].iloc[0]
            self.assertEqual(row["ground_truth_latent_risk"], 1, f"Component {uid} should be 1 (Latent Risk)")
            self.assertTrue(row["future_spec_violation"])

    # Test 2: Feature Matrix Temporal Quarantining
    def test_02_screening_features_quarantine(self):
        feats = extract_screening_features(self.df, t_screen=self.t_screen)
        self.assertEqual(len(feats), 6, "Expected 6 feature rows")
        self.assertTrue((feats["t_cutoff_hours"] <= self.t_screen).all(), "Feature timestamp exceeds t_screen!")

        # Verify expected causal features exist
        expected_cols = [
            "component_id", "lot_id", "t_cutoff_hours", "delta_capacitance_pct",
            "delta_esr_pct", "drift_velocity_c", "drift_velocity_esr", "c_to_esr_ratio"
        ]
        for col in expected_cols:
            self.assertIn(col, feats.columns)

    # Test 3: Anti-Leakage Guard - Component Overlap Detection
    def test_03_leakage_detection_component_overlap(self):
        feats = extract_screening_features(self.df, t_screen=self.t_screen)
        # Deliberately violate component isolation
        train_comps = {"C1", "C2", "C3", "C4"}
        test_comps = {"C4", "C5"}  # C4 is in both!
        with self.assertRaises(DataLeakageError):
            verify_anti_leakage(train_comps, test_comps, feats, t_screen=self.t_screen)

    # Test 4: Anti-Leakage Guard - Lookahead Detection
    def test_04_leakage_detection_temporal_lookahead(self):
        feats = extract_screening_features(self.df, t_screen=self.t_screen)
        # Corrupt feature table with future timestamp
        bad_feats = feats.copy()
        bad_feats.loc[0, "t_cutoff_hours"] = 94.0
        with self.assertRaises(DataLeakageError):
            verify_anti_leakage({"C1", "C2"}, {"C3"}, bad_feats, t_screen=self.t_screen)

    # Test 5: Anti-Leakage Guard - Target Column Detection
    def test_05_leakage_detection_target_column(self):
        feats = extract_screening_features(self.df, t_screen=self.t_screen)
        bad_feats = feats.copy()
        bad_feats["ground_truth_latent_risk"] = 1  # Inadvertently include label!
        with self.assertRaises(DataLeakageError):
            verify_anti_leakage({"C1", "C2"}, {"C3"}, bad_feats, t_screen=self.t_screen)

    # Test 6: Config YAML Exists and is Valid
    def test_06_config_yaml_validity(self):
        self.assertTrue(CONFIG_FILE.is_file(), "experiment_config.yaml missing")
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        self.assertIn("predictive_ess_comparative_screening", text)
        self.assertIn("isolation_forest", text)
        self.assertIn("leave_one_component_out", text)


if __name__ == "__main__":
    unittest.main()
