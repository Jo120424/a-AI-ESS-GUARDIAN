#!/usr/bin/env python3
"""
Automated Data Pipeline Validation Test Suite for Step 2.
Verifies raw integrity, schema validity, component uniqueness, non-leakage,
temporal ordering, and processed output reproducibility.
"""

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd
import scipy.io

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "nasa_capacitor_electrical_stress"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
METADATA_DIR = BASE_DIR / "data" / "metadata"


class TestDataPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_mat_file = RAW_DIR / "EOS_DataSet.mat"
        cls.raw_zip_file = RAW_DIR / "EOS_DataSet.zip"
        cls.processed_csv = PROCESSED_DIR / "dataset_processed.csv"
        cls.manifest_json = METADATA_DIR / "dataset_manifest.json"
        cls.acq_meta_json = METADATA_DIR / "acquisition_metadata.json"

    # Test 1: Raw Dataset Untouched & Present
    def test_01_raw_dataset_untouched_and_present(self):
        self.assertTrue(self.raw_mat_file.is_file(), "Raw MAT file is missing")
        self.assertTrue(self.raw_zip_file.is_file(), "Raw ZIP archive is missing")

        # Verify raw MAT SHA256 matches acquisition record
        with open(self.raw_mat_file, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()

        with open(self.acq_meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        raw_records = {item["file_name"]: item["sha256"] for item in meta["raw_files"]}
        self.assertIn("EOS_DataSet.mat", raw_records)
        self.assertEqual(sha, raw_records["EOS_DataSet.mat"], "Raw file hash mismatch! File was modified in place.")

    # Test 2: Schema Validity
    def test_02_schema_validity(self):
        mat = scipy.io.loadmat(self.raw_mat_file)
        expected_keys = {"aging_time", "C", "ESR"}
        actual_keys = {k for k in mat if not k.startswith("__")}
        self.assertTrue(expected_keys.issubset(actual_keys), f"Missing required raw keys: {expected_keys - actual_keys}")
        self.assertEqual(mat["C"].shape, mat["ESR"].shape)
        self.assertEqual(len(mat["aging_time"]), mat["C"].shape[0])

    # Test 3: Required Columns in Processed Data
    def test_03_required_columns_in_processed(self):
        self.assertTrue(self.processed_csv.is_file(), "Processed CSV is missing")
        df = pd.read_csv(self.processed_csv)
        required_cols = [
            "component_id", "lot_id", "test_step", "aging_time_hours",
            "delta_capacitance_pct", "delta_esr_pct", "capacitance_uf", "esr_ohms",
            "stress_voltage_v", "static_spec_fail", "drift_velocity_c",
            "drift_velocity_esr", "component_ever_fails", "first_failure_step"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

    # Test 4: Numeric Conversion & Physical Plausibility
    def test_04_numeric_conversion_and_bounds(self):
        df = pd.read_csv(self.processed_csv)
        numeric_cols = [
            "test_step", "aging_time_hours", "delta_capacitance_pct",
            "delta_esr_pct", "capacitance_uf", "esr_ohms", "stress_voltage_v"
        ]
        for col in numeric_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric")

        # Check physical bounds
        self.assertTrue((df["aging_time_hours"] >= 0.0).all(), "Negative aging time detected")
        self.assertTrue((df["delta_capacitance_pct"] >= 0.0).all(), "Negative capacitance drop detected")
        self.assertTrue((df["delta_esr_pct"] >= 0.0).all(), "Negative ESR rise detected")
        self.assertTrue((df["capacitance_uf"] > 0.0).all(), "Non-positive capacitance detected")
        self.assertTrue((df["esr_ohms"] > 0.0).all(), "Non-positive ESR detected")

    # Test 5: Missing Values Handling
    def test_05_missing_values_zero(self):
        df = pd.read_csv(self.processed_csv)
        total_missing = df.isna().sum().sum()
        self.assertEqual(total_missing, 0, f"Unexpected missing values detected: {total_missing}")

    # Test 6: Component Identifiers Integrity
    def test_06_component_identifiers(self):
        df = pd.read_csv(self.processed_csv)
        expected_comps = {"C1", "C2", "C3", "C4", "C5", "C6"}
        actual_comps = set(df["component_id"].unique())
        self.assertEqual(actual_comps, expected_comps, "Component IDs mismatch")

        # Verify equal observations per component
        counts = df["component_id"].value_counts()
        self.assertEqual(len(set(counts.values)), 1, "Unequal observation count across components")
        self.assertEqual(counts.iloc[0], 11, "Expected 11 observations per component")

    # Test 7: No Duplicate Component-Time Records
    def test_07_no_duplicate_component_time(self):
        df = pd.read_csv(self.processed_csv)
        duplicates = df.duplicated(subset=["component_id", "aging_time_hours"]).sum()
        self.assertEqual(duplicates, 0, "Duplicate component-time records found")

    # Test 8: Temporal Ordering
    def test_08_temporal_ordering(self):
        df = pd.read_csv(self.processed_csv)
        for comp in df["component_id"].unique():
            sub = df[df["component_id"] == comp]
            times = sub["aging_time_hours"].tolist()
            self.assertEqual(times, sorted(times), f"Non-monotonic time steps for component {comp}")
            steps = sub["test_step"].tolist()
            self.assertEqual(steps, list(range(len(steps))), f"Non-sequential test steps for component {comp}")

    # Test 9: No Target Leakage in Feature Columns
    def test_09_no_future_information_leakage(self):
        df = pd.read_csv(self.processed_csv)
        # Verify that drift_velocity at t=0 is 0.0 (no forward difference)
        t0_records = df[df["aging_time_hours"] == 0.0]
        self.assertTrue((t0_records["drift_velocity_c"] == 0.0).all())
        self.assertTrue((t0_records["drift_velocity_esr"] == 0.0).all())

    # Test 10: Processed Dataset Reproducibility
    def test_10_processed_reproducibility(self):
        from backend.data_pipeline.preprocess import PreprocessingPipeline
        pipeline = PreprocessingPipeline()
        fresh_df = pipeline.load_and_transform_tidy()
        disk_df = pd.read_csv(self.processed_csv)
        pd.testing.assert_frame_equal(fresh_df, disk_df, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
