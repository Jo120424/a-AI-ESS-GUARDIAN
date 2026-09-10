#!/usr/bin/env python3
"""
Automated Foundation & Metadata Verification Tests for Step 1.
"""

import json
import os
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class TestProjectFoundation(unittest.TestCase):
    def test_directory_structure(self):
        required_dirs = [
            "data/raw",
            "data/processed",
            "data/metadata",
            "data/sample",
            "experiments/configs",
            "experiments/results",
            "experiments/manifests",
            "models",
            "reports",
            "research/datasets",
            "research/literature",
            "research/notes",
            "tests",
            "docs",
            "frontend",
            "backend"
        ]
        for d in required_dirs:
            p = BASE_DIR / d
            self.assertTrue(p.is_dir(), f"Required directory missing: {d}")

    def test_dataset_metadata_schema(self):
        metadata_file = BASE_DIR / "data" / "metadata" / "dataset_metadata.json"
        self.assertTrue(metadata_file.is_file(), "dataset_metadata.json missing")

        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = [
            "dataset_name",
            "source",
            "source_url",
            "original_paper",
            "classification",
            "component_count",
            "features",
            "time_series_available",
            "component_id_available",
            "lot_id_available",
            "failure_labels_available",
            "degradation_labels_available",
            "environmental_variables",
            "static_threshold_available",
            "anomaly_detection_supported",
            "degradation_prediction_supported",
            "latent_risk_proxy_supported",
            "limitations"
        ]
        for k in required_keys:
            self.assertIn(k, data, f"Key {k} missing in dataset_metadata.json")

        self.assertEqual(data["classification"], "C — ACCELERATED LIFE / DEGRADATION DATASET")
        self.assertTrue(data["time_series_available"])
        self.assertTrue(data["component_id_available"])
        self.assertTrue(data["static_threshold_available"])
        self.assertEqual(data["component_count"], 24)

    def test_candidate_datasets_catalog(self):
        candidates_file = BASE_DIR / "data" / "metadata" / "candidate_datasets.json"
        self.assertTrue(candidates_file.is_file(), "candidate_datasets.json missing")

        with open(candidates_file, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        self.assertIsInstance(candidates, list)
        self.assertGreaterEqual(len(candidates), 3)

        roles = [c.get("role") for c in candidates]
        self.assertIn("PRIMARY", roles)
        self.assertIn("BACKUP 1", roles)
        self.assertIn("BACKUP 2", roles)

    def test_documentation_files(self):
        required_docs = [
            "research/datasets/dataset_selection.md",
            "research/research_plan.md",
            "docs/architecture.md",
            "README.md"
        ]
        for doc in required_docs:
            p = BASE_DIR / doc
            self.assertTrue(p.is_file(), f"Document missing: {doc}")
            self.assertGreater(p.stat().st_size, 200, f"Document too short: {doc}")

    def test_frontend_files(self):
        required_frontend = [
            "frontend/index.html",
            "frontend/css/style.css",
            "frontend/js/app.js"
        ]
        for f in required_frontend:
            p = BASE_DIR / f
            self.assertTrue(p.is_file(), f"Frontend file missing: {f}")


if __name__ == "__main__":
    unittest.main()
