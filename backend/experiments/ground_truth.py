#!/usr/bin/env python3
"""
Ground Truth Label Generation and Anti-Leakage Feature Extraction Module.
Generates research-grade ground-truth target labels strictly from observations
occurring AFTER the screening cutoff horizon (t > t_screen).
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "dataset_processed.csv"


class DataLeakageError(Exception):
    """Raised when data leakage between train/test or past/future is detected."""
    pass


def load_processed_data(file_path: Path = None) -> pd.DataFrame:
    path = file_path or PROCESSED_FILE
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")
    return pd.read_csv(path)


def generate_ground_truth_labels(
    df: pd.DataFrame,
    t_screen: float,
    spec_limit_delta_c: float = 20.0,
    spec_limit_delta_esr: float = 100.0
) -> pd.DataFrame:
    """
    Generates component-level ground-truth labels evaluated strictly on the
    future operational horizon (t > t_screen).

    Ground Truth Definition:
    y = 1 (Latent Defect / Future Risk):
        Component satisfies static limits at t_screen (Delta C < 20%),
        but violates MIL-PRF-62F spec (Delta C >= 20% or Delta ESR >= 100%)
        at some time t > t_screen.
    y = 0 (Safe Survivor):
        Component satisfies static limits at t_screen AND never violates
        the specification throughout the remainder of the test (up to t_final).

    Returns:
        pd.DataFrame with columns: ['component_id', 'lot_id', 'passes_early_static',
                                   'future_max_delta_c', 'future_max_delta_esr',
                                   'future_spec_violation', 'ground_truth_latent_risk',
                                   'future_failure_time_h']
    """
    components = sorted(df["component_id"].unique())
    label_records = []

    for comp in components:
        comp_data = df[df["component_id"] == comp].sort_values("aging_time_hours")

        # 1. Early screening state (t <= t_screen)
        early_data = comp_data[comp_data["aging_time_hours"] <= t_screen]
        if len(early_data) == 0:
            raise ValueError(f"No early observations for component {comp} at t_screen={t_screen}")

        early_max_c = early_data["delta_capacitance_pct"].max()
        early_max_esr = early_data["delta_esr_pct"].max()
        passes_early = bool(early_max_c < spec_limit_delta_c and early_max_esr < spec_limit_delta_esr)

        # 2. Future operational state (t > t_screen)
        future_data = comp_data[comp_data["aging_time_hours"] > t_screen]
        if len(future_data) == 0:
            future_max_c = early_max_c
            future_max_esr = early_max_esr
            future_fail = False
            first_fail_t = None
        else:
            future_max_c = future_data["delta_capacitance_pct"].max()
            future_max_esr = future_data["delta_esr_pct"].max()
            future_fail_records = future_data[
                (future_data["delta_capacitance_pct"] >= spec_limit_delta_c) |
                (future_data["delta_esr_pct"] >= spec_limit_delta_esr)
            ]
            future_fail = bool(len(future_fail_records) > 0)
            first_fail_t = float(future_fail_records["aging_time_hours"].min()) if future_fail else None

        # 3. Formulate Ground-Truth Latent Risk Label
        # A component is a latent risk if it passed early screening but failed in the future
        is_latent_risk = 1 if (passes_early and future_fail) else 0

        label_records.append({
            "component_id": comp,
            "lot_id": comp_data["lot_id"].iloc[0],
            "t_screen_hours": float(t_screen),
            "passes_early_static": passes_early,
            "early_max_delta_c": round(float(early_max_c), 4),
            "future_max_delta_c": round(float(future_max_c), 4),
            "future_max_delta_esr": round(float(future_max_esr), 4),
            "future_spec_violation": future_fail,
            "future_failure_time_h": first_fail_t,
            "ground_truth_latent_risk": is_latent_risk
        })

    return pd.DataFrame(label_records)


def extract_screening_features(df: pd.DataFrame, t_screen: float) -> pd.DataFrame:
    """
    Extracts strictly pre-screening features for all components at t = t_screen.
    Guarantees that no future telemetry (t > t_screen) enters the feature matrix.

    Returns:
        pd.DataFrame indexed by component_id with strictly causal features.
    """
    early_df = df[df["aging_time_hours"] <= t_screen].copy()
    feature_rows = []

    for comp in sorted(early_df["component_id"].unique()):
        sub = early_df[early_df["component_id"] == comp].sort_values("aging_time_hours")
        latest = sub.iloc[-1]
        first = sub.iloc[0]

        feat = {
            "component_id": comp,
            "lot_id": latest["lot_id"],
            "t_cutoff_hours": float(latest["aging_time_hours"]),
            "delta_capacitance_pct": float(latest["delta_capacitance_pct"]),
            "delta_esr_pct": float(latest["delta_esr_pct"]),
            "capacitance_uf": float(latest["capacitance_uf"]),
            "esr_ohms": float(latest["esr_ohms"]),
            "drift_velocity_c": float(latest["drift_velocity_c"]),
            "drift_velocity_esr": float(latest["drift_velocity_esr"]),
            "cum_delta_c": float(latest["delta_capacitance_pct"] - first["delta_capacitance_pct"]),
            "cum_delta_esr": float(latest["delta_esr_pct"] - first["delta_esr_pct"]),
            "c_to_esr_ratio": float((latest["delta_capacitance_pct"] + 1e-4) / (latest["delta_esr_pct"] + 1e-4))
        }
        feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows)
    return feat_df


def verify_anti_leakage(
    train_comps: Set[str],
    test_comps: Set[str],
    X_screen: pd.DataFrame,
    t_screen: float
) -> bool:
    """
    Programmatic audit verifying all anti-leakage constraints.
    Raises DataLeakageError on any violation.
    """
    # 1. Check Component Disjointness
    overlap = train_comps.intersection(test_comps)
    if overlap:
        raise DataLeakageError(f"Component Leakage Detected! Overlapping components: {overlap}")

    # 2. Check Temporal Lookahead
    if (X_screen["t_cutoff_hours"] > t_screen).any():
        bad_times = X_screen[X_screen["t_cutoff_hours"] > t_screen]["t_cutoff_hours"].tolist()
        raise DataLeakageError(f"Temporal Leakage Detected! Observations exceed t_screen ({t_screen}h): {bad_times}")

    # 3. Check for Prohibited Target Columns in X
    prohibited_cols = {"ground_truth_latent_risk", "future_spec_violation", "future_failure_time_h", "component_ever_fails"}
    found_targets = prohibited_cols.intersection(set(X_screen.columns))
    if found_targets:
        raise DataLeakageError(f"Label Leakage Detected! Target columns found in feature matrix: {found_targets}")

    return True


if __name__ == "__main__":
    df = load_processed_data()
    t_screen = 47.0
    print(f"=== Testing Ground Truth & Feature Extraction (t_screen = {t_screen}h) ===")

    gt_df = generate_ground_truth_labels(df, t_screen=t_screen)
    print("\nGenerated Ground-Truth Labels:")
    print(gt_df[["component_id", "passes_early_static", "future_spec_violation", "future_failure_time_h", "ground_truth_latent_risk"]])

    feats_df = extract_screening_features(df, t_screen=t_screen)
    print("\nExtracted Screening Feature Matrix (Head):")
    print(feats_df[["component_id", "t_cutoff_hours", "delta_capacitance_pct", "drift_velocity_c", "c_to_esr_ratio"]])

    # Test anti-leakage check
    train_c = {"C1", "C2", "C3", "C5", "C6"}
    test_c = {"C4"}
    verify_anti_leakage(train_c, test_c, feats_df, t_screen=t_screen)
    print("\nAnti-leakage audit: PASSED successfully!")
