#!/usr/bin/env python3
"""
Master Experiment Runner for Predictive AI-Based Environmental Stress Screening (Step 4).
Executes the complete, leak-free 6-fold Leave-One-Component-Out (LOCO) comparative experiment:
- Level 1: Traditional Static Screening (MIL-PRF-62F & Tightened Baseline)
- Level 2: Dynamic Statistical Screening (MAD Robust Z-score & Ledoit-Wolf Mahalanobis)
- Level 3: AI/ML Anomaly Detection (Isolation Forest & One-Class SVM)
- Level 4: Early Drift Prediction (Ridge & Gradient Boosting Regressor)
- Level 5: Risk Fusion Engine (Multi-Tier Decision Synthesis)

Saves all raw and processed artifacts to results/step4/ and generates publication figures.
Executable via:
    python -m backend.experiments.run_experiment
"""

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.experiments.anomaly import MLAnomalyScreening
from backend.experiments.drift import DriftPredictor
from backend.experiments.evaluation import (
    build_master_comparison_table,
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
from backend.experiments.visualizer import generate_all_figures

CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "experiment_config.yaml"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_processed.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "step4"
FIGURES_DIR = RESULTS_DIR / "figures"


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for cryptographic provenance."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def run_full_experiment(
    config_path: Path = CONFIG_PATH,
    results_dir: Path = RESULTS_DIR,
    t_screen: float = 47.0,
    seed: int = 42
) -> Dict:
    """
    Executes the reproducible 6-fold LOCO comparative screening experiment.
    """
    start_time = time.time()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"================================================================================")
    print(f" PREDICTIVE AI-BASED ESS: STEP 4 COMPARATIVE EXPERIMENT EXECUTION")
    print(f" Screening Cutoff Horizon: t_screen = {t_screen} hours")
    print(f" Evaluation Protocol: 6-Fold Leave-One-Component-Out (LOCO) Cross-Validation")
    print(f" Random Seed: {seed}")
    print(f"================================================================================")

    # 1. Ingest Data and Verify Checksum
    df_processed = load_processed_data(PROCESSED_DATA_PATH)
    data_hash = compute_file_sha256(PROCESSED_DATA_PATH)
    print(f"[1/8] Ingested Processed Data: {len(df_processed)} records across {df_processed['component_id'].nunique()} components.")
    print(f"      Dataset SHA-256: {data_hash}")

    # 2. Generate Frozen Ground Truth (Definition B: Future Spec Violation at t > t_screen)
    df_ground_truth = generate_ground_truth_labels(df_processed, t_screen=t_screen)
    print(f"[2/8] Ground Truth Generated: {df_ground_truth['ground_truth_latent_risk'].sum()} Latent Defect Risks, "
          f"{len(df_ground_truth) - df_ground_truth['ground_truth_latent_risk'].sum()} Safe Survivors.")

    # 3. Extract Causal Pre-Screening Features (t <= 47.0h only)
    df_features = extract_screening_features(df_processed, t_screen=t_screen)
    df_drift_features = DriftPredictor.extract_drift_features(df_processed, t_screen=t_screen)
    print(f"[3/8] Extracted {len(df_features.columns)} causal features strictly constrained to t <= {t_screen}h.")

    components = sorted(df_processed["component_id"].unique())
    n_components = len(components)

    # 4. Execute 6-Fold Leave-One-Component-Out (LOCO)
    print(f"[4/8] Executing {n_components}-Fold Leave-One-Component-Out (LOCO) Evaluation Loop...")

    oof_traditional_records = []
    oof_statistical_records = []
    oof_aiml_records = []
    oof_drift_records = []
    oof_fusion_records = []

    for fold_idx, test_comp in enumerate(components, start=1):
        train_comps = [c for c in components if c != test_comp]
        test_comps = [test_comp]

        # Verify strict anti-leakage constraints for this fold
        verify_anti_leakage(set(train_comps), set(test_comps), df_features, t_screen=t_screen)

        # Split features
        train_feat_df = df_features[df_features["component_id"].isin(train_comps)].copy()
        test_feat_df = df_features[df_features["component_id"] == test_comp].copy()

        train_drift_df = df_drift_features[df_drift_features["component_id"].isin(train_comps)].copy()
        test_drift_df = df_drift_features[df_drift_features["component_id"] == test_comp].copy()

        # -------------------------------------------------------------
        # Level 1: Traditional Static Screening (MIL-PRF-62F & Tightened)
        # -------------------------------------------------------------
        trad_screener = TraditionalScreening(spec_limit_delta_c=20.0, tightened_limit_delta_c=5.0)
        trad_res = trad_screener.evaluate_component(test_feat_df.iloc[0].to_dict())
        oof_traditional_records.append(trad_res)

        # -------------------------------------------------------------
        # Level 2: Dynamic Statistical Screening (MAD & Mahalanobis)
        # -------------------------------------------------------------
        stat_screener = StatisticalScreening(mad_threshold=2.5, alpha_significance=0.05)
        stat_screener.fit(train_feat_df)
        stat_res = stat_screener.evaluate_component(test_feat_df.iloc[0].to_dict())
        oof_statistical_records.append(stat_res)

        # -------------------------------------------------------------
        # Level 3: AI/ML Anomaly Screening (Isolation Forest & OC-SVM)
        # -------------------------------------------------------------
        ml_screener = MLAnomalyScreening(contamination=0.20, random_state=seed + fold_idx)
        ml_screener.fit(train_feat_df)
        ml_res = ml_screener.evaluate_component(test_feat_df.iloc[0].to_dict())
        oof_aiml_records.append(ml_res)

        # -------------------------------------------------------------
        # Level 4: Early Drift Prediction (Ridge & GBR)
        # -------------------------------------------------------------
        drift_predictor = DriftPredictor(target_horizon_hours=194.0, spec_limit=20.0, random_state=seed)
        drift_predictor.fit(train_drift_df, train_drift_df["target_delta_c_194h"].values)
        drift_res = drift_predictor.predict_component(test_drift_df.iloc[0].to_dict())
        oof_drift_records.append(drift_res)

        # -------------------------------------------------------------
        # Level 5: Risk Fusion Engine
        # -------------------------------------------------------------
        fusion_engine = RiskFusionEngine()
        fusion_res = fusion_engine.fuse_component(test_comp, trad_res, stat_res, ml_res, drift_res)
        oof_fusion_records.append(fusion_res)

        print(f"      Fold {fold_idx}/{n_components}: Test Component {test_comp} -> "
              f"Trad: {trad_res['traditional_status']}, Stat: {stat_res['statistical_status']}, "
              f"AI: {ml_res['aiml_status']}, Drift: {drift_res['drift_status']}, "
              f"Fusion: {fusion_res['risk_tier']}")

    # 5. Assemble Out-of-Fold Component Predictions
    print(f"[5/8] Assembling Out-of-Fold Master Predictions Table...")
    df_oof_trad = pd.DataFrame(oof_traditional_records).rename(columns={"explanation": "trad_explanation"})
    df_oof_stat = pd.DataFrame(oof_statistical_records).rename(columns={"explanation": "stat_explanation"})
    df_oof_ml = pd.DataFrame(oof_aiml_records).rename(columns={"explanation": "aiml_explanation"})
    df_oof_drift = pd.DataFrame(oof_drift_records).rename(columns={"explanation": "drift_explanation"})
    df_oof_fusion = pd.DataFrame(oof_fusion_records).rename(columns={"explanation": "fusion_explanation"})

    # Merge into comprehensive component_level_results.csv
    df_component_master = df_ground_truth.merge(df_oof_trad, on="component_id")
    df_component_master = df_component_master.merge(df_oof_stat, on="component_id")
    df_component_master = df_component_master.merge(df_oof_ml, on="component_id")
    df_component_master = df_component_master.merge(df_oof_drift, on="component_id")
    df_component_master = df_component_master.merge(df_oof_fusion, on="component_id")

    # 6. Compute Comparative Screening Metrics on Identical Test Partitions
    print(f"[6/8] Computing Comparative Reliability and Yield Metrics...")
    y_true = df_ground_truth["ground_truth_latent_risk"].values
    failure_times = df_ground_truth["future_failure_time_h"].tolist()

    predictions_map = {
        "Traditional (MIL-PRF-62F 20%)": df_oof_trad["traditional_decision"].values,
        "Traditional (5% Tightened)": df_oof_trad["traditional_decision_tightened"].values,
        "Dynamic Statistical (MAD/Maha)": df_oof_stat["statistical_decision"].values,
        "AI/ML Anomaly (Isolation Forest)": df_oof_ml["aiml_decision"].values,
        "AI/ML Anomaly (One-Class SVM)": df_oof_ml["ocsvm_flag"].values,
        "Early Drift Forecast (194h)": df_oof_drift["drift_decision"].values,
        "Risk Fusion Engine (Multi-Tier)": df_oof_fusion["fusion_decision"].values
    }

    scores_map = {
        "Traditional (MIL-PRF-62F 20%)": df_oof_trad["delta_c_at_screen"].values / 20.0,
        "Traditional (5% Tightened)": df_oof_trad["delta_c_at_screen"].values / 5.0,
        "Dynamic Statistical (MAD/Maha)": df_oof_stat["mahalanobis_dist_sq"].values / df_oof_stat["chi2_cutoff"].iloc[0],
        "AI/ML Anomaly (Isolation Forest)": df_oof_ml["iforest_anomaly_score"].values,
        "AI/ML Anomaly (One-Class SVM)": df_oof_ml["ocsvm_anomaly_score"].values,
        "Early Drift Forecast (194h)": df_oof_drift["predicted_delta_c_194h"].values / 20.0,
        "Risk Fusion Engine (Multi-Tier)": df_oof_fusion["fusion_risk_score"].values
    }

    df_comparison = build_master_comparison_table(
        y_true=y_true,
        predictions_dict=predictions_map,
        scores_dict=scores_map,
        failure_times=failure_times,
        t_screen=t_screen
    )

    # Statistical significance testing (McNemar and Bootstrap)
    y_trad = predictions_map["Traditional (MIL-PRF-62F 20%)"]
    y_aiml = predictions_map["AI/ML Anomaly (Isolation Forest)"]
    y_ocsvm = predictions_map["AI/ML Anomaly (One-Class SVM)"]
    y_drift = predictions_map["Early Drift Forecast (194h)"]
    y_stat = predictions_map["Dynamic Statistical (MAD/Maha)"]
    y_fusion = predictions_map["Risk Fusion Engine (Multi-Tier)"]

    mcnemar_trad_vs_ml = mcnemar_exact_test(y_true, y_trad, y_aiml)
    mcnemar_trad_vs_ocsvm = mcnemar_exact_test(y_true, y_trad, y_ocsvm)
    mcnemar_trad_vs_drift = mcnemar_exact_test(y_true, y_trad, y_drift)
    mcnemar_trad_vs_stat = mcnemar_exact_test(y_true, y_trad, y_stat)
    mcnemar_trad_vs_fusion = mcnemar_exact_test(y_true, y_trad, y_fusion)

    bootstrap_trad_vs_ml = paired_bootstrap_ci(y_true, y_trad, y_aiml, n_iterations=2000, seed=seed)
    bootstrap_trad_vs_ocsvm = paired_bootstrap_ci(y_true, y_trad, y_ocsvm, n_iterations=2000, seed=seed)
    bootstrap_trad_vs_drift = paired_bootstrap_ci(y_true, y_trad, y_drift, n_iterations=2000, seed=seed)
    bootstrap_trad_vs_stat = paired_bootstrap_ci(y_true, y_trad, y_stat, n_iterations=2000, seed=seed)
    bootstrap_trad_vs_fusion = paired_bootstrap_ci(y_true, y_trad, y_fusion, n_iterations=2000, seed=seed)

    # Drift prediction regression metrics
    drift_errors = df_oof_drift["prediction_abs_error"].dropna().values
    drift_mae = float(np.mean(drift_errors)) if len(drift_errors) > 0 else 0.0
    drift_rmse = float(np.sqrt(np.mean(drift_errors ** 2))) if len(drift_errors) > 0 else 0.0

    # 7. Persist All Artifacts to results/step4/
    print(f"[7/8] Persisting Results, Data Tables, and Config to {results_dir}...")
    df_oof_trad.to_csv(results_dir / "traditional_predictions.csv", index=False)
    df_oof_stat.to_csv(results_dir / "statistical_predictions.csv", index=False)
    df_oof_ml.to_csv(results_dir / "anomaly_predictions.csv", index=False)
    df_oof_drift.to_csv(results_dir / "drift_predictions.csv", index=False)
    df_oof_fusion.to_csv(results_dir / "risk_predictions.csv", index=False)
    df_component_master.to_csv(results_dir / "component_level_results.csv", index=False)
    df_comparison.to_csv(results_dir / "comparison.csv", index=False)
    df_comparison.to_csv(results_dir / "metrics.csv", index=False)

    # Feature Importance table
    dev_cols = [c for c in df_component_master.columns if c.startswith("dev_")]
    feat_imp_rows = []
    for c in dev_cols:
        feat_name = c.replace("dev_", "")
        mean_dev = float(df_component_master[c].mean())
        max_dev = float(df_component_master[c].max())
        feat_imp_rows.append({
            "feature": feat_name,
            "mean_standardized_deviation": round(mean_dev, 4),
            "max_standardized_deviation": round(max_dev, 4)
        })
    df_feat_imp = pd.DataFrame(feat_imp_rows).sort_values("mean_standardized_deviation", ascending=False)
    df_feat_imp.to_csv(results_dir / "feature_importance.csv", index=False)

    # Copy config used
    if CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, results_dir / "config_used.yaml")

    # Experiment metadata JSON
    metadata = {
        "experiment_name": "predictive_ess_comparative_screening_step4",
        "timestamp": datetime.now().isoformat(),
        "execution_duration_seconds": round(time.time() - start_time, 2),
        "dataset": "NASA Capacitor Electrical Stress Degradation Dataset (EOS_DataSet.mat)",
        "dataset_sha256": data_hash,
        "screening_window_hours": t_screen,
        "total_test_duration_hours": 194.0,
        "number_of_components": n_components,
        "ground_truth_latent_defects": int(np.sum(y_true == 1)),
        "ground_truth_safe_survivors": int(np.sum(y_true == 0)),
        "cross_validation": "6-Fold Leave-One-Component-Out (LOCO)",
        "random_seed": seed,
        "drift_prediction": {
            "target_horizon_hours": 194.0,
            "mae_pct": round(drift_mae, 4),
            "rmse_pct": round(drift_rmse, 4)
        },
        "statistical_hypothesis_tests": {
            "mcnemar_trad_vs_isolation_forest": mcnemar_trad_vs_ml,
            "mcnemar_trad_vs_one_class_svm": mcnemar_trad_vs_ocsvm,
            "mcnemar_trad_vs_drift_forecast": mcnemar_trad_vs_drift,
            "mcnemar_trad_vs_dynamic_statistical": mcnemar_trad_vs_stat,
            "mcnemar_trad_vs_risk_fusion": mcnemar_trad_vs_fusion,
            "bootstrap_trad_vs_isolation_forest_95ci": bootstrap_trad_vs_ml,
            "bootstrap_trad_vs_one_class_svm_95ci": bootstrap_trad_vs_ocsvm,
            "bootstrap_trad_vs_drift_forecast_95ci": bootstrap_trad_vs_drift,
            "bootstrap_trad_vs_dynamic_statistical_95ci": bootstrap_trad_vs_stat,
            "bootstrap_trad_vs_risk_fusion_95ci": bootstrap_trad_vs_fusion
        }
    }

    with open(results_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 8. Generate 12 Publication-Quality Visualizations
    print(f"[8/8] Generating 12 Publication-Quality Plots in {figures_dir}...")
    fig_paths = generate_all_figures(
        comparison_df=df_comparison,
        component_results_df=df_component_master,
        drift_features_df=df_drift_features,
        processed_df=df_processed,
        output_dir=figures_dir
    )
    print(f"      Successfully generated {len(fig_paths)} figures.")

    # Format master results summary for terminal
    print("\n" + "=" * 80)
    print(" STEP 4 EXPERIMENT RESULTS SUMMARY (EVALUATION ON UNSEEN TEST COMPONENTS)")
    print("=" * 80)
    print(df_comparison[["method", "recall", "false_negative_rate", "precision", "f1_score", "false_positive_rate", "pr_auc", "mean_lead_time_hours"]].to_string(index=False))
    print("-" * 80)
    print(f"Drift Trajectory Forecast Error (MAE): {drift_mae:.3f}%, RMSE: {drift_rmse:.3f}%")
    print(f"McNemar Test (Traditional vs Isolation Forest): p-value = {mcnemar_trad_vs_ml['exact_p_value']} (Discordant: {mcnemar_trad_vs_ml['total_discordant']})")
    print(f"McNemar Test (Traditional vs One-Class SVM):   p-value = {mcnemar_trad_vs_ocsvm['exact_p_value']} (Discordant: {mcnemar_trad_vs_ocsvm['total_discordant']})")
    print(f"McNemar Test (Traditional vs Drift Forecast):  p-value = {mcnemar_trad_vs_drift['exact_p_value']} (Discordant: {mcnemar_trad_vs_drift['total_discordant']})")
    print(f"McNemar Test (Traditional vs Risk Fusion):     p-value = {mcnemar_trad_vs_fusion['exact_p_value']} (Discordant: {mcnemar_trad_vs_fusion['total_discordant']})")
    print(f"Bootstrap 95% CI (Delta Recall OC-SVM - Trad):  [{bootstrap_trad_vs_ocsvm['delta_recall_ci_95'][0]}, {bootstrap_trad_vs_ocsvm['delta_recall_ci_95'][1]}] (Mean: +{bootstrap_trad_vs_ocsvm['delta_recall_mean']})")
    print(f"Bootstrap 95% CI (Delta Recall Drift - Trad):   [{bootstrap_trad_vs_drift['delta_recall_ci_95'][0]}, {bootstrap_trad_vs_drift['delta_recall_ci_95'][1]}] (Mean: +{bootstrap_trad_vs_drift['delta_recall_mean']})")
    print(f"Bootstrap 95% CI (Delta Recall Fusion - Trad):  [{bootstrap_trad_vs_fusion['delta_recall_ci_95'][0]}, {bootstrap_trad_vs_fusion['delta_recall_ci_95'][1]}] (Mean: +{bootstrap_trad_vs_fusion['delta_recall_mean']})")
    print("=" * 80 + "\n")

    return {
        "metadata": metadata,
        "comparison": df_comparison.to_dict(orient="records"),
        "component_results": df_component_master.to_dict(orient="records"),
        "drift_metrics": {"mae": drift_mae, "rmse": drift_rmse},
        "figures": fig_paths
    }


if __name__ == "__main__":
    run_full_experiment()
