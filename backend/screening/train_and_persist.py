#!/usr/bin/env python3
"""
TRAINING & MODEL ARTIFACT PERSISTENCE ORCHESTRATOR (Phase 9 & 12).
Trains Module A (Dynamic Outlier Detector), Module B (168h Drift Predictor),
and fits the Dynamic Safety Envelope on reference data.
Saves all artifacts into models/ with version metadata.
Generates comprehensive comparative validation metrics across the 4 paradigms.
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, precision_recall_curve, auc

from typing import Dict, List, Tuple
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "burnin_semiconductor_screening.csv"
MODELS_DIR = PROJECT_ROOT / "models"

from backend.screening.module_a import DynamicOutlierDetector
from backend.screening.module_b import TimeSeriesDriftPredictor
from backend.screening.safety_envelope import DynamicSafetyEnvelope
from backend.screening.decision_engine import DecisionEngine
from backend.screening.artifact_manager import ModelArtifactManager


def train_and_persist_all(seed: int = 42) -> Dict:
    print("=" * 75)
    print(" AI-ESS GUARDIAN: MODEL TRAINING & ARTIFACT PERSISTENCE PIPELINE")
    print("=" * 75)
    t0 = time.time()

    # 1. Ingest Dataset
    if not DATA_PATH.exists():
        from data.synthetic.generate_synthetic_data import generate_burnin_dataset
        print("[1/5] Benchmark dataset missing. Generating synthetic dataset...")
        df_raw = generate_burnin_dataset(seed=seed)
    else:
        print(f"[1/5] Ingesting dataset from {DATA_PATH}...")
        df_raw = pd.read_csv(DATA_PATH)

    print(f"      Total records: {len(df_raw)} across {df_raw['component_id'].nunique()} components.")

    # 2. Extract Multi-Checkpoint Trajectories
    print("[2/5] Preparing feature representations...")
    feat_df = DynamicOutlierDetector.extract_features_from_panel(df_raw, t_early=24.0)
    wide_df, y_targets = TimeSeriesDriftPredictor.prepare_training_features(df_raw)

    # Component-level ground truth (true_latent_defect)
    gt_map = df_raw.groupby("component_id")["true_latent_defect"].max().to_dict()
    y_true = np.array([gt_map.get(cid, 0) for cid in feat_df["component_id"]])

    # 3. Fit Models
    print("[3/5] Fitting Module A, Module B, and Dynamic Safety Envelope...")
    # Reference training cohort: Healthy stable and mild drift components
    healthy_mask = np.array([df_raw[df_raw["component_id"] == cid]["archetype"].iloc[0] in ["HEALTHY_STABLE", "MILD_DRIFT"] for cid in feat_df["component_id"]])
    ref_wide_df = wide_df[healthy_mask]

    # Fit Module A
    mod_a = DynamicOutlierDetector(random_state=seed)
    mod_a.fit(feat_df)

    # Fit Module B
    mod_b = TimeSeriesDriftPredictor(random_state=seed)
    mod_b.fit(wide_df, y_targets)

    # Fit Dynamic Safety Envelope
    safety_env = DynamicSafetyEnvelope(
        target_parameter="leakage_current_ua",
        engineering_limit=50.0,
        prototype_percentile=95.0
    )
    safety_env.fit(
        ref_wide_df,
        val_0h_col="val_0h",
        val_24h_col="val_24h",
        val_168h_col="val_168h_target"
    )

    # 4. Save Artifacts to models/
    print("[4/5] Serializing artifacts into models/...")
    artifact_mgr = ModelArtifactManager(MODELS_DIR)
    meta = {
        "training_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "burnin_semiconductor_screening.csv",
        "n_components": int(len(feat_df)),
        "n_reference_healthy": int(len(ref_wide_df)),
        "target_parameter": "leakage_current_ua",
        "datasheet_spec_limit": 50.0,
        "prototype_safety_threshold": safety_env.data_driven_prototype_limit_,
        "early_slope_max": safety_env.healthy_max_early_slope_,
        "drift_model_mae": round(mod_b.mae_train_, 4),
        "drift_model_rmse": round(mod_b.rmse_train_, 4)
    }
    saved_files = artifact_mgr.save_artifacts(mod_a, mod_b, safety_env, metadata=meta)

    # 5. Evaluate Multi-Paradigm Comparison Table
    print("[5/5] Evaluating Baseline vs Hybrid Multi-Paradigm Comparison...")
    # Level 1: Traditional Static Screening (Leakage at 24h >= 50.0 uA)
    y_trad = (feat_df["leakage_24h"] >= 50.0).astype(int).values

    # Level 2: Robust Statistical Screening (Robust Z >= 3.0)
    mod_a_res_df = mod_a.evaluate_batch(df_raw)
    y_stat = mod_a_res_df["statistical_outlier_flag"].values

    # Level 3: AI/ML Anomaly Detection (Isolation Forest)
    y_aiml = mod_a_res_df["ml_anomaly_flag"].values

    # Level 4: Time-Series Drift Forecast
    mod_b_res_df = mod_b.predict_batch(wide_df)
    y_drift = mod_b_res_df["breaches_spec_limit"].values

    # Level 5 / Hybrid: Dynamic Safety Envelope + Decision Engine
    env_eval_records = []
    for idx in range(len(feat_df)):
        v0 = feat_df["leakage_0h"].iloc[idx]
        v24 = feat_df["leakage_24h"].iloc[idx]
        p168 = mod_b_res_df["predicted_168h"].iloc[idx]
        env_eval_records.append(safety_env.evaluate(v0, v24, p168))
    env_df = pd.DataFrame(env_eval_records)

    decision_engine = DecisionEngine()
    decision_df = decision_engine.evaluate_batch(mod_a_res_df, mod_b_res_df, env_df)
    # Binary rejection or review (safety screening flags)
    y_hybrid = (decision_df["final_decision"].isin(["REJECT", "REVIEW"])).astype(int).values

    # Metrics computation function
    def compute_metrics(y_true, y_pred, scores=None):
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        recall = tp / max(tp + fn, 1)
        fnr = 1.0 - recall
        prec = tp / max(tp + fp, 1)
        f1 = (2 * prec * recall) / max(prec + recall, 1e-6)
        fpr = fp / max(fp + tn, 1)

        pr_auc = 0.0
        if scores is not None and len(np.unique(y_true)) > 1:
            try:
                p_arr, r_arr, _ = precision_recall_curve(y_true, scores)
                pr_auc = auc(r_arr, p_arr)
            except Exception:
                pr_auc = 0.0

        return {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Recall": round(recall * 100.0, 1),
            "FNR": round(fnr * 100.0, 1),
            "Precision": round(prec * 100.0, 1),
            "F1_Score": round(f1, 4),
            "FPR": round(fpr * 100.0, 1),
            "PR_AUC": round(pr_auc, 4)
        }

    comparison_rows = [
        {"Paradigm": "BASELINE 1: Static Datasheet Limits (Traditional)", **compute_metrics(y_true, y_trad, feat_df["leakage_24h"] / 50.0)},
        {"Paradigm": "BASELINE 2: Robust Statistical Screening (MAD Z-Score)", **compute_metrics(y_true, y_stat, mod_a_res_df["max_robust_z"] / 3.0)},
        {"Paradigm": "MODEL: ML Anomaly Detection (Isolation Forest)", **compute_metrics(y_true, y_aiml, mod_a_res_df["anomaly_score"])},
        {"Paradigm": "HYBRID: AI-ESS GUARDIAN (Static + Anomaly + Drift + Envelope)", **compute_metrics(y_true, y_hybrid, decision_df["drift_risk_score"])}
    ]

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = MODELS_DIR / "comparison_benchmark.csv"
    comp_df.to_csv(comp_csv_path, index=False)

    duration = time.time() - t0
    print("\n" + "=" * 80)
    print(" MASTER COMPARATIVE VALIDATION BENCHMARK (90 COMPONENTS)")
    print("=" * 80)
    print(comp_df[["Paradigm", "Recall", "FNR", "Precision", "F1_Score", "FPR", "PR_AUC"]].to_string(index=False))
    print("-" * 80)
    print(f"Drift Model In-Fold MAE: {mod_b.mae_train_:.3f} uA, RMSE: {mod_b.rmse_train_:.3f} uA")
    print(f"Prototype Safety Limit:  {safety_env.data_driven_prototype_limit_:.2f} uA (Datasheet: 50.00 uA)")
    print(f"Training completed in {duration:.2f}s. All models persisted to: {MODELS_DIR}")
    print("=" * 80 + "\n")

    return {
        "saved_files": saved_files,
        "comparison": comp_df.to_dict(orient="records"),
        "models": {
            "module_a": mod_a,
            "module_b": mod_b,
            "safety_envelope": safety_env,
            "decision_engine": decision_engine
        }
    }


if __name__ == "__main__":
    train_and_persist_all()
