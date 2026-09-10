#!/usr/bin/env python3
"""
Publication-Quality Visualization Generator for Predictive ESS Experiment (Step 4).
Generates 12 clean, publication-ready figures saved in results/step4/figures/:
1. Traditional vs AI/ML performance comparison (bar chart)
2. Confusion matrix — Traditional
3. Confusion matrix — AI/ML
4. Precision-Recall curves
5. ROC curves
6. Anomaly-score distribution
7. Component risk ranking
8. Actual vs predicted future values
9. Drift trajectories with forecasts
10. Early-warning lead time timeline
11. Feature importance / physical precursor attribution
12. Multi-method decision comparison heatmap
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_all_figures(
    comparison_df: pd.DataFrame,
    component_results_df: pd.DataFrame,
    drift_features_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    output_dir: Path
) -> List[str]:
    """
    Generates all 12 publication-grade figures using actual experiment outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # Common styling configuration
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--"
    })

    # Color palette
    palette = ["#d97706", "#f59e0b", "#0284c7", "#7c3aed", "#8b5cf6", "#059669", "#dc2626"]

    # -------------------------------------------------------------
    # Fig 1: Performance Comparison Bar Chart
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    methods = comparison_df["method"].tolist()
    x = np.arange(len(methods))
    width = 0.20

    metrics = ["recall", "precision", "f1_score", "false_positive_rate"]
    metric_labels = ["Recall (Defect Detection)", "Precision", "F1-Score", "FPR (Yield Loss)"]
    m_colors = ["#2563eb", "#059669", "#7c3aed", "#e11d48"]

    for i, (m, label, color) in enumerate(zip(metrics, metric_labels, m_colors)):
        vals = comparison_df[m].values
        rects = ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, alpha=0.88, edgecolor="black", linewidth=0.5)
        for rect in rects:
            h = rect.get_height()
            if h > 0.05:
                ax.annotate(f"{h:.2f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, rotation=45)

    ax.set_ylabel("Metric Value (0.0 to 1.0)")
    ax.set_title("Figure 1: Comparative Screening Performance Across Methods on Identical Unseen Components", pad=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontweight="medium")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    f1_path = output_dir / "fig1_method_comparison_bars.png"
    plt.savefig(f1_path)
    plt.close()
    generated_files.append(str(f1_path))

    # -------------------------------------------------------------
    # Fig 2: Confusion Matrix — Traditional
    # -------------------------------------------------------------
    trad_m = comparison_df[comparison_df["method"].str.contains("Traditional", case=False)].iloc[0]
    cm_trad = np.array([[trad_m["true_negatives"], trad_m["false_positives"]],
                        [trad_m["false_negatives"], trad_m["true_positives"]]])

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    cax = ax.matshow(cm_trad, cmap="Blues", alpha=0.85)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm_trad[i, j])), ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if cm_trad[i, j] > 3 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: PASS", "Pred: REJECT"])
    ax.set_yticklabels(["True: SURVIVOR", "True: DEFECT"])
    ax.set_title("Figure 2: Confusion Matrix — Traditional Screening\n(100% False Negative Escape Rate)", pad=14, fontweight="bold")
    plt.colorbar(cax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    f2_path = output_dir / "fig2_confusion_matrix_traditional.png"
    plt.savefig(f2_path)
    plt.close()
    generated_files.append(str(f2_path))

    # -------------------------------------------------------------
    # Fig 3: Confusion Matrix — AI/ML & Risk Fusion
    # -------------------------------------------------------------
    ml_row = comparison_df[comparison_df["method"].str.contains("Risk Fusion|Isolation", case=False)].iloc[0]
    cm_ml = np.array([[ml_row["true_negatives"], ml_row["false_positives"]],
                      [ml_row["false_negatives"], ml_row["true_positives"]]])

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=300)
    cax = ax.matshow(cm_ml, cmap="Greens", alpha=0.85)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm_ml[i, j])), ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if cm_ml[i, j] > 3 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: PASS", "Pred: REJECT"])
    ax.set_yticklabels(["True: SURVIVOR", "True: DEFECT"])
    ax.set_title(f"Figure 3: Confusion Matrix — {ml_row['method']}\n(Early Latent Defect Detection)", pad=14, fontweight="bold")
    plt.colorbar(cax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    f3_path = output_dir / "fig3_confusion_matrix_aiml.png"
    plt.savefig(f3_path)
    plt.close()
    generated_files.append(str(f3_path))

    # -------------------------------------------------------------
    # Fig 4: Precision-Recall Comparison
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for idx, row in comparison_df.iterrows():
        rec = row["recall"]
        prec = row["precision"]
        ax.scatter(rec, prec, s=140, color=palette[idx % len(palette)], label=f"{row['method']} (PR-AUC: {row['pr_auc']:.2f})", zorder=3, edgecolors="black")
        ax.annotate(row["method"], (rec, prec), textcoords="offset points", xytext=(8, -4), fontsize=8.5)

    ax.set_xlabel("Recall (Latent Defect Detection)")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Figure 4: Precision vs Recall Across Screening Paradigms", pad=12, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    f4_path = output_dir / "fig4_precision_recall_curves.png"
    plt.savefig(f4_path)
    plt.close()
    generated_files.append(str(f4_path))

    # -------------------------------------------------------------
    # Fig 5: ROC Comparison (FPR vs Recall)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")
    for idx, row in comparison_df.iterrows():
        fpr = row["false_positive_rate"]
        tpr = row["recall"]
        ax.scatter(fpr, tpr, s=140, color=palette[idx % len(palette)], label=f"{row['method']} (ROC-AUC: {row['roc_auc']:.2f})", zorder=3, edgecolors="black")
        ax.annotate(row["method"], (fpr, tpr), textcoords="offset points", xytext=(8, -4), fontsize=8.5)

    ax.set_xlabel("False Positive Rate (Yield Loss)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Figure 5: Operating Characteristics (Yield Loss vs Defect Recall)", pad=12, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    f5_path = output_dir / "fig5_roc_curves.png"
    plt.savefig(f5_path)
    plt.close()
    generated_files.append(str(f5_path))

    # -------------------------------------------------------------
    # Fig 6: Anomaly Score Distribution (Survivor vs Latent Defects)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    cids = component_results_df["component_id"].tolist()
    scores = component_results_df["iforest_anomaly_score"].values
    gts = component_results_df["ground_truth_latent_risk"].values
    thresh = component_results_df["iforest_threshold"].iloc[0]

    colors = ["#059669" if g == 0 else "#dc2626" for g in gts]
    bars = ax.bar(cids, scores, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    ax.axhline(thresh, color="#2563eb", linestyle="--", linewidth=1.5, label=f"Anomaly Threshold ({thresh:.3f})")

    for bar, score in zip(bars, scores):
        ax.annotate(f"{score:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, score),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="medium")

    ax.set_ylabel("Isolation Forest Anomaly Score")
    ax.set_xlabel("Component ID (Green = Safe Survivor, Red = Latent Defect Risk)")
    ax.set_title("Figure 6: Unsupervised Anomaly Score Separation at Screening Cutoff (t = 47.0h)", pad=12, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    f6_path = output_dir / "fig6_anomaly_score_distribution.png"
    plt.savefig(f6_path)
    plt.close()
    generated_files.append(str(f6_path))

    # -------------------------------------------------------------
    # Fig 7: Component Risk Ranking
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    sorted_df = component_results_df.sort_values("fusion_risk_score", ascending=True)
    c_bars = ["#059669" if r == "LOW RISK" else ("#d97706" if r == "MEDIUM RISK" else "#dc2626") for r in sorted_df["risk_tier"]]
    ax.barh(sorted_df["component_id"], sorted_df["fusion_risk_score"], color=c_bars, edgecolor="black", alpha=0.85, height=0.5)

    for i, (_, row) in enumerate(sorted_df.iterrows()):
        ax.text(row["fusion_risk_score"] + 0.02, i, f"{row['risk_tier']} ({row['fusion_risk_score']:.2f})", va="center", fontsize=9, fontweight="medium")

    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Fused Multi-Paradigm Risk Score [0.0 to 1.0]")
    ax.set_title("Figure 7: Component Prioritization & Risk Tier Ranking", pad=12, fontweight="bold")
    plt.tight_layout()
    f7_path = output_dir / "fig7_component_risk_ranking.png"
    plt.savefig(f7_path)
    plt.close()
    generated_files.append(str(f7_path))

    # -------------------------------------------------------------
    # Fig 8: Actual vs Predicted Future Values (Delta C at 194h)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    actuals = component_results_df["actual_delta_c_194h"].values
    preds = component_results_df["predicted_delta_c_194h"].values
    c_labels = component_results_df["component_id"].tolist()

    ax.scatter(actuals, preds, s=120, color="#7c3aed", edgecolors="black", zorder=4)
    for c, act, pred in zip(c_labels, actuals, preds):
        ax.annotate(c, (act, pred), textcoords="offset points", xytext=(8, -4), fontsize=9, fontweight="bold")

    # Diagonal unity line
    min_v = min(min(actuals), min(preds)) - 1
    max_v = max(max(actuals), max(preds)) + 1
    ax.plot([min_v, max_v], [min_v, max_v], "k--", alpha=0.6, label="Perfect Forecast (1:1)")
    ax.axhline(20.0, color="#dc2626", linestyle=":", label="MIL-PRF-62F Limit (20.0%)")
    ax.axvline(20.0, color="#dc2626", linestyle=":")

    ax.set_xlabel("Actual Future Delta C (%) at 194h")
    ax.set_ylabel("Predicted Future Delta C (%) at 194h (from t <= 47h)")
    ax.set_title("Figure 8: Early Drift Trajectory Forecast Accuracy", pad=12, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    f8_path = output_dir / "fig8_actual_vs_predicted_drift.png"
    plt.savefig(f8_path)
    plt.close()
    generated_files.append(str(f8_path))

    # -------------------------------------------------------------
    # Fig 9: Longitudinal Trajectories with Early Cutoff Line
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    c_colors = {"C1": "#059669", "C2": "#ea580c", "C3": "#d97706", "C4": "#dc2626", "C5": "#9333ea", "C6": "#4f46e5"}

    for cid in sorted(processed_df["component_id"].unique()):
        sub = processed_df[processed_df["component_id"] == cid].sort_values("aging_time_hours")
        ax.plot(sub["aging_time_hours"], sub["delta_capacitance_pct"], marker="o", markersize=4,
                linewidth=1.8, label=f"Unit {cid} ({'Survivor' if cid=='C1' else 'Failed'})", color=c_colors.get(cid, "black"))

    # Cutoff vertical line
    ax.axvline(47.0, color="#2563eb", linestyle="--", linewidth=2.0, label="Screening Cutoff (t = 47.0h)")
    ax.axhline(20.0, color="#dc2626", linestyle="-", linewidth=1.5, label="MIL-PRF-62F Specification Limit (20.0%)")

    # Shaded screening phase
    ax.axvspan(0, 47.0, alpha=0.10, color="#2563eb", label="Early Screening Information Horizon")

    ax.set_xlabel("Elapsed Accelerated Stress Time (Hours)")
    ax.set_ylabel("Percentage Capacitance Drop (Delta C %)")
    ax.set_title("Figure 9: Longitudinal Degradation Curves & Early Screening Horizon", pad=12, fontweight="bold")
    ax.set_xlim(-5, 205)
    ax.set_ylim(-1, 25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    f9_path = output_dir / "fig9_degradation_trajectories_with_forecast.png"
    plt.savefig(f9_path)
    plt.close()
    generated_files.append(str(f9_path))

    # -------------------------------------------------------------
    # Fig 10: Early Warning Lead Time Timeline
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    defects = component_results_df[component_results_df["ground_truth_latent_risk"] == 1].sort_values("component_id")
    y_pos = np.arange(len(defects))

    for idx, (_, row) in enumerate(defects.iterrows()):
        t_fail = row["future_failure_time_h"]
        t_scr = 47.0
        lead = t_fail - t_scr
        ax.barh(idx, lead, left=t_scr, color="#2563eb", edgecolor="black", alpha=0.8, height=0.45)
        ax.scatter([t_scr], [idx], color="#059669", s=80, zorder=3, marker="s")
        ax.scatter([t_fail], [idx], color="#dc2626", s=90, zorder=3, marker="X")
        ax.text(t_scr + lead / 2, idx + 0.22, f"Lead Time: {lead:.0f}h", ha="center", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(defects["component_id"], fontweight="medium")
    ax.set_xlabel("Stress Time Horizon (Hours)")
    ax.set_xlim(0, 215)
    ax.set_title("Figure 10: Early-Warning Lead Time Provided by AI/ML Prior to Failure", pad=12, fontweight="bold")

    # Custom legend elements
    p_green = ax.scatter([], [], color="#059669", marker="s", s=60, label="AI Screening Detection (47.0h)")
    p_red = ax.scatter([], [], color="#dc2626", marker="X", s=70, label="Physical Spec Breach (171h–194h)")
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    f10_path = output_dir / "fig10_early_lead_time_timeline.png"
    plt.savefig(f10_path)
    plt.close()
    generated_files.append(str(f10_path))

    # -------------------------------------------------------------
    # Fig 11: Feature Importance & Physical Precursor Attribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    dev_cols = [c for c in component_results_df.columns if c.startswith("dev_")]
    if dev_cols:
        feat_labels = [c.replace("dev_", "") for c in dev_cols]
        mean_devs = [component_results_df[c].mean() for c in dev_cols]
        sorted_indices = np.argsort(mean_devs)
        ax.barh(range(len(sorted_indices)), [mean_devs[i] for i in sorted_indices], color="#0284c7", edgecolor="black", alpha=0.85, height=0.5)
        ax.set_yticks(range(len(sorted_indices)))
        ax.set_yticklabels([feat_labels[i] for i in sorted_indices], fontweight="medium")
        ax.set_xlabel("Mean Standardized Deviation Across Defect Components (Z-Score)")
        ax.set_title("Figure 11: Kinetic Precursor Feature Attribution Driving Early Anomaly Detection", pad=12, fontweight="bold")
    plt.tight_layout()
    f11_path = output_dir / "fig11_feature_importance_attribution.png"
    plt.savefig(f11_path)
    plt.close()
    generated_files.append(str(f11_path))

    # -------------------------------------------------------------
    # Fig 12: Multi-Method Decision Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    comp_list = component_results_df["component_id"].tolist()
    decision_matrix = np.array([
        component_results_df["traditional_decision"].values,
        component_results_df["traditional_decision_tightened"].values,
        component_results_df["statistical_decision"].values,
        component_results_df["aiml_decision"].values,
        component_results_df["drift_decision"].values,
        component_results_df["fusion_decision"].values,
        component_results_df["ground_truth_latent_risk"].values
    ])
    method_labels = [
        "Traditional (20% Spec)",
        "Traditional (5% Tightened)",
        "Dynamic Statistical",
        "AI/ML Anomaly (iForest)",
        "Early Drift Forecast",
        "Risk Fusion Engine",
        "Ground Truth Proxy"
    ]

    cax = ax.imshow(decision_matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    for i in range(len(method_labels)):
        for j in range(len(comp_list)):
            val = int(decision_matrix[i, j])
            label_text = "REJECT" if val == 1 else "PASS"
            ax.text(j, i, label_text, ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="white" if val == 1 else "black")

    ax.set_xticks(range(len(comp_list)))
    ax.set_xticklabels(comp_list, fontweight="bold")
    ax.set_yticks(range(len(method_labels)))
    ax.set_yticklabels(method_labels, fontweight="medium")
    ax.set_title("Figure 12: Decision Matrix by Component Across All 5 Screening Paradigms", pad=12, fontweight="bold")
    plt.tight_layout()
    f12_path = output_dir / "fig12_decision_boundary_comparison.png"
    plt.savefig(f12_path)
    plt.close()
    generated_files.append(str(f12_path))

    return generated_files
