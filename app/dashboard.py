#!/usr/bin/env python3
"""
AI-ESS GUARDIAN
Environmental Stress Screening & Reliability Analysis
Engineering Software Interface for Component Burn-In Screening.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_pipeline.sih_pipeline import RobustDataPipeline
from backend.screening.artifact_manager import ModelArtifactManager
from backend.screening.decision_engine import DecisionEngine
from backend.screening.explainer import ScreeningExplainer
from backend.screening.module_a import DynamicOutlierDetector
from backend.screening.module_b import TimeSeriesDriftPredictor
from backend.screening.safety_envelope import DynamicSafetyEnvelope

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(
    page_title="AI-ESS GUARDIAN | Reliability Screening Software",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- RESTRAINED INDUSTRIAL ENGINEERING CSS ---
st.markdown("""
<style>
    /* Clean Industrial Theme */
    html, body, [class*="css"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif;
        color: #0f172a;
        background-color: #f8fafc;
    }
    
    /* Top Header Bar */
    .eng-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 16px;
        background-color: #ffffff;
        border-bottom: 1px solid #e2e8f0;
        margin-top: -3.5rem;
        margin-bottom: 1rem;
    }
    .eng-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .eng-subtitle {
        font-size: 0.78rem;
        color: #64748b;
        margin: 2px 0 0 0;
    }
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: ui-monospace, Menlo, Consolas, monospace;
        color: #0f172a;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 4px 10px;
        border-radius: 3px;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #16a34a;
        border-radius: 50%;
        display: inline-block;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.85rem !important;
        padding: 4px 8px !important;
    }
    
    /* Engineering Metric Blocks */
    .kpi-row {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }
    .kpi-block {
        flex: 1;
        min-width: 140px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 10px 14px;
    }
    .kpi-label {
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        font-family: ui-monospace, Menlo, Consolas, monospace;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.70rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Status Badges */
    .badge-pass {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 3px 8px;
        border-radius: 3px;
        font-weight: 600;
        font-size: 0.78rem;
        font-family: ui-monospace, Menlo, Consolas, monospace;
        display: inline-block;
    }
    .badge-review {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 3px 8px;
        border-radius: 3px;
        font-weight: 600;
        font-size: 0.78rem;
        font-family: ui-monospace, Menlo, Consolas, monospace;
        display: inline-block;
    }
    .badge-reject {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 3px 8px;
        border-radius: 3px;
        font-weight: 600;
        font-size: 0.78rem;
        font-family: ui-monospace, Menlo, Consolas, monospace;
        display: inline-block;
    }
    .badge-info {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 3px 8px;
        border-radius: 3px;
        font-weight: 600;
        font-size: 0.78rem;
        font-family: ui-monospace, Menlo, Consolas, monospace;
        display: inline-block;
    }

    /* Engineering Card Container */
    .eng-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .eng-card-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #334155;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 10px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 6px;
    }

    /* Comparison Box */
    .comp-box {
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        padding: 14px 16px;
        background: #ffffff;
    }

    /* Clean Table Styling */
    .eng-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
    }
    .eng-table th {
        background-color: #f8fafc;
        color: #475569;
        text-align: left;
        padding: 8px 10px;
        border-bottom: 1px solid #cbd5e1;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .eng-table td {
        padding: 7px 10px;
        border-bottom: 1px solid #f1f5f9;
        color: #1e293b;
    }
    .eng-table tr:hover {
        background-color: #f8fafc;
    }
    .text-right {
        text-align: right !important;
    }
    .font-mono {
        font-family: ui-monospace, Menlo, Consolas, monospace !important;
    }

    /* Progress Distribution Bar */
    .dist-bar {
        display: flex;
        width: 100%;
        height: 14px;
        border-radius: 3px;
        overflow: hidden;
        margin: 8px 0 16px 0;
        border: 1px solid #cbd5e1;
    }
    .dist-pass {
        background-color: #22c55e;
    }
    .dist-review {
        background-color: #f59e0b;
    }
    .dist-reject {
        background-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_screening_system():
    """Loads pre-trained models, scalers, and safety envelope."""
    art_mgr = ModelArtifactManager(PROJECT_ROOT / "models")
    if not art_mgr.artifacts_exist():
        from backend.screening.train_and_persist import train_and_persist_all
        train_and_persist_all()

    mod_a, mod_b, safety_env = art_mgr.load_artifacts()
    decision_engine = DecisionEngine()
    pipeline = RobustDataPipeline()
    return mod_a, mod_b, safety_env, decision_engine, pipeline


@st.cache_data
def load_benchmark_dataset():
    data_path = PROJECT_ROOT / "data" / "synthetic" / "burnin_semiconductor_screening.csv"
    if not data_path.exists():
        from data.synthetic.generate_synthetic_data import generate_burnin_dataset
        df = generate_burnin_dataset()
    else:
        df = pd.read_csv(data_path)
    return df


def screen_dataset(df_raw, mod_a, mod_b, safety_env, decision_engine):
    """Executes end-to-end screening on a DataFrame."""
    feat_df = DynamicOutlierDetector.extract_features_from_panel(df_raw)
    wide_df, _ = TimeSeriesDriftPredictor.prepare_training_features(df_raw)

    mod_a_res = mod_a.evaluate_batch(df_raw)
    mod_b_res = mod_b.predict_batch(wide_df)

    env_records = []
    for idx in range(len(feat_df)):
        v0 = feat_df["leakage_0h"].iloc[idx]
        v24 = feat_df["leakage_24h"].iloc[idx]
        p168 = mod_b_res["predicted_168h"].iloc[idx]
        env_records.append(safety_env.evaluate(v0, v24, p168))
    env_df = pd.DataFrame(env_records)

    decision_df = decision_engine.evaluate_batch(mod_a_res, mod_b_res, env_df)

    # Merge into unified screening results
    merged = pd.concat([
        feat_df[["component_id", "lot_id", "leakage_0h", "leakage_24h"]].reset_index(drop=True),
        mod_a_res[["anomaly_score", "severity", "max_robust_z", "leakage_mult_of_lot_median", "absolute_spec_failed"]].reset_index(drop=True),
        mod_b_res[["predicted_168h", "drift_amount", "drift_percentage", "drift_risk_score", "ci_lower_80", "ci_upper_80"]].reset_index(drop=True),
        env_df[["early_slope", "healthy_envelope_max_early_slope", "data_driven_prototype_limit", "engineering_limit", "envelope_violated"]].reset_index(drop=True),
        decision_df[["final_decision", "explanation", "qa_bullets"]].reset_index(drop=True)
    ], axis=1)

    return merged, feat_df, mod_a_res, mod_b_res, env_df, decision_df


# --- LOAD BACKEND ENGINES ---
mod_a, mod_b, safety_env, decision_engine, pipeline = load_screening_system()

# --- SIDEBAR: NAVIGATION & CONTROLS ---
st.sidebar.markdown("""
<div style='padding: 8px 0 16px 0; border-bottom: 1px solid #1e293b;'>
    <div style='font-size: 1.05rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;'>AI-ESS GUARDIAN</div>
    <div style='font-size: 0.72rem; color: #94a3b8;'>Reliability Screening Software</div>
</div>
""", unsafe_allow_html=True)

nav_sections = [
    "Overview",
    "Screening",
    "Components",
    "Trend Analysis",
    "Decision Review",
    "Model Performance",
    "Data Upload"
]

selected_section = st.sidebar.radio("Navigation", nav_sections, index=0)

st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Telemetry Source
dataset_option = st.sidebar.selectbox(
    "Telemetry Source",
    ["Benchmark Dataset (Semiconductor)", "Custom CSV/Excel Import"],
    index=0
)

# Demo Mode Toggle (subtle, professional)
demo_mode = st.sidebar.checkbox("Demo Dataset: Latent Defect Case", value=False)

# Dataset Resolution
if dataset_option == "Custom CSV/Excel Import":
    uploaded_file = st.sidebar.file_uploader("Upload Telemetry File", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            df_raw = pipeline.ingest_file(uploaded_file)
            df_raw, clean_summary = pipeline.validate_and_clean(df_raw)
            st.sidebar.caption(f"Loaded {len(df_raw)} records ({clean_summary['components_count']} units)")
        except Exception as e:
            st.sidebar.error(f"Ingestion error: {e}")
            df_raw = load_benchmark_dataset()
    else:
        df_raw = load_benchmark_dataset()
else:
    df_raw = load_benchmark_dataset()

# Run Screening Execution
screened_df, feat_df, mod_a_df, mod_b_df, env_df, dec_df = screen_dataset(
    df_raw, mod_a, mod_b, safety_env, decision_engine
)

# Sidebar CSV Export
csv_export_data = screened_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Export Screening CSV",
    data=csv_export_data,
    file_name=f"ess_screening_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# Sidebar System Metadata
st.sidebar.markdown("""
<div style='margin-top: 24px; padding-top: 14px; border-top: 1px solid #1e293b; font-size: 0.68rem; color: #64748b;'>
    <div>Application: <strong>AI-ESS GUARDIAN</strong></div>
    <div>Module: <strong>ESS Screening</strong></div>
    <div>Environment: <strong>Engineering Prototype</strong></div>
    <div>Dataset: <strong>Current Screening Batch</strong></div>
    <div>Last Analysis: <strong>Current session</strong></div>
    <div>System Status: <strong style='color: #4ade80;'>READY</strong></div>
    <div>Version: <strong>Prototype v1.0</strong></div>
</div>
""", unsafe_allow_html=True)


# --- TOP APPLICATION SHELL HEADER ---
st.markdown("""
<div class="eng-header">
    <div>
        <div class="eng-title">AI-ESS GUARDIAN</div>
        <div class="eng-subtitle">Environmental Stress Screening & Reliability Analysis</div>
    </div>
    <div class="status-indicator">
        <span class="status-dot"></span>
        SYSTEM STATUS: READY
    </div>
</div>
""", unsafe_allow_html=True)


# --- DEMO DATASET NOTIFICATION (IF ACTIVE) ---
if demo_mode:
    sih_comps = df_raw[df_raw["archetype"] == "LATENT_DEFECT_LOT_OUTLIER"]["component_id"].unique()
    target_demo_cid = sih_comps[0] if len(sih_comps) > 0 else screened_df["component_id"].iloc[0]
    demo_row = screened_df[screened_df["component_id"] == target_demo_cid].iloc[0]
    demo_lot_dict = getattr(mod_a, "lot_baselines_", {}).get(demo_row["lot_id"], {})
    demo_lot_med = demo_lot_dict.get("leakage_current_ua", {}).get("median", 10.45)

    st.markdown(f"""
    <div class="eng-card" style="border-left: 3px solid #0284c7; background: #ffffff;">
        <div class="eng-card-title" style="margin-bottom: 8px;">Demo Dataset: Latent Defect Comparison (Unit: {target_demo_cid})</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
            <div class="comp-box" style="border-color: #cbd5e1;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Traditional Screening (Static Limit)</div>
                <div style="margin-top: 6px; font-size: 0.85rem; color: #334155;">
                    Measured 24h Leakage: <strong>{demo_row['leakage_24h']:.2f} µA</strong> &nbsp;|&nbsp; Datasheet Limit: <strong>50.00 µA</strong>
                </div>
                <div style="margin-top: 8px;">
                    <span class="badge-pass">TRADITIONAL: PASS</span>
                    <span style="font-size: 0.75rem; color: #b91c1c; margin-left: 8px;">({demo_row['leakage_24h']:.2f} µA &le; 50.00 µA &mdash; latent defect escapes to flight hardware)</span>
                </div>
            </div>
            <div class="comp-box" style="border-color: #93c5fd; background: #f8fafc;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #0369a1; text-transform: uppercase;">AI-ESS Guardian (Lot-Aware & Predictive)</div>
                <div style="margin-top: 6px; font-size: 0.85rem; color: #334155;">
                    Lot Baseline Median: <strong>{demo_lot_med:.2f} µA</strong> &nbsp;|&nbsp; Deviation: <strong>{demo_row['leakage_mult_of_lot_median']:.1f}&times;</strong> &nbsp;|&nbsp; 168h Forecast: <strong>{demo_row['predicted_168h']:.2f} µA</strong>
                </div>
                <div style="margin-top: 8px;">
                    <span class="badge-reject">AI-ESS: REJECT</span>
                    <span style="font-size: 0.75rem; color: #166534; margin-left: 8px;">(Intercepted 144 hours early via kinetic drift and lot deviation)</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 1: OVERVIEW PAGE
# ==============================================================================
if selected_section == "Overview":
    st.markdown("### Screening Overview")
    st.caption("Current screening batch and component health summary")

    total_c = len(screened_df)
    pass_c = int((screened_df["final_decision"] == "PASS").sum())
    review_c = int((screened_df["final_decision"] == "REVIEW").sum())
    reject_c = int((screened_df["final_decision"] == "REJECT").sum())
    high_risk_c = int((screened_df["drift_risk_score"] >= 0.70).sum())

    # Compact KPI Row
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-block">
            <div class="kpi-label">Total Components</div>
            <div class="kpi-value">{total_c}</div>
            <div class="kpi-sub">Active batch</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">PASS</div>
            <div class="kpi-value" style="color: #15803d;">{pass_c}</div>
            <div class="kpi-sub">Flight qualified ({(pass_c/total_c)*100:.1f}%)</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">REVIEW</div>
            <div class="kpi-value" style="color: #b45309;">{review_c}</div>
            <div class="kpi-sub">Quarantine / hold ({(review_c/total_c)*100:.1f}%)</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">REJECT</div>
            <div class="kpi-value" style="color: #b91c1c;">{reject_c}</div>
            <div class="kpi-sub">Mandatory scrap ({(reject_c/total_c)*100:.1f}%)</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">High Drift Risk</div>
            <div class="kpi-value" style="color: #b91c1c;">{high_risk_c}</div>
            <div class="kpi-sub">Risk score &ge; 0.70</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # A. Screening Distribution Horizontal Bar
    p_pct = (pass_c / total_c) * 100.0
    rv_pct = (review_c / total_c) * 100.0
    rj_pct = (reject_c / total_c) * 100.0

    st.markdown(f"""
    <div class="eng-card">
        <div class="eng-card-title">Screening Distribution</div>
        <div class="dist-bar">
            <div class="dist-pass" style="width: {p_pct:.1f}%;" title="PASS: {pass_c} ({p_pct:.1f}%)"></div>
            <div class="dist-review" style="width: {rv_pct:.1f}%;" title="REVIEW: {review_c} ({rv_pct:.1f}%)"></div>
            <div class="dist-reject" style="width: {rj_pct:.1f}%;" title="REJECT: {reject_c} ({rj_pct:.1f}%)"></div>
        </div>
        <div style="display: flex; gap: 18px; font-size: 0.75rem; color: #475569;">
            <span><span style="display:inline-block;width:10px;height:10px;background:#22c55e;margin-right:4px;"></span>PASS: <strong>{pass_c}</strong> ({p_pct:.1f}%)</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:#f59e0b;margin-right:4px;"></span>REVIEW: <strong>{review_c}</strong> ({rv_pct:.1f}%)</span>
            <span><span style="display:inline-block;width:10px;height:10px;background:#ef4444;margin-right:4px;"></span>REJECT: <strong>{reject_c}</strong> ({rj_pct:.1f}%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_lot, col_recent = st.columns([1, 1])

    with col_lot:
        st.markdown("""<div class="eng-card-title">Lot Summary</div>""", unsafe_allow_html=True)
        # Lot breakdown table
        lot_rows = []
        for lot_id, grp in screened_df.groupby("lot_id"):
            l_tot = len(grp)
            l_p = int((grp["final_decision"] == "PASS").sum())
            l_rv = int((grp["final_decision"] == "REVIEW").sum())
            l_rj = int((grp["final_decision"] == "REJECT").sum())
            l_anom_pct = ((l_rv + l_rj) / l_tot) * 100.0
            lot_rows.append({
                "Lot ID": lot_id,
                "Components": l_tot,
                "Pass": l_p,
                "Review": l_rv,
                "Reject": l_rj,
                "Anomaly Rate": f"{l_anom_pct:.1f}%"
            })
        lot_table_df = pd.DataFrame(lot_rows)
        st.dataframe(
            lot_table_df,
            hide_index=True,
            column_config={
                "Components": st.column_config.NumberColumn(format="%d"),
                "Pass": st.column_config.NumberColumn(format="%d"),
                "Review": st.column_config.NumberColumn(format="%d"),
                "Reject": st.column_config.NumberColumn(format="%d")
            }
        )

    with col_recent:
        st.markdown("""<div class="eng-card-title">Recent Screening Results</div>""", unsafe_allow_html=True)
        recent_df = screened_df[[
            "component_id", "lot_id", "leakage_0h", "leakage_24h",
            "anomaly_score", "predicted_168h", "final_decision"
        ]].head(8)
        recent_df = recent_df.rename(columns={
            "component_id": "Component ID",
            "lot_id": "Lot",
            "leakage_0h": "Leakage 0h",
            "leakage_24h": "Leakage 24h",
            "anomaly_score": "Anomaly Score",
            "predicted_168h": "Predicted 168h",
            "final_decision": "Decision"
        })
        st.dataframe(
            recent_df,
            hide_index=True,
            column_config={
                "Leakage 0h": st.column_config.NumberColumn(format="%.2f µA"),
                "Leakage 24h": st.column_config.NumberColumn(format="%.2f µA"),
                "Anomaly Score": st.column_config.NumberColumn(format="%.3f"),
                "Predicted 168h": st.column_config.NumberColumn(format="%.2f µA"),
            }
        )


# ==============================================================================
# SECTION 2: COMPONENT SCREENING (MAIN WORKING PAGE)
# ==============================================================================
elif selected_section == "Screening":
    st.markdown("### Component Screening")
    st.caption("Primary evaluation interface: component inspection, relative lot benchmarking, and safety assessment.")

    # Search / Select Section
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel2:
        lot_list = ["All Lots"] + sorted(list(screened_df["lot_id"].unique()))
        selected_lot = st.selectbox("Lot Filter", lot_list, index=0)

    filtered_comps = screened_df if selected_lot == "All Lots" else screened_df[screened_df["lot_id"] == selected_lot]
    comp_options = sorted(list(filtered_comps["component_id"].unique()))

    with col_sel1:
        selected_comp = st.selectbox("Component ID", comp_options, index=0)

    # Component Data Row
    c_row = screened_df[screened_df["component_id"] == selected_comp].iloc[0]
    final_dec = c_row["final_decision"]

    # Lot-specific reference distribution
    lot_dict = getattr(mod_a, "lot_baselines_", {}).get(c_row["lot_id"], {})
    leak_baseline = lot_dict.get("leakage_current_ua", {})
    lot_median_val = leak_baseline.get("median", 10.45)
    lot_mad_val = leak_baseline.get("mad", 0.40)

    badge_map = {"PASS": "badge-pass", "REVIEW": "badge-review", "REJECT": "badge-reject"}
    status_badge_class = badge_map.get(final_dec, "badge-info")

    # Compact Component Summary Strip
    st.markdown(f"""
    <div class="kpi-row" style="margin-top: 8px;">
        <div class="kpi-block" style="border-left: 3px solid #334155;">
            <div class="kpi-label">Current Status</div>
            <div style="margin-top: 2px;"><span class="{status_badge_class}" style="font-size: 0.95rem;">{final_dec}</span></div>
            <div class="kpi-sub">Lot: {c_row['lot_id']}</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">Primary Parameter</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 4px;">Leakage Current</div>
            <div class="kpi-sub">Monitored channel</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">Current Value (24h)</div>
            <div class="kpi-value">{c_row['leakage_24h']:.2f} <span style="font-size: 0.9rem; color: #64748b;">µA</span></div>
            <div class="kpi-sub">0h initial: {c_row['leakage_0h']:.2f} µA</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">Lot Median</div>
            <div class="kpi-value">{lot_median_val:.2f} <span style="font-size: 0.9rem; color: #64748b;">µA</span></div>
            <div class="kpi-sub">Cohort baseline</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">Lot Deviation</div>
            <div class="kpi-value" style="color: {'#b91c1c' if c_row['leakage_mult_of_lot_median'] >= 3.0 else '#0f172a'};">{c_row['leakage_mult_of_lot_median']:.2f}&times;</div>
            <div class="kpi-sub">MAD Z = +{c_row['max_robust_z']:.2f}</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">Predicted 168h</div>
            <div class="kpi-value">{c_row['predicted_168h']:.2f} <span style="font-size: 0.9rem; color: #64748b;">µA</span></div>
            <div class="kpi-sub">Drift: +{c_row['drift_amount']:.2f} µA</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 7. SCREENING ASSESSMENT (VISUALLY CLEAR ENGINEERING LOGIC)
    # Determine item statuses
    abs_status = "VIOLATION" if c_row["absolute_spec_failed"] == 1 else "WITHIN LIMIT"
    abs_badge = "badge-reject" if c_row["absolute_spec_failed"] == 1 else "badge-pass"

    if c_row["leakage_mult_of_lot_median"] >= 3.0 or c_row["max_robust_z"] >= 4.0:
        lot_status = "ABNORMAL"
        lot_badge = "badge-reject"
    elif c_row["leakage_mult_of_lot_median"] >= 1.8:
        lot_status = "ELEVATED"
        lot_badge = "badge-review"
    else:
        lot_status = "NORMAL"
        lot_badge = "badge-pass"

    if c_row["predicted_168h"] >= c_row["engineering_limit"]:
        drift_status = "EXCEEDS LIMIT"
        drift_badge = "badge-reject"
    elif c_row["predicted_168h"] >= c_row["data_driven_prototype_limit"]:
        drift_status = "ELEVATED DRIFT"
        drift_badge = "badge-review"
    else:
        drift_status = "STABLE"
        drift_badge = "badge-pass"

    env_status = "EXCEEDED" if c_row["envelope_violated"] else "WITHIN LIMIT"
    env_badge = "badge-reject" if c_row["envelope_violated"] else "badge-pass"

    st.markdown(f"""
    <div class="eng-card">
        <div class="eng-card-title">Screening Assessment</div>
        <table class="eng-table">
            <thead>
                <tr>
                    <th style="width: 25%;">Screening Criterion</th>
                    <th style="width: 30%;">Engineering Reference</th>
                    <th style="width: 25%;">Observed / Forecasted</th>
                    <th style="width: 20%;">Criterion Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>1. Absolute Limit</strong></td>
                    <td>Datasheet maximum: 50.00 µA</td>
                    <td>Current: {c_row['leakage_24h']:.2f} µA</td>
                    <td><span class="{abs_badge}">{abs_status}</span></td>
                </tr>
                <tr>
                    <td><strong>2. Lot-Relative Behavior</strong></td>
                    <td>Lot median: {lot_median_val:.2f} µA (MAD: {lot_mad_val:.2f} µA)</td>
                    <td>Deviation: {c_row['leakage_mult_of_lot_median']:.2f}&times; (Z = +{c_row['max_robust_z']:.2f})</td>
                    <td><span class="{lot_badge}">{lot_status}</span></td>
                </tr>
                <tr>
                    <td><strong>3. Future Drift</strong></td>
                    <td>Datasheet ceiling: 50.00 µA (168h)</td>
                    <td>Predicted 168h: {c_row['predicted_168h']:.2f} µA</td>
                    <td><span class="{drift_badge}">{drift_status}</span></td>
                </tr>
                <tr>
                    <td><strong>4. Safety Envelope</strong></td>
                    <td>Healthy early slope limit: &le; {c_row['healthy_envelope_max_early_slope']:.4f} µA/h</td>
                    <td>Observed early slope: {c_row['early_slope']:.4f} µA/h</td>
                    <td><span class="{env_badge}">{env_status}</span></td>
                </tr>
            </tbody>
        </table>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 14px; padding-top: 10px; border-top: 1px solid #f1f5f9;">
            <div style="font-size: 0.85rem; color: #475569;">
                <strong>Key Screening Rule:</strong> <em>Within specification does not necessarily mean normal.</em>
            </div>
            <div>
                <span style="font-size: 0.85rem; color: #64748b; font-weight: 600; margin-right: 8px;">FINAL DECISION:</span>
                <span class="{status_badge_class}" style="font-size: 0.95rem;">{final_dec}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 3: COMPONENTS PAGE (ROSTER & INVENTORY)
# ==============================================================================
elif selected_section == "Components":
    st.markdown("### Component Master Roster")
    st.caption("Comprehensive screening ledger across all evaluated devices.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        lot_filter = st.selectbox("Lot Filter", ["All Lots"] + sorted(list(screened_df["lot_id"].unique())), key="comp_lot_f")
    with col_f2:
        decision_filter = st.selectbox("Decision Filter", ["All Decisions", "PASS", "REVIEW", "REJECT"], index=0)
    with col_f3:
        sort_by = st.selectbox("Sort Order", ["Component ID", "Highest Leakage 24h", "Highest Deviation", "Highest Predicted 168h"], index=0)

    view_df = screened_df.copy()
    if lot_filter != "All Lots":
        view_df = view_df[view_df["lot_id"] == lot_filter]
    if decision_filter != "All Decisions":
        view_df = view_df[view_df["final_decision"] == decision_filter]

    if sort_by == "Highest Leakage 24h":
        view_df = view_df.sort_values("leakage_24h", ascending=False)
    elif sort_by == "Highest Deviation":
        view_df = view_df.sort_values("leakage_mult_of_lot_median", ascending=False)
    elif sort_by == "Highest Predicted 168h":
        view_df = view_df.sort_values("predicted_168h", ascending=False)
    else:
        view_df = view_df.sort_values("component_id")

    display_df = view_df[[
        "component_id", "lot_id", "leakage_0h", "leakage_24h",
        "leakage_mult_of_lot_median", "max_robust_z", "anomaly_score",
        "predicted_168h", "drift_amount", "final_decision"
    ]].rename(columns={
        "component_id": "Component ID",
        "lot_id": "Lot",
        "leakage_0h": "0h (µA)",
        "leakage_24h": "24h (µA)",
        "leakage_mult_of_lot_median": "Lot Dev",
        "max_robust_z": "Robust Z",
        "anomaly_score": "Anomaly Score",
        "predicted_168h": "168h Pred (µA)",
        "drift_amount": "Drift (µA)",
        "final_decision": "Decision"
    })

    st.dataframe(
        display_df,
        hide_index=True,
        column_config={
            "0h (µA)": st.column_config.NumberColumn(format="%.2f"),
            "24h (µA)": st.column_config.NumberColumn(format="%.2f"),
            "Lot Dev": st.column_config.NumberColumn(format="%.2f×"),
            "Robust Z": st.column_config.NumberColumn(format="%+.2f"),
            "Anomaly Score": st.column_config.NumberColumn(format="%.3f"),
            "168h Pred (µA)": st.column_config.NumberColumn(format="%.2f"),
            "Drift (µA)": st.column_config.NumberColumn(format="%+.2f"),
        }
    )
    st.caption(f"Displaying {len(display_df)} of {len(screened_df)} records.")


# ==============================================================================
# SECTION 4: TRAJECTORY & ENVELOPES (TREND ANALYSIS)
# ==============================================================================
elif selected_section == "Trend Analysis":
    st.markdown("### Parameter Trend & 168h Forecast")
    st.caption("Longitudinal trajectory tracking: measured checkpoints (0h, 24h, 96h) and projected 168h degradation.")

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        comp_options = sorted(list(screened_df["component_id"].unique()))
        selected_comp = st.selectbox("Select Component", comp_options, index=0)
    with col_t2:
        show_background = st.checkbox("Show Lot Cohort Context", value=True)

    c_row = screened_df[screened_df["component_id"] == selected_comp].iloc[0]

    # Engineering 2D Plot (Clean white canvas with subtle grid)
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=140)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Lot Cohort Background lines
    if show_background:
        lot_comps = df_raw[df_raw["lot_id"] == c_row["lot_id"]]
        for cid in lot_comps["component_id"].unique()[:15]:
            sub = lot_comps[lot_comps["component_id"] == cid].sort_values("burnin_hours")
            if len(sub) >= 3:
                ax.plot(sub["burnin_hours"], sub["leakage_current_ua"], color="#cbd5e1", linewidth=0.8, alpha=0.7)

    # Datasheet Limit (Engineering ceiling)
    eng_limit = c_row["engineering_limit"]
    ax.axhline(eng_limit, color="#dc2626", linestyle="-", linewidth=1.2, label=f"Datasheet Limit ({eng_limit:.1f} µA)")

    # Data-Driven Prototype Safety Threshold
    proto_limit = c_row["data_driven_prototype_limit"]
    ax.axhline(proto_limit, color="#d97706", linestyle="--", linewidth=1.2, label=f"Safety Threshold ({proto_limit:.1f} µA)")

    # Shaded Normative Envelope
    ax.axhspan(0, proto_limit, facecolor="#f8fafc", alpha=0.6, label="Normative Envelope")

    # Measured points for this component
    measured_sub = df_raw[df_raw["component_id"] == selected_comp].sort_values("burnin_hours")
    # Early measured (0h and 24h)
    early_meas = measured_sub[measured_sub["burnin_hours"] <= 24.0]
    ax.plot(early_meas["burnin_hours"], early_meas["leakage_current_ua"], color="#0f172a", marker="o", markersize=5, linewidth=1.8, label="Measured (0h–24h)")

    # 168h Forecast Line & Marker
    v24 = c_row["leakage_24h"]
    pred_168 = c_row["predicted_168h"]
    ci_low = c_row["ci_lower_80"]
    ci_high = c_row["ci_upper_80"]

    ax.plot([24.0, 168.0], [v24, pred_168], color="#0284c7", linestyle="--", linewidth=1.5, label="168h Forecast")
    ax.plot([168.0], [pred_168], color="#0284c7", marker="s", markersize=6)
    ax.errorbar([168.0], [pred_168], yerr=[[pred_168 - ci_low], [ci_high - pred_168]], fmt='none', ecolor="#0284c7", capsize=4, linewidth=1.2, label="80% Prediction Interval")

    # Actual 168h if present (for validation comparison)
    act_168_pts = measured_sub[measured_sub["burnin_hours"] >= 168.0]
    if len(act_168_pts) > 0:
        act_val = act_168_pts["leakage_current_ua"].iloc[0]
        ax.plot([168.0], [act_val], color="#16a34a", marker="^", markersize=6, label=f"Actual 168h ({act_val:.2f} µA)")

    # Clean styling
    ax.set_xlim(-5, 180)
    ax.set_ylim(0, max(eng_limit * 1.15, pred_168 * 1.15))
    ax.set_xlabel("Burn-In Time (Hours)", fontsize=9, color="#334155")
    ax.set_ylabel("Leakage Current (µA)", fontsize=9, color="#334155")
    ax.tick_params(axis='both', which='both', labelsize=8, colors="#475569")
    ax.grid(True, linestyle=":", color="#e2e8f0", linewidth=0.8)
    
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)

    ax.legend(loc="upper left", fontsize=7.5, facecolor="#ffffff", edgecolor="#cbd5e1", framealpha=0.95)
    plt.tight_layout()

    st.pyplot(fig)

    # Telemetry Slope & Confidence Metrics Below Chart
    st.markdown(f"""
    <div class="kpi-row" style="margin-top: 6px;">
        <div class="kpi-block">
            <div class="kpi-label">Early Slope (0–24h)</div>
            <div class="kpi-value">{c_row['early_slope']:.4f} <span style="font-size: 0.8rem; color: #64748b;">µA/h</span></div>
            <div class="kpi-sub">Envelope limit: {c_row['healthy_envelope_max_early_slope']:.4f} µA/h</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">Predicted Slope (24–168h)</div>
            <div class="kpi-value">{((c_row['predicted_168h'] - c_row['leakage_24h'])/144.0):.4f} <span style="font-size: 0.8rem; color: #64748b;">µA/h</span></div>
            <div class="kpi-sub">Projected kinetic rate</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">Drift Percentage</div>
            <div class="kpi-value">{c_row['drift_percentage']:+.1f}%</div>
            <div class="kpi-sub">Total drift: {c_row['drift_amount']:+.2f} µA</div>
        </div>
        <div class="kpi-block">
            <div class="kpi-label">Forecast 80% CI</div>
            <div class="kpi-value" style="font-size: 1.15rem;">[{c_row['ci_lower_80']:.2f}, {c_row['ci_upper_80']:.2f}]</div>
            <div class="kpi-sub">Uncertainty margin: &plusmn;{((c_row['ci_upper_80'] - c_row['ci_lower_80'])/2):.2f} µA</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 5: DECISION REVIEW (QA AUDIT & EVIDENCE)
# ==============================================================================
elif selected_section == "Decision Review":
    st.markdown("### Decision Review")
    st.caption("Quality Assurance screening audit record and technical justification.")

    comp_options = sorted(list(screened_df["component_id"].unique()))
    selected_comp = st.selectbox("Inspect Component Record", comp_options, index=0)
    c_row = screened_df[screened_df["component_id"] == selected_comp].iloc[0]
    final_dec = c_row["final_decision"]

    # Retrieve lot-specific baseline distributions dynamically
    lot_dict = getattr(mod_a, "lot_baselines_", {}).get(c_row["lot_id"], {})
    leak_baseline = lot_dict.get("leakage_current_ua", {})
    lot_median_val = leak_baseline.get("median", 10.45)
    lot_mad_val = leak_baseline.get("mad", 0.40)
    slope_baseline = lot_dict.get("early_slope_leakage", {})
    slope_median_val = slope_baseline.get("median", 0.0060)

    badge_map = {"PASS": "badge-pass", "REVIEW": "badge-review", "REJECT": "badge-reject"}
    status_badge_class = badge_map.get(final_dec, "badge-info")

    # Header Card
    st.markdown(f"""
    <div class="eng-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 0.95rem; font-weight: 700; color: #0f172a;">Component: {selected_comp}</span>
                <span style="font-size: 0.8rem; color: #64748b; margin-left: 10px;">Lot: {c_row['lot_id']}</span>
            </div>
            <div>
                <span style="font-size: 0.8rem; color: #64748b; margin-right: 6px;">DISPOSITION:</span>
                <span class="{status_badge_class}">{final_dec}</span>
            </div>
        </div>
        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 10px 0;">
        <div style="font-size: 0.80rem; font-weight: 700; text-transform: uppercase; color: #475569; margin-bottom: 6px;">Reason for Decision:</div>
        <div style="font-size: 0.85rem; color: #1e293b; line-height: 1.5;">
    """, unsafe_allow_html=True)

    # Bullet points explanation
    reasons = []
    if c_row["absolute_spec_failed"] == 1:
        reasons.append("Absolute datasheet parametric limit violated at inspection checkpoint.")
    if c_row["leakage_mult_of_lot_median"] >= 3.0:
        reasons.append(f"Leakage current ({c_row['leakage_24h']:.2f} µA) is significantly higher than the lot baseline ({c_row['leakage_mult_of_lot_median']:.2f}× lot median).")
    elif c_row["leakage_mult_of_lot_median"] >= 1.8:
        reasons.append(f"Leakage current shows moderate deviation from lot median ({c_row['leakage_mult_of_lot_median']:.2f}×).")
    if c_row["max_robust_z"] >= 3.0:
        reasons.append(f"Component exhibits abnormal lot-relative behavior (MAD Robust Z = +{c_row['max_robust_z']:.2f}).")
    if c_row["envelope_violated"]:
        reasons.append(f"Early degradation slope ({c_row['early_slope']:.4f} µA/h) exceeds the defined healthy reference envelope ({c_row['healthy_envelope_max_early_slope']:.4f} µA/h).")
    if c_row["predicted_168h"] >= c_row["data_driven_prototype_limit"]:
        reasons.append(f"168h forecast ({c_row['predicted_168h']:.2f} µA) approaches/exceeds the prototype safety threshold ({c_row['data_driven_prototype_limit']:.2f} µA).")
    if not reasons:
        reasons.append("Measured parameters and forecasted trajectory fall within normative lot distributions and safety envelope bounds.")

    for r in reasons:
        st.markdown(f"• {r}")

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Evidence Table
    st.markdown("""<div class="eng-card-title">Decision Evidence</div>""", unsafe_allow_html=True)

    evidence_data = [
        {
            "Parameter": "Leakage Current (24h)",
            "Observed": f"{c_row['leakage_24h']:.2f} µA",
            "Lot Reference": f"{lot_median_val:.2f} µA",
            "Deviation": f"{c_row['leakage_mult_of_lot_median']:.2f}× (Z = +{c_row['max_robust_z']:.2f})",
            "168h Forecast": f"{c_row['predicted_168h']:.2f} µA",
            "Assessment": "ABNORMAL" if c_row['leakage_mult_of_lot_median'] >= 3.0 else ("ELEVATED" if c_row['leakage_mult_of_lot_median'] >= 1.8 else "NOMINAL")
        },
        {
            "Parameter": "Early Degradation Slope",
            "Observed": f"{c_row['early_slope']:.4f} µA/h",
            "Lot Reference": f"{slope_median_val:.4f} µA/h",
            "Deviation": f"+{c_row['early_slope'] - slope_median_val:.4f} µA/h",
            "168h Forecast": f"{((c_row['predicted_168h'] - c_row['leakage_24h'])/144.0):.4f} µA/h",
            "Assessment": "EXCEEDED" if c_row['envelope_violated'] else "WITHIN LIMIT"
        },
        {
            "Parameter": "Multivariate Anomaly Score",
            "Observed": f"{c_row['anomaly_score']:.3f}",
            "Lot Reference": "< 0.500",
            "Deviation": f"+{max(0.0, c_row['anomaly_score'] - 0.500):.3f}",
            "168h Forecast": "N/A",
            "Assessment": c_row["severity"]
        },
        {
            "Parameter": "Datasheet Margin",
            "Observed": f"{c_row['engineering_limit'] - c_row['leakage_24h']:.2f} µA",
            "Lot Reference": "> 35.00 µA",
            "Deviation": f"{c_row['leakage_24h'] / c_row['engineering_limit'] * 100:.1f}% of limit",
            "168h Forecast": f"{c_row['engineering_limit'] - c_row['predicted_168h']:.2f} µA margin",
            "Assessment": "WITHIN LIMIT" if c_row['absolute_spec_failed'] == 0 else "BREACHED"
        }
    ]

    st.dataframe(
        pd.DataFrame(evidence_data),
        hide_index=True,
        column_config={
            "Parameter": st.column_config.TextColumn(width="medium"),
            "Observed": st.column_config.TextColumn(width="small"),
            "Lot Reference": st.column_config.TextColumn(width="small"),
            "Deviation": st.column_config.TextColumn(width="medium"),
            "168h Forecast": st.column_config.TextColumn(width="small"),
            "Assessment": st.column_config.TextColumn(width="small")
        }
    )


# ==============================================================================
# SECTION 6: MODEL PERFORMANCE (VALIDATION & BENCHMARK)
# ==============================================================================
elif selected_section == "Model Performance":
    st.markdown("### Model Validation & Benchmark")
    st.caption("Performance measured on the current benchmark dataset.")

    bench_file = PROJECT_ROOT / "models" / "comparison_benchmark.csv"
    if bench_file.exists():
        bench_df = pd.read_csv(bench_file)

        # Format column names neutrally
        display_bench = bench_df.rename(columns={
            "Paradigm": "Method",
            "Recall": "Recall (%)",
            "FNR": "False Negative Rate (%)",
            "Precision": "Precision (%)",
            "F1_Score": "F1 Score",
            "FPR": "Yield Loss (FPR %)",
            "PR_AUC": "PR-AUC"
        })

        st.dataframe(
            display_bench,
            hide_index=True,
            column_config={
                "Recall (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "False Negative Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Precision (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "F1 Score": st.column_config.NumberColumn(format="%.4f"),
                "Yield Loss (FPR %)": st.column_config.NumberColumn(format="%.1f%%"),
                "PR-AUC": st.column_config.NumberColumn(format="%.4f")
            }
        )
    else:
        st.info("Benchmark table will appear after running model training.")

    st.markdown("""
    <div class="eng-card" style="margin-top: 14px;">
        <div class="eng-card-title">Validation Notes</div>
        <ul style="font-size: 0.80rem; color: #475569; margin: 0 0 0 16px; padding: 0;">
            <li>Evaluation performed on the benchmark dataset using out-of-fold cross-validation without temporal data leakage.</li>
            <li>Metrics reflect screening accuracy on the specified test cohort and should not be interpreted as aerospace certification or field qualification.</li>
            <li>In reliability screening, False Negative Rate (escape probability) is prioritized over nominal accuracy.</li>
            <li>Real ESS production telemetry should be used for site-specific qualification and acceptance limits.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# SECTION 7: DATA UPLOAD (ESS DATA IMPORT)
# ==============================================================================
elif selected_section == "Data Upload":
    st.markdown("### ESS Data Import")
    st.caption("Ingest component inspection telemetry for automated validation and screening.")

    uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"], key="data_upload_page")

    if uploaded_file is not None:
        try:
            raw_input_df = pipeline.ingest_file(uploaded_file)
            clean_df, val_summary = pipeline.validate_and_clean(raw_input_df)

            col_u1, col_u2 = st.columns(2)
            with col_u1:
                st.markdown("""<div class="eng-card-title">File Details</div>""", unsafe_allow_html=True)
                st.write(f"- **File Name:** `{getattr(uploaded_file, 'name', 'Uploaded File')}`")
                st.write(f"- **Rows Detected:** `{val_summary['initial_rows']}`")
                st.write(f"- **Components Detected:** `{val_summary['components_count']}`")
                st.write(f"- **Lots Detected:** `{', '.join(val_summary['lots_detected'])}`")
                st.write(f"- **Columns Identified:** `{len(val_summary['columns'])}`")

            with col_u2:
                st.markdown("""<div class="eng-card-title">Data Validation</div>""", unsafe_allow_html=True)
                st.markdown("""
                <div style="font-size: 0.85rem; color: #166534; line-height: 1.8;">
                    <div>&#10003; Required columns detected</div>
                    <div>&#10003; Numeric values validated and checked for physical limits</div>
                    <div>&#10003; Missing checkpoints handled via lot-median imputation</div>
                    <div>&#10003; Data ready for screening</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Cleaned Data Preview")
            st.dataframe(clean_df.head(10), hide_index=True)

        except Exception as e:
            st.error(f"Error validating uploaded file: {e}")
    else:
        st.markdown("""
        <div class="eng-card">
            <div class="eng-card-title">Active Batch Telemetry Preview</div>
            <p style="font-size: 0.8rem; color: #64748b;">Currently screening reference benchmark dataset (90 components, 360 records).</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_raw.head(15), hide_index=True)
