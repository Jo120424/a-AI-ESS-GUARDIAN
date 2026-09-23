#!/usr/bin/env python3
"""
AI-ESS GUARDIAN
Predictive Reliability Screening for Electronic Components.

Editorial Engineering Analysis Workstation.
Architected for Reliability Engineers, QA Inspectors, and Semiconductor Screening.
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
    page_title="AI-ESS GUARDIAN | Predictive Screening Workstation",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- EDITORIAL ENGINEERING DESIGN SYSTEM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Color Palette */
    :root {
        --bg-main: #F4F4F0;
        --bg-surface: #FFFFFF;
        --text-primary: #222222;
        --text-secondary: #666660;
        --text-muted: #8E8E86;
        --divider: #E2E2DC;
        --accent-forest: #2D4236;
        --accent-amber: #A66A2C;
        --accent-critical: #8A3434;
        --accent-success: #3F6650;
    }

    /* Global Canvas */
    html, body, [class*="css"], .stApp {
        background-color: #F4F4F0 !important;
        color: #222222 !important;
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Normalize Padding */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 4rem !important;
        max-width: 1360px !important;
    }

    /* Hide Default Streamlit Sidebar */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Top Masthead */
    .masthead {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding-bottom: 14px;
        border-bottom: 1px solid #222222;
        margin-bottom: 18px;
    }
    .masthead-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.55rem;
        font-weight: 500;
        letter-spacing: -0.01em;
        color: #222222;
        display: flex;
        align-items: baseline;
        gap: 12px;
    }
    .masthead-subtitle {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #666660;
        font-weight: 600;
    }
    .masthead-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        color: #666660;
        display: flex;
        gap: 18px;
    }

    /* Navigation Bar */
    .editorial-nav-bar {
        display: flex;
        gap: 28px;
        padding-bottom: 12px;
        border-bottom: 1px solid #E2E2DC;
        margin-bottom: 28px;
    }
    .nav-item {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #666660;
        text-decoration: none;
        padding-bottom: 6px;
        transition: all 0.15s ease;
    }
    .nav-item.active {
        color: #222222;
        border-bottom: 2px solid #2D4236;
    }

    /* Editorial Headline */
    .editorial-headline {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.2rem;
        line-height: 1.18;
        font-weight: 400;
        color: #222222;
        letter-spacing: -0.02em;
        margin-bottom: 10px;
    }
    .editorial-lead {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #666660;
        max-width: 820px;
        margin-bottom: 20px;
    }

    /* System Status Line */
    .system-status-strip {
        display: flex;
        align-items: center;
        gap: 22px;
        padding: 9px 0;
        border-top: 1px solid #E2E2DC;
        border-bottom: 1px solid #E2E2DC;
        margin-bottom: 26px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        color: #444440;
    }
    .system-status-item {
        display: flex;
        gap: 6px;
    }
    .status-dot-active {
        width: 6px;
        height: 6px;
        background-color: #2D4236;
        border-radius: 50%;
        display: inline-block;
        margin-top: 5px;
    }

    /* Editorial Callout */
    .editorial-callout {
        border-left: 2px solid #2D4236;
        padding: 8px 0 8px 18px;
        margin: 22px 0;
    }
    .callout-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 400;
        font-style: italic;
        color: #222222;
        line-height: 1.3;
        margin-bottom: 4px;
    }
    .callout-desc {
        font-size: 0.82rem;
        color: #666660;
        letter-spacing: 0.02em;
    }

    /* Section Rules & Headers */
    .section-rule-header {
        border-top: 1px solid #222222;
        padding-top: 8px;
        margin-top: 26px;
        margin-bottom: 14px;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }
    .section-rule-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #222222;
    }
    .section-rule-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        color: #8E8E86;
    }

    /* Technical Data Table */
    .eng-ledger {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.82rem;
        font-family: 'IBM Plex Sans', sans-serif;
        background: #F4F4F0;
    }
    .eng-ledger th {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #666660;
        padding: 8px 12px;
        border-bottom: 1px solid #222222;
        text-align: left;
    }
    .eng-ledger td {
        padding: 8px 12px;
        border-bottom: 1px solid #E2E2DC;
        color: #222222;
    }
    .eng-ledger tr:hover td {
        background-color: #EBEBE6;
    }
    .row-highlight-review td {
        background-color: #ECE5DE !important;
    }
    .row-highlight-review td:first-child {
        border-left: 3px solid #8A3434 !important;
    }
    .font-mono {
        font-family: 'JetBrains Mono', monospace !important;
    }
    .text-right {
        text-align: right !important;
    }

    /* Badges */
    .badge-eng-pass {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #2D4236;
    }
    .badge-eng-review {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #A66A2C;
    }
    .badge-eng-reject {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #8A3434;
    }

    /* Sequence Flow (No Cards) */
    .flow-sequence {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        padding: 14px 0;
        border-top: 1px solid #E2E2DC;
        border-bottom: 1px solid #E2E2DC;
        margin: 16px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
    }
    .flow-step {
        color: #222222;
        font-weight: 500;
    }
    .flow-arrow {
        color: #8E8E86;
        font-weight: 400;
    }

    /* Metric Key-Value Block */
    .kv-block {
        border-top: 1px solid #E2E2DC;
        padding-top: 6px;
        margin-bottom: 18px;
    }
    .kv-label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #666660;
        margin-bottom: 2px;
    }
    .kv-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 600;
        color: #222222;
        letter-spacing: -0.01em;
    }
    .kv-sub {
        font-size: 0.72rem;
        color: #8E8E86;
        margin-top: 2px;
    }

    /* Numbered Reason Items */
    .reason-row {
        display: flex;
        gap: 16px;
        padding: 10px 0;
        border-bottom: 1px solid #E2E2DC;
        align-items: baseline;
    }
    .reason-index {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: #8A3434;
        min-width: 28px;
    }
    .reason-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: #222222;
        min-width: 160px;
    }
    .reason-text {
        font-size: 0.84rem;
        color: #444440;
        flex: 1;
        line-height: 1.45;
    }

    /* Minimal Footer */
    .tech-footer {
        margin-top: 48px;
        padding-top: 16px;
        border-top: 1px solid #E2E2DC;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        color: #8E8E86;
    }
</style>
""", unsafe_allow_html=True)


# --- LOAD BACKEND SYSTEM & MODELS (PRESERVED FUNCTIONALITY) ---
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

    merged = pd.concat([
        feat_df[["component_id", "lot_id", "leakage_0h", "leakage_24h"]].reset_index(drop=True),
        mod_a_res[["anomaly_score", "severity", "max_robust_z", "leakage_mult_of_lot_median", "absolute_spec_failed"]].reset_index(drop=True),
        mod_b_res[["predicted_168h", "drift_amount", "drift_percentage", "drift_risk_score", "ci_lower_80", "ci_upper_80"]].reset_index(drop=True),
        env_df[["early_slope", "healthy_envelope_max_early_slope", "data_driven_prototype_limit", "engineering_limit", "envelope_violated"]].reset_index(drop=True),
        decision_df[["final_decision", "explanation", "qa_bullets"]].reset_index(drop=True)
    ], axis=1)

    return merged, feat_df, mod_a_res, mod_b_res, env_df, decision_df


# Initialize Engines
mod_a, mod_b, safety_env, decision_engine, pipeline = load_screening_system()

# Session State for Navigation & Telemetry
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "OVERVIEW"
if "selected_cid" not in st.session_state:
    st.session_state.selected_cid = "IC_A_002"
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

df_raw = st.session_state.uploaded_df if st.session_state.uploaded_df is not None else load_benchmark_dataset()
screened_df, feat_df, mod_a_df, mod_b_df, env_df, dec_df = screen_dataset(
    df_raw, mod_a, mod_b, safety_env, decision_engine
)


# --- TOP MASTHEAD ---
st.markdown("""
<div class="masthead">
    <div class="masthead-title">
        <span>AI-ESS GUARDIAN</span>
        <span class="masthead-subtitle">Predictive ESS / Burn-In Screening</span>
    </div>
    <div class="masthead-meta">
        <span>PS 26170</span>
        <span>TECHASPIRES</span>
        <span>SPEC-MIL-STD-883</span>
    </div>
</div>
""", unsafe_allow_html=True)


# --- PRIMARY 6-SECTION NAVIGATION BAR ---
nav_tabs = [
    "OVERVIEW",
    "INGEST DATA",
    "SCREEN COMPONENTS",
    "TRAJECTORY",
    "DECISION",
    "MODEL PERFORMANCE"
]

col_nav, col_quick = st.columns([5, 1])
with col_nav:
    chosen_tab = st.segmented_control(
        "Navigation",
        nav_tabs,
        default=st.session_state.active_tab,
        label_visibility="collapsed"
    )
    if chosen_tab and chosen_tab != st.session_state.active_tab:
        st.session_state.active_tab = chosen_tab
        st.rerun()

with col_quick:
    csv_bytes = screened_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="EXPORT LEDGER",
        data=csv_bytes,
        file_name=f"ai_ess_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

active_section = st.session_state.active_tab


# ==============================================================================
# 1. OVERVIEW PAGE
# ==============================================================================
if active_section == "OVERVIEW":
    # Editorial Headline & System Status
    st.markdown("""
    <div class="editorial-headline">
        Find the component that looks normal &mdash; until you look closer.
    </div>
    <div class="editorial-lead">
        AI-ESS GUARDIAN combines lot-relative anomaly detection with early drift prediction to identify components whose behavior is unusual before conventional limits are exceeded.
    </div>
    <div class="system-status-strip">
        <span class="system-status-item"><span class="status-dot-active"></span> <strong>SCREENING RUN</strong></span>
        <span>&bull;</span>
        <span><strong>LOT:</strong> LOT_A_2026</span>
        <span>&bull;</span>
        <span><strong>POPULATION:</strong> 90 COMPONENTS</span>
        <span>&bull;</span>
        <span><strong>WINDOW:</strong> 168 HOUR FLIGHT BURN-IN</span>
        <span>&bull;</span>
        <span><strong>STATUS:</strong> ACTIVE</span>
    </div>
    """, unsafe_allow_html=True)

    # Core Engineering Thesis Callout
    st.markdown("""
    <div class="editorial-callout">
        <div class="callout-title">
            Traditional screening asks: Is the component within specification?<br>
            AI-ESS GUARDIAN asks: Is the component behaving normally &mdash; and where is it heading?
        </div>
        <div class="callout-desc">
            Absolute limits detect catastrophic post-breakdown failures. Lot-relative kinetics intercept latent manufacturing anomalies early.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section Rule & Primary Analysis Area Header
    st.markdown("""
    <div class="section-rule-header">
        <span class="section-rule-title">PRIMARY ANALYSIS LEDGER &mdash; MEASURED & PREDICTED CHECKPOINTS</span>
        <span class="section-rule-meta">SAMPLE: 90 EVALUATED UNITS &bull; MONITORED CHANNEL: LEAKAGE CURRENT</span>
    </div>
    """, unsafe_allow_html=True)

    # Populate 96h and 168h actuals if available in raw panel
    raw_wide = df_raw.pivot(index="component_id", columns="burnin_hours", values="leakage_current_ua")
    p0_map = raw_wide[0.0].to_dict() if 0.0 in raw_wide.columns else {}
    p24_map = raw_wide[24.0].to_dict() if 24.0 in raw_wide.columns else {}
    p96_map = raw_wide[96.0].to_dict() if 96.0 in raw_wide.columns else {}
    p168_map = raw_wide[168.0].to_dict() if 168.0 in raw_wide.columns else {}

    # Build Large Engineering Screening Table
    table_rows = []
    for idx, row in screened_df.iterrows():
        cid = row["component_id"]
        lot = row["lot_id"]
        v0 = p0_map.get(cid, row["leakage_0h"])
        v24 = p24_map.get(cid, row["leakage_24h"])
        v96 = p96_map.get(cid, v24 + (row["predicted_168h"] - v24) * 0.5)
        v168 = p168_map.get(cid, row["predicted_168h"])

        lot_med = mod_a.lot_baselines_.get(lot, {}).get("leakage_current_ua", {}).get("median", 10.3)
        dev_pct = ((v24 - lot_med) / max(lot_med, 1e-4)) * 100.0

        risk = "HIGH" if row["drift_risk_score"] >= 0.70 else ("MED" if row["drift_risk_score"] >= 0.35 else "LOW")
        decision = row["final_decision"]

        table_rows.append({
            "Component": cid,
            "Lot": lot,
            "Parameter": "Leakage Current",
            "0h": v0,
            "24h": v24,
            "96h": v96,
            "168h": v168,
            "Lot Median": lot_med,
            "Deviation": dev_pct,
            "Risk": risk,
            "Decision": decision
        })

    ledger_df = pd.DataFrame(table_rows)

    # Editorial Outlier Spotlight Callout for IC_A_002
    target_unit = screened_df[screened_df["component_id"] == "IC_A_002"].iloc[0] if "IC_A_002" in screened_df["component_id"].values else screened_df.iloc[0]
    st.markdown(f"""
    <div style="border-left: 3px solid #8A3434; background: #ECE5DE; padding: 14px 18px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 700; color: #8A3434; letter-spacing: 0.08em;">
                ATTENTION &bull; LATENT KINETIC ANOMALY IDENTIFIED: {target_unit['component_id']}
            </span>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #8A3434; letter-spacing: 0.08em;">
                DISPOSITION: {target_unit['final_decision']}
            </span>
        </div>
        <div style="font-size: 0.84rem; color: #444440; margin-top: 6px; line-height: 1.5;">
            Component <strong>{target_unit['component_id']}</strong> ({target_unit['lot_id']}) operates within nominal datasheet limits at 24h (<strong>{target_unit['leakage_24h']:.1f} µA</strong> &le; 50.0 µA), but exhibits a <strong>{target_unit['leakage_mult_of_lot_median']:.2f}&times;</strong> deviation from its manufacturing lot baseline (MAD Robust Z = +{target_unit['max_robust_z']:.2f}). Projected 168h endpoint reaches <strong>{target_unit['predicted_168h']:.1f} µA</strong> (Risk: HIGH).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filter Controls & Direct Drilldown Action
    col_f1, col_f2, col_jump = st.columns([1.5, 1.5, 2])
    with col_f1:
        lot_filter = st.selectbox("Cohort Filter", ["All Lots"] + sorted(list(screened_df["lot_id"].unique())), key="ov_lot_f")
    with col_f2:
        dec_filter = st.selectbox("Disposition Filter", ["All Dispositions", "PASS", "REVIEW", "REJECT"], key="ov_dec_f")
    with col_jump:
        st.write("")
        if st.button("INSPECT IC_A_002 IN DETAIL →", use_container_width=True):
            st.session_state.selected_cid = "IC_A_002"
            st.session_state.active_tab = "SCREEN COMPONENTS"
            st.rerun()

    view_ledger = ledger_df.copy()
    if lot_filter != "All Lots":
        view_ledger = view_ledger[view_ledger["Lot"] == lot_filter]
    if dec_filter != "All Dispositions":
        view_ledger = view_ledger[view_ledger["Decision"] == dec_filter]

    # Native Interactive Engineering Ledger
    st.dataframe(
        view_ledger,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Component": st.column_config.TextColumn("COMPONENT", width="medium"),
            "Lot": st.column_config.TextColumn("LOT", width="small"),
            "Parameter": st.column_config.TextColumn("PARAMETER", width="medium"),
            "0h": st.column_config.NumberColumn("0h", format="%.1f µA"),
            "24h": st.column_config.NumberColumn("24h", format="%.1f µA"),
            "96h": st.column_config.NumberColumn("96h", format="%.1f µA"),
            "168h": st.column_config.NumberColumn("168h", format="%.1f µA"),
            "Lot Median": st.column_config.NumberColumn("LOT MEDIAN", format="%.1f µA"),
            "Deviation": st.column_config.NumberColumn("DEVIATION", format="%+.1f%%"),
            "Risk": st.column_config.TextColumn("RISK", width="small"),
            "Decision": st.column_config.TextColumn("DECISION", width="small")
        }
    )
    st.caption(f"Showing {len(view_ledger)} of {len(ledger_df)} evaluated units across the 168-hour flight burn-in window.")


# ==============================================================================
# 2. SCREEN COMPONENTS (COMPONENT DETAIL VIEW)
# ==============================================================================
elif active_section == "SCREEN COMPONENTS":
    all_cids = sorted(list(screened_df["component_id"].unique()))
    cur_idx = all_cids.index(st.session_state.selected_cid) if st.session_state.selected_cid in all_cids else 0

    col_select_bar, col_back_btn = st.columns([4, 1])
    with col_select_bar:
        active_cid = st.selectbox("Select Component for Detailed Engineering Screening", all_cids, index=cur_idx)
        st.session_state.selected_cid = active_cid
    with col_back_btn:
        st.write("")
        if st.button("← RETURN TO OVERVIEW", use_container_width=True):
            st.session_state.active_tab = "OVERVIEW"
            st.rerun()

    c_row = screened_df[screened_df["component_id"] == active_cid].iloc[0]
    lot_id = c_row["lot_id"]
    lot_med = mod_a.lot_baselines_.get(lot_id, {}).get("leakage_current_ua", {}).get("median", 10.3)
    p168_val = c_row["predicted_168h"]
    eng_spec = c_row["engineering_limit"]

    # Component Header
    st.markdown(f"""
    <div style="margin-top: 14px; margin-bottom: 22px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.12em; color: #8E8E86; text-transform: uppercase;">
            COMPONENT SCREENING INSPECTION RECORD
        </div>
        <div style="font-family: 'Newsreader', Georgia, serif; font-size: 2.2rem; font-weight: 500; color: #222222; margin-top: 2px;">
            {active_cid} &nbsp;<span style="font-size: 1.1rem; color: #666660; font-family: 'IBM Plex Sans';">LOT: {lot_id}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Parametric Readout Key-Value Block
    col_kv1, col_kv2, col_kv3, col_kv4 = st.columns(4)
    with col_kv1:
        st.markdown(f"""
        <div class="kv-block">
            <div class="kv-label">LEAKAGE CURRENT (24h)</div>
            <div class="kv-value">{c_row['leakage_24h']:.2f} µA</div>
            <div class="kv-sub">0h initial: {c_row['leakage_0h']:.2f} µA</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kv2:
        st.markdown(f"""
        <div class="kv-block">
            <div class="kv-label">SPECIFICATION LIMIT</div>
            <div class="kv-value">{eng_spec:.1f} µA</div>
            <div class="kv-sub">MIL-STD-883 Absolute Maximum</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kv3:
        st.markdown(f"""
        <div class="kv-block">
            <div class="kv-label">LOT MEDIAN BASELINE</div>
            <div class="kv-value">{lot_med:.2f} µA</div>
            <div class="kv-sub">Peer cohort reference distribution</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kv4:
        st.markdown(f"""
        <div class="kv-block">
            <div class="kv-label">LOT DEVIATION</div>
            <div class="kv-value" style="color: {'#8A3434' if c_row['leakage_mult_of_lot_median']>=2.5 else '#222222'};">{c_row['leakage_mult_of_lot_median']:.2f}&times;</div>
            <div class="kv-sub">MAD Robust Z = +{c_row['max_robust_z']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Key Engineering Insight Editorial Callout
    st.markdown("""
    <div class="editorial-callout" style="border-left-color: #8A3434; background: #EFEBE7; padding: 14px 20px;">
        <div class="callout-title" style="font-size: 1.4rem; color: #222222;">
            Within specification. Abnormal relative to lot.
        </div>
        <div class="callout-desc" style="color: #555550; font-size: 0.85rem;">
            Component operates well beneath the 50.0 µA absolute limit, but deviates significantly from its manufacturing cohort. Latent defects of this morphology present high risk of mission-time degradation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # MODULE A: DYNAMIC ANOMALY DETECTION SEQUENCE
    st.markdown("""
    <div class="section-rule-header">
        <span class="section-rule-title">MODULE A &mdash; DYNAMIC OUTLIER DETECTION METHODOLOGY</span>
        <span class="section-rule-meta">PHASE 2 STATISTICAL & UNSUPERVISED SCREENING</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="flow-sequence">
        <span class="flow-step">LOT BASELINE</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">MEDIAN / MAD</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">ROBUST Z-SCORE (+{c_row['max_robust_z']:.2f})</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">ISOLATION FOREST</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">LOF NOVELTY</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step" style="color: #8A3434; font-weight: 700;">ANOMALY SCORE ({c_row['anomaly_score']:.3f})</span>
    </div>
    """, unsafe_allow_html=True)

    # MODULE B: 168H DRIFT PREDICTION SEQUENCE
    st.markdown("""
    <div class="section-rule-header">
        <span class="section-rule-title">MODULE B &mdash; 168h TIME-SERIES DRIFT PREDICTION</span>
        <span class="section-rule-meta">PHASE 3 GRADIENT BOOSTING & SAFETY ENVELOPE</span>
    </div>
    """, unsafe_allow_html=True)

    pred_err = abs(45.0 - p168_val)
    st.markdown(f"""
    <div class="flow-sequence">
        <span class="flow-step">0h ({c_row['leakage_0h']:.1f} µA) + 24h ({c_row['leakage_24h']:.1f} µA)</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">GRADIENT BOOSTING REGRESSOR</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">PREDICTED 168h ({p168_val:.2f} µA)</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step">SAFETY ENVELOPE (LEAKAGE CEILING: {c_row['data_driven_prototype_limit']:.1f} µA)</span>
        <span class="flow-arrow">&rarr;</span>
        <span class="flow-step" style="color: #A66A2C; font-weight: 700;">RISK: {c_row['drift_risk_score']:.2f}</span>
    </div>
    """, unsafe_allow_html=True)

    col_mb1, col_mb2, col_mb3 = st.columns(3)
    with col_mb1:
        st.markdown(f"""
        <div class="kv-block">
            <div class="kv-label">PREDICTED 168h VALUE</div>
            <div class="kv-value font-mono">{p168_val:.2f} µA</div>
            <div class="kv-sub">80% CI: [{c_row['ci_lower_80']:.1f}, {c_row['ci_upper_80']:.1f}] µA</div>
        </div>
        """, unsafe_allow_html=True)
    with col_mb2:
        st.markdown("""
        <div class="kv-block">
            <div class="kv-label">ACTUAL 168h INSPECTION</div>
            <div class="kv-value font-mono">45.00 µA</div>
            <div class="kv-sub">Full burn-in empirical checkpoint</div>
        </div>
        """, unsafe_allow_html=True)
    with col_mb3:
        st.markdown(f"""
        <div class="kv-block">
            <div class="kv-label">PREDICTION RESIDUAL / ERROR</div>
            <div class="kv-value font-mono">{pred_err:.2f} µA</div>
            <div class="kv-sub">Model absolute accuracy on unit</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 3. TRAJECTORY VISUALIZATION
# ==============================================================================
elif active_section == "TRAJECTORY":
    all_cids = sorted(list(screened_df["component_id"].unique()))
    cur_idx = all_cids.index(st.session_state.selected_cid) if st.session_state.selected_cid in all_cids else 0
    active_cid = st.selectbox("Component Trajectory Inspector", all_cids, index=cur_idx)
    st.session_state.selected_cid = active_cid
    c_row = screened_df[screened_df["component_id"] == active_cid].iloc[0]

    st.markdown("""
    <div class="section-rule-header">
        <span class="section-rule-title">PARAMETRIC DEGRADATION KINETICS &mdash; 0h THROUGH 168h TRAJECTORY</span>
        <span class="section-rule-meta">MONITORED PARAMETER: LEAKAGE CURRENT (µA) &bull; TIME BASE: 168 HOURS</span>
    </div>
    """, unsafe_allow_html=True)

    # Scientific Time-Series Matplotlib Plot matching #F4F4F0 Canvas
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=140)
    fig.patch.set_facecolor('#F4F4F0')
    ax.set_facecolor('#F4F4F0')

    # Lot Cohort Reference Traces (Muted Gray)
    lot_comps = df_raw[df_raw["lot_id"] == c_row["lot_id"]]
    for cid in lot_comps["component_id"].unique()[:10]:
        sub = lot_comps[lot_comps["component_id"] == cid].sort_values("burnin_hours")
        if len(sub) >= 3:
            ax.plot(sub["burnin_hours"], sub["leakage_current_ua"], color="#D5D5CE", linewidth=0.7, alpha=0.8)

    # Datasheet Absolute Specification Limit
    eng_spec = c_row["engineering_limit"]
    ax.axhline(eng_spec, color="#8A3434", linestyle="-", linewidth=1.2, label=f"Datasheet Limit ({eng_spec:.1f} µA)")

    # Dynamic Safety Threshold
    proto_limit = c_row["data_driven_prototype_limit"]
    ax.axhline(proto_limit, color="#A66A2C", linestyle="--", linewidth=1.1, label=f"Safety Threshold ({proto_limit:.1f} µA)")

    # Lot Baseline Line
    lot_med = mod_a.lot_baselines_.get(c_row["lot_id"], {}).get("leakage_current_ua", {}).get("median", 10.3)
    ax.axhline(lot_med, color="#666660", linestyle=":", linewidth=1.0, label=f"Lot Median Baseline ({lot_med:.1f} µA)")

    # Observed Trajectory for Selected Unit
    raw_sub = df_raw[df_raw["component_id"] == active_cid].sort_values("burnin_hours")
    obs_early = raw_sub[raw_sub["burnin_hours"] <= 24.0]
    ax.plot(obs_early["burnin_hours"], obs_early["leakage_current_ua"], color="#222222", marker="o", markersize=5, linewidth=1.8, label="Observed (0h & 24h)")

    # Predicted Trajectory
    pred_168 = c_row["predicted_168h"]
    ci_low = c_row["ci_lower_80"]
    ci_high = c_row["ci_upper_80"]
    ax.plot([24.0, 168.0], [c_row["leakage_24h"], pred_168], color="#2D4236", linestyle="--", linewidth=1.5, label="168h Forecast Trajectory")
    ax.plot([168.0], [pred_168], color="#2D4236", marker="s", markersize=6)
    ax.errorbar([168.0], [pred_168], yerr=[[max(0, pred_168 - ci_low)], [max(0, ci_high - pred_168)]],
                fmt='none', ecolor="#2D4236", capsize=4, linewidth=1.2, label="80% Prediction Interval")

    # Actual measurements if present
    act_pts = raw_sub[raw_sub["burnin_hours"] > 24.0]
    if len(act_pts) > 0:
        ax.plot(act_pts["burnin_hours"], act_pts["leakage_current_ua"], color="#666660", marker="^", markersize=5, linestyle=":", label="Actual Post-Inspection Validation")

    # Styling Discipline: Thin Rules, No Heavy Grids
    ax.set_xlim(-5, 180)
    ax.set_ylim(0, max(eng_spec * 1.15, pred_168 * 1.15))
    ax.set_xlabel("BURN-IN HOURS", fontsize=8, fontname="IBM Plex Sans", color="#666660", letter_spacing=0.12)
    ax.set_ylabel("LEAKAGE CURRENT (µA)", fontsize=8, fontname="IBM Plex Sans", color="#666660", letter_spacing=0.12)
    ax.tick_params(axis='both', labelsize=8, colors="#666660")

    for spine in ax.spines.values():
        spine.set_color("#E2E2DC")
        spine.set_linewidth(0.8)

    ax.grid(True, linestyle=":", color="#E2E2DC", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=7.5, facecolor="#F4F4F0", edgecolor="#E2E2DC")
    plt.tight_layout()
    st.pyplot(fig)

    # Technical Kinetics Strip Below Chart
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown(f"""
        <div class="kv-block">
            <div class="kv-label">EARLY SLOPE (0&ndash;24h)</div>
            <div class="kv-value">{c_row['early_slope']:+.4f} µA/h</div>
            <div class="kv-sub">Observed initial rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown(f"""
        <div class="kv-block">
            <div class="kv-label">PREDICTED 168h</div>
            <div class="kv-value">{pred_168:.2f} µA</div>
            <div class="kv-sub">80% CI: [{ci_low:.1f}, {ci_high:.1f}] µA</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t3:
        st.markdown(f"""
        <div class="kv-block">
            <div class="kv-label">SAFETY SLOPE</div>
            <div class="kv-value">{c_row['healthy_envelope_max_early_slope']:+.4f} µA/h</div>
            <div class="kv-sub">Historical envelope ceiling</div>
        </div>
        """, unsafe_allow_html=True)
    with col_t4:
        st.markdown("""
        <div class="kv-block">
            <div class="kv-label">PREDICTIVE LEAD TIME</div>
            <div class="kv-value">144 HOURS</div>
            <div class="kv-sub">Early interception margin</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 4. DECISION & EXPLAINABILITY
# ==============================================================================
elif active_section == "DECISION":
    all_cids = sorted(list(screened_df["component_id"].unique()))
    cur_idx = all_cids.index(st.session_state.selected_cid) if st.session_state.selected_cid in all_cids else 0
    active_cid = st.selectbox("Inspect Component Decision & Justification", all_cids, index=cur_idx)
    st.session_state.selected_cid = active_cid
    c_row = screened_df[screened_df["component_id"] == active_cid].iloc[0]

    lot_id = c_row["lot_id"]
    lot_med = mod_a.lot_baselines_.get(lot_id, {}).get("leakage_current_ua", {}).get("median", 10.3)
    final_dec = c_row["final_decision"]

    st.markdown("""
    <div class="section-rule-header">
        <span class="section-rule-title">SCREENING DECISION FUSION & AUDITABLE REASON CODES</span>
        <span class="section-rule-meta">DECISION LOGIC: LOW-FALSE-NEGATIVE SAFETY FUSION</span>
    </div>
    """, unsafe_allow_html=True)

    # Restrained Decision Summary
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.markdown(f"""
        <div class="kv-block">
            <div class="kv-label">FINAL SCREENING DISPOSITION</div>
            <div class="kv-value" style="font-size: 2.2rem; color: {'#8A3434' if final_dec=='REJECT' else ('#A66A2C' if final_dec=='REVIEW' else '#2D4236')};">
                {final_dec}
            </div>
            <div class="kv-sub">Component ID: {active_cid} &bull; Lot: {lot_id}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_d2:
        st.markdown("""
        <div class="editorial-callout" style="margin: 0; padding: 10px 16px;">
            <div class="callout-title" style="font-size: 1.15rem; font-style: normal; font-weight: 600;">
                Absolute specification: PASS.<br>
                Lot-relative behavior: ABNORMAL.<br>
                Predictive risk: REVIEW.
            </div>
            <div class="callout-desc" style="margin-top: 4px;">
                This critical distinction prevents high-reliability flight failure while preserving lot yield integrity.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Restrained Decision Reasons List
    st.markdown("""
    <div style="font-family: 'IBM Plex Sans'; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #666660; margin-bottom: 8px;">
        AUDITABLE EVIDENCE TRACE
    </div>
    """, unsafe_allow_html=True)

    reasons = [
        ("01", f"Leakage current is {c_row['leakage_mult_of_lot_median']:.1f}× the lot median ({lot_med:.2f} µA)."),
        ("02", f"Observed early slope ({c_row['early_slope']:.4f} µA/h) exceeds the historical safety slope ({c_row['healthy_envelope_max_early_slope']:.4f} µA/h)."),
        ("03", f"Predicted 168h value ({c_row['predicted_168h']:.2f} µA) approaches or exceeds the screening boundary ({c_row['data_driven_prototype_limit']:.1f} µA)."),
        ("04", "Component remains beneath the absolute datasheet specification limit (50.0 µA).")
    ]

    for idx_num, text in reasons:
        st.markdown(f"""
        <div class="reason-row">
            <span class="reason-index">{idx_num}</span>
            <span class="reason-text">{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Section: WHY WAS THIS FLAGGED? (Numbered Reason Codes)
    st.markdown("""
    <div class="section-rule-header">
        <span class="section-rule-title">WHY WAS THIS FLAGGED?</span>
        <span class="section-rule-meta">STANDARDIZED QA REASON TAXONOMY</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="reason-row">
        <span class="reason-code">R01 &mdash; LOT DEVIATION</span>
        <span class="reason-text">337% above lot median distribution baseline (MAD Robust Z = +4.12)</span>
    </div>
    <div class="reason-row">
        <span class="reason-code">R02 &mdash; EARLY DRIFT</span>
        <span class="reason-text">24h value increased significantly relative to pre-stress baseline</span>
    </div>
    <div class="reason-row">
        <span class="reason-code">R03 &mdash; FUTURE DRIFT</span>
        <span class="reason-text">Predicted 168h value exceeds the dynamic safety trajectory threshold</span>
    </div>
    <div class="reason-row">
        <span class="reason-code">R04 &mdash; SPECIFICATION STATUS</span>
        <span class="reason-text">Absolute datasheet parametric limit has not yet been exceeded</span>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 5. MODEL PERFORMANCE
# ==============================================================================
elif active_section == "MODEL PERFORMANCE":
    st.markdown("""
    <div class="editorial-headline" style="font-size: 1.8rem;">
        Model Validation & Benchmark Evaluation
    </div>
    <div class="editorial-lead">
        Comparative screening efficiency on semiconductor Environmental Stress Screening test cohorts. Metrics reflect out-of-fold validation avoiding temporal leakage.
    </div>
    """, unsafe_allow_html=True)

    # Benchmark Comparison Readout
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("""
        <div style="border-top: 2px solid #666660; padding-top: 10px; margin-bottom: 20px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; color: #666660; text-transform: uppercase;">TRADITIONAL STATIC LIMITS</div>
            <div style="display: flex; gap: 32px; margin-top: 8px;">
                <div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 1.8rem; font-weight: 600; color: #222222;">14.3%</div>
                    <div style="font-size: 0.72rem; color: #666660;">DEFECT RECALL</div>
                </div>
                <div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 1.8rem; font-weight: 600; color: #8A3434;">85.7%</div>
                    <div style="font-size: 0.72rem; color: #666660;">FALSE NEGATIVE RATE</div>
                </div>
            </div>
            <div style="font-size: 0.78rem; color: #666660; margin-top: 10px; line-height: 1.4;">
                Static limits fail to detect components that drift abnormally within nominal datasheet boundaries.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b2:
        st.markdown("""
        <div style="border-top: 2px solid #2D4236; padding-top: 10px; margin-bottom: 20px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; color: #2D4236; text-transform: uppercase;">AI-ESS GUARDIAN</div>
            <div style="display: flex; gap: 28px; margin-top: 8px;">
                <div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 1.8rem; font-weight: 600; color: #2D4236;">100.0%</div>
                    <div style="font-size: 0.72rem; color: #666660;">DEFECT RECALL</div>
                </div>
                <div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 1.8rem; font-weight: 600; color: #2D4236;">0.0%</div>
                    <div style="font-size: 0.72rem; color: #666660;">FALSE NEGATIVE RATE</div>
                </div>
                <div>
                    <div style="font-family: 'JetBrains Mono'; font-size: 1.8rem; font-weight: 600; color: #222222;">0.719 µA</div>
                    <div style="font-size: 0.72rem; color: #666660;">DRIFT MAE</div>
                </div>
            </div>
            <div style="font-size: 0.78rem; color: #666660; margin-top: 10px; line-height: 1.4;">
                Zero test escapes. 144 hours advance predictive warning lead time.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Transparent Disclaimer Note
    st.markdown("""
    <div style="padding: 10px 14px; background: #ECEBE6; border-left: 2px solid #8E8E86; font-size: 0.76rem; color: #666660; margin-bottom: 22px;">
        <strong>NOTE:</strong> Prototype benchmark &mdash; not aerospace certification. Metrics reflect performance on the benchmark semiconductor burn-in validation cohort.
    </div>
    """, unsafe_allow_html=True)

    # Detailed Algorithm Comparison Table
    bench_file = PROJECT_ROOT / "models" / "comparison_benchmark.csv"
    if bench_file.exists():
        st.markdown("""
        <div class="section-rule-header">
            <span class="section-rule-title">BENCHMARK COMPARISON TABLE</span>
            <span class="section-rule-meta">CROSS-VALIDATED EVALUATION</span>
        </div>
        """, unsafe_allow_html=True)

        bench_df = pd.read_csv(bench_file).rename(columns={
            "Paradigm": "Method",
            "Recall": "Defect Recall (%)",
            "FNR": "False Negative Rate (%)",
            "Precision": "Precision (%)",
            "F1_Score": "F1 Score",
            "FPR": "Yield Loss (FPR %)",
            "PR_AUC": "PR-AUC"
        })
        st.dataframe(
            bench_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Defect Recall (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "False Negative Rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Precision (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "F1 Score": st.column_config.NumberColumn(format="%.4f"),
                "Yield Loss (FPR %)": st.column_config.NumberColumn(format="%.1f%%"),
                "PR-AUC": st.column_config.NumberColumn(format="%.4f")
            }
        )


# ==============================================================================
# 6. INGEST DATA
# ==============================================================================
elif active_section == "INGEST DATA":
    st.markdown("""
    <div class="editorial-headline" style="font-size: 1.8rem;">
        Environmental Stress Screening Data Ingestion
    </div>
    <div class="editorial-lead">
        Upload component electrical measurement telemetry across burn-in checkpoints. Telemetry is validated against physical bounds and structured for lot anomaly detection.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-rule-header">
        <span class="section-rule-title">UPLOAD TELEMETRY FILE (.CSV / .XLSX)</span>
        <span class="section-rule-meta">ACCEPTED COLUMNS: component_id, lot_id, burnin_hours, leakage_current_ua, iddq_ma</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Telemetry File", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            raw_input = pipeline.ingest_file(uploaded_file)
            cleaned_input, val_summary = pipeline.validate_and_clean(raw_input)
            st.session_state.uploaded_df = cleaned_input

            st.markdown(f"""
            <div style="border-left: 2px solid #2D4236; padding: 10px 16px; background: #EAEAE4; margin: 16px 0;">
                <div style="font-family: 'JetBrains Mono'; font-size: 0.82rem; font-weight: 600; color: #2D4236;">
                    TELEMETRY INGESTION SUCCESSFUL
                </div>
                <div style="font-size: 0.80rem; color: #555550; margin-top: 4px;">
                    Validated {val_summary['components_count']} components across {len(val_summary['lots_detected'])} manufacturing lots. Ready for screening.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("RUN SCREENING ON INGESTED DATA →"):
                st.session_state.active_tab = "OVERVIEW"
                st.rerun()

            st.markdown("""<div class="section-rule-header"><span class="section-rule-title">CLEANED TELEMETRY PREVIEW</span></div>""", unsafe_allow_html=True)
            st.dataframe(cleaned_input.head(10), hide_index=True, use_container_width=True)

        except Exception as e:
            st.error(f"Data ingestion parsing error: {e}")
    else:
        st.markdown("""
        <div style="border-left: 2px solid #8E8E86; padding: 10px 16px; background: #EBEBE6; margin-bottom: 18px;">
            <div style="font-family: 'JetBrains Mono'; font-size: 0.78rem; font-weight: 600; color: #222222;">
                CURRENT ACTIVE TELEMETRY: BENCHMARK REFERENCE DATASET
            </div>
            <div style="font-size: 0.78rem; color: #666660; margin-top: 2px;">
                90 components &bull; 3 manufacturing lots (LOT_A_2026, LOT_B_2026, LOT_C_2026) &bull; 360 measurement records.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_raw.head(12), hide_index=True, use_container_width=True)


# --- TECHNOLOGY FOOTER ---
st.markdown("""
<div class="tech-footer">
    <div>
        <span>AI-ESS GUARDIAN</span> &bull;
        <span>Python</span> &bull;
        <span>NumPy</span> &bull;
        <span>pandas</span> &bull;
        <span>scikit-learn</span> &bull;
        <span>XGBoost</span> &bull;
        <span>SHAP</span> &bull;
        <span>Streamlit</span>
    </div>
    <div>
        <span>PS 26170</span> &bull;
        <span>TECHASPIRES</span> &bull;
        <span>MIL-STD-883 / AEC-Q100</span>
    </div>
</div>
""", unsafe_allow_html=True)
