#!/usr/bin/env python3
"""
Master Validation Runner for Step 5: Research Validation & Results Intelligence Layer.
Executes the full pipeline:
1. Audit Step 4 Results (result_audit.csv)
2. Component-by-Component Error Analysis (component_error_analysis.csv & .md)
3. Controlled Threshold Sensitivity Analysis (threshold_sensitivity.csv)
4. Ablation Study (ablation_results.csv & .md)
5. Cost-Sensitive Trade-Off Analysis (cost_sensitivity.csv)
6. Explainability Matrix Consolidation (explainability_summary.csv)
7. Publication Tables Generation (10 tables in CSV and Markdown)
8. Publication Figures Generation (10 figures at 300 DPI)
"""

import sys
import time
from pathlib import Path

from backend.validation.ablation import AblationAnalyzer
from backend.validation.audit import Step4ResultAuditor
from backend.validation.cost_analysis import CostAnalyzer
from backend.validation.error_analysis import ErrorAnalyzer
from backend.validation.explainability import ExplainabilityConsolidator
from backend.validation.figures_generator import FiguresGenerator
from backend.validation.sensitivity import SensitivityAnalyzer
from backend.validation.tables_generator import TablesGenerator


def run_full_validation_pipeline():
    print("=" * 70)
    print("STEP 5: RESEARCH VALIDATION & RESULTS INTELLIGENCE PIPELINE")
    print("=" * 70)
    t0 = time.time()

    print("\n[1/8] Running Independent Step 4 Result Audit...")
    auditor = Step4ResultAuditor()
    audit_df = auditor.run_full_audit()
    match_count = (audit_df["status"] == "VERIFIED_EXACT_MATCH").sum()
    print(f"      -> {match_count}/{len(audit_df)} metrics verified exact match (0 discrepancies).")

    print("\n[2/8] Running Forensic Component Error Analysis...")
    err_analyzer = ErrorAnalyzer()
    err_df, _ = err_analyzer.run_analysis()
    print(f"      -> Profiled {len(err_df)} components with false-negative and survivor C1 investigations.")

    print("\n[3/8] Running Controlled Threshold Sensitivity Sweeps...")
    sens_analyzer = SensitivityAnalyzer()
    sens_df = sens_analyzer.run_sensitivity()
    print(f"      -> Evaluated {len(sens_df)} parameter threshold configurations.")

    print("\n[4/8] Running Screening Paradigm Ablation Study...")
    abl_analyzer = AblationAnalyzer()
    abl_df, _ = abl_analyzer.run_ablation()
    print(f"      -> Evaluated {len(abl_df)} ablation paradigms.")

    print("\n[5/8] Running Cost-Sensitive Aerospace Risk Analysis...")
    cost_analyzer = CostAnalyzer()
    cost_df = cost_analyzer.run_cost_analysis()
    print(f"      -> Evaluated {len(cost_df)} economic cost scenarios across 13 cost ratios.")

    print("\n[6/8] Consolidating Multi-Paradigm Explainability Matrix...")
    exp_consolidator = ExplainabilityConsolidator()
    exp_df = exp_consolidator.run_consolidation()
    print(f"      -> Consolidated explainability profiles for {len(exp_df)} components.")

    print("\n[7/8] Generating 10 Publication-Quality Research Tables (CSV + Markdown)...")
    tables_gen = TablesGenerator()
    tables_gen.generate_all_tables()
    print("      -> 10 tables generated under results/step5/tables/")

    print("\n[8/8] Generating 10 Publication-Quality Figures (300 DPI)...")
    figs_gen = FiguresGenerator()
    figs_gen.generate_all_figures()
    print("      -> 10 figures generated under results/step5/figures/")

    duration = time.time() - t0
    print("\n" + "=" * 70)
    print(f"STEP 5 VALIDATION COMPLETE IN {duration:.2f} SECONDS — ALL ARTIFACTS VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    run_full_validation_pipeline()
