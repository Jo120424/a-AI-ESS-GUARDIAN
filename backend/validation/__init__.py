"""
Research Validation, Robustness, Ablation & Scientific Results Package.
Provides audit, error analysis, sensitivity testing, ablation study,
cost-sensitive trade-off modeling, and publication asset generation for Step 5.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STEP4_RESULTS_DIR = PROJECT_ROOT / "results" / "step4"
STEP5_RESULTS_DIR = PROJECT_ROOT / "results" / "step5"
STEP5_TABLES_DIR = STEP5_RESULTS_DIR / "tables"
STEP5_FIGURES_DIR = STEP5_RESULTS_DIR / "figures"

# Ensure output directories exist
STEP5_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STEP5_TABLES_DIR.mkdir(parents=True, exist_ok=True)
STEP5_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "PROJECT_ROOT",
    "STEP4_RESULTS_DIR",
    "STEP5_RESULTS_DIR",
    "STEP5_TABLES_DIR",
    "STEP5_FIGURES_DIR",
]
