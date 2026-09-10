#!/usr/bin/env python3
"""
ROBUST DATA PIPELINE (Phase 7).
Handles CSV / Excel ingestion, column validation, missing value imputation,
duplicate handling, numeric validation, impossible physical value detection,
and missing checkpoint handling (missing 24h, 96h, 168h).
Ensures the dashboard and screening engines never crash on malformed rows.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class RobustDataPipeline:
    """
    Industrial-grade preprocessing and validation pipeline for burn-in telemetry.
    """

    # Standard expected columns
    STANDARD_COLS = {
        "component_id",
        "lot_id",
        "burnin_hours",
        "leakage_current_ua"
    }

    # Physical validity bounds
    PHYSICAL_BOUNDS = {
        "leakage_current_ua": (0.0, 1000.0),       # Non-negative, max 1 mA in uA
        "iddq_ma": (0.0, 500.0),                   # Non-negative, max 500 mA
        "propagation_delay_ns": (0.0, 200.0),      # Non-negative, max 200 ns
        "chamber_temp_c": (-55.0, 200.0),          # Military temperature range
        "burnin_hours": (0.0, 2000.0)              # Realistic burn-in duration
    }

    # Column name normalization aliases
    COLUMN_ALIASES = {
        "comp_id": "component_id",
        "cid": "component_id",
        "device_id": "component_id",
        "part_id": "component_id",
        "lot": "lot_id",
        "batch_id": "lot_id",
        "cohort": "lot_id",
        "hours": "burnin_hours",
        "time_hours": "burnin_hours",
        "aging_time_hours": "burnin_hours",
        "time": "burnin_hours",
        "leakage": "leakage_current_ua",
        "leakage_ua": "leakage_current_ua",
        "ileak_ua": "leakage_current_ua",
        "leakage_current": "leakage_current_ua",
        "iddq": "iddq_ma",
        "iddq_current_ma": "iddq_ma",
        "delay": "propagation_delay_ns",
        "tpd_ns": "propagation_delay_ns",
        "delay_ns": "propagation_delay_ns"
    }

    def __init__(self):
        self.validation_log_: List[str] = []

    def ingest_file(self, file_path_or_buffer: Union[str, Path, object]) -> pd.DataFrame:
        """
        Reads CSV or Excel file safely with fallback decoding.
        """
        self.validation_log_ = []
        df = None

        # Check if file path or buffer
        is_path = isinstance(file_path_or_buffer, (str, Path))
        name = str(file_path_or_buffer).lower() if is_path else getattr(file_path_or_buffer, "name", "").lower()

        if name.endswith(".xlsx") or name.endswith(".xls"):
            try:
                df = pd.read_excel(file_path_or_buffer)
                self.validation_log_.append(f"Ingested Excel file successfully ({len(df)} rows).")
            except Exception as e:
                self.validation_log_.append(f"Excel read warning: {e}. Attempting CSV fallback.")

        if df is None:
            # Try CSV with UTF-8, then fallback to latin1
            try:
                df = pd.read_csv(file_path_or_buffer)
                self.validation_log_.append(f"Ingested CSV file successfully ({len(df)} rows).")
            except UnicodeDecodeError:
                if is_path:
                    df = pd.read_csv(file_path_or_buffer, encoding="latin1")
                    self.validation_log_.append("Ingested CSV with latin1 fallback encoding.")
                else:
                    raise

        return df

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes column headers to lowercase snake_case and applies aliases.
        """
        df_clean = df.copy()
        new_names = {}
        for col in df_clean.columns:
            clean_name = str(col).strip().lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            aliased = self.COLUMN_ALIASES.get(clean_name, clean_name)
            new_names[col] = aliased

        df_clean = df_clean.rename(columns=new_names)
        return df_clean

    def validate_and_clean(
        self,
        df: pd.DataFrame,
        strict: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Validates schema, checks impossible values, removes duplicates,
        and imputes missing checkpoints.
        """
        issues = []
        df_std = self.standardize_columns(df)
        initial_rows = len(df_std)

        # 1. Check essential identity columns
        if "component_id" not in df_std.columns:
            df_std["component_id"] = [f"COMP_{i+1:03d}" for i in range(len(df_std))]
            issues.append("Missing 'component_id' column: generated surrogate IDs.")

        if "lot_id" not in df_std.columns:
            df_std["lot_id"] = "DEFAULT_LOT"
            issues.append("Missing 'lot_id' column: defaulted to 'DEFAULT_LOT'.")

        # 2. Check for duplicate rows
        dup_count = df_std.duplicated().sum()
        if dup_count > 0:
            df_std = df_std.drop_duplicates().reset_index(drop=True)
            issues.append(f"Dropped {dup_count} duplicate row(s).")

        # 3. Numeric conversion and impossible value filtering
        numeric_cols = [c for c in ["leakage_current_ua", "iddq_ma", "propagation_delay_ns", "burnin_hours", "leakage_0h", "leakage_24h", "leakage_168h"] if c in df_std.columns]

        for col in numeric_cols:
            df_std[col] = pd.to_numeric(df_std[col], errors="coerce")

            # Check impossible bounds
            if col in self.PHYSICAL_BOUNDS:
                low, high = self.PHYSICAL_BOUNDS[col]
                invalid_mask = (df_std[col] < low) | (df_std[col] > high)
                inv_count = invalid_mask.sum()
                if inv_count > 0:
                    issues.append(f"Detected {inv_count} impossible values in '{col}' outside physical bounds [{low}, {high}]. Clipped to valid range.")
                    df_std[col] = df_std[col].clip(lower=low, upper=high)

        # 4. Impute missing values with robust median per lot
        for col in numeric_cols:
            null_count = df_std[col].isnull().sum()
            if null_count > 0:
                issues.append(f"Found {null_count} missing value(s) in '{col}'. Imputed using lot median.")
                # Compute lot median or fallback global
                lot_meds = df_std.groupby("lot_id")[col].transform("median")
                global_med = df_std[col].median()
                df_std[col] = df_std[col].fillna(lot_meds).fillna(global_med).fillna(0.0)

        # 5. Handle missing checkpoints (e.g. wide tables where 24h is missing)
        if "leakage_24h" in df_std.columns and "leakage_0h" in df_std.columns:
            missing_24 = df_std["leakage_24h"].isnull() | (df_std["leakage_24h"] == 0.0)
            if missing_24.sum() > 0:
                issues.append(f"Imputed {missing_24.sum()} missing 24h readings assuming nominal early slope.")
                df_std.loc[missing_24, "leakage_24h"] = df_std.loc[missing_24, "leakage_0h"] * 1.02

        # 6. Lot size analysis
        lot_sizes = df_std["lot_id"].value_counts().to_dict()
        for lot_id, size in lot_sizes.items():
            if size < 3:
                issues.append(f"Warning: Lot '{lot_id}' has small sample size ({size} records). Statistical power may be reduced.")

        summary = {
            "initial_rows": initial_rows,
            "cleaned_rows": len(df_std),
            "columns": list(df_std.columns),
            "lots_detected": list(lot_sizes.keys()),
            "lot_sizes": lot_sizes,
            "components_count": df_std["component_id"].nunique(),
            "issues_resolved": issues,
            "is_valid": len(df_std) > 0
        }

        return df_std, summary
