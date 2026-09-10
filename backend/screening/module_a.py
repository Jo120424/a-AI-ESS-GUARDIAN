#!/usr/bin/env python3
"""
MODULE A: Dynamic Outlier Detection (Phase 2).
Detects components that are abnormal RELATIVE to their lot or peer group,
even when they remain comfortably inside static absolute datasheet limits.

Combines:
1. Absolute specification checks
2. Lot-aware robust statistics (Median / MAD / Robust Z-scores)
3. Unsupervised ML: Isolation Forest and Local Outlier Factor (LOF)

Features:
- raw parameter values
- normalized lot-relative values
- deviation from lot median
- robust Z-scores
- delta 0h->24h, delta 24h->96h
- percentage drift
- early slope
- cross-parameter interactions (e.g. leakage to iddq ratio)

Avoids data leakage: lot baselines and model scalers are fitted strictly on training reference cohorts.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


class DynamicOutlierDetector:
    """
    Lot-aware dynamic anomaly screening engine combining robust statistics and unsupervised ML.
    """

    DEFAULT_FEATURES = [
        "leakage_current_ua",
        "iddq_ma",
        "propagation_delay_ns",
        "early_slope_leakage",
        "delta_0_24_leakage_pct",
        "leakage_to_iddq_ratio"
    ]

    DEFAULT_SPEC_LIMITS = {
        "leakage_current_ua": 50.0,
        "iddq_ma": 5.0,
        "propagation_delay_ns": 15.0
    }

    def __init__(
        self,
        features: Optional[List[str]] = None,
        spec_limits: Optional[Dict[str, float]] = None,
        mad_threshold: float = 3.0,
        contamination: float = 0.15,
        random_state: int = 42
    ):
        self.features = features or self.DEFAULT_FEATURES
        self.spec_limits = spec_limits or self.DEFAULT_SPEC_LIMITS
        self.mad_threshold = mad_threshold
        self.contamination = contamination
        self.random_state = random_state

        self.scaler = RobustScaler()
        self.iforest = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=min(15, 20),
            contamination=self.contamination,
            novelty=True
        )

        # Lot baselines (lot_id -> {param: {"median": val, "mad": val}})
        self.lot_baselines_: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.global_baselines_: Dict[str, Dict[str, float]] = {}
        self.iforest_thresh_: float = 0.50
        self.is_fitted_: bool = False

    @staticmethod
    def extract_features_from_panel(
        df: pd.DataFrame,
        t_early: float = 24.0
    ) -> pd.DataFrame:
        """
        Extracts causal screening features strictly up to t_early.
        Accepts wide or tidy panel formats.
        """
        if "burnin_hours" in df.columns:
            # Tidy format with burnin_hours
            components = df["component_id"].unique()
            rows = []
            for cid in components:
                sub = df[df["component_id"] == cid].sort_values("burnin_hours")
                sub_early = sub[sub["burnin_hours"] <= t_early]

                p0 = sub_early[sub_early["burnin_hours"] == 0.0]
                p24 = sub_early[sub_early["burnin_hours"] == t_early]

                leak_0 = float(p0["leakage_current_ua"].iloc[0]) if len(p0) > 0 else float(sub_early["leakage_current_ua"].iloc[0])
                leak_24 = float(p24["leakage_current_ua"].iloc[0]) if len(p24) > 0 else float(sub_early["leakage_current_ua"].iloc[-1])
                iddq_24 = float(p24["iddq_ma"].iloc[0]) if len(p24) > 0 else float(sub_early["iddq_ma"].iloc[-1])
                delay_24 = float(p24["propagation_delay_ns"].iloc[0]) if len(p24) > 0 else float(sub_early["propagation_delay_ns"].iloc[-1])
                lot = str(sub["lot_id"].iloc[0])

                early_slope = (leak_24 - leak_0) / t_early if t_early > 0 else 0.0
                pct_drift = ((leak_24 - leak_0) / max(leak_0, 1e-4)) * 100.0
                ratio = leak_24 / max(iddq_24 * 1000.0, 1.0)  # uA / uA

                rows.append({
                    "component_id": cid,
                    "lot_id": lot,
                    "leakage_0h": leak_0,
                    "leakage_24h": leak_24,
                    "leakage_current_ua": leak_24,
                    "iddq_ma": iddq_24,
                    "propagation_delay_ns": delay_24,
                    "early_slope_leakage": early_slope,
                    "delta_0_24_leakage_pct": pct_drift,
                    "leakage_to_iddq_ratio": ratio
                })
            return pd.DataFrame(rows)
        else:
            # Already feature-extracted wide format
            df_out = df.copy()
            if "early_slope_leakage" not in df_out.columns and "leakage_0h" in df_out.columns and "leakage_24h" in df_out.columns:
                df_out["early_slope_leakage"] = (df_out["leakage_24h"] - df_out["leakage_0h"]) / 24.0
            if "delta_0_24_leakage_pct" not in df_out.columns and "leakage_0h" in df_out.columns and "leakage_24h" in df_out.columns:
                df_out["delta_0_24_leakage_pct"] = ((df_out["leakage_24h"] - df_out["leakage_0h"]) / df_out["leakage_0h"].clip(lower=1e-4)) * 100.0
            if "leakage_to_iddq_ratio" not in df_out.columns and "leakage_current_ua" in df_out.columns and "iddq_ma" in df_out.columns:
                df_out["leakage_to_iddq_ratio"] = df_out["leakage_current_ua"] / (df_out["iddq_ma"].clip(lower=1e-4) * 1000.0)
            return df_out

    def fit(self, X_train: pd.DataFrame) -> "DynamicOutlierDetector":
        """
        Fits lot baselines, robust scalers, and ML anomaly models on training cohort.
        """
        if len(X_train) < 3:
            raise ValueError(f"Need at least 3 training records, got {len(X_train)}")

        # 1. Compute robust lot-aware baselines (Median and MAD per lot and global)
        params_to_track = ["leakage_current_ua", "iddq_ma", "propagation_delay_ns", "early_slope_leakage"]
        self.lot_baselines_ = {}
        self.global_baselines_ = {}

        for p in params_to_track:
            if p in X_train.columns:
                vals = X_train[p].astype(float).values
                g_med = float(np.median(vals))
                g_mad = float(np.median(np.abs(vals - g_med)))
                self.global_baselines_[p] = {
                    "median": round(g_med, 4),
                    "mad": round(max(g_mad, 1e-4), 4)
                }

        if "lot_id" in X_train.columns:
            for lot_id, lot_group in X_train.groupby("lot_id"):
                self.lot_baselines_[str(lot_id)] = {}
                for p in params_to_track:
                    if p in lot_group.columns:
                        vals = lot_group[p].astype(float).values
                        med = float(np.median(vals))
                        mad = float(np.median(np.abs(vals - med)))
                        self.lot_baselines_[str(lot_id)][p] = {
                            "median": round(med, 4),
                            "mad": round(max(mad, 1e-4), 4)
                        }

        # 2. Fit Robust Scaler and Unsupervised ML Models
        avail_features = [f for f in self.features if f in X_train.columns]
        self.features = avail_features
        X_mat = X_train[self.features].astype(float).values

        X_scaled = self.scaler.fit_transform(X_mat)
        self.iforest.fit(X_scaled)
        self.lof.fit(X_scaled)

        # Calibrate Isolation Forest score threshold from in-fold distribution
        raw_scores = -self.iforest.score_samples(X_scaled)
        self.iforest_thresh_ = float(np.percentile(raw_scores, (1.0 - self.contamination) * 100))

        self.is_fitted_ = True
        return self

    def evaluate_component(self, comp: Dict) -> Dict:
        """
        Evaluates a single component for dynamic lot-relative anomalies.
        """
        cid = str(comp.get("component_id", "UNKNOWN"))
        lot = str(comp.get("lot_id", "UNKNOWN_LOT"))

        # 1. Absolute Specification Checks
        abs_violations = []
        for param, limit in self.spec_limits.items():
            val = float(comp.get(param, 0.0))
            if val >= limit:
                abs_violations.append(f"{param} ({val:.2f} >= spec limit {limit:.2f})")
        abs_failed = 1 if len(abs_violations) > 0 else 0

        # 2. Lot-Aware Robust Z-scores & Median Multiplier
        lot_base = self.lot_baselines_.get(lot, self.global_baselines_)
        z_scores = {}
        multipliers = {}
        z_violators = []

        for param in ["leakage_current_ua", "iddq_ma", "propagation_delay_ns", "early_slope_leakage"]:
            if param in comp and param in lot_base:
                val = float(comp[param])
                med = lot_base[param]["median"]
                mad = lot_base[param]["mad"]
                z = (val - med) / (1.4826 * mad)
                z_scores[f"z_{param}"] = round(float(z), 3)

                mult = val / max(med, 1e-4) if med > 0 else 1.0
                multipliers[f"mult_{param}"] = round(float(mult), 2)

                # Directional check: significantly higher than lot median
                if z > self.mad_threshold:
                    z_violators.append((param, z, mult))

        # Max robust Z-score
        max_z = max(z_scores.values()) if z_scores else 0.0
        primary_abnormal_param = max(z_scores, key=z_scores.get).replace("z_", "") if z_scores else "None"
        leakage_mult = multipliers.get("mult_leakage_current_ua", 1.0)

        # Auto-derive missing kinetic and ratio features if raw measurements present
        comp_dict = dict(comp)
        l0 = float(comp_dict.get("leakage_0h", comp_dict.get("leakage_current_ua", 10.0)))
        l24 = float(comp_dict.get("leakage_24h", comp_dict.get("leakage_current_ua", l0)))
        iddq = float(comp_dict.get("iddq_ma", 1.5))
        if "early_slope_leakage" not in comp_dict:
            comp_dict["early_slope_leakage"] = (l24 - l0) / 24.0
        if "delta_0_24_leakage_pct" not in comp_dict:
            comp_dict["delta_0_24_leakage_pct"] = ((l24 - l0) / max(l0, 1e-4)) * 100.0
        if "leakage_to_iddq_ratio" not in comp_dict:
            comp_dict["leakage_to_iddq_ratio"] = l24 / max(iddq * 1000.0, 1.0)

        # 3. Unsupervised ML Anomaly Evaluation
        x_vec = np.array([float(comp_dict.get(f, 0.0)) for f in self.features]).reshape(1, -1)
        x_scaled = self.scaler.transform(x_vec)

        # Isolation Forest raw score
        iforest_raw = float(-self.iforest.score_samples(x_scaled)[0])
        # Normalized anomaly score mapped to [0, 1]
        norm_score = float(1.0 / (1.0 + np.exp(-4.0 * (iforest_raw - self.iforest_thresh_))))
        iforest_pred = int(self.iforest.predict(x_scaled)[0])
        iforest_flag = 1 if (iforest_pred == -1 or iforest_raw > self.iforest_thresh_) else 0

        # LOF anomaly prediction (-1 = outlier, 1 = inlier)
        lof_pred = int(self.lof.predict(x_scaled)[0])
        lof_flag = 1 if lof_pred == -1 else 0

        # Composite anomaly indicator
        statistical_outlier = 1 if len(z_violators) > 0 else 0
        ml_anomaly = 1 if (iforest_flag == 1 or lof_flag == 1) else 0
        composite_outlier = 1 if (abs_failed == 1 or statistical_outlier == 1 or ml_anomaly == 1) else 0

        # Severity Classification
        if abs_failed == 1 or max_z >= 5.0 or norm_score >= 0.75:
            severity = "SEVERE"
        elif statistical_outlier == 1 or ml_anomaly == 1 or max_z >= 2.5:
            severity = "SUSPICIOUS"
        else:
            severity = "NORMAL"

        # Formulate structured plain-language explanation
        reasons = []
        if abs_failed == 1:
            reasons.append("Absolute datasheet limit breach: " + "; ".join(abs_violations))
        if z_violators:
            for p, z_val, mult_val in z_violators:
                reasons.append(f"{p} is {mult_val:.1f}x lot median (Robust Z = +{z_val:.2f})")
        elif leakage_mult >= 2.0:
            reasons.append(f"Leakage is elevated at {leakage_mult:.1f}x lot baseline")

        if iforest_flag == 1:
            reasons.append(f"Multivariate behavior differs from healthy population (iForest score = {iforest_raw:.3f} > {self.iforest_thresh_:.3f})")

        early_slope = float(comp.get("early_slope_leakage", 0.0))
        if early_slope > 0.05:
            reasons.append(f"Rapid early drift detected ({early_slope:.4f} uA/h)")

        if not reasons:
            explanation = f"NORMAL: Telemetry aligns with lot distribution (max Z = +{max_z:.2f}, anomaly score = {norm_score:.2f})."
        else:
            explanation = f"ANOMALY ({severity}): " + "; ".join(reasons) + "."

        result = {
            "component_id": cid,
            "lot_id": lot,
            "anomaly_score": round(norm_score, 4),
            "iforest_raw_score": round(iforest_raw, 4),
            "iforest_threshold": round(self.iforest_thresh_, 4),
            "anomaly_label": composite_outlier,
            "severity": severity,
            "primary_abnormal_parameter": primary_abnormal_param,
            "max_robust_z": round(float(max_z), 3),
            "leakage_mult_of_lot_median": leakage_mult,
            "statistical_outlier_flag": statistical_outlier,
            "ml_anomaly_flag": ml_anomaly,
            "absolute_spec_failed": abs_failed,
            "explanation": explanation
        }
        result.update(z_scores)
        result.update(multipliers)
        return result

    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        features_df = self.extract_features_from_panel(df)
        results = [self.evaluate_component(r) for r in features_df.to_dict(orient="records")]
        return pd.DataFrame(results)
