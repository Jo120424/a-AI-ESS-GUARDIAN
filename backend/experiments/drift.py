#!/usr/bin/env python3
"""
Model D: Early Drift Prediction and Trajectory Forecasting.
Predicts end-of-life degradation (Delta C at t = 194.0h) using exclusively early
telemetry (t <= 47.0h). Evaluates whether forecasted degradation breaches the
MIL-PRF-62F 20% failure threshold, providing early lead-time warnings.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge


class DriftPredictor:
    """
    Time-Series Degradation Forecaster.
    Trained out-of-fold using early measurements to forecast downstream degradation.
    """

    def __init__(
        self,
        target_horizon_hours: float = 194.0,
        spec_limit: float = 20.0,
        random_state: int = 42
    ):
        self.target_horizon_hours = target_horizon_hours
        self.spec_limit = spec_limit
        self.random_state = random_state

        self.ridge_model = Ridge(alpha=1.0)
        self.gbr_model = GradientBoostingRegressor(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.1,
            random_state=self.random_state
        )
        self.feature_names = [
            "delta_c_24h",
            "delta_c_47h",
            "delta_esr_47h",
            "drift_velocity_c",
            "drift_velocity_esr",
            "drift_accel_c"
        ]
        self.is_fitted_ = False

    @staticmethod
    def extract_drift_features(df: pd.DataFrame, t_screen: float = 47.0) -> pd.DataFrame:
        """
        Extracts multi-point early trajectory features strictly up to t_screen.
        """
        components = sorted(df["component_id"].unique())
        rows = []
        for comp in components:
            sub = df[df["component_id"] == comp].sort_values("aging_time_hours")
            early = sub[sub["aging_time_hours"] <= t_screen]

            # Get points at 0, 24, 47 if available
            p0 = early[early["aging_time_hours"] == 0.0]
            p24 = early[early["aging_time_hours"] == 24.0]
            p47 = early[early["aging_time_hours"] == t_screen]

            c0 = float(p0["delta_capacitance_pct"].iloc[0]) if len(p0) > 0 else 0.0
            c24 = float(p24["delta_capacitance_pct"].iloc[0]) if len(p24) > 0 else 0.0
            c47 = float(p47["delta_capacitance_pct"].iloc[0]) if len(p47) > 0 else float(early["delta_capacitance_pct"].iloc[-1])
            esr47 = float(p47["delta_esr_pct"].iloc[0]) if len(p47) > 0 else float(early["delta_esr_pct"].iloc[-1])

            vel_early = (c24 - c0) / 24.0 if 24.0 > 0 else 0.0
            vel_late = (c47 - c24) / (t_screen - 24.0) if (t_screen - 24.0) > 0 else 0.0
            accel = vel_late - vel_early

            # Get actual target at 194h (for ground-truth comparison ONLY, not in features)
            p_final = sub[sub["aging_time_hours"] >= 194.0]
            c_final = float(p_final["delta_capacitance_pct"].iloc[0]) if len(p_final) > 0 else float(sub["delta_capacitance_pct"].iloc[-1])

            rows.append({
                "component_id": comp,
                "lot_id": sub["lot_id"].iloc[0],
                "delta_c_0h": c0,
                "delta_c_24h": c24,
                "delta_c_47h": c47,
                "delta_esr_47h": esr47,
                "drift_velocity_c": vel_late,
                "drift_velocity_esr": float(early["drift_velocity_esr"].iloc[-1]),
                "drift_accel_c": accel,
                "target_delta_c_194h": c_final
            })
        return pd.DataFrame(rows)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "DriftPredictor":
        """
        Fits regression models on training components.
        """
        if len(X_train) < 2:
            raise ValueError("Need at least 2 training components for drift forecasting.")

        X_mat = X_train[self.feature_names].astype(float).values
        y_vec = np.array(y_train, dtype=float)

        self.ridge_model.fit(X_mat, y_vec)
        self.gbr_model.fit(X_mat, y_vec)
        self.is_fitted_ = True
        return self

    def predict_component(self, comp_features: Dict[str, float]) -> Dict:
        """
        Predicts future degradation for an out-of-fold component.
        """
        if not self.is_fitted_:
            raise RuntimeError("DriftPredictor must be fitted before making predictions.")

        cid = comp_features.get("component_id", "UNKNOWN")
        x_vec = np.array([float(comp_features.get(f, 0.0)) for f in self.feature_names]).reshape(1, -1)

        # Predict with Ridge and GBR
        pred_ridge = float(self.ridge_model.predict(x_vec)[0])
        pred_gbr = float(self.gbr_model.predict(x_vec)[0])
        # Ensemble average prediction
        pred_future_c = 0.5 * (pred_ridge + pred_gbr)

        actual_future_c = float(comp_features.get("target_delta_c_194h", np.nan))
        abs_error = abs(pred_future_c - actual_future_c) if not np.isnan(actual_future_c) else np.nan
        signed_error = (pred_future_c - actual_future_c) if not np.isnan(actual_future_c) else np.nan

        # Safety threshold check
        exceeds_spec = 1 if pred_future_c >= self.spec_limit else 0
        forecast_margin = self.spec_limit - pred_future_c

        # Formulate plain-language explanation
        if exceeds_spec == 1:
            explanation = (
                f"DRIFT WARNING: Early drift trajectory forecasts end-of-life Delta C = {pred_future_c:.2f}%, "
                f"breaching the 20.0% MIL-PRF-62F limit by {pred_future_c - self.spec_limit:.2f}%. "
                f"Current drift rate = {comp_features.get('drift_velocity_c', 0.0):.4f}%/h."
            )
        else:
            explanation = (
                f"DRIFT SAFE: Forecasted end-of-life Delta C = {pred_future_c:.2f}%, "
                f"remaining safely below the 20.0% limit (forecast margin: {forecast_margin:.2f}%)."
            )

        return {
            "component_id": cid,
            "drift_decision": exceeds_spec,
            "drift_status": "FUTURE_VIOLATION" if exceeds_spec == 1 else "SAFE_TRAJECTORY",
            "predicted_delta_c_194h": round(pred_future_c, 4),
            "predicted_ridge_194h": round(pred_ridge, 4),
            "predicted_gbr_194h": round(pred_gbr, 4),
            "actual_delta_c_194h": round(actual_future_c, 4) if not np.isnan(actual_future_c) else None,
            "prediction_abs_error": round(abs_error, 4) if not np.isnan(abs_error) else None,
            "prediction_signed_error": round(signed_error, 4) if not np.isnan(signed_error) else None,
            "forecast_safety_margin": round(forecast_margin, 4),
            "explanation": explanation
        }

    def predict_batch(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts future degradation for a batch of components.
        """
        results = []
        for _, row in X_test.iterrows():
            res = self.predict_component(row.to_dict())
            results.append(res)
        return pd.DataFrame(results)
