#!/usr/bin/env python3
"""
MODULE B: Time-Series Drift Prediction (Phase 3).
Uses early measurements (Value_0h, Value_24h, and Value_96h when available)
to predict later component behavior, especially Value_168h.

Uses practical, robust tabular models:
- XGBoost Regressor (with HistGradientBoosting fallback)
- Quantile Regression / Prediction Intervals for Uncertainty Quantification.

Outputs for every component:
- predicted 168h value
- prediction error during evaluation
- drift amount
- drift percentage
- risk score
- uncertainty / confidence intervals
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class TimeSeriesDriftPredictor:
    """
    Predicts 168h parametric degradation from early checkpoints (0h and 24h).
    """

    FEATURE_NAMES = [
        "val_0h",
        "val_24h",
        "early_delta",
        "early_slope",
        "early_drift_pct",
        "lot_median_val_24h",
        "ratio_to_lot_median_24h"
    ]

    def __init__(
        self,
        target_param: str = "leakage_current_ua",
        spec_limit: float = 50.0,
        prototype_limit: float = 45.0,
        random_state: int = 42
    ):
        self.target_param = target_param
        self.spec_limit = float(spec_limit)
        self.prototype_limit = float(prototype_limit)
        self.random_state = random_state

        # Primary point predictor: XGBoost or GradientBoosting
        if HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=40,
                max_depth=3,
                learning_rate=0.08,
                random_state=self.random_state
            )
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=40,
                max_depth=3,
                random_state=self.random_state
            )

        # Lower and Upper Quantile Predictors for Confidence Intervals
        self.q_lower_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.10,
            n_estimators=30,
            max_depth=2,
            random_state=self.random_state
        )
        self.q_upper_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.90,
            n_estimators=30,
            max_depth=2,
            random_state=self.random_state
        )

        self.lot_medians_24h_: Dict[str, float] = {}
        self.global_median_24h_: float = 10.0
        self.mae_train_: float = 0.0
        self.rmse_train_: float = 0.0
        self.is_fitted_: bool = False

    @staticmethod
    def prepare_training_features(
        df: pd.DataFrame,
        param: str = "leakage_current_ua"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Builds wide training table from panel containing 0h, 24h, and 168h.
        """
        if "burnin_hours" in df.columns:
            comps = df["component_id"].unique()
            rows = []
            for cid in comps:
                sub = df[df["component_id"] == cid].sort_values("burnin_hours")
                p0 = sub[sub["burnin_hours"] == 0.0]
                p24 = sub[sub["burnin_hours"] == 24.0]
                p168 = sub[sub["burnin_hours"] >= 168.0]

                v0 = float(p0[param].iloc[0]) if len(p0) > 0 else float(sub[param].iloc[0])
                v24 = float(p24[param].iloc[0]) if len(p24) > 0 else float(sub[param].iloc[-1])
                v168 = float(p168[param].iloc[0]) if len(p168) > 0 else float(sub[param].iloc[-1])
                lot = str(sub["lot_id"].iloc[0])

                rows.append({
                    "component_id": cid,
                    "lot_id": lot,
                    "val_0h": v0,
                    "val_24h": v24,
                    "val_168h_target": v168
                })
            wide_df = pd.DataFrame(rows)
        else:
            wide_df = pd.DataFrame({
                "component_id": df["component_id"],
                "lot_id": df["lot_id"],
                "val_0h": df[f"leakage_0h"] if "leakage_0h" in df.columns else df["val_0h"],
                "val_24h": df[f"leakage_24h"] if "leakage_24h" in df.columns else df["val_24h"],
                "val_168h_target": df[f"leakage_168h"] if "leakage_168h" in df.columns else df["val_168h"]
            })

        # Feature engineering
        wide_df["early_delta"] = wide_df["val_24h"] - wide_df["val_0h"]
        wide_df["early_slope"] = wide_df["early_delta"] / 24.0
        wide_df["early_drift_pct"] = (wide_df["early_delta"] / wide_df["val_0h"].clip(lower=1e-4)) * 100.0

        # In-fold lot median
        lot_meds = wide_df.groupby("lot_id")["val_24h"].median().to_dict()
        wide_df["lot_median_val_24h"] = wide_df["lot_id"].map(lot_meds)
        wide_df["ratio_to_lot_median_24h"] = wide_df["val_24h"] / wide_df["lot_median_val_24h"].clip(lower=1e-4)

        X = wide_df[TimeSeriesDriftPredictor.FEATURE_NAMES].astype(float)
        y = wide_df["val_168h_target"].astype(float).values
        return wide_df, y

    def fit(self, wide_train_df: pd.DataFrame, y_train: np.ndarray) -> "TimeSeriesDriftPredictor":
        """
        Fits point prediction and quantile prediction models on training components.
        """
        if len(wide_train_df) < 3:
            raise ValueError(f"Need at least 3 components to fit drift model, got {len(wide_train_df)}")

        # Track lot medians
        self.lot_medians_24h_ = wide_train_df.groupby("lot_id")["val_24h"].median().to_dict()
        self.global_median_24h_ = float(wide_train_df["val_24h"].median())

        X_mat = wide_train_df[self.FEATURE_NAMES].astype(float).values
        y_vec = np.array(y_train, dtype=float)

        self.model.fit(X_mat, y_vec)
        self.q_lower_model.fit(X_mat, y_vec)
        self.q_upper_model.fit(X_mat, y_vec)

        # In-fold error metrics
        preds = self.model.predict(X_mat)
        abs_err = np.abs(preds - y_vec)
        self.mae_train_ = float(np.mean(abs_err))
        self.rmse_train_ = float(np.sqrt(np.mean(abs_err ** 2)))

        self.is_fitted_ = True
        return self

    def predict_component(
        self,
        comp_data: Dict,
        actual_168h: Optional[float] = None
    ) -> Dict:
        """
        Generates 168h prediction, confidence intervals, drift metrics, and risk score.
        """
        cid = str(comp_data.get("component_id", "UNKNOWN"))
        lot = str(comp_data.get("lot_id", "UNKNOWN_LOT"))

        v0 = float(comp_data.get("val_0h", comp_data.get("leakage_0h", 10.0)))
        v24 = float(comp_data.get("val_24h", comp_data.get("leakage_24h", v0)))

        early_delta = v24 - v0
        early_slope = early_delta / 24.0
        early_pct = (early_delta / max(v0, 1e-4)) * 100.0

        lot_med = self.lot_medians_24h_.get(lot, self.global_median_24h_)
        ratio_to_lot = v24 / max(lot_med, 1e-4)

        feat_vector = np.array([
            v0,
            v24,
            early_delta,
            early_slope,
            early_pct,
            lot_med,
            ratio_to_lot
        ]).reshape(1, -1)

        # 1. Point Prediction
        pred_168h = float(self.model.predict(feat_vector)[0]) if self.is_fitted_ else (v24 + early_slope * 144.0)
        # Physical lower bound
        pred_168h = max(pred_168h, v24)

        # 2. Prediction Intervals (Quantile Regression 10th and 90th percentiles)
        if self.is_fitted_:
            ci_lower = float(self.q_lower_model.predict(feat_vector)[0])
            ci_upper = float(self.q_upper_model.predict(feat_vector)[0])
            # Ensure proper ordering
            ci_lower = min(ci_lower, pred_168h)
            ci_upper = max(ci_upper, pred_168h)
        else:
            ci_lower = pred_168h * 0.92
            ci_upper = pred_168h * 1.08

        confidence_margin = ci_upper - ci_lower

        # 3. Drift calculations
        total_drift = pred_168h - v0
        total_drift_pct = (total_drift / max(v0, 1e-4)) * 100.0

        # 4. Error calculation if ground truth 168h is provided
        act_168 = actual_168h if actual_168h is not None else comp_data.get("val_168h_target", comp_data.get("leakage_168h", None))
        abs_err = None
        signed_err = None
        if act_168 is not None and not np.isnan(float(act_168)):
            act_168 = float(act_168)
            abs_err = round(abs(pred_168h - act_168), 4)
            signed_err = round(pred_168h - act_168, 4)

        # 5. Risk Score formulation [0.0 to 1.0]
        # Combines margin to specification limit and drift severity
        spec_ratio = pred_168h / self.spec_limit
        if spec_ratio >= 1.0:
            risk_score = 1.0
        elif spec_ratio >= 0.90:
            risk_score = 0.70 + 0.30 * ((spec_ratio - 0.90) / 0.10)
        elif spec_ratio >= 0.70:
            risk_score = 0.30 + 0.40 * ((spec_ratio - 0.70) / 0.20)
        else:
            risk_score = max(0.0, spec_ratio * 0.40)
        risk_score = round(float(min(max(risk_score, 0.0), 1.0)), 4)

        # Violation check
        breaches_spec = 1 if pred_168h >= self.spec_limit else 0
        breaches_proto = 1 if pred_168h >= self.prototype_limit else 0

        # Reason code
        if breaches_spec == 1:
            explanation = (
                f"PREDICTED SPEC VIOLATION: Forecasted 168h {self.target_param} = {pred_168h:.2f} "
                f"exceeds datasheet limit ({self.spec_limit:.2f}) by {pred_168h - self.spec_limit:.2f}. "
                f"Drift = +{total_drift:.2f} ({total_drift_pct:.1f}%). 80% CI: [{ci_lower:.2f}, {ci_upper:.2f}]."
            )
        elif breaches_proto == 1:
            explanation = (
                f"BORDERLINE PROTOTYPE RISK: Forecasted 168h {self.target_param} = {pred_168h:.2f} "
                f"approaches datasheet limit ({self.spec_limit:.2f}) and breaches prototype limit ({self.prototype_limit:.2f}). "
                f"Drift = +{total_drift:.2f} ({total_drift_pct:.1f}%)."
            )
        else:
            explanation = (
                f"SAFE FORECAST: Forecasted 168h {self.target_param} = {pred_168h:.2f} "
                f"remains comfortably within specification (safety margin = {self.spec_limit - pred_168h:.2f})."
            )

        return {
            "component_id": cid,
            "lot_id": lot,
            "predicted_168h": round(pred_168h, 4),
            "actual_168h": round(act_168, 4) if act_168 is not None else None,
            "prediction_error_abs": abs_err,
            "prediction_error_signed": signed_err,
            "drift_amount": round(total_drift, 4),
            "drift_percentage": round(total_drift_pct, 2),
            "drift_risk_score": risk_score,
            "ci_lower_80": round(ci_lower, 4),
            "ci_upper_80": round(ci_upper, 4),
            "confidence_margin": round(confidence_margin, 4),
            "breaches_spec_limit": breaches_spec,
            "breaches_prototype_limit": breaches_proto,
            "explanation": explanation
        }

    def predict_batch(self, wide_df: pd.DataFrame) -> pd.DataFrame:
        results = [self.predict_component(r) for r in wide_df.to_dict(orient="records")]
        return pd.DataFrame(results)
