#!/usr/bin/env python3
"""
Model B: Statistical / Lot-Relative Baseline.
Implements non-AI statistical outlier detection using:
1. Univariate Hampel / MAD Robust Z-scores.
2. Multivariate Regularized Mahalanobis Distance (Ledoit-Wolf shrinkage).
Provides a meaningful intermediate baseline between static limits and machine learning.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf


class StatisticalScreening:
    """
    Population-relative statistical screening engine.
    Calibrated strictly on training components (out-of-fold).
    """

    def __init__(
        self,
        mad_features: Optional[List[str]] = None,
        mahalanobis_features: Optional[List[str]] = None,
        mad_threshold: float = 2.5,
        alpha_significance: float = 0.05
    ):
        self.mad_features = mad_features or ["delta_capacitance_pct", "drift_velocity_c", "drift_velocity_esr"]
        self.mahalanobis_features = mahalanobis_features or ["delta_capacitance_pct", "delta_esr_pct", "drift_velocity_c"]
        self.mad_threshold = mad_threshold
        self.alpha_significance = alpha_significance

        # Fitted training statistics
        self.medians_: Dict[str, float] = {}
        self.mads_: Dict[str, float] = {}
        self.mean_vector_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None
        self.cov_estimator_: Optional[LedoitWolf] = None
        self.chi2_cutoff_: Optional[float] = None
        self.is_fitted_: bool = False

    def fit(self, X_train: pd.DataFrame) -> "StatisticalScreening":
        """
        Fits population statistics strictly on training components.
        """
        if len(X_train) < 2:
            raise ValueError(f"Need at least 2 training components to estimate statistical parameters, got {len(X_train)}")

        # 1. Fit Univariate Robust Z-score (MAD) parameters
        self.medians_ = {}
        self.mads_ = {}
        for feat in self.mad_features:
            vals = X_train[feat].astype(float).values
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            # Guard against zero MAD in homogeneous samples
            effective_mad = max(mad, 1e-4)
            self.medians_[feat] = med
            self.mads_[feat] = effective_mad

        # 2. Fit Multivariate Ledoit-Wolf Regularized Covariance
        X_maha = X_train[self.mahalanobis_features].astype(float).values
        p = X_maha.shape[1]
        self.mean_vector_ = np.mean(X_maha, axis=0)

        lw = LedoitWolf()
        lw.fit(X_maha)
        self.cov_estimator_ = lw

        # Compute regularized inverse covariance (precision matrix)
        cov_matrix = lw.covariance_
        # Add small ridge if necessary for numerical stability
        ridge = 1e-6 * np.eye(p)
        self.cov_inv_ = np.linalg.pinv(cov_matrix + ridge)

        # Critical chi-square cutoff for p degrees of freedom
        self.chi2_cutoff_ = float(stats.chi2.ppf(1.0 - self.alpha_significance, df=p))
        self.is_fitted_ = True
        return self

    def evaluate_component(self, comp_row: Dict[str, float]) -> Dict:
        """
        Evaluates an out-of-fold component against the fitted training statistics.
        """
        if not self.is_fitted_:
            raise RuntimeError("StatisticalScreening must be fitted before evaluating components.")

        cid = comp_row.get("component_id", "UNKNOWN")

        # 1. Compute Robust Z-scores
        z_scores = {}
        max_abs_z = 0.0
        z_violating_feats = []
        for feat in self.mad_features:
            val = float(comp_row.get(feat, 0.0))
            med = self.medians_[feat]
            mad = self.mads_[feat]
            z = (val - med) / (1.4826 * mad)
            z_scores[f"z_{feat}"] = round(float(z), 4)
            if abs(z) > max_abs_z:
                max_abs_z = abs(z)
            if abs(z) > self.mad_threshold:
                z_violating_feats.append((feat, z))

        mad_decision = 1 if len(z_violating_feats) > 0 else 0

        # 2. Compute Mahalanobis Distance
        x_vec = np.array([float(comp_row.get(f, 0.0)) for f in self.mahalanobis_features])
        diff = x_vec - self.mean_vector_
        d_m_sq = float(diff.T @ self.cov_inv_ @ diff)
        d_m = float(np.sqrt(max(d_m_sq, 0.0)))
        maha_decision = 1 if d_m_sq > self.chi2_cutoff_ else 0

        # Partial distance attribution per feature
        attributions = diff * (self.cov_inv_ @ diff)
        attr_dict = {
            f"maha_attr_{feat}": round(float(attr), 4)
            for feat, attr in zip(self.mahalanobis_features, attributions)
        }

        # Overall statistical decision (flagged if either MAD or Mahalanobis triggers)
        stat_decision = 1 if (mad_decision == 1 or maha_decision == 1) else 0

        # Generate transparent explanation
        reasons = []
        if mad_decision == 1:
            for feat, z in z_violating_feats:
                direction = "higher" if z > 0 else "lower"
                reasons.append(f"{feat} is {abs(z):.2f} MAD deviations {direction} than lot median")
        if maha_decision == 1:
            reasons.append(f"multivariate Mahalanobis D_M^2={d_m_sq:.2f} exceeds chi-square limit ({self.chi2_cutoff_:.2f})")

        if stat_decision == 1:
            explanation = f"STATISTICAL OUTLIER FLAGGED: {'; '.join(reasons)}."
        else:
            explanation = (
                f"STATISTICAL NORMAL: Telemetry aligns with lot distribution. "
                f"Max MAD Z-score = {max_abs_z:.2f} (<= {self.mad_threshold}), "
                f"Mahalanobis D_M^2 = {d_m_sq:.2f} (<= {self.chi2_cutoff_:.2f})."
            )

        result = {
            "component_id": cid,
            "statistical_decision": stat_decision,
            "statistical_status": "OUTLIER" if stat_decision == 1 else "NORMAL",
            "mad_decision": mad_decision,
            "mahalanobis_decision": maha_decision,
            "max_abs_zscore": round(max_abs_z, 4),
            "mahalanobis_dist_sq": round(d_m_sq, 4),
            "mahalanobis_dist": round(d_m, 4),
            "chi2_cutoff": round(self.chi2_cutoff_, 4),
            "contributing_parameters": [f[0] for f in z_violating_feats],
            "explanation": explanation
        }
        result.update(z_scores)
        result.update(attr_dict)
        return result

    def evaluate_batch(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates a batch of test components.
        """
        results = []
        for _, row in X_test.iterrows():
            res = self.evaluate_component(row.to_dict())
            results.append(res)
        return pd.DataFrame(results)
