#!/usr/bin/env python3
"""
Model C: AI/ML Unsupervised Anomaly Detection.
Implements:
1. Primary: Isolation Forest (iForest) - Tree-based recursive spatial isolation.
2. Secondary: One-Class Support Vector Machine (OC-SVM) - Non-linear RBF kernel boundary.
Trained strictly out-of-fold on training components (no test-set contamination).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


class MLAnomalyScreening:
    """
    Unsupervised AI/ML Screening Engine.
    Detects complex multivariate degradation patterns before threshold breaches.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        n_estimators: int = 100,
        contamination: float = 0.20,
        random_state: int = 42,
        ocsvm_nu: float = 0.20,
        ocsvm_kernel: str = "rbf"
    ):
        self.features = features or [
            "delta_capacitance_pct",
            "delta_esr_pct",
            "drift_velocity_c",
            "drift_velocity_esr",
            "c_to_esr_ratio"
        ]
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.ocsvm_nu = ocsvm_nu
        self.ocsvm_kernel = ocsvm_kernel

        self.scaler = StandardScaler()
        self.iforest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.ocsvm = OneClassSVM(
            kernel=self.ocsvm_kernel,
            nu=self.ocsvm_nu,
            gamma="scale"
        )

        self.train_features_mean_: Optional[np.ndarray] = None
        self.train_features_std_: Optional[np.ndarray] = None
        self.iforest_threshold_: float = 0.0
        self.ocsvm_threshold_: float = 0.0
        self.is_fitted_: bool = False

    def fit(self, X_train: pd.DataFrame) -> "MLAnomalyScreening":
        """
        Fits scalers and anomaly models strictly on training components.
        """
        if len(X_train) < 2:
            raise ValueError(f"Need at least 2 training components to fit ML anomaly models, got {len(X_train)}")

        X_mat = X_train[self.features].astype(float).values
        self.train_features_mean_ = np.mean(X_mat, axis=0)
        self.train_features_std_ = np.std(X_mat, axis=0)
        # Prevent division by zero
        self.train_features_std_[self.train_features_std_ < 1e-6] = 1.0

        # Fit standard scaler and models
        X_scaled = self.scaler.fit_transform(X_mat)
        self.iforest.fit(X_scaled)
        self.ocsvm.fit(X_scaled)

        # Calibrate threshold on training distribution
        train_iforest_scores = -self.iforest.score_samples(X_scaled)
        self.iforest_threshold_ = float(np.percentile(train_iforest_scores, (1.0 - self.contamination) * 100))

        train_ocsvm_scores = -self.ocsvm.decision_function(X_scaled)
        self.ocsvm_threshold_ = float(np.percentile(train_ocsvm_scores, (1.0 - self.contamination) * 100))

        self.is_fitted_ = True
        return self

    def evaluate_component(self, comp_row: Dict[str, float]) -> Dict:
        """
        Evaluates an out-of-fold test component.
        """
        if not self.is_fitted_:
            raise RuntimeError("MLAnomalyScreening must be fitted before evaluating components.")

        cid = comp_row.get("component_id", "UNKNOWN")
        x_vec = np.array([float(comp_row.get(f, 0.0)) for f in self.features]).reshape(1, -1)
        x_scaled = self.scaler.transform(x_vec)

        # 1. Isolation Forest Anomaly Score
        # Negative of score_samples: higher value = more anomalous
        iforest_raw = float(-self.iforest.score_samples(x_scaled)[0])
        iforest_pred = int(self.iforest.predict(x_scaled)[0])  # -1 for anomaly, 1 for inlier
        iforest_flag = 1 if (iforest_pred == -1 or iforest_raw > self.iforest_threshold_) else 0

        # Sigmoidal normalization of anomaly score to [0, 1] range
        iforest_score_norm = float(1.0 / (1.0 + np.exp(-3.0 * (iforest_raw - self.iforest_threshold_))))

        # 2. One-Class SVM Anomaly Score
        ocsvm_raw = float(-self.ocsvm.decision_function(x_scaled)[0])
        ocsvm_pred = int(self.ocsvm.predict(x_scaled)[0])  # -1 for anomaly, 1 for inlier
        ocsvm_flag = 1 if (ocsvm_pred == -1 or ocsvm_raw > self.ocsvm_threshold_) else 0

        # 3. Feature Importance / Attribution
        # Standardized deviation from training population mean
        z_devs = np.abs((x_vec[0] - self.train_features_mean_) / self.train_features_std_)
        top_feat_idx = int(np.argmax(z_devs))
        top_feature = self.features[top_feat_idx]
        top_feat_z = float(z_devs[top_feat_idx])

        feature_attributions = {
            f"dev_{feat}": round(float(dev), 4)
            for feat, dev in zip(self.features, z_devs)
        }

        # Combined AI/ML Screening Decision
        # Primary decision driven by Isolation Forest
        ml_decision = iforest_flag

        # Formulate plain-language explanation
        if ml_decision == 1:
            explanation = (
                f"FLAGGED AS AI ANOMALY: Isolation Forest anomaly score ({iforest_raw:.3f}) exceeds threshold ({self.iforest_threshold_:.3f}). "
                f"Primary precursor signal: '{top_feature}' deviates {top_feat_z:.2f} standard deviations from training baseline."
            )
        else:
            explanation = (
                f"CLEARED BY AI: Telemetry falls within normative training subspace. "
                f"Isolation Forest score ({iforest_raw:.3f} <= {self.iforest_threshold_:.3f}). "
                f"Max feature deviation: {top_feat_z:.2f} std devs on '{top_feature}'."
            )

        result = {
            "component_id": cid,
            "aiml_decision": ml_decision,
            "aiml_status": "ANOMALY" if ml_decision == 1 else "NORMAL",
            "iforest_anomaly_score": round(iforest_raw, 4),
            "iforest_score_normalized": round(iforest_score_norm, 4),
            "iforest_threshold": round(self.iforest_threshold_, 4),
            "iforest_flag": iforest_flag,
            "ocsvm_anomaly_score": round(ocsvm_raw, 4),
            "ocsvm_threshold": round(self.ocsvm_threshold_, 4),
            "ocsvm_flag": ocsvm_flag,
            "primary_contributing_feature": top_feature,
            "max_feature_deviation_z": round(top_feat_z, 4),
            "explanation": explanation
        }
        result.update(feature_attributions)
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
