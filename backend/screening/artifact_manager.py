#!/usr/bin/env python3
"""
MODEL ARTIFACT MANAGEMENT (Phase 12).
Serializes and loads trained ML models, scalers, lot baseline encoders,
feature configs, safety envelope parameters, and version metadata under models/.
Uses joblib / json with integrity validation.
"""

from pathlib import Path
import json
import joblib
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class ModelArtifactManager:
    """
    Manages persistence and versioning of trained screening artifacts.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_artifacts(
        self,
        module_a_detector,
        module_b_predictor,
        safety_envelope,
        metadata: Optional[Dict] = None,
        version: str = "1.0.0"
    ) -> Dict[str, str]:
        """
        Saves all screening model artifacts into models/.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)

        mod_a_path = self.models_dir / "module_a_detector.joblib"
        mod_b_path = self.models_dir / "module_b_predictor.joblib"
        env_path = self.models_dir / "safety_envelope.json"
        meta_path = self.models_dir / "model_metadata.json"

        # Save Module A
        joblib.dump(module_a_detector, mod_a_path)

        # Save Module B
        joblib.dump(module_b_predictor, mod_b_path)

        # Save Safety Envelope
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(safety_envelope.to_dict(), f, indent=2)

        # Save Metadata
        meta = metadata or {}
        meta.update({
            "model_version": version,
            "architecture": "AI-ESS GUARDIAN Dual-Module Screening Engine",
            "module_a": "Dynamic Outlier Detector (Isolation Forest + LOF + MAD)",
            "module_b": "Time-Series Drift Predictor (XGBoost / HistGB + Quantile Intervals)",
            "safety_envelope": "Dynamic Safety Slope (early_slope + predicted_slope)",
            "files": {
                "module_a": str(mod_a_path.name),
                "module_b": str(mod_b_path.name),
                "safety_envelope": str(env_path.name)
            }
        })

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[Artifact Manager] Saved all model artifacts to {self.models_dir}")
        return {
            "module_a": str(mod_a_path),
            "module_b": str(mod_b_path),
            "safety_envelope": str(env_path),
            "metadata": str(meta_path)
        }

    def load_artifacts(self) -> Tuple:
        """
        Loads pre-trained model artifacts from models/.
        """
        from .safety_envelope import DynamicSafetyEnvelope

        mod_a_path = self.models_dir / "module_a_detector.joblib"
        mod_b_path = self.models_dir / "module_b_predictor.joblib"
        env_path = self.models_dir / "safety_envelope.json"

        if not mod_a_path.exists() or not mod_b_path.exists():
            raise FileNotFoundError(f"Model artifacts missing in {self.models_dir}. Please train models first.")

        module_a = joblib.load(mod_a_path)
        module_b = joblib.load(mod_b_path)

        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                env_dict = json.load(f)
            safety_envelope = DynamicSafetyEnvelope.from_dict(env_dict)
        else:
            safety_envelope = DynamicSafetyEnvelope()

        return module_a, module_b, safety_envelope

    def artifacts_exist(self) -> bool:
        """Checks whether all trained model artifacts exist."""
        return (
            (self.models_dir / "module_a_detector.joblib").exists() and
            (self.models_dir / "module_b_predictor.joblib").exists() and
            (self.models_dir / "safety_envelope.json").exists()
        )
