#!/usr/bin/env python3
"""
AI-ESS GUARDIAN: Synthetic Demonstration Dataset Generator.
Generates an industrial semiconductor / IC burn-in screening dataset
modeling high-reliability microelectronics across 0h, 24h, 96h, and 168h checkpoints.

Parameters:
- leakage_current_ua (Datasheet max = 50.0 uA, Lot baseline ~10.0 uA)
- iddq_ma (Datasheet max = 5.0 mA, Lot baseline ~1.5 mA)
- propagation_delay_ns (Datasheet max = 15.0 ns, Lot baseline ~8.5 ns)
- operating_temp_c (85 C - 125 C elevated burn-in chamber temperature)

Clearly modeled defect archetypes:
1. HEALTHY_STABLE: Flat, nominal trajectories close to lot median.
2. MILD_DRIFT: Natural, safe thermal aging within acceptable safety envelope.
3. LATENT_DEFECT_LOT_OUTLIER: The Classic SIH Case (45 uA <= 50 uA limit -> Traditional PASS, but 4.5x lot median -> AI-ESS GUARDIAN REJECT).
4. ACCELERATING_DRIFT: Subtle early divergence at 24h predicting downstream failure at 168h.
5. SUDDEN_JUMP: Early micro-defect step-jump at 24h.
6. ABSOLUTE_SPEC_VIOLATION: Gross defect exceeding datasheet limits early.
7. LOT_TO_LOT_VARIATION: Distinct baseline offsets across manufacturing lots (LOT_A, LOT_B, LOT_C).
8. MEASUREMENT_NOISE: Realistic sensor / test-fixture Gaussian noise.

NOTE: Clearly labeled as a SYNTHETIC DEMONSTRATION & BENCHMARK DATASET for research & hackathon evaluation.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_CSV = OUTPUT_DIR / "burnin_semiconductor_screening.csv"
OUTPUT_METADATA = OUTPUT_DIR / "dataset_metadata.json"


def generate_burnin_dataset(seed: int = 42, n_per_lot: int = 30) -> pd.DataFrame:
    np.random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints = [0.0, 24.0, 96.0, 168.0]
    lots = {
        "LOT_A_2026": {"leakage_base": 10.0, "iddq_base": 1.45, "delay_base": 8.4, "temp": 125.0},
        "LOT_B_2026": {"leakage_base": 12.5, "iddq_base": 1.60, "delay_base": 8.7, "temp": 125.0},
        "LOT_C_2026": {"leakage_base": 9.2,  "iddq_base": 1.38, "delay_base": 8.2, "temp": 125.0},
    }

    # Datasheet Absolute Specification Limits (Engineering Limits)
    SPEC_LIMITS = {
        "leakage_current_ua": 50.0,
        "iddq_ma": 5.0,
        "propagation_delay_ns": 15.0,
    }

    records = []
    comp_counter = 1

    for lot_id, lot_params in lots.items():
        base_leak = lot_params["leakage_base"]
        base_iddq = lot_params["iddq_base"]
        base_delay = lot_params["delay_base"]
        temp_c = lot_params["temp"]

        for i in range(n_per_lot):
            comp_id = f"IC_{lot_id.split('_')[1]}_{comp_counter:03d}"
            comp_counter += 1

            # Assign Archetype
            # 65% Healthy Stable, 15% Mild Drift, 7% Latent Outlier (SIH Case),
            # 7% Accelerating Drift, 3% Sudden Jump, 3% Absolute Violation
            rand_val = np.random.rand()
            if rand_val < 0.60:
                archetype = "HEALTHY_STABLE"
                is_defect = 0
            elif rand_val < 0.75:
                archetype = "MILD_DRIFT"
                is_defect = 0
            elif rand_val < 0.85:
                archetype = "LATENT_DEFECT_LOT_OUTLIER"  # Crucial SIH benchmark case
                is_defect = 1
            elif rand_val < 0.93:
                archetype = "ACCELERATING_DRIFT"
                is_defect = 1
            elif rand_val < 0.97:
                archetype = "SUDDEN_JUMP"
                is_defect = 1
            else:
                archetype = "ABSOLUTE_SPEC_VIOLATION"
                is_defect = 1

            # Component random individual initial offsets
            comp_noise_leak = np.random.normal(0.0, 0.4)
            comp_noise_iddq = np.random.normal(0.0, 0.04)
            comp_noise_delay = np.random.normal(0.0, 0.1)

            c_leak_0 = base_leak + comp_noise_leak
            c_iddq_0 = base_iddq + comp_noise_iddq
            c_delay_0 = base_delay + comp_noise_delay

            # Generate trajectories across checkpoints
            leakage_vals = []
            iddq_vals = []
            delay_vals = []

            for t in checkpoints:
                measurement_noise = np.random.normal(0.0, 0.15)

                if archetype == "HEALTHY_STABLE":
                    # Minimal drift: ~0.005 uA/h
                    l_val = c_leak_0 + 0.006 * t + measurement_noise
                    i_val = c_iddq_0 + 0.0003 * t + np.random.normal(0, 0.01)
                    d_val = c_delay_0 + 0.001 * t + np.random.normal(0, 0.03)

                elif archetype == "MILD_DRIFT":
                    # Normal acceptable burn-in drift: ~0.015 uA/h
                    l_val = c_leak_0 + 0.018 * t + measurement_noise
                    i_val = c_iddq_0 + 0.0008 * t + np.random.normal(0, 0.01)
                    d_val = c_delay_0 + 0.003 * t + np.random.normal(0, 0.03)

                elif archetype == "LATENT_DEFECT_LOT_OUTLIER":
                    # CLASSIC SIH PROBLEM:
                    # Starts high relative to lot (e.g. 38 uA when lot is ~10 uA)
                    # Drifts towards 45 - 48 uA at 168h.
                    # ALWAYS stays <= 50.0 uA datasheet max!
                    # Traditional screening PASSES this!
                    start_leak = 37.0 + np.random.uniform(0, 4.0)
                    l_val = start_leak + 0.045 * t + measurement_noise
                    # Ensure stays strictly <= 49.5 uA at 168h so static pass is guaranteed
                    l_val = min(l_val, 48.9)
                    i_val = c_iddq_0 * 1.8 + 0.002 * t + np.random.normal(0, 0.02)
                    d_val = c_delay_0 + 0.010 * t + np.random.normal(0, 0.05)

                elif archetype == "ACCELERATING_DRIFT":
                    # Starts close to normal at 0h (11 uA), slight rise at 24h (15 uA),
                    # then accelerating thermal runaway breaches 50 uA at 168h (e.g. 56 uA).
                    drift_factor = 0.0015 * (t ** 2) / 24.0
                    l_val = c_leak_0 + 0.08 * t + drift_factor + measurement_noise
                    i_val = c_iddq_0 + 0.004 * t + np.random.normal(0, 0.02)
                    d_val = c_delay_0 + 0.015 * t + np.random.normal(0, 0.05)

                elif archetype == "SUDDEN_JUMP":
                    # Step jump at t=24h due to early micro-defect
                    jump = 22.0 if t >= 24.0 else 0.0
                    l_val = c_leak_0 + jump + 0.02 * t + measurement_noise
                    i_val = c_iddq_0 + (0.8 if t >= 24.0 else 0.0) + np.random.normal(0, 0.02)
                    d_val = c_delay_0 + 0.004 * t + np.random.normal(0, 0.04)

                elif archetype == "ABSOLUTE_SPEC_VIOLATION":
                    # Gross defect breaching datasheet limit early
                    l_val = 52.0 + 0.08 * t + measurement_noise
                    i_val = 5.2 + 0.005 * t + np.random.normal(0, 0.02)
                    d_val = 15.5 + 0.01 * t + np.random.normal(0, 0.05)

                leakage_vals.append(max(round(float(l_val), 3), 0.1))
                iddq_vals.append(max(round(float(i_val), 4), 0.01))
                delay_vals.append(max(round(float(d_val), 3), 1.0))

            # Add longitudinal records
            for idx, t in enumerate(checkpoints):
                l_cur = leakage_vals[idx]
                i_cur = iddq_vals[idx]
                d_cur = delay_vals[idx]

                # Static specification failure flag
                spec_fail = 1 if (
                    l_cur >= SPEC_LIMITS["leakage_current_ua"] or
                    i_cur >= SPEC_LIMITS["iddq_ma"] or
                    d_cur >= SPEC_LIMITS["propagation_delay_ns"]
                ) else 0

                records.append({
                    "component_id": comp_id,
                    "lot_id": lot_id,
                    "burnin_hours": float(t),
                    "leakage_current_ua": l_cur,
                    "iddq_ma": i_cur,
                    "propagation_delay_ns": d_cur,
                    "chamber_temp_c": temp_c,
                    "static_spec_fail": spec_fail,
                    "archetype": archetype,
                    "true_latent_defect": is_defect,
                    "leakage_0h": leakage_vals[0],
                    "leakage_24h": leakage_vals[1],
                    "leakage_96h": leakage_vals[2],
                    "leakage_168h": leakage_vals[3]
                })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    metadata = {
        "dataset_name": "AI-ESS Guardian Semiconductor Burn-In Benchmark Dataset",
        "type": "SYNTHETIC DEMONSTRATION DATASET",
        "description": "Longitudinal burn-in screening panel with 0h, 24h, 96h, and 168h checkpoints across 3 manufacturing lots.",
        "parameters": {
            "leakage_current_ua": {"unit": "uA", "spec_limit": 50.0, "nominal": 10.0},
            "iddq_ma": {"unit": "mA", "spec_limit": 5.0, "nominal": 1.5},
            "propagation_delay_ns": {"unit": "ns", "spec_limit": 15.0, "nominal": 8.5}
        },
        "checkpoints_hours": checkpoints,
        "total_records": len(df),
        "total_components": df["component_id"].nunique(),
        "lots": list(lots.keys()),
        "archetypes": df["archetype"].value_counts().to_dict(),
        "classic_sih_example_component": df[df["archetype"] == "LATENT_DEFECT_LOT_OUTLIER"]["component_id"].iloc[0]
    }

    with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Synthetic Generator] Generated {len(df)} records for {df['component_id'].nunique()} components.")
    print(f"                      Saved CSV to: {OUTPUT_CSV}")
    print(f"                      Saved Metadata to: {OUTPUT_METADATA}")
    return df


if __name__ == "__main__":
    generate_burnin_dataset()
